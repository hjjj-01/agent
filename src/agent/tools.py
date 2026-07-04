"""
LangChain 1.x 中定义工具的两种方式：
  方式1：@tool 装饰器（推荐，更简洁）:
    from langchain_core.tools import tool

    @tool
    def query_employee_work(node_id: str = "", date: str = "") -> str:
        \"\"\"查询员工工作量。当用户询问工作量时使用。\"\"\"
        return wms_client.query_employee_api(params)

  方式2：StructuredTool.from_function（更灵活）：
    from langchain_core.tools import StructuredTool

    StructuredTool.from_function(
        func=query_employee_work,
        name="wms_query_employee_work",
        description="查询员工工作量数据..."
    )

  本文件使用方式2（StructuredTool.from_function），更直观地展示
  name 和 description 的配置，适合学习理解 Tool 的工作原理。
  实际项目中推荐使用方式1（@tool），代码更简洁。

Agent 使用工具的过程（示例）:
  用户: "查询今天的员工工作量"
    -> Agent 分析意图，看工具描述，识别需要使用 wms_query_employee_work
    -> 调用 query_employee_work(node_id=None, date="2026-07-02")
    -> 获取结果: "总记录数: 45条, 员工人数: 12人..."
    -> 基于结果生成回答: "今天共有12名员工完成工作..."

设计思路:
  1. 每个 WMS 功能方法包装成独立工具（职责单一）
  2. RAG 检索功能包装成知识查询工具
  3. 每个工具的 description 写清楚功能、适用场景和参数
  4. Agent 通过阅读 description 来决定调用哪个工具
"""

from typing import Optional, Dict, Any
from loguru import logger

# LangChain 1.x 工具定义
# StructuredTool：通过 from_function() 将普通函数包装成 Tool
# Tool 接口是 LangChain 1.x 中的标准工具接口
from langchain_core.tools import StructuredTool

from ..wms import WMSClient
from ..rag import RAGSystem


class WMSTools:
    """
    WMS 工具集合

    将 WMSClient 的功能包装成 LangChain 工具。

    为什么需要包装？
      WMSClient 是普通的 Python 类（有方法如 query_employee_api()），
      Agent（LLM）无法直接理解和调用 Python 对象的方法。
      必须通过 Tool 提供标准接口：
        - name: Agent 用来引用工具（如 "wms_query_employee_work"）
        - description: Agent 用来判断何时使用（自然语言描述）
        - func: 工具的实际执行逻辑

      类比：WMSClient 是一台机器，Tool 是这台机器的"使用说明书+按钮"，
      Agent 读了说明书，按了按钮，机器就干活了。

    设计原则：
      1. 每个方法返回字符串（不是对象），因为要直接给 LLM 阅读
      2. 返回的内容要结构化、易读（用列表、标题、分隔线）
      3. 参数全部设为 Optional（可选），让 Agent 灵活使用
      4. 错误不要抛出异常，而是返回错误信息字符串

    ID 映射查找机制（知识库驱动）：
      =============================================================
      WMS API 只认 ID，不认姓名/岗位名。

      知识库存放了员工姓名→worker_id 和岗位名→node_id 的映射表。
      查询流程：
        1. 用户说"张三的工作量"或"打包岗位的工作量"
        2. LLM 识别出员工姓名或岗位名，传入 worker_name 或 node_name
        3. 本工具先查知识库，找到对应的 worker_id 或 node_id
        4. 用找到的 ID 调用 WMS API 查询
        5. 如查具体员工，在返回结果中过滤该员工的数据
      =============================================================
    """

    def __init__(self, wms_client: WMSClient, rag_system: RAGSystem = None):
        """
        初始化 WMS 工具集

        Args:
            wms_client: WMS 客户端实例（已配置好 API 地址和 Token）
            rag_system: RAG 系统实例（用于知识库 ID 映射查找）
        """
        self.wms_client = wms_client
        self.rag_system = rag_system
        logger.info("WMS 工具集初始化完成" + (f"，已连接知识库" if rag_system else ""))

    def _lookup_id_from_knowledge(
        self,
        worker_name: str = None,
        node_name: str = None
    ) -> tuple:
        """
        从知识库中查找员工ID或岗位ID

        知识库存放了员工姓名→worker_id 和岗位名→node_id 的映射。
        当用户说"张三的工作量"或"打包岗位的工作量"时，
        先查知识库找到对应的 ID，再用 ID 查 WMS API。

        Args:
            worker_name: 员工姓名（可选，如"张三"、"LS-温明瑶"）
            node_name: 岗位名称（可选，如"打包"、"压包"、"备货打包"）

        Returns:
            (worker_id, node_id_list):
              worker_id: 员工ID（字符串），如果没找到返回 None
              node_id_list: 岗位ID列表，如果没找到返回 ["1928371092732805121"]（默认打包岗位）
        """
        default_node_id = "1928371092732805121"
        found_worker_id = None
        found_node_id_list = [default_node_id]

        # 如果提供了员工姓名，优先查员工ID
        if worker_name and self.rag_system:
            # 在知识库中检索该员工姓名，找到对应的 worker_id
            kb_results = self.rag_system.retrieve(
                f"员工 {worker_name} 的ID编号",
                k=3
            )
            for doc in kb_results:
                content = doc.page_content
                # 知识库格式：员工姓名\tworker_id（如"LS-温明瑶\t2072639244575629313"）
                if worker_name in content:
                    # 提取 worker_id（数字串）
                    import re
                    id_match = re.search(r'\d{16,}', content)
                    if id_match:
                        found_worker_id = id_match.group()
                        logger.info(f"从知识库找到员工 {worker_name} 的ID: {found_worker_id}")
                        break

        # 如果提供了岗位名称，查找岗位ID
        if node_name and self.rag_system:
            kb_results = self.rag_system.retrieve(
                f"岗位 {node_name} 的ID编号",
                k=3
            )
            for doc in kb_results:
                content = doc.page_content
                # 知识库格式：岗位名：node_id（如"打包：1928371092732805121"）
                # 也支持模糊匹配，如"备货打包"能匹配到"打包"
                if node_name in content or any(alias in content for alias in node_name.split()):
                    import re
                    id_match = re.search(r'\d{16,}', content)
                    if id_match:
                        found_node_id_list = [id_match.group()]
                        logger.info(f"从知识库找到岗位 {node_name} 的ID: {found_node_id_list}")
                        break

        return found_worker_id, found_node_id_list

    def _lookup_worker_name_from_knowledge(self, worker_id: str) -> str:
        """
        从知识库反向查找员工姓名（根据 worker_id 找姓名）

        WMS API 返回的 workerName 是 null，但知识库中存储了 姓名→ID 的映射。
        这个方法做反向查找：根据 worker_id，在知识库中找到对应的员工姓名。

        知识库格式（Tab分隔）：
            LS-温明瑶    2072639244575629313
            孙悦轩    2072197399236304898

        Args:
            worker_id: 员工ID（字符串，如 "2072639244575629313"）

        Returns:
            员工姓名，如果没找到返回 worker_id
        """
        if not worker_id or not self.rag_system:
            return worker_id

        kb_results = self.rag_system.retrieve(
            f"员工ID {worker_id} 对应的姓名",
            k=3
        )

        for doc in kb_results:
            content = doc.page_content
            # 知识库格式：Tab分隔 "姓名\tworker_id"
            # 查找包含该 worker_id 的行，然后提取姓名
            if worker_id in content:
                import re
                # 匹配 "姓名\t数字ID" 或 "姓名 数字ID" 格式
                # 姓名在ID前面的部分
                parts = re.split(r'[\t\s]+', content.strip())
                for i, part in enumerate(parts):
                    if part == worker_id and i > 0:
                        # worker_id 前面的部分就是姓名
                        name = parts[i - 1].strip()
                        if name and not name.isdigit():
                            logger.info(f"从知识库找到 worker_id {worker_id} 的姓名: {name}")
                            return name
                    # 也支持 "数字ID\t姓名" 格式
                    if i < len(parts) - 1 and part == worker_id:
                        name = parts[i + 1].strip()
                        if name and not name.isdigit():
                            logger.info(f"从知识库找到 worker_id {worker_id} 的姓名: {name}")
                            return name

        return worker_id

    def query_employee_work(
        self,
        node_id: Optional[str] = None,
        node_name: Optional[str] = None,
        worker_name: Optional[str] = None,
        date: Optional[str] = None
    ) -> str:
        """
        员工工作量查询工具的实现函数

        =============================================================
        查询流程（知识库驱动）：
        =============================================================

          用户说"打包岗位的工作量"：
            1. LLM 识别出岗位名 = "打包"，调用此工具
            2. _lookup_id_from_knowledge(node_name="打包")
               → 查知识库，找到 node_id = "1928371092732805121"
            3. 调用 WMS API: nodeIdList=["1928371092732805121"]
            4. 返回打包岗位所有人的数据
        =============================================================

        WMS API 请求参数（来自 models.EmployeeApiRequest）：
          - pageNo: 页码（从1开始）
          - pageSize: 每页数量（默认100）
          - nodeIdList: 岗位 ID 列表（如 ["1928371049917349889"]）
          - endTime: 结束时间范围，格式 ['YYYY-MM-DD HH:mm:ss', 'YYYY-MM-DD HH:mm:ss']

        WMS API 返回数据（来自 models.EmployeeApiResponse）：
          - code: 状态码（0表示成功）
          - data.list: WorkRecord 列表
          - data.total: 总记录数

        WorkRecord 包含字段：
          createTime, worker, workerName, nodeName, nodeCode,
          workQty, outputQty, unit, itemQty, weight, volume,
          listingCount, skuCategory, packCount
        """
        logger.info(
            f"执行员工工作量查询，"
            f"岗位ID: {node_id}, 岗位名: {node_name}, "
            f"员工名: {worker_name}, 日期: {date}"
        )

        try:
            import datetime
            from ..wms.models import EmployeeApiRequest

            # 第1步：如果传了 worker_name 或 node_name，先查知识库找 ID
            worker_id, resolved_node_list = self._lookup_id_from_knowledge(
                worker_name=worker_name,
                node_name=node_name
            )

            # 第2步：确定最终查询参数
            # node_id（直接传） > node_name（知识库查） > 默认岗位
            query_date = date or datetime.datetime.now().strftime("%Y-%m-%d")
            final_node_id_list = [node_id] if node_id else resolved_node_list

            params = EmployeeApiRequest(
                pageNo=1,
                pageSize=100,
                nodeIdList=final_node_id_list,
                endTime=[f"{query_date} 00:00:00", f"{query_date} 23:59:59"]
            )

            # 第3步：调用 WMS API
            result = self.wms_client.query_employee_api(params)

            # 第4步：格式化返回结果
            if not result or not hasattr(result, 'data') or not result.data.list:
                return f"没有找到 {query_date} 的员工工作量数据，可能当天没有工作记录。"

            all_records = result.data.list
            total = result.data.total

            # 第5步：如果指定了员工姓名，过滤出该员工的数据
            if worker_id:
                records = [r for r in all_records if r.worker == worker_id]
                if not records:
                    return (
                        f"在知识库中找到了员工 {worker_name}（ID: {worker_id}），"
                        f"但在 {query_date} 的工作量数据中没有找到该员工的记录。"
                    )
                logger.info(f"知识库ID映射：从 {worker_name} → {worker_id}，过滤到 {len(records)} 条记录")
            else:
                records = all_records

            # 计算汇总
            total_work_qty = sum(r.workQty for r in records)
            total_output_qty = sum(r.outputQty for r in records)
            total_weight = sum(r.weight for r in records)
            total_volume = sum(r.volume for r in records)
            unique_workers = set(r.worker for r in records)

            # 获取所有员工的姓名（从知识库反向查找）
            worker_names = {}
            for worker_id in unique_workers:
                name = self._lookup_worker_name_from_knowledge(worker_id)
                worker_names[worker_id] = name

            output_lines = [
                f"员工工作量统计（{query_date}）：",
            ]

            # 如果查的是具体员工，标题更明确
            if worker_name:
                output_lines[0] = f"员工「{worker_name}」工作量统计（{query_date}）："
                output_lines.append(f"  - 工作记录数: {len(records)} 条")
            elif node_name:
                output_lines[0] = f"「{node_name}」岗位工作量统计（{query_date}）："
                output_lines.extend([
                    f"\n【总体概览】",
                    f"  - 总记录数: {total} 条",
                    f"  - 员工人数: {len(unique_workers)} 人",
                ])
            else:
                output_lines.extend([
                    f"\n【总体概览】",
                    f"  - 总记录数: {total} 条",
                    f"  - 员工人数: {len(unique_workers)} 人",
                ])

            output_lines.extend([
                f"  - 总工作数量: {total_work_qty} 件",
                f"  - 总产出数量: {total_output_qty} 件",
                f"  - 产出率: {total_output_qty/total_work_qty*100:.1f}%" if total_work_qty > 0 else "  - 产出率: N/A",
                f"  - 总重量: {total_weight:.2f} KG",
                f"  - 总体积: {total_volume:.2f} m3",
            ])

            # 详细记录（按工作量降序排列，最多50条）
            display_records = sorted(records, key=lambda r: r.workQty, reverse=True)[:50]
            output_lines.append(f"\n【详细记录】（共{len(records)}条，按工作量降序）")

            for record in display_records:
                create_time = getattr(record, 'createTime', [])
                if create_time and len(create_time) >= 3:
                    date_str = f"{create_time[0]}-{create_time[1]:02d}-{create_time[2]:02d}"
                else:
                    date_str = query_date

                # 用知识库查找员工姓名
                worker_id = getattr(record, 'worker', '未知')
                worker_display_name = worker_names.get(worker_id, worker_id)

                output_lines.extend([
                    f"\n  员工: {worker_display_name}",
                    f"    员工ID: {worker_id}",
                    f"    岗位: {getattr(record, 'nodeName', '未知')} ({getattr(record, 'nodeCode', '')})",
                    f"    工作数量: {getattr(record, 'workQty', 0)} {getattr(record, 'unit', '件')}",
                    f"    产出数量: {getattr(record, 'outputQty', 0)} {getattr(record, 'unit', '件')}",
                    f"    重量: {getattr(record, 'weight', 0):.2f} KG",
                    f"    体积: {getattr(record, 'volume', 0):.2f} m3",
                    f"    上架数量: {getattr(record, 'listingCount', 0)}",
                    f"    打包数量: {getattr(record, 'packCount', 0)}",
                    f"    SKU类别: {getattr(record, 'skuCategory', 0)}",
                ])

            return "\n".join(output_lines)

        except Exception as e:
            logger.error(f"员工工作量查询失败: {str(e)}")
            return f"员工工作量查询失败：{str(e)}"

    def get_tools(self) -> list:
        """
        获取所有 WMS 工具

        Returns:
            LangChain Tool 对象列表

        StructuredTool.from_function 参数说明：
          - func: 要包装的 Python 函数
          - name: 工具的唯一标识名称（Agent 用这个引用工具）
          - description: 自然语言描述，告诉 Agent：
              1. 这个工具是做什么的
              2. 什么时候应该使用它
              3. 需要什么参数
              4. 参数的含义

        description 的写法技巧：
          好的 description 能显著提升 Agent 的工具选择准确率：
            用完整的句子，不要缩写
            包含使用场景的关键词（"员工"、"工作量"、"KPI"）
            说明参数的用途（"可选参数：node_id（岗位ID），不提供则查询所有岗位"）
            给出使用示例（"示例：查询今天的员工工作量"）
        """
        tools = [
            StructuredTool.from_function(
                func=self.query_employee_work,
                name="wms_query_employee_work",
                description=(
                    "查询 WMS 员工工作量数据。\n"
                    "当用户询问岗位作业情况、员工工作统计、工作量统计、"
                    "打包数量、工作产出、重量统计、KPI 信息时使用此工具。\n"
                    "\n"
                    "重要——知识库ID映射机制：\n"
                    "  WMS API 只认ID（如 1928371092732805121），不认姓名/岗位名。\n"
                    "  知识库存放了员工姓名→worker_id 和岗位名→node_id 的映射表。\n"
                    "  工具内部会自动查知识库完成ID映射，并从知识库反向查找员工姓名。\n"
                    "\n"
                    "参数（全部可选，传岗位名或员工名即可自动查知识库）：\n"
                    "  - node_name：岗位名称，如'备货打包'、'打包'、'压包'，会查知识库找岗位ID\n"
                    "  - node_id：岗位ID（直接传ID，不查知识库），与 node_name 二选一\n"
                    "  - worker_name：员工姓名，如'张三'、'LS-温明瑶'，会查知识库找员工ID\n"
                    "  - date：查询日期，格式 YYYY-MM-DD，默认今天\n"
                    "\n"
                    "返回：总记录数、员工人数、总工作数量、总产出数量、产出率、总重量、总体积，"
                    "以及每条详细记录（员工姓名、岗位、工作数量、产出数量、重量、体积等）。\n"
                    "\n"
                    "示例用法：\n"
                    "  - 查询备货打包岗位今天的工作量：node_name=备货打包\n"
                    "  - 查询压包岗位的工作量：node_name=压包\n"
                    "  - 查询张三的工作量：worker_name=张三\n"
                    "  - 查询打包岗位今天的数据：node_name=打包, date=2026-07-02\n"
                    "  - 查询温明瑶的工作量：worker_name=LS-温明瑶"
                ),
            ),
        ]

        logger.info(f"创建了 {len(tools)} 个 WMS 工具")
        return tools


class RAGTools:

    def __init__(self, rag_system: RAGSystem):
        """
        初始化 RAG 工具集

        Args:
            rag_system: RAG 系统实例（已构建好知识库）
        """
        self.rag_system = rag_system
        logger.info("RAG 工具集初始化完成")

    def search_knowledge(self, query: str) -> str:
        """
        知识检索工具的实现函数

        Args:
            query: 查询文本（用户想问的问题）

        Returns:
            检索到的知识内容（前 500 字符的摘要）

        检索原理（RAG 的 R：Retrieval）：
          1. 将 query 向量化（转为数字向量）
          2. 在向量数据库中搜索最相似的文档块
          3. 返回最相关的前 k 个结果
          4. LLM 基于这些结果生成回答（RAG 的 G：Generation）

          向量搜索 vs 关键词搜索：
            - 关键词搜索："绩效考核" 匹配不到 "KPI怎么算"
            - 向量搜索：能理解语义相似性，"绩效考核" 约等于 "KPI怎么算"
        """
        logger.info(f"执行知识检索，查询: {query}")

        try:
            # 使用 RAG 系统检索相关文档
            # k=3 表示返回最相关的 3 个文档块
            documents = self.rag_system.retrieve(query, k=3)

            if not documents:
                return "未找到相关知识内容"

            # 将检索结果格式化为易读的文本
            output_lines = ["相关知识内容："]

            for i, doc in enumerate(documents):
                source = doc.metadata.get('source', 'unknown')
                title = doc.metadata.get('title', '未知')

                output_lines.append(f"\n【来源 {i+1}】: {source}")
                output_lines.append(f"  标题: {title}")
                output_lines.append(f"\n  内容摘要：")

                # 只返回前 500 字符，避免太长影响 LLM 处理
                content_preview = doc.page_content[:500]
                if len(doc.page_content) > 500:
                    content_preview += "..."
                output_lines.append(f"  {content_preview}")

            return "\n".join(output_lines)

        except Exception as e:
            logger.error(f"知识检索失败: {str(e)}")
            return f"知识检索失败：{str(e)}"

    def get_tools(self) -> list:
        """
        获取 RAG 工具列表

        Returns:
            包含一个知识检索工具的列表
        """
        tools = [
            StructuredTool.from_function(
                func=self.search_knowledge,
                name="rag_search_knowledge",
                description=(
                    "从知识库中检索相关知识内容。\n"
                    "知识库当前包含两大类内容：\n"
                    "  1. ID映射表（最主要）：员工姓名→worker_id、岗位名→node_id 的对应关系\n"
                    "     用于 wms_query_employee_work 工具将姓名/岗位名转换为API所需的ID\n"
                    "  2. 业务知识：操作流程、绩效考核标准、KPI计算方法、常见问题等\n"
                    "\n"
                    "何时使用此工具 vs wms_query_employee_work：\n"
                    "  - 查员工工作量 → 用 wms_query_employee_work（会自己查知识库找ID）\n"
                    "  - 查员工/岗位的ID → 用 rag_search_knowledge（直接查映射表）\n"
                    "  - 查操作流程、考核标准等 → 用 rag_search_knowledge\n"
                    "\n"
                    "参数：query（查询问题，用自然语言描述）。\n"
                    "\n"
                    "示例用法：\n"
                    "  - 'LS-温明瑶的ID是多少' → 返回员工ID\n"
                    "  - '打包岗位的ID是什么' → 返回岗位ID\n"
                    "  - '绩效考核标准是什么' → 返回考核制度内容\n"
                    "  - '打包岗位工作流程' → 返回操作手册内容"
                ),
            ),
        ]

        logger.info(f"创建了 {len(tools)} 个 RAG 工具")
        return tools


def create_all_tools(wms_client: WMSClient, rag_system: RAGSystem) -> list:
    """
    创建所有工具（WMS 数据工具 + RAG 知识工具）

    Args:
        wms_client: WMS 客户端（实时数据来源）
        rag_system: RAG 系统（知识库来源）

    Returns:
        所有工具的列表，包含：
          - 1 个 WMS 工具（员工工作量查询）
          - 1 个 RAG 工具（知识检索）

    工具的数量建议：
      - 太少（< 3 个）：Agent 功能受限，很多问题回答不了
      - 合适（5-10 个）：Agent 能覆盖主要功能，且选择准确
      - 太多（> 15 个）：Agent 选择困难，容易选错工具，响应变慢

      本项目的 2 个工具是最精简配置，适合学习阶段。
      实际生产中可以根据业务需求增加更多工具。
    """
    logger.info("创建所有工具（WMS + RAG）")

    # 创建 WMS 工具（传入 rag_system 用于知识库ID映射查找）
    wms_tools = WMSTools(wms_client, rag_system)
    wms_tool_list = wms_tools.get_tools()

    # 创建 RAG 工具
    rag_tools = RAGTools(rag_system)
    rag_tool_list = rag_tools.get_tools()

    # 合并所有工具
    all_tools = wms_tool_list + rag_tool_list

    logger.success(f"总共创建了 {len(all_tools)} 个工具（{len(wms_tool_list)} 个 WMS + {len(rag_tool_list)} 个 RAG）")
    return all_tools
