"""
  LangChain工具定义模块

  功能:
  将WMS客户端和RAG系统的功能包装成LangChain Tool，
  让Agent能够调用这些工具来完成用户任务。

  LangChain Tool的概念:
  - Tool是Agent可以调用的功能单元
  - 每个Tool包含：
    - name: 工具名称（Agent用于识别）
    - description: 工具描述（告诉Agent何时使用）
    - func: 工具执行函数

  设计思路:
  1. 将WMS客户端的每个功能方法包装成独立工具
  2. 将RAG检索功能包装成知识查询工具
  3. 每个工具都有清晰的描述，帮助Agent理解何时调用

  Agent使用工具的过程:
  用户: "查询SKU001的库存"
  → Agent分析意图，识别需要使用"库存查询工具"
  → 调用wms_query_inventory工具，传入sku="SKU001"
  → 获取结果，生成回答: "SKU001当前库存150件"
"""

from typing import Optional, Dict, Any
from loguru import logger
from langchain.tools import Tool, StructuredTool

# 导入WMS客户端和RAG系统
from ..wms import WMSClient
from ..rag import RAGSystem


class WMSTools:
    """
    WMS工具集合

    将WMSClient的功能包装成LangChain工具。

    为什么需要包装？
    - WMSClient是Python类方法，Agent无法直接理解
    - Tool提供标准接口和描述，Agent能理解何时调用
    - Tool支持参数验证和错误处理
    """

    def __init__(self, wms_client: WMSClient):
        """
        初始化WMS工具集

        Args:
            wms_client: WMS客户端实例
        """
        self.wms_client = wms_client
        logger.info("WMS工具集初始化完成")

    def query_inventory(self, sku: Optional[str] = None) -> str:
        """
        库存查询工具的实现函数

        Args:
            sku: SKU编码（可选）

        Returns:
            格式化的库存信息文本

        说明:
            这是Tool的实际执行函数。
            当Agent决定调用库存查询工具时，会执行这个函数。
        """
        logger.info(f"执行库存查询，SKU: {sku}")

        try:
            # 调用WMS客户端查询库存
            result = self.wms_client.query_inventory(sku)

            # 将结果格式化为易读的文本
            # Agent会将这个文本作为"Observation"，用于生成最终回答
            if not result.items:
                return "未找到相关库存信息"

            # 构建返回文本
            output_lines = ["库存查询结果："]
            for item in result.items:
                output_lines.append(f"\nSKU: {item.sku}")
                output_lines.append(f"商品名称: {item.product_name}")
                output_lines.append(f"当前库存: {item.quantity}件")
                output_lines.append(f"可用库存: {item.available_quantity}件")
                output_lines.append(f"安全库存: {item.safety_stock}件")

                # 添加库存状态提示
                if item.quantity < item.safety_stock:
                    output_lines.append(f"[警告] 库存低于安全库存，需要补货")
                else:
                    output_lines.append(f"[正常] 库存充足")

                output_lines.append(f"库位: {item.warehouse_location}")
                output_lines.append(f"单位成本: {item.unit_cost}元")

            return "\n".join(output_lines)

        except Exception as e:
            logger.error(f"库存查询失败: {str(e)}")
            return f"库存查询失败：{str(e)}"

    def query_orders(
        self,
        order_id: Optional[str] = None,
        status: Optional[str] = None,
        order_type: Optional[str] = None
    ) -> str:
        """
        订单查询工具的实现函数

        Args:
            order_id: 订单号
            status: 订单状态
            order_type: 订单类型

        Returns:
            格式化的订单信息文本
        """
        logger.info(f"执行订单查询，订单号: {order_id}, 状态: {status}, 类型: {order_type}")

        try:
            result = self.wms_client.query_orders(order_id, status, order_type)

            if not result.orders:
                return "未找到相关订单信息"

            output_lines = ["订单查询结果："]
            for order in result.orders:
                output_lines.append(f"\n订单号: {order.order_id}")
                output_lines.append(f"类型: {order.order_type}")
                output_lines.append(f"客户: {order.customer_name}")
                output_lines.append(f"状态: {order.status}")
                output_lines.append(f"总金额: {order.total_amount}元")
                output_lines.append(f"创建时间: {order.created_time.strftime('%Y-%m-%d %H:%M')}")

                if order.items:
                    output_lines.append("商品明细:")
                    for item in order.items:
                        output_lines.append(
                            f"  - {item.sku}: {item.quantity}件 x {item.unit_price}元 = {item.total_price}元"
                        )

                if order.notes:
                    output_lines.append(f"备注: {order.notes}")

            return "\n".join(output_lines)

        except Exception as e:
            logger.error(f"订单查询失败: {str(e)}")
            return f"订单查询失败：{str(e)}"

    def query_inbounds(
        self,
        inbound_id: Optional[str] = None,
        sku: Optional[str] = None
    ) -> str:
        """
        入库记录查询工具的实现函数
        """
        logger.info(f"执行入库查询，入库单号: {inbound_id}, SKU: {sku}")

        try:
            result = self.wms_client.query_inbounds(inbound_id, sku)

            if not result.records:
                return "未找到相关入库记录"

            output_lines = ["入库记录查询结果："]
            for record in result.records:
                output_lines.append(f"\n入库单号: {record.inbound_id}")
                output_lines.append(f"SKU: {record.sku}")
                output_lines.append(f"商品: {record.product_name}")
                output_lines.append(f"数量: {record.quantity}")
                output_lines.append(f"供应商: {record.supplier_name}")
                output_lines.append(f"批次号: {record.batch_number}")
                output_lines.append(f"状态: {record.status}")
                output_lines.append(f"库位: {record.warehouse_location}")

                if record.received_time:
                    output_lines.append(
                        f"接收时间: {record.received_time.strftime('%Y-%m-%d %H:%M')}"
                    )

                if record.notes:
                    output_lines.append(f"备注: {record.notes}")

            return "\n".join(output_lines)

        except Exception as e:
            logger.error(f"入库查询失败: {str(e)}")
            return f"入库查询失败：{str(e)}"

    def get_statistics(self) -> str:
        """
        统计信息工具的实现函数

        说明:
            这是一个综合统计工具，返回库存和订单的汇总信息。
        """
        logger.info("执行统计信息查询")

        try:
            # 获取库存统计
            inventory_stats = self.wms_client.get_inventory_statistics()

            # 获取订单统计
            order_stats = self.wms_client.get_order_statistics()

            output_lines = [
                "仓库统计信息：",
                "\n【库存统计】",
                f"总SKU数: {inventory_stats.total_items}",
                f"总库存数量: {inventory_stats.total_quantity}件",
                f"总库存价值: {inventory_stats.total_value}元",
                f"低库存SKU数: {inventory_stats.low_stock_items}",
                f"库存过多SKU数: {inventory_stats.overstock_items}",
                "\n【订单统计】",
                f"总订单数: {order_stats.total_orders}",
                f"待处理订单: {order_stats.pending_orders}",
                f"已完成订单: {order_stats.completed_orders}",
                f"订单总金额: {order_stats.total_amount}元"
            ]

            return "\n".join(output_lines)

        except Exception as e:
            logger.error(f"统计查询失败: {str(e)}")
            return f"统计查询失败：{str(e)}"

    def check_low_stock(self) -> str:
        """
        低库存检查工具的实现函数

        说明:
            检查哪些SKU低于安全库存，并提供补货建议。
        """
        logger.info("执行低库存检查")

        try:
            result = self.wms_client.check_low_stock()

            if result['low_stock_count'] == 0:
                return "所有SKU库存充足，无需补货"

            output_lines = [
                f"低库存检查结果：发现 {result['low_stock_count']} 个SKU需要补货"
            ]

            for item in result['items']:
                output_lines.append(f"\nSKU: {item['sku']}")
                output_lines.append(f"商品名称: {item['product_name']}")
                output_lines.append(f"当前库存: {item['current_quantity']}件")
                output_lines.append(f"安全库存: {item['safety_stock']}件")
                output_lines.append(f"建议补货: {item['suggested_replenishment']}件")
                output_lines.append(f"紧急程度: {item['urgency']}")

            return "\n".join(output_lines)

        except Exception as e:
            logger.error(f"低库存检查失败: {str(e)}")
            return f"低库存检查失败：{str(e)}"

    def get_warehouse_summary(self) -> str:
        """
        仓库概览工具的实现函数

        说明:
            提供仓库的综合运营信息和建议。
        """
        logger.info("执行仓库概览查询")

        try:
            result = self.wms_client.get_warehouse_summary()

            output_lines = [
                "仓库运营概览：",
                "\n【库存概况】",
                f"总SKU数: {result['inventory_summary']['total_items']}",
                f"总库存数量: {result['inventory_summary']['total_quantity']}件",
                f"总库存价值: {result['inventory_summary']['total_value']}元",
                f"低库存警告: {result['inventory_summary']['low_stock_alert']}个SKU",
                "\n【订单概况】",
                f"总订单数: {result['order_summary']['total_orders']}",
                f"待处理订单: {result['order_summary']['pending_orders']}",
                f"订单总金额: {result['order_summary']['total_amount']}元",
                "\n【运营建议】"
            ]

            for rec in result['recommendations']:
                output_lines.append(f"  - {rec}")

            return "\n".join(output_lines)

        except Exception as e:
            logger.error(f"仓库概览查询失败: {str(e)}")
            return f"仓库概览查询失败：{str(e)}"

    def get_tools(self) -> list:
        """
        获取所有WMS工具

        Returns:
            LangChain Tool对象列表

        说明:
            这个方法将所有功能包装成Tool对象，
            每个Tool包含：
            - name: 工具名称（Agent用这个来识别工具）
            - description: 工具描述（Agent用这个判断何时调用）
            - func: 执行函数

            description非常重要：
            - 要清晰描述工具的功能
            - 要说明何时应该使用
            - 要说明需要什么参数
            - Agent通过阅读description来决定调用哪个工具
        """
        tools = [
            # 库存查询工具
            Tool(
                name="wms_query_inventory",
                description=(
                    "查询WMS库存信息。"
                    "当用户询问库存、商品数量、库位等信息时使用此工具。"
                    "可选参数：sku（SKU编码），如果不提供则查询所有库存。"
                    "示例：查询SKU001的库存、查询所有库存"
                ),
                func=self.query_inventory
            ),

            # 订单查询工具
            Tool(
                name="wms_query_orders",
                description=(
                    "查询WMS订单信息。"
                    "当用户询问订单、订单状态、订单明细时使用此工具。"
                    "可选参数：order_id（订单号）、status（订单状态）、order_type（订单类型）。"
                    "示例：查询订单ORD20240115001、查询所有待处理订单、查询采购订单"
                ),
                func=lambda x: self.query_orders(**x) if isinstance(x, dict) else self.query_orders(x)
            ),

            # 入库查询工具
            Tool(
                name="wms_query_inbounds",
                description=(
                    "查询WMS入库记录。"
                    "当用户询问入库历史、入库单信息时使用此工具。"
                    "可选参数：inbound_id（入库单号）、sku（SKU编码）。"
                    "示例：查询入库单IN20240114001、查询SKU001的入库记录"
                ),
                func=lambda x: self.query_inbounds(**x) if isinstance(x, dict) else self.query_inbounds(x)
            ),

            # 统计信息工具
            Tool(
                name="wms_get_statistics",
                description=(
                    "获取WMS统计信息。"
                    "当用户询问库存统计、订单统计、汇总数据时使用此工具。"
                    "无需参数。"
                    "示例：库存统计、订单统计、统计数据"
                ),
                func=self.get_statistics
            ),

            # 低库存检查工具
            Tool(
                name="wms_check_low_stock",
                description=(
                    "检查WMS低库存商品。"
                    "当用户询问哪些商品需要补货、库存预警时使用此工具。"
                    "无需参数。"
                    "示例：检查低库存、哪些商品需要补货"
                ),
                func=self.check_low_stock
            ),

            # 仓库概览工具
            Tool(
                name="wms_get_summary",
                description=(
                    "获取WMS仓库综合概览。"
                    "当用户询问仓库概况、运营状况、整体信息时使用此工具。"
                    "无需参数。"
                    "示例：仓库概览、运营概况、整体情况"
                ),
                func=self.get_warehouse_summary
            ),
        ]

        logger.info(f"创建了 {len(tools)} 个WMS工具")
        return tools


class RAGTools:
    """
    RAG知识检索工具集合

    将RAG系统的检索功能包装成LangChain工具。
    """

    def __init__(self, rag_system: RAGSystem):
        """
        初始化RAG工具集

        Args:
            rag_system: RAG系统实例
        """
        self.rag_system = rag_system
        logger.info("RAG工具集初始化完成")

    def search_knowledge(self, query: str) -> str:
        """
        知识检索工具的实现函数

        Args:
            query: 查询文本

        Returns:
            检索到的知识内容

        说明:
            当用户询问操作流程、业务规则、产品信息等知识性问题时使用。
        """
        logger.info(f"执行知识检索，查询: {query}")

        try:
            # 使用RAG系统检索相关文档
            documents = self.rag_system.retrieve(query, k=3)

            if not documents:
                return "未找到相关知识内容"

            # 将检索结果格式化
            output_lines = ["相关知识内容："]

            for i, doc in enumerate(documents):
                output_lines.append(f"\n【来源{i+1}】: {doc.metadata.get('source', 'unknown')}")
                output_lines.append(f"标题: {doc.metadata.get('title', '未知')}")
                output_lines.append(f"\n内容摘要：")
                # 只返回前500字符，避免太长
                content_preview = doc.page_content[:500]
                output_lines.append(content_preview)

            return "\n".join(output_lines)

        except Exception as e:
            logger.error(f"知识检索失败: {str(e)}")
            return f"知识检索失败：{str(e)}"

    def get_tools(self) -> list:
        """
        获取RAG工具列表
        """
        tools = [
            Tool(
                name="rag_search_knowledge",
                description=(
                    "从知识库中检索相关知识内容。"
                    "当用户询问操作流程、业务规则、产品信息、常见问题等知识性问题时使用此工具。"
                    "参数：query（查询问题）。"
                    "示例：如何查询库存、退货订单处理流程、SKU001的产品信息"
                ),
                func=self.search_knowledge
            ),
        ]

        logger.info(f"创建了 {len(tools)} 个RAG工具")
        return tools


def create_all_tools(wms_client: WMSClient, rag_system: RAGSystem) -> list:
    """
    创建所有工具（WMS + RAG）

    Args:
        wms_client: WMS客户端
        rag_system: RAG系统

    Returns:
        所有工具的列表

    说明:
        这个函数整合了所有工具，供Agent使用。
    """
    logger.info("创建所有工具")

    # 创建WMS工具
    wms_tools = WMSTools(wms_client)
    wms_tool_list = wms_tools.get_tools()

    # 创建RAG工具
    rag_tools = RAGTools(rag_system)
    rag_tool_list = rag_tools.get_tools()

    # 合并所有工具
    all_tools = wms_tool_list + rag_tool_list

    logger.success(f"总共创建了 {len(all_tools)} 个工具")
    return all_tools