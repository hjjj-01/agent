from typing import Optional, List
from loguru import logger

# LangChain 1.x 工具定义
# StructuredTool：通过 from_function() 将普通函数包装成 Tool
# Tool 接口是 LangChain 1.x 中的标准工具接口
from langchain_core.tools import StructuredTool

from ..wms import WMSClient
from ..wms.models import InventoryQueryRequest


class WMSTools:
    """
    WMS 工具集

    本类把"库位库存查询"这一个业务能力，包装成 Agent（LLM）可以调用的工具。
    当前项目只保留这一种查询能力：
      - 查询库位库存（按库位 / 按商品SKU / 按商品SPU）

    设计说明：
      - 工具层是 LLM 和底层 WMSClient 之间的"翻译官 + 适配器"。
      - LLM 只看得懂"工具名 + 描述 + 字符串返回值"，看不懂 WMSClient 对象。
      - 本类把 WMS API 的 JSON 响应，整理成一段清晰的中文文本交给 LLM 阅读。
    """

    def __init__(self, wms_client: WMSClient):
        """
        初始化工具集

        Args:
            wms_client: WMS 客户端实例（负责真正发起 HTTP 请求）
        """
        self.wms_client = wms_client
        logger.info("WMS 工具集初始化完成")

    # =========================================================
    # 库位库存查询（唯一的核心业务方法）
    # =========================================================
    def query_inventory(
        self,
        location_code: Optional[str] = None,
        sku_code: Optional[str] = None,
        spu_code: Optional[str] = None,
        filter_non_zero: Optional[str] = "Y",
        warehouse_areas_code: Optional[str] = None,
        low_stock_threshold: Optional[int] = None,
    ) -> str:
        """
        查询 WMS 库位库存。

        支持三种问法（由参数组合决定）：
          1) 按库位查：传 location_code（库位编码），如 "W3-C1-01-01"
          2) 按SKU查：传 sku_code（SKU编码），如 "PT236DBKM"
          3) 按SPU查：传 spu_code（商品编码），如 "PT236D"
        也可以组合：既传 location_code 又传 sku_code，即"某个库位上某个SKU的库存"。

        Args:
            location_code: 库位编码，如 "W3-C1-01-01"
            sku_code: SKU编码，如 "PT236DBKM"
            spu_code: SPU编码（商品编码），如 "PT236D"，与 sku_code 二选一
            filter_non_zero: 是否只看有库存的数据，"Y"=过滤零库存（默认），空字符串=不过滤
            warehouse_areas_code: 库区编码（可选，缩小查询范围）
            low_stock_threshold: 低库存预警阈值（可选）。当某条记录的总库存 <= 该值时，
                                 在结果中标记为"⚠️低库存预警"

        Returns:
            一段中文文本，包含汇总数据和每条库存明细。
            任何异常都会返回错误字符串而不是抛异常（友好失败，避免 Agent 崩溃）。
        """
        logger.info(
            f"执行库位库存查询，"
            f"库位: {location_code}, SKU: {sku_code}, SPU: {spu_code}, "
            f"过滤零库存: {filter_non_zero}, 库区: {warehouse_areas_code}, "
            f"低库存阈值: {low_stock_threshold}"
        )

        try:
            # 第1步：确定"按商品查询"的维度（SKU 优先于 SPU）
            if sku_code:
                item_label, item_value = "skuCode", sku_code
            elif spu_code:
                item_label, item_value = "spuCode", spu_code
            else:
                item_label, item_value = None, None

            # 第2步：构造请求模型
            params = InventoryQueryRequest(
                pageNo=1,
                pageSize=100,
                locationCode=location_code,
                itemLabel=item_label,
                itemValue=item_value,
                filterNonZero=filter_non_zero,
                warehouseAreasCode=warehouse_areas_code,
            )

            # 第3步：调用 WMS API
            result = self.wms_client.query_inventory(params)

            # 第4步：检查返回
            if result.code != 0:
                return f"库位库存查询失败：接口返回 code={result.code}，msg={result.msg}"

            records = result.list
            total = result.total

            if not records:
                return "未查询到符合条件的库存记录（可能该库位/商品当前没有库存）。"

            # 第5步：计算汇总
            total_qty_sum = sum((r.totalQty or 0) for r in records)
            available_qty_sum = sum((r.availableQty or 0) for r in records)
            frozen_qty_sum = sum((r.frozenQty or 0) for r in records)
            locked_qty_sum = sum((r.lockedQty or 0) for r in records)

            # 去重统计涉及的库位数和SKU数
            locations = set(r.locationCode for r in records if r.locationCode)
            skus = set(r.skuCode for r in records if r.skuCode)

            # 第6步：组装输出文本
            # 标题：根据查询维度动态生成
            if location_code and not (sku_code or spu_code):
                title = f"库位「{location_code}」库存明细"
            elif sku_code:
                title = f"SKU「{sku_code}」库存分布"
            elif spu_code:
                title = f"商品「{spu_code}」库存分布"
            else:
                title = "库位库存查询结果"

            output_lines = [
                f"{title}（共 {total} 条记录）：",
                f"\n【总体概览】",
                f"  - 库位数: {len(locations)} 个" if locations else "  - 库位数: 1 个",
                f"  - SKU数: {len(skus)} 个" if skus else "  - SKU数: -",
                f"  - 总库存合计: {total_qty_sum}",
                f"  - 可用库存合计: {available_qty_sum}",
                f"  - 冻结库存合计: {frozen_qty_sum}",
                f"  - 锁定库存合计: {locked_qty_sum}",
            ]

            # 第7步：逐条明细（按总库存降序，最多50条）
            display_records = sorted(
                records, key=lambda r: (r.totalQty or 0), reverse=True
            )[:50]

            output_lines.append(f"\n【库存明细】（共{len(records)}条，按总库存降序）")

            for r in display_records:
                # 低库存预警判断
                warning = ""
                if low_stock_threshold is not None:
                    if (r.totalQty or 0) <= low_stock_threshold:
                        warning = "  ⚠️低库存预警"

                name_parts = []
                if r.spuName:
                    name_parts.append(r.spuName)
                if r.color:
                    name_parts.append(r.color)
                if r.size:
                    name_parts.append(r.size)
                goods_desc = "-".join(name_parts) if name_parts else (r.skuName or "未命名商品")

                output_lines.extend([
                    f"\n  库位: {r.locationCode or '-'}  | SKU: {r.skuCode or '-'}{warning}",
                    f"    商品: {goods_desc}",
                    f"    总库存: {r.totalQty or 0}  | 可用: {r.availableQty or 0}  | 冻结: {r.frozenQty or 0}  | 锁定: {r.lockedQty or 0}",
                    f"    库区: {r.warehouseAreasCode or '-'}  | 库道: {r.warehousePassageCode or '-'}  | 分类: {r.levelThreeName or '-'}",
                ])

            return "\n".join(output_lines)

        except Exception as e:
            logger.error(f"库位库存查询失败: {str(e)}")
            return f"库位库存查询失败：{str(e)}"

    def get_tools(self) -> List[StructuredTool]:
        """
        把 query_inventory 方法包装成 LangChain 工具对象。

        Returns:
            包含一个工具的列表（wms_query_inventory）。
        """
        tools = [
            StructuredTool.from_function(
                func=self.query_inventory,
                name="wms_query_inventory",
                description=(
                    "查询 WMS 库位库存数据。\n"
                    "当用户询问库存数量、库位库存、某个商品/SKU的库存、哪个库位有某商品、"
                    "低库存预警、可用库存等信息时使用此工具。\n"
                    "\n"
                    "支持三种查询维度（参数可组合）：\n"
                    "  - 按库位查：传 location_code，如 'W3-C1-01-01'\n"
                    "  - 按SKU查：传 sku_code，如 'PT236DBKM'\n"
                    "  - 按商品(SPU)查：传 spu_code，如 'PT236D'\n"
                    "\n"
                    "其他可选参数：\n"
                    "  - filter_non_zero：是否只看有库存的数据，'Y'(默认)过滤零库存，传空字符串''不过滤\n"
                    "  - warehouse_areas_code：库区编码，缩小查询范围\n"
                    "  - low_stock_threshold：低库存阈值，某条记录总库存<=该值时标记为⚠️低库存预警\n"
                    "\n"
                    "返回：总库存、可用库存、冻结库存、锁定库存的汇总，以及每条库存明细"
                    "（库位、SKU、商品名称、颜色、尺码、各状态库存数量等）。\n"
                    "\n"
                    "示例用法：\n"
                    "  - 查询库位 W3-C1-01-01 的库存：location_code=W3-C1-01-01\n"
                    "  - 查询 SKU PT236DBKM 的库存分布：sku_code=PT236DBKM\n"
                    "  - 查询 W3-C1-01-01 库位上 PT236DBKM 的库存：location_code=W3-C1-01-01, sku_code=PT236DBKM\n"
                    "  - 查询低库存（总库存<=10）：sku_code=PT236DBKM, low_stock_threshold=10\n"
                    "  - 查询某库区库存：warehouse_areas_code=C1"
                ),
            ),
        ]

        logger.info(f"创建了 {len(tools)} 个 WMS 工具")
        return tools


def create_all_tools(wms_client: WMSClient) -> List[StructuredTool]:
    """
    创建所有工具（当前只有库位库存查询一个工具）

    这是 tools 模块对外的唯一入口，wms_agent.py 会调用它。

    Args:
        wms_client: WMS 客户端实例

    Returns:
        工具列表（LangChain StructuredTool 对象）
    """
    logger.info("创建所有工具（仅库位库存查询）")

    wms_tools = WMSTools(wms_client)
    all_tools = wms_tools.get_tools()

    logger.success(f"总共创建了 {len(all_tools)} 个工具")
    return all_tools
