"""
WMS客户端模块

这个模块封装了与WMS系统交互的所有逻辑。
设计思路：
1. 模拟WMS API调用（目前使用模拟数据，未来替换为真实API）
2. 提供统一的接口方法
3. 处理错误和异常
4. 记录日志便于调试

当有真实WMS接口时，只需要修改这个文件中的实现，其他代码无需改动。
"""
import json
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
from loguru import logger

from .models import (
    InventoryItem,
    InventoryQueryResult,
    Order,
    OrderQueryResult,
    InboundRecord,
    InboundQueryResult,
    InventoryStatistics,
    OrderStatistics,
    ErrorResponse
)


class WMSClient:
    """
    WMS系统客户端

    用于与WMS系统进行交互，获取库存、订单、入库等数据。

    当前实现：使用模拟数据
    未来改进：替换为真实的API调用

    使用方式：
        client = WMSClient(api_base_url="http://wms-api.com", api_token="token")
        inventory = client.query_inventory(sku="SKU001")
    """

    def __init__(self, api_base_url: str, api_token: str):
        """
        初始化WMS客户端

        Args:
            api_base_url: WMS系统API地址
            api_token: 认证token

        说明:
            在真实场景中，这里会设置HTTP客户端配置，如：
            - 设置请求头（Authorization等）
            - 配置超时时间
            - 设置重试策略
        """
        self.api_base_url = api_base_url
        self.api_token = api_token
        logger.info(f"WMS客户端初始化完成，API地址: {api_base_url}")

        # 初始化模拟数据（真实场景中不需要这部分）
        self._init_mock_data()

    def _init_mock_data(self):
        """
        初始化模拟数据

        说明:
            这是为了演示和学习准备的模拟数据。
            当有真实WMS接口时，这个方法会被删除，数据将从真实API获取。

        模拟数据包括：
            - 库存数据：5个SKU的库存信息
            - 订单数据：几个示例订单
            - 入库记录：一些入库历史
        """
        logger.info("初始化模拟数据（用于演示和学习）")

        # 模拟库存数据
        self.mock_inventory = {
            "SKU001": {
                "sku": "SKU001",
                "product_name": "商品A - 电子元件",
                "quantity": 150,
                "available_quantity": 120,
                "reserved_quantity": 30,
                "warehouse_location": "A-01-02",
                "safety_stock": 100,
                "unit_cost": 25.50,
            },
            "SKU002": {
                "sku": "SKU002",
                "product_name": "商品B - 包装材料",
                "quantity": 80,
                "available_quantity": 80,
                "reserved_quantity": 0,
                "warehouse_location": "B-02-03",
                "safety_stock": 100,
                "unit_cost": 15.00,
            },
            "SKU003": {
                "sku": "SKU003",
                "product_name": "商品C - 办公用品",
                "quantity": 250,
                "available_quantity": 200,
                "reserved_quantity": 50,
                "warehouse_location": "C-01-01",
                "safety_stock": 150,
                "unit_cost": 8.50,
            },
            "SKU004": {
                "sku": "SKU004",
                "product_name": "商品D - 家居用品",
                "quantity": 40,
                "available_quantity": 40,
                "reserved_quantity": 0,
                "warehouse_location": "D-03-02",
                "safety_stock": 50,
                "unit_cost": 45.00,
            },
            "SKU005": {
                "sku": "SKU005",
                "product_name": "商品E - 食品原料",
                "quantity": 500,
                "available_quantity": 450,
                "reserved_quantity": 50,
                "warehouse_location": "E-01-03",
                "safety_stock": 200,
                "unit_cost": 12.00,
            },
        }

        # 模拟订单数据
        self.mock_orders = [
            {
                "order_id": "ORD20240115001",
                "order_type": "sale",
                "customer_id": "CUST001",
                "customer_name": "客户A - 某电子公司",
                "items": [
                    {
                        "sku": "SKU001",
                        "product_name": "商品A - 电子元件",
                        "quantity": 50,
                        "unit_price": 30.00,
                        "total_price": 1500.00,
                    }
                ],
                "total_amount": 1500.00,
                "status": "completed",
                "created_time": datetime.now() - timedelta(days=2),
                "notes": "客户急需，优先发货",
            },
            {
                "order_id": "ORD20240115002",
                "order_type": "sale",
                "customer_id": "CUST002",
                "customer_name": "客户B - 某贸易公司",
                "items": [
                    {
                        "sku": "SKU002",
                        "product_name": "商品B - 包装材料",
                        "quantity": 100,
                        "unit_price": 18.00,
                        "total_price": 1800.00,
                    },
                    {
                        "sku": "SKU003",
                        "product_name": "商品C - 办公用品",
                        "quantity": 200,
                        "unit_price": 10.00,
                        "total_price": 2000.00,
                    }
                ],
                "total_amount": 3800.00,
                "status": "processing",
                "created_time": datetime.now() - timedelta(days=1),
                "notes": "",
            },
            {
                "order_id": "ORD20240115003",
                "order_type": "purchase",
                "customer_id": "SUP001",
                "customer_name": "供应商A",
                "items": [
                    {
                        "sku": "SKU001",
                        "product_name": "商品A - 电子元件",
                        "quantity": 200,
                        "unit_price": 25.50,
                        "total_price": 5100.00,
                    }
                ],
                "total_amount": 5100.00,
                "status": "pending",
                "created_time": datetime.now(),
                "notes": "需要补货",
            },
        ]

        # 模拟入库记录
        self.mock_inbounds = [
            {
                "inbound_id": "IN20240114001",
                "sku": "SKU001",
                "product_name": "商品A - 电子元件",
                "quantity": 300,
                "supplier_id": "SUP001",
                "supplier_name": "供应商A",
                "batch_number": "BATCH20240114",
                "warehouse_location": "A-01-02",
                "status": "completed",
                "received_time": datetime.now() - timedelta(days=1, hours=8),
                "completed_time": datetime.now() - timedelta(days=1, hours=2),
                "notes": "质检合格",
            },
            {
                "inbound_id": "IN20240113001",
                "sku": "SKU003",
                "product_name": "商品C - 办公用品",
                "quantity": 100,
                "supplier_id": "SUP002",
                "supplier_name": "供应商B",
                "batch_number": "BATCH20240113",
                "warehouse_location": "C-01-01",
                "status": "completed",
                "received_time": datetime.now() - timedelta(days=2, hours=10),
                "completed_time": datetime.now() - timedelta(days=2, hours=4),
                "notes": "",
            },
        ]

    def query_inventory(self, sku: Optional[str] = None) -> InventoryQueryResult:
        """
        查询库存信息

        Args:
            sku: SKU编码，如果指定则查询特定SKU，否则查询所有库存

        Returns:
            InventoryQueryResult: 库存查询结果

        实现逻辑:
            1. 真实场景：调用WMS API的库存查询接口
            2. 模拟场景：从模拟数据中获取

        示例API调用（真实场景）:
            response = requests.get(
                f"{self.api_base_url}/inventory",
                headers={"Authorization": f"Bearer {self.api_token}"},
                params={"sku": sku} if sku else {}
            )
            return InventoryQueryResult(**response.json())
        """
        logger.info(f"查询库存，SKU: {sku if sku else '全部'}")

        try:
            items = []

            # 从模拟数据中获取
            if sku:
                # 查询特定SKU
                if sku in self.mock_inventory:
                    data = self.mock_inventory[sku].copy()
                    data["last_updated"] = datetime.now()
                    items.append(InventoryItem(**data))
                else:
                    logger.warning(f"未找到SKU: {sku}")
            else:
                # 查询所有库存
                for data in self.mock_inventory.values():
                    data_copy = data.copy()
                    data_copy["last_updated"] = datetime.now()
                    items.append(InventoryItem(**data_copy))

            # 构建返回结果
            result = InventoryQueryResult(
                items=items,
                total_count=len(items),
                query_time=datetime.now()
            )

            logger.success(f"库存查询成功，返回 {len(items)} 条记录")
            return result

        except Exception as e:
            logger.error(f"库存查询失败: {str(e)}")
            raise

    def query_orders(
        self,
        order_id: Optional[str] = None,
        status: Optional[str] = None,
        order_type: Optional[str] = None
    ) -> OrderQueryResult:
        """
        查询订单信息

        Args:
            order_id: 订单号，如果指定则查询特定订单
            status: 订单状态筛选（pending, processing, completed, cancelled）
            order_type: 订单类型筛选（sale, purchase, return）

        Returns:
            OrderQueryResult: 订单查询结果

        实现逻辑:
            支持多种查询条件组合，灵活筛选订单
        """
        logger.info(f"查询订单，订单号: {order_id}, 状态: {status}, 类型: {order_type}")

        try:
            orders = []

            # 从模拟数据中筛选
            for order_data in self.mock_orders:
                # 应用筛选条件
                if order_id and order_data["order_id"] != order_id:
                    continue
                if status and order_data["status"] != status:
                    continue
                if order_type and order_data["order_type"] != order_type:
                    continue

                # 复制数据并转换
                data_copy = order_data.copy()
                data_copy["updated_time"] = datetime.now()
                orders.append(Order(**data_copy))

            result = OrderQueryResult(
                orders=orders,
                total_count=len(orders),
                query_time=datetime.now()
            )

            logger.success(f"订单查询成功，返回 {len(orders)} 条记录")
            return result

        except Exception as e:
            logger.error(f"订单查询失败: {str(e)}")
            raise

    def query_inbounds(
        self,
        inbound_id: Optional[str] = None,
        sku: Optional[str] = None
    ) -> InboundQueryResult:
        """
        查询入库记录

        Args:
            inbound_id: 入库单号
            sku: SKU编码

        Returns:
            InboundQueryResult: 入库记录查询结果
        """
        logger.info(f"查询入库记录，入库单号: {inbound_id}, SKU: {sku}")

        try:
            records = []

            # 从模拟数据中筛选
            for inbound_data in self.mock_inbounds:
                if inbound_id and inbound_data["inbound_id"] != inbound_id:
                    continue
                if sku and inbound_data["sku"] != sku:
                    continue

                records.append(InboundRecord(**inbound_data))

            result = InboundQueryResult(
                records=records,
                total_count=len(records),
                query_time=datetime.now()
            )

            logger.success(f"入库记录查询成功，返回 {len(records)} 条记录")
            return result

        except Exception as e:
            logger.error(f"入库记录查询失败: {str(e)}")
            raise

    def get_inventory_statistics(self) -> InventoryStatistics:
        """
        获取库存统计信息

        Returns:
            InventoryStatistics: 库存统计数据

        说明:
            对所有库存进行汇总统计，包括：
            - 总SKU数
            - 总库存数量
            - 总库存价值
            - 低库存警告（低于安全库存的SKU数量）
        """
        logger.info("获取库存统计信息")

        try:
            # 查询所有库存
            inventory_result = self.query_inventory()

            # 计算统计数据
            total_items = len(inventory_result.items)
            total_quantity = sum(item.quantity for item in inventory_result.items)
            total_value = sum(item.quantity * item.unit_cost for item in inventory_result.items)
            low_stock_items = sum(1 for item in inventory_result.items if item.quantity < item.safety_stock)
            overstock_items = sum(1 for item in inventory_result.items if item.quantity > item.safety_stock * 2)

            statistics = InventoryStatistics(
                total_items=total_items,
                total_quantity=total_quantity,
                total_value=total_value,
                low_stock_items=low_stock_items,
                overstock_items=overstock_items,
                statistics_time=datetime.now()
            )

            logger.success(f"库存统计完成：总SKU数 {total_items}, 总库存数量 {total_quantity}")
            return statistics

        except Exception as e:
            logger.error(f"库存统计失败: {str(e)}")
            raise

    def get_order_statistics(self) -> OrderStatistics:
        """
        获取订单统计信息

        Returns:
            OrderStatistics: 订单统计数据
        """
        logger.info("获取订单统计信息")

        try:
            # 查询所有订单
            order_result = self.query_orders()

            # 计算统计数据
            total_orders = len(order_result.orders)
            pending_orders = sum(1 for order in order_result.orders if order.status == "pending")
            completed_orders = sum(1 for order in order_result.orders if order.status == "completed")
            total_amount = sum(order.total_amount for order in order_result.orders)

            statistics = OrderStatistics(
                total_orders=total_orders,
                pending_orders=pending_orders,
                completed_orders=completed_orders,
                total_amount=total_amount,
                statistics_time=datetime.now()
            )

            logger.success(f"订单统计完成：总订单数 {total_orders}, 待处理 {pending_orders}")
            return statistics

        except Exception as e:
            logger.error(f"订单统计失败: {str(e)}")
            raise

    def check_low_stock(self) -> Dict[str, Any]:
        """
        检查低库存商品

        Returns:
            Dict: 低库存商品信息，包括SKU列表和建议补货数量

        说明:
            这是一个业务逻辑方法，基于库存数据进行分析，
            返回需要补货的商品和建议补货数量。
        """
        logger.info("检查低库存商品")

        try:
            inventory_result = self.query_inventory()

            low_stock_items = []
            for item in inventory_result.items:
                if item.quantity < item.safety_stock:
                    # 计算建议补货数量（补充到安全库存的1.5倍）
                    suggested_quantity = int(item.safety_stock * 1.5 - item.quantity)

                    low_stock_items.append({
                        "sku": item.sku,
                        "product_name": item.product_name,
                        "current_quantity": item.quantity,
                        "safety_stock": item.safety_stock,
                        "suggested_replenishment": suggested_quantity,
                        "urgency": "high" if item.quantity < item.safety_stock * 0.5 else "medium"
                    })

            result = {
                "low_stock_count": len(low_stock_items),
                "items": low_stock_items,
                "check_time": datetime.now().isoformat()
            }

            logger.success(f"低库存检查完成，发现 {len(low_stock_items)} 个SKU需要补货")
            return result

        except Exception as e:
            logger.error(f"低库存检查失败: {str(e)}")
            raise

    def get_warehouse_summary(self) -> Dict[str, Any]:
        """
        获取仓库概览

        Returns:
            Dict: 仓库综合信息摘要

        说明:
            这是一个综合查询方法，整合了库存、订单、入库等多方面的信息，
            为用户提供仓库运营的整体概览。
        """
        logger.info("获取仓库概览")

        try:
            # 获取各项统计数据
            inventory_stats = self.get_inventory_statistics()
            order_stats = self.get_order_statistics()
            low_stock = self.check_low_stock()

            # 构建综合概览
            summary = {
                "inventory_summary": {
                    "total_items": inventory_stats.total_items,
                    "total_quantity": inventory_stats.total_quantity,
                    "total_value": inventory_stats.total_value,
                    "low_stock_alert": low_stock["low_stock_count"],
                },
                "order_summary": {
                    "total_orders": order_stats.total_orders,
                    "pending_orders": order_stats.pending_orders,
                    "total_amount": order_stats.total_amount,
                },
                "recommendations": self._generate_recommendations(inventory_stats, order_stats, low_stock),
                "summary_time": datetime.now().isoformat()
            }

            logger.success("仓库概览生成成功")
            return summary

        except Exception as e:
            logger.error(f"获取仓库概览失败: {str(e)}")
            raise

    def _generate_recommendations(
        self,
        inventory_stats: InventoryStatistics,
        order_stats: OrderStatistics,
        low_stock: Dict
    ) -> List[str]:
        """
        生成运营建议

        Args:
            inventory_stats: 库存统计数据
            order_stats: 订单统计数据
            low_stock: 低库存检查结果

        Returns:
            List[str]: 运营建议列表

        说明:
            这是一个私有方法，基于数据分析生成运营建议。
            在真实场景中，这些逻辑会更复杂，可能涉及机器学习模型。
        """
        recommendations = []

        # 低库存建议
        if low_stock["low_stock_count"] > 0:
            recommendations.append(
                f"建议立即处理 {low_stock['low_stock_count']} 个低库存SKU的补货需求"
            )

        # 待处理订单建议
        if order_stats.pending_orders > 5:
            recommendations.append(
                f"当前有 {order_stats.pending_orders} 个待处理订单，建议加快订单处理速度"
            )

        # 库存周转建议
        if inventory_stats.overstock_items > 2:
            recommendations.append(
                f"有 {inventory_stats.overstock_items} 个SKU库存过多，建议优化库存周转"
            )

        if not recommendations:
            recommendations.append("仓库运营状况良好，继续保持当前运营策略")

        return recommendations