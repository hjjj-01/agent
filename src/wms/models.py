"""
WMS数据模型定义

这个文件定义了WMS系统中各种数据的结构。
使用Pydantic来定义数据模型的好处：
1. 数据验证：自动验证数据类型和格式
2. 序列化：方便转换为JSON或其他格式
3. 文档化：清晰的字段说明
4. IDE支持：代码提示和类型检查
"""
from typing import Optional, List
from datetime import datetime
from pydantic import BaseModel, Field


# ==================== 库存相关模型 ====================

class InventoryItem(BaseModel):
    """
    库存物品模型

    用于表示单个SKU的库存信息
    """
    sku: str = Field(..., description="SKU编码，唯一标识一个商品")
    product_name: str = Field(..., description="商品名称")
    quantity: int = Field(..., description="当前库存数量")
    available_quantity: int = Field(..., description="可用库存数量（扣除预留）")
    reserved_quantity: int = Field(default=0, description="预留库存数量")
    warehouse_location: str = Field(..., description="仓库位置编码")
    safety_stock: int = Field(default=100, description="安全库存阈值")
    unit_cost: float = Field(..., description="单位成本")
    last_updated: datetime = Field(default_factory=datetime.now, description="最后更新时间")

    class Config:
        json_schema_extra = {
            "example": {
                "sku": "SKU001",
                "product_name": "商品A",
                "quantity": 150,
                "available_quantity": 120,
                "reserved_quantity": 30,
                "warehouse_location": "A-01-02",
                "safety_stock": 100,
                "unit_cost": 25.50,
                "last_updated": "2024-01-15T10:30:00"
            }
        }


class InventoryQueryResult(BaseModel):
    """
    库存查询结果模型

    用于返回库存查询的结果列表
    """
    items: List[InventoryItem] = Field(default_factory=list, description="库存物品列表")
    total_count: int = Field(..., description="总记录数")
    query_time: datetime = Field(default_factory=datetime.now, description="查询时间")


# ==================== 订单相关模型 ====================

class OrderItem(BaseModel):
    """
    订单明细模型

    订单中的单个商品项
    """
    sku: str = Field(..., description="SKU编码")
    product_name: str = Field(..., description="商品名称")
    quantity: int = Field(..., description="订购数量")
    unit_price: float = Field(..., description="单价")
    total_price: float = Field(..., description="该项总价")


class Order(BaseModel):
    """
    订单模型

    表示一个完整的订单信息
    """
    order_id: str = Field(..., description="订单号")
    order_type: str = Field(..., description="订单类型：sale(销售), purchase(采购), return(退货)")
    customer_id: str = Field(..., description="客户/供应商ID")
    customer_name: str = Field(..., description="客户/供应商名称")
    items: List[OrderItem] = Field(default_factory=list, description="订单明细列表")
    total_amount: float = Field(..., description="订单总金额")
    status: str = Field(..., description="订单状态：pending, processing, completed, cancelled")
    created_time: datetime = Field(default_factory=datetime.now, description="创建时间")
    updated_time: datetime = Field(default_factory=datetime.now, description="更新时间")
    notes: Optional[str] = Field(None, description="备注信息")

    class Config:
        json_schema_extra = {
            "example": {
                "order_id": "ORD20240115001",
                "order_type": "sale",
                "customer_id": "CUST001",
                "customer_name": "客户A",
                "items": [
                    {
                        "sku": "SKU001",
                        "product_name": "商品A",
                        "quantity": 100,
                        "unit_price": 30.00,
                        "total_price": 3000.00
                    }
                ],
                "total_amount": 3000.00,
                "status": "pending",
                "created_time": "2024-01-15T14:30:00",
                "updated_time": "2024-01-15T14:30:00",
                "notes": "客户急需，优先发货"
            }
        }


class OrderQueryResult(BaseModel):
    """
    订单查询结果模型
    """
    orders: List[Order] = Field(default_factory=list, description="订单列表")
    total_count: int = Field(..., description="总记录数")
    query_time: datetime = Field(default_factory=datetime.now, description="查询时间")


# ==================== 入库相关模型 ====================

class InboundRecord(BaseModel):
    """
    入库记录模型

    表示一个入库单的信息
    """
    inbound_id: str = Field(..., description="入库单号")
    sku: str = Field(..., description="SKU编码")
    product_name: str = Field(..., description="商品名称")
    quantity: int = Field(..., description="入库数量")
    supplier_id: str = Field(..., description="供应商ID")
    supplier_name: str = Field(..., description="供应商名称")
    batch_number: str = Field(..., description="批次号")
    warehouse_location: str = Field(..., description="入库仓库位置")
    status: str = Field(..., description="状态：pending, received, completed")
    received_time: Optional[datetime] = Field(None, description="接收时间")
    completed_time: Optional[datetime] = Field(None, description="完成时间")
    notes: Optional[str] = Field(None, description="备注")

    class Config:
        json_schema_extra = {
            "example": {
                "inbound_id": "IN20240115001",
                "sku": "SKU001",
                "product_name": "商品A",
                "quantity": 500,
                "supplier_id": "SUP001",
                "supplier_name": "供应商A",
                "batch_number": "BATCH20240115",
                "warehouse_location": "A-01-02",
                "status": "completed",
                "received_time": "2024-01-15T08:00:00",
                "completed_time": "2024-01-15T10:00:00",
                "notes": "质检合格"
            }
        }


class InboundQueryResult(BaseModel):
    """
    入库查询结果模型
    """
    records: List[InboundRecord] = Field(default_factory=list, description="入库记录列表")
    total_count: int = Field(..., description="总记录数")
    query_time: datetime = Field(default_factory=datetime.now, description="查询时间")


# ==================== 统计分析模型 ====================

class InventoryStatistics(BaseModel):
    """
    库存统计模型

    用于库存汇总统计
    """
    total_items: int = Field(..., description="总SKU数")
    total_quantity: int = Field(..., description="总库存数量")
    total_value: float = Field(..., description="总库存价值")
    low_stock_items: int = Field(..., description="低于安全库存的SKU数")
    overstock_items: int = Field(..., description="库存过多的SKU数")
    statistics_time: datetime = Field(default_factory=datetime.now, description="统计时间")


class OrderStatistics(BaseModel):
    """
    订单统计模型

    用于订单汇总统计
    """
    total_orders: int = Field(..., description="总订单数")
    pending_orders: int = Field(..., description="待处理订单数")
    completed_orders: int = Field(..., description="已完成订单数")
    total_amount: float = Field(..., description="订单总金额")
    statistics_time: datetime = Field(default_factory=datetime.now, description="统计时间")


# ==================== 错误响应模型 ====================

class ErrorResponse(BaseModel):
    """
    错误响应模型

    用于API调用失败时返回的错误信息
    """
    error_code: str = Field(..., description="错误代码")
    error_message: str = Field(..., description="错误描述")
    details: Optional[str] = Field(None, description="详细错误信息")
    timestamp: datetime = Field(default_factory=datetime.now, description="错误发生时间")