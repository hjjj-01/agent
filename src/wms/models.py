"""
WMS数据模型定义

本文件定义库位库存查询相关的 Pydantic 模型：
  - InventoryQueryRequest：库位库存查询请求参数
  - InventoryRecord：单条库位库存记录
  - InventoryResponse：库位库存查询响应（整体包装）
  - ErrorResponse：错误响应
"""
from typing import Optional, List
from datetime import datetime
from pydantic import BaseModel, Field


# ==================== 库位库存查询相关模型 ====================

class InventoryQueryRequest(BaseModel):
    """
    库位库存查询请求模型

    对应 WMS 接口：
      GET /wms/inventory-common/location-inventory-page

    支持两种查询维度：
      1) 按库位查：传 locationCode（库位编码），如 "W3-C1-01-01"
      2) 按商品查：传 itemLabel + itemValue
         - itemLabel="skuCode"，itemValue="PT236DBKM" 表示按SKU查
         - itemLabel="spuCode"，itemValue="PT236D"   表示按SPU查
    两者可以组合使用（既指定库位，又指定商品）。
    """
    pageNo: int = Field(default=1, description="页码，默认第1页")
    pageSize: int = Field(default=100, description="每页数量，默认100")

    # —— 维度一：按库位查询 ——
    locationCode: Optional[str] = Field(None, description="库位编码，如 W3-C1-01-01")

    # —— 维度二：按商品查询 ——
    itemLabel: Optional[str] = Field(
        default="skuCode",
        description="商品查询维度标签：skuCode(SKU编码) / spuCode(SPU编码)"
    )
    itemValue: Optional[str] = Field(None, description="商品编码值，配合 itemLabel 使用")

    # —— 其他可选过滤条件 ——
    lpnNo: Optional[str] = Field(None, description="容器号(LPN)")
    filterNonZero: Optional[str] = Field(
        default="Y",
        description="是否过滤零库存：Y=只看有库存的数据，空=不过滤"
    )
    isIncludeVirtual: Optional[str] = Field(default="N", description="是否包含虚拟库存：N=不包含")
    warehousePassageCode: Optional[str] = Field(None, description="库道编码")
    warehouseAreasCode: Optional[str] = Field(None, description="库区编码")
    storageMode: Optional[str] = Field(None, description="存储方式")
    spuName: Optional[str] = Field(None, description="商品名称(SPU名称)，支持模糊")
    season: Optional[str] = Field(None, description="季节")
    activeType: Optional[str] = Field(None, description="活性类型")
    levelOneName: Optional[str] = Field(None, description="一级分类")
    levelTwoName: Optional[str] = Field(None, description="二级分类")
    levelThreeName: Optional[str] = Field(None, description="三级分类")
    fnsku: Optional[str] = Field(None, description="FNSKU")
    shopName: Optional[str] = Field(None, description="店铺名称")
    totalQtyGe: Optional[str] = Field(None, description="总库存 >= 该值")
    totalQtyLe: Optional[str] = Field(None, description="总库存 <= 该值")

    class Config:
        json_schema_extra = {
            "example": {
                "pageNo": 1,
                "pageSize": 100,
                "locationCode": "W3-C1-01-01",
                "itemLabel": "skuCode",
                "itemValue": "PT236DBKM",
                "filterNonZero": "Y",
                "isIncludeVirtual": "N",
            }
        }


class InventoryRecord(BaseModel):
    """
    单条库位库存记录

    对应接口返回的 data.list 中的一条数据。
    字段均为 Optional，因为示例中存在 null 值。
    """
    id: Optional[str] = Field(None, description="记录ID")
    warehouseAreasInfoId: Optional[str] = Field(None, description="库区信息ID")
    warehouseAreasCode: Optional[str] = Field(None, description="库区编码")
    floorCode: Optional[str] = Field(None, description="楼层/库区代码")
    warehouseAreasName: Optional[str] = Field(None, description="库区名称")
    locationId: Optional[str] = Field(None, description="库位ID")
    locationCode: Optional[str] = Field(None, description="库位编码")
    skuId: Optional[str] = Field(None, description="SKU ID")
    spuId: Optional[str] = Field(None, description="SPU ID")
    skuCode: Optional[str] = Field(None, description="SKU编码")
    spuCode: Optional[str] = Field(None, description="SPU编码")
    spuName: Optional[str] = Field(None, description="商品名称(SPU名称)")
    color: Optional[str] = Field(None, description="颜色")
    size: Optional[str] = Field(None, description="尺码")
    skuName: Optional[str] = Field(None, description="SKU完整名称(名称-颜色-尺码)")
    totalQty: Optional[int] = Field(None, description="总库存数量")
    availableQty: Optional[int] = Field(None, description="可用库存数量")
    frozenQty: Optional[int] = Field(default=0, description="冻结库存数量")
    lockedQty: Optional[int] = Field(default=0, description="锁定库存数量")
    inflatedQty: Optional[int] = Field(default=0, description="膨胀库存数量")
    generalQty: Optional[int] = Field(None, description="常规库存数量")
    prepressQty: Optional[int] = Field(None, description="预压库存数量")
    createTime: Optional[int] = Field(None, description="创建时间戳(毫秒)")
    creator: Optional[str] = Field(None, description="创建人ID")
    creatorName: Optional[str] = Field(None, description="创建人姓名")
    sourceNO: Optional[str] = Field(None, description="来源单号")
    lpnId: Optional[str] = Field(None, description="容器ID")
    lpnNo: Optional[str] = Field(None, description="容器号")
    warehousePassageCode: Optional[str] = Field(None, description="库道编码")
    storageMode: Optional[str] = Field(None, description="存储方式")
    season: Optional[str] = Field(None, description="季节")
    activeType: Optional[str] = Field(None, description="活性类型")
    levelOneName: Optional[str] = Field(None, description="一级分类")
    levelTwoName: Optional[str] = Field(None, description="二级分类")
    levelThreeName: Optional[str] = Field(None, description="三级分类")
    fnsku: Optional[str] = Field(None, description="FNSKU")
    shopName: Optional[str] = Field(None, description="店铺名称")


class InventoryResponse(BaseModel):
    """
    库位库存查询响应模型

    对应接口整体返回的 JSON：
        { "code": 0, "data": { "list": [...], "total": 1 }, "msg": "" }
    """
    code: int = Field(..., description="状态码，0表示成功")
    data: Optional[dict] = Field(None, description="数据对象，含 list 和 total")
    msg: str = Field(default="", description="提示信息")

    @property
    def list(self) -> List[InventoryRecord]:
        """从 data.list 解析出库存记录列表（带类型转换，保证安全）"""
        if not self.data or "list" not in self.data:
            return []
        records = []
        for item in self.data["list"]:
            try:
                records.append(InventoryRecord(**item))
            except Exception:
                # 单条解析失败不影响其他记录
                continue
        return records

    @property
    def total(self) -> int:
        """总记录数"""
        if not self.data or "total" not in self.data:
            return 0
        return self.data["total"]


class ErrorResponse(BaseModel):
    """
    错误响应模型
    """
    error_code: str = Field(..., description="错误代码")
    error_message: str = Field(..., description="错误描述")
    details: Optional[str] = Field(None, description="详细错误信息")
    timestamp: datetime = Field(default_factory=datetime.now, description="错误发生时间")
