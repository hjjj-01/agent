"""
WMS数据模型定义
"""
from typing import Optional, List
from datetime import datetime
from pydantic import BaseModel, Field


# ==================== 员工API相关模型 ====================

class EmployeeApiRequest(BaseModel):
    """
    员工API请求模型
    """
    pageNo: int = Field(..., description="页码")
    pageSize: int = Field(..., description="每页数量")
    nodeIdList: List[str] = Field(..., description="岗位id列表")
    endTime: List[str] = Field(..., description="结束时间范围，格式：['YYYY-MM-DD HH:mm:ss', 'YYYY-MM-DD HH:mm:ss']")

    class Config:
        json_schema_extra = {
            "example": {
                "pageNo": 1,
                "pageSize": 100,
                "nodeIdList": ["1928371049917349889"],
                "endTime": ["2026-07-02 00:00:00", "2026-07-02 23:59:59"],
            }
        }


class WorkRecord(BaseModel):
    """
    单个员工工作量记录
    """
    createTime: List[int] = Field(..., description="创建日期 [年,月,日]")
    worker: str = Field(..., description="员工ID")
    workerName: Optional[str] = Field(None, description="员工姓名")
    nodeName: str = Field(..., description="岗位名称")
    nodeCode: str = Field(..., description="岗位编码")
    workQty: int = Field(..., description="工作数量")
    outputQty: int = Field(..., description="产出数量")
    unit: str = Field(..., description="单位")
    itemQty: Optional[int] = Field(None, description="单品数量")
    weight: float = Field(..., description="重量")
    volume: float = Field(..., description="体积")
    listingCount: int = Field(..., description="上架数量")
    skuCategory: int = Field(..., description="SKU类别")
    packCount: int = Field(..., description="打包数量")


class EmployeeApiData(BaseModel):
    """
    员工API返回数据对象
    """
    list: List[WorkRecord] = Field(..., description="工作记录列表")
    total: int = Field(..., description="总记录数")

class EmployeeApiResponse(BaseModel):
    """
    员工API响应模型
    """
    code: int = Field(..., description="状态码，0表示成功")
    data: EmployeeApiData = Field(..., description="数据对象")
    msg: str = Field(..., description="提示信息")


class ErrorResponse(BaseModel):
    """
    错误响应模型
    """
    error_code: str = Field(..., description="错误代码")
    error_message: str = Field(..., description="错误描述")
    details: Optional[str] = Field(None, description="详细错误信息")
    timestamp: datetime = Field(default_factory=datetime.now, description="错误发生时间")