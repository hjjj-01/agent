"""
WMS客户端模块

这个模块封装了与WMS系统交互的所有逻辑。
使用requests库发送HTTP请求，处理API响应。
"""
import json
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
from loguru import logger
import requests

from .models import (
    EmployeeApiRequest,
    EmployeeApiResponse,
    ErrorResponse,
)


class WMSClient:
    """
    WMS系统客户端
    用于与WMS系统进行交互，获取员工工作量等数据。
    """
    # WMSClient 类的构造函数（初始化方法），在创建 WMSClient 对象时自动调用，用于设置客户端的基本配置并准备数据。
    def __init__(self, api_base_url: str, api_token: str):
        """
        初始化WMS客户端

        Args:
            api_base_url: WMS系统API地址
            api_token: 认证token
        """
        self.api_base_url = api_base_url
        self.api_token = api_token
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {api_token}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        })
        logger.info(f"WMS客户端初始化完成，API地址: {api_base_url}")

    

    def query_employee_api(self, params: EmployeeApiRequest) -> EmployeeApiResponse:
        """
        查询员工工作量API
        """
        logger.info(f"执行员工API查询，参数: {params}")

        try:
            request_model = EmployeeApiRequest(**params)

            payload = request_model.dict()
            # 调用http请求通用方法
            response_data = self._make_request(
                method="GET",
                endpoint="wms/work-record/person-kpi",
                params=payload
            )
            # 解析响应数据
            result = EmployeeApiResponse(**response_data)

            logger.info(f"员工API查询成功，返回 {result.total} 条记录")
            return result

        except Exception as e:
            logger.error(f"员工API查询失败: {str(e)}")
            raise





    # 发送HTTP请求的通用方法，_make_request 以下划线开头，表示这是一个 私有方法，只能在类内部调用。
    def _make_request(self, method: str, endpoint: str, **kwargs) -> Dict[str, Any]:

        # 去掉api_base_url和endpoint的首尾斜杠，确保URL格式正确
        url = f"{self.api_base_url.rstrip('/')}/{endpoint.lstrip('/')}"

        try:
            logger.info(f"发送{method}请求: {url}")
            logger.debug(f"请求参数: {json.dumps(kwargs.get('json', {}), ensure_ascii=False)}")

            response = self.session.request(
                method=method,
                url=url,
                timeout=30,
                **kwargs
            )

            # 检查响应状态，非2xx范围抛出异常
            response.raise_for_status()

            result = response.json()
            logger.debug(f"响应数据: {json.dumps(result, ensure_ascii=False)[:500]}")

            return result

        except requests.exceptions.RequestException as e:
            logger.error(f"HTTP请求失败: {str(e)}")
            if response is not None:
                logger.error(f"响应状态码: {response.status_code}")
                try:
                    logger.error(f"响应内容: {response.text[:500]}")
                except:
                    pass
            raise

        except json.JSONDecodeError as e:
            logger.error(f"响应解析失败: {str(e)}")
            raise

    