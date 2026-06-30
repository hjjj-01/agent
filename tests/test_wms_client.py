"""
WMS客户端测试脚本

这个脚本用于测试WMS客户端的功能。
运行方式：python tests/test_wms_client.py

通过这个脚本，你可以看到：
1. 如何创建WMS客户端
2. 如何查询库存
3. 如何查询订单
4. 如何获取统计数据
"""
import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.wms import WMSClient
from src.utils import setup_logger
import json


def test_wms_client():
    """测试WMS客户端的各项功能"""

    # 设置日志
    logger = setup_logger("DEBUG")

    print("=" * 60)
    print("WMS客户端功能测试")
    print("=" * 60)

    # 创建客户端（使用模拟配置）
    client = WMSClient(
        api_base_url="http://mock-wms-api.com",
        api_token="mock-token-123"
    )

    print("\n1. 测试库存查询功能")
    print("-" * 60)

    # 查询特定SKU的库存
    print("\n查询SKU001的库存：")
    result = client.query_inventory(sku="SKU001")
    for item in result.items:
        print(f"  SKU: {item.sku}")
        print(f"  商品名称: {item.product_name}")
        print(f"  当前库存: {item.quantity}")
        print(f"  可用库存: {item.available_quantity}")
        print(f"  安全库存: {item.safety_stock}")
        print(f"  库存位置: {item.warehouse_location}")

    # 查询所有库存
    print("\n查询所有库存：")
    all_inventory = client.query_inventory()
    print(f"  总共 {all_inventory.total_count} 个SKU")
    for item in all_inventory.items:
        status = "[正常]" if item.quantity >= item.safety_stock else "[低库存]"
        print(f"  {item.sku} - {item.product_name}: {item.quantity}件 [{status}]")

    print("\n2. 测试订单查询功能")
    print("-" * 60)

    # 查询所有订单
    print("\n查询所有订单：")
    all_orders = client.query_orders()
    print(f"  总共 {all_orders.total_count} 个订单")
    for order in all_orders.orders:
        print(f"\n  订单号: {order.order_id}")
        print(f"  类型: {order.order_type}")
        print(f"  客户: {order.customer_name}")
        print(f"  状态: {order.status}")
        print(f"  总金额: {order.total_amount}元")
        print(f"  商品明细:")
        for item in order.items:
            print(f"    - {item.sku}: {item.quantity}件 x {item.unit_price}元")

    # 查询特定状态的订单
    print("\n查询待处理订单：")
    pending_orders = client.query_orders(status="pending")
    print(f"  待处理订单数: {pending_orders.total_count}")

    print("\n3. 测试入库记录查询")
    print("-" * 60)

    inbounds = client.query_inbounds()
    print(f"  总共 {inbounds.total_count} 条入库记录")
    for record in inbounds.records:
        print(f"\n  入库单号: {record.inbound_id}")
        print(f"  SKU: {record.sku}")
        print(f"  数量: {record.quantity}")
        print(f"  供应商: {record.supplier_name}")
        print(f"  状态: {record.status}")

    print("\n4. 测试库存统计功能")
    print("-" * 60)

    stats = client.get_inventory_statistics()
    print(f"  总SKU数: {stats.total_items}")
    print(f"  总库存数量: {stats.total_quantity}")
    print(f"  总库存价值: {stats.total_value}元")
    print(f"  低库存SKU数: {stats.low_stock_items}")
    print(f"  库存过多SKU数: {stats.overstock_items}")

    print("\n5. 测试订单统计功能")
    print("-" * 60)

    order_stats = client.get_order_statistics()
    print(f"  总订单数: {order_stats.total_orders}")
    print(f"  待处理订单: {order_stats.pending_orders}")
    print(f"  已完成订单: {order_stats.completed_orders}")
    print(f"  订单总金额: {order_stats.total_amount}元")

    print("\n6. 测试低库存检查功能")
    print("-" * 60)

    low_stock = client.check_low_stock()
    print(f"  低库存SKU数: {low_stock['low_stock_count']}")
    if low_stock['low_stock_count'] > 0:
        print(f"  需要补货的商品:")
        for item in low_stock['items']:
            print(f"\n    SKU: {item['sku']}")
            print(f"    商品名称: {item['product_name']}")
            print(f"    当前库存: {item['current_quantity']}件")
            print(f"    安全库存: {item['safety_stock']}件")
            print(f"    建议补货: {item['suggested_replenishment']}件")
            print(f"    紧急程度: {item['urgency']}")

    print("\n7. 测试仓库概览功能")
    print("-" * 60)

    summary = client.get_warehouse_summary()
    print("\n仓库运营概览:")
    print(f"  库存概况:")
    inv_summary = summary['inventory_summary']
    print(f"    - 总SKU数: {inv_summary['total_items']}")
    print(f"    - 总库存数量: {inv_summary['total_quantity']}")
    print(f"    - 总库存价值: {inv_summary['total_value']}元")
    print(f"    - 低库存警告: {inv_summary['low_stock_alert']}个")

    print(f"  订单概况:")
    ord_summary = summary['order_summary']
    print(f"    - 总订单数: {ord_summary['total_orders']}")
    print(f"    - 待处理订单: {ord_summary['pending_orders']}")
    print(f"    - 订单总金额: {ord_summary['total_amount']}元")

    print(f"\n  运营建议:")
    for rec in summary['recommendations']:
        print(f"    - {rec}")

    print("\n" + "=" * 60)
    print("[OK] 所有测试完成！")
    print("=" * 60)


if __name__ == "__main__":
    test_wms_client()