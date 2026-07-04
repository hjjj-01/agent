"""
WMS AI Agent - 主入口文件

整合所有模块，启动完整的WMS智能助手系统。

系统组成：
1. WMS客户端 - 获取实时数据
2. RAG知识库 - 检索知识信息
3. LangChain Agent - 智能对话引擎
4. 飞书机器人 - 用户交互界面

启动方式：
1. 配置.env文件（填写API密钥等）
2. 运行: python main.py

飞书机器人模式：
    系统启动Web服务器，接收飞书消息，实时处理和回复。

测试模式：
    通过/test接口直接测试Agent功能，无需飞书配置。
"""
import sys
import io
from pathlib import Path

# 设置控制台编码为UTF-8（解决Windows GBK编码问题）
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
elif sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# 将src目录添加到Python路径
sys.path.insert(0, str(Path(__file__).parent))

from src.utils import load_config, setup_logger
from src.agent.wms_agent import create_wms_agent
from src.feishu.server import create_feishu_server
from loguru import logger


def main():
    """
    主函数 - 启动WMS AI Agent系统

    启动流程：
        1. 加载配置
        2. 设置日志
        3. 创建Agent
        4. 创建飞书服务器
        5. 启动服务
    """
    print("=" * 70)
    print("WMS AI Agent - 仓库管理智能助手")
    print("=" * 70)
    print()

    # 1. 加载配置
    print("[1/5] 加载配置文件...")
    try:
        config = load_config()
        print(f"  ✓ 配置加载成功")
        print(f"  - OpenAI模型: {config.openai.model}")
        print(f"  - 服务端口: {config.server.port}")
        print(f"  - 日志级别: {config.server.log_level}")
    except Exception as e:
        print(f"  ✗ 配置加载失败: {str(e)}")
        print("  提示: 请检查.env文件是否配置正确")
        print("  参考: .env.example文件中的配置说明")
        return

    print()

    # 2. 设置日志
    print("[2/5] 设置日志系统...")
    logger = setup_logger(config.server.log_level, "./logs/app.log")
    print(f"  ✓ 日志系统设置成功")
    print(f"  - 日志文件: ./logs/app.log")
    print()

    logger.info("=" * 70)
    logger.info("WMS AI Agent 启动中...")
    logger.info("=" * 70)

    # 3. 创建WMS Agent
    print("[3/5] 创建WMS Agent...")
    logger.info("创建WMS Agent")

    try:
        agent = create_wms_agent(config)
        print(f"  ✓ WMS Agent创建成功")
        print(f"  - 工具数量: {len(agent.get_available_tools())}")
        print(f"  - 工具列表: {', '.join(agent.get_available_tools())}")
        logger.success("WMS Agent创建成功")
    except Exception as e:
        print(f"  ✗ WMS Agent创建失败: {str(e)}")
        logger.error(f"WMS Agent创建失败: {str(e)}")
        print("  提示: 请检查OpenAI API密钥是否正确")
        return

    print()

    # 4. 创建飞书服务器
    print("[4/5] 创建飞书机器人服务器...")
    logger.info("创建飞书服务器")

    try:
        server = create_feishu_server(config, agent)
        print(f"  ✓ 飞书服务器创建成功")
        print(f"  - 服务地址: http://0.0.0.0:{config.server.port}")
        print(f"  - Webhook路径: /webhook")
        print(f"  - 健康检查: /health")
        print(f"  - 测试接口: /test")
        logger.success("飞书服务器创建成功")
    except Exception as e:
        print(f"  ✗ 飞书服务器创建失败: {str(e)}")
        logger.error(f"飞书服务器创建失败: {str(e)}")
        return

    print()

    # 5. 启动服务
    print("[5/5] 启动服务...")
    print()
    print("=" * 70)
    print("服务已启动！")
    print("=" * 70)
    print()
    print("使用方式：")
    print()
    print("1. 飞书机器人模式：")
    print("   - 在飞书开放平台配置事件订阅URL:")
    print(f"     http://your-server-ip:{config.server.port}/webhook")
    print("   - 用户在飞书中与机器人对话")
    print()
    print("2. 测试模式（无需飞书）：")
    print(f"   - 访问测试接口: http://localhost:{config.server.port}/test")
    print("   - POST请求示例:")
    print("     {")
    print("       \"message\": \"查询SKU001的库存\"")
    print("     }")
    print()
    print("3. API文档：")
    print(f"   - 访问: http://localhost:{config.server.port}/docs")
    print()
    print("=" * 70)
    print()

    logger.success("WMS AI Agent 启动成功！")
    logger.info(f"服务端口: {config.server.port}")
    logger.info(f"访问地址: http://localhost:{config.server.port}")

    # 启动Web服务器
    try:
        server.start()
    except KeyboardInterrupt:
        logger.info("收到停止信号，服务关闭")
        print()
        print("服务已停止")
    except Exception as e:
        logger.error(f"服务运行错误: {str(e)}")
        print()
        print(f"服务运行错误: {str(e)}")


def test_agent_directly():
    """
    直接测试Agent功能（不启动服务器）

    用于快速测试Agent是否正常工作。
    """
    print("=" * 70)
    print("WMS Agent 直接测试模式")
    print("=" * 70)
    print()

    # 加载配置
    config = load_config()
    logger = setup_logger("DEBUG")

    # 创建Agent
    print("创建WMS Agent...")
    agent = create_wms_agent(config)
    print("Agent创建成功！")
    print()

    # 测试对话
    test_questions = [
        "查询SKU001的库存",
        "查询所有库存",
        "检查低库存商品",
        "仓库概览"
    ]

    print("开始测试对话...")
    print("-" * 70)

    for question in test_questions:
        print(f"\n问题: {question}")
        print("-" * 70)
        response = agent.chat(question)
        print(f"回答: {response}")
        print("-" * 70)

    print("\n测试完成！")


if __name__ == "__main__":
    # 判断运行模式
    import argparse

    parser = argparse.ArgumentParser(description="WMS AI Agent")
    parser.add_argument(
        "--test",
        action="store_true",
        help="直接测试Agent，不启动服务器"
    )

    args = parser.parse_args()

    if args.test:
        # 测试模式
        test_agent_directly()
    else:
        # 正常启动模式
        main()