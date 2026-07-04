"""
日志配置模块
功能：配置应用日志系统
"""
import sys
from pathlib import Path
# loguru 的 logger 是一个全局共享的单例对象！
from loguru import logger


def setup_logger(log_level: str = "INFO", log_file: str = None):
    """
    设置日志系统

    Args:
        log_level: 日志级别（DEBUG, INFO, WARNING, ERROR, CRITICAL）
        log_file: 日志文件路径（可选）

    说明:
        日志格式：
        - 时间戳
        - 日志级别
        - 模块名
        - 日志消息

        输出：
        - 控制台输出（彩色）
        - 文件输出（如果指定）
    """
    # 设置控制台编码为UTF-8（解决Windows GBK编码问题）
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    elif sys.platform == 'win32':
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

    # 清空默认格式
    logger.remove()

    # 添加控制台输出（带颜色）
    logger.add(
        sys.stdout,
        level=log_level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
               "<level>{level: <8}</level> | "
               "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
               "<level>{message}</level>",
        colorize=True
    )

    # 如果指定了日志文件，添加文件输出
    if log_file:
        # 确保日志目录存在
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        logger.add(
            log_file,
            level=log_level,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
            rotation="10 MB",  # 日志文件达到10MB时轮转
            retention="7 days",  # 保留7天的日志
            compression="zip"  # 压缩旧日志
        )

    return logger