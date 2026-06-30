"""
文档处理器模块

功能:
1. 加载各种格式的文档（文本、Markdown、JSON等）
2. 将长文档分割成小块（chunks）
3. 为每个文档块添加元数据

为什么要分割文档？
- LLM有上下文长度限制，无法处理超长文档
- 小块文档检索更精确，能找到最相关的部分
- 向量化时小块的效果更好

分割策略:
- 按字符数分割：简单直接，适合结构简单的文档
- 按段落分割：保持语义完整性
- 递归分割：根据文档结构智能分割
"""
from typing import List, Dict, Any
from pathlib import Path
from loguru import logger

# LangChain文档加载器和分割器
from langchain.schema import Document
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter,
    MarkdownHeaderTextSplitter
)


class DocumentProcessor:
    """
    文档处理器

    用于加载和处理文档，将其分割成适合向量化的块。

    使用方式:
        processor = DocumentProcessor(chunk_size=500, chunk_overlap=50)
        documents = processor.load_text_file("manual.txt")
        chunks = processor.split_documents(documents)
    """

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        separators: List[str] = None
    ):
        """
        初始化文档处理器

        Args:
            chunk_size: 每个文档块的最大字符数
            chunk_overlap: 文档块之间的重叠字符数（确保上下文连续）
            separators: 分割符列表，按优先级尝试分割

        说明:
            chunk_overlap的作用：
            例如chunk_size=500，overlap=50
            第一个块: 字符0-500
            第二个块: 字符450-950（与前一个块重叠50字符）
            这样可以避免重要信息在分割处被切断
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # 默认分割符：优先按段落、句子分割，最后才按字符分割
        if separators is None:
            separators = ["\\n\\n", "\\n", " ", ""]

        # 初始化文本分割器（递归分割，智能选择分割点）
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=separators,
            length_function=len
        )

        # Markdown分割器（按标题结构分割）
        self.markdown_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[
                ("#", "header1"),
                ("##", "header2"),
                ("###", "header3"),
            ]
        )

        logger.info(f"文档处理器初始化完成，块大小: {chunk_size}, 重叠: {chunk_overlap}")

    def load_text_file(self, file_path: str, metadata: Dict[str, Any] = None) -> List[Document]:
        """
        加载文本文件

        Args:
            file_path: 文件路径
            metadata: 文档元数据（如标题、来源、作者等）

        Returns:
            Document对象列表

        说明:
            Document对象包含两部分：
            - page_content: 文本内容
            - metadata: 元数据字典
        """
        logger.info(f"加载文本文件: {file_path}")

        path = Path(file_path)
        if not path.exists():
            logger.error(f"文件不存在: {file_path}")
            raise FileNotFoundError(f"文件不存在: {file_path}")

        # 读取文件内容
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        # 构建元数据
        if metadata is None:
            metadata = {}

        metadata["source"] = file_path
        metadata["filename"] = path.name
        metadata["file_type"] = "text"

        # 创建Document对象
        document = Document(page_content=content, metadata=metadata)

        logger.success(f"文件加载成功，内容长度: {len(content)}字符")
        return [document]

    def load_markdown_file(self, file_path: str, metadata: Dict[str, Any] = None) -> List[Document]:
        """
        加载Markdown文件

        Args:
            file_path: 文件路径
            metadata: 文档元数据

        Returns:
            Document对象列表

        说明:
            Markdown文件会按标题结构分割，保留文档的组织结构
        """
        logger.info(f"加载Markdown文件: {file_path}")

        path = Path(file_path)
        if not path.exists():
            logger.error(f"文件不存在: {file_path}")
            raise FileNotFoundError(f"文件不存在: {file_path}")

        # 读取文件内容
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        # 构建元数据
        if metadata is None:
            metadata = {}

        metadata["source"] = file_path
        metadata["filename"] = path.name
        metadata["file_type"] = "markdown"

        # 按标题分割Markdown
        md_documents = self.markdown_splitter.split_text(content)

        # 为每个分割后的文档添加元数据
        for doc in md_documents:
            doc.metadata.update(metadata)

        logger.success(f"Markdown文件加载成功，按标题分割为 {len(md_documents)} 个部分")
        return md_documents

    def load_json_data(self, data: Dict[str, Any], source: str = "json_data") -> List[Document]:
        """
        从JSON数据创建Document

        Args:
            data: JSON数据字典
            source: 数据来源标识

        Returns:
            Document对象列表

        说明:
            用于将结构化数据转换为文档，例如：
            - 将WMS的产品信息转为文档
            - 将业务规则转为文档
        """
        logger.info(f"从JSON数据创建Document")

        documents = []

        # 遍历JSON数据，为每个条目创建Document
        for key, value in data.items():
            if isinstance(value, dict):
                # 如果value是字典，将其转为格式化的文本
                content = self._dict_to_text(value)
                metadata = {
                    "source": source,
                    "key": key,
                    "file_type": "json"
                }
                doc = Document(page_content=content, metadata=metadata)
                documents.append(doc)

        logger.success(f"从JSON数据创建了 {len(documents)} 个Document")
        return documents

    def _dict_to_text(self, data: Dict[str, Any]) -> str:
        """
        将字典转换为文本格式

        Args:
            data: 字典数据

        Returns:
            格式化的文本字符串
        """
        lines = []
        for key, value in data.items():
            lines.append(f"{key}: {value}")
        return "\\n".join(lines)

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        分割文档

        Args:
            documents: Document对象列表

        Returns:
            分割后的Document对象列表

        说明:
            使用递归分割器，智能选择分割点：
            1. 优先尝试按段落分割（\\n\\n）
            2. 如果段落太长，按行分割（\\n）
            3. 如果行太长，按句子分割（空格）
            4. 最后按字符分割
        """
        logger.info(f"开始分割 {len(documents)} 个文档")

        # 使用分割器分割文档
        chunks = self.text_splitter.split_documents(documents)

        # 为每个块添加分割信息
        for i, chunk in enumerate(chunks):
            chunk.metadata["chunk_id"] = i
            chunk.metadata["chunk_size"] = len(chunk.page_content)

        logger.success(f"文档分割完成，从 {len(documents)} 个文档分割为 {len(chunks)} 个块")
        return chunks

    def process_directory(self, directory_path: str, file_types: List[str] = None) -> List[Document]:
        """
        处理目录中的所有文件

        Args:
            directory_path: 目录路径
            file_types: 要处理的文件类型列表，如[".txt", ".md"]

        Returns:
            所有文档分割后的块列表
        """
        logger.info(f"处理目录: {directory_path}")

        if file_types is None:
            file_types = [".txt", ".md"]

        directory = Path(directory_path)
        if not directory.exists():
            logger.error(f"目录不存在: {directory_path}")
            raise FileNotFoundError(f"目录不存在: {directory_path}")

        all_chunks = []

        # 遍历目录中的文件
        for file_path in directory.rglob("*"):
            if file_path.is_file() and file_path.suffix in file_types:
                logger.info(f"处理文件: {file_path}")

                # 根据文件类型选择加载方法
                if file_path.suffix == ".txt":
                    documents = self.load_text_file(str(file_path))
                elif file_path.suffix == ".md":
                    documents = self.load_markdown_file(str(file_path))
                else:
                    continue

                # 分割文档
                chunks = self.split_documents(documents)
                all_chunks.extend(chunks)

        logger.success(f"目录处理完成，总共生成 {len(all_chunks)} 个文档块")
        return all_chunks

    def create_sample_documents(self) -> List[Document]:
        """
        创建示例文档（用于演示和学习）

        Returns:
            示例Document对象列表

        说明:
            这个方法创建了一些WMS相关的示例文档，包括：
            - 操作手册
            - 业务规则
            - 产品信息

            当你有真实文档时，可以使用load_text_file等方法加载。
        """
        logger.info("创建示例文档（用于演示和学习）")

        documents = []

        # 示例1：WMS操作手册
        manual_content = """
# WMS系统操作手册

## 库存查询操作

### 1. 查询单个SKU库存
在飞书机器人中输入：查询SKU001的库存
系统会返回该SKU的详细信息，包括当前库存、可用库存、安全库存等。

### 2. 查询所有库存
输入：查询所有库存
系统会列出所有SKU的库存状况，并标注低库存商品。

### 3. 检查低库存
输入：检查低库存
系统会分析库存数据，返回需要补货的商品列表和建议补货数量。

## 订单管理操作

### 1. 查询订单
可以按订单号、状态或类型查询订单：
- 查询订单号ORD20240115001
- 查询所有待处理订单
- 查询所有采购订单

### 2. 订单统计
输入：订单统计
系统会返回订单的汇总信息，包括总订单数、待处理订单数、总金额等。

## 入库管理操作

### 1. 查询入库记录
可以按入库单号或SKU查询入库历史：
- 查询入库单IN20240114001
- 查询SKU001的入库记录

### 2. 入库流程
标准入库流程：
1. 供应商送货到仓库
2. 仓库接收并核对数量
3. 进行质量检查
4. 上架入库
5. 更新库存系统

## 仓库概览

输入：仓库概览
系统会提供仓库的整体运营状况，包括库存概况、订单概况和运营建议。

## 常见问题

### Q1: 如何判断是否需要补货？
系统会自动对比当前库存和安全库存，低于安全库存时提示需要补货。

### Q2: 订单状态说明
- pending: 待处理，订单已创建但未开始执行
- processing: 处理中，订单正在执行
- completed: 已完成，订单已成功完成
- cancelled: 已取消

### Q3: 库存位置编码规则
位置编码格式：区域-排-层
例如：A-01-02 表示A区域第1排第2层
"""
        documents.append(Document(
            page_content=manual_content,
            metadata={
                "source": "wms_manual",
                "title": "WMS系统操作手册",
                "category": "manual",
                "file_type": "markdown"
            }
        ))

        # 示例2：业务规则
        rules_content = """
# WMS业务规则

## 库存管理规则

### 安全库存设置
每个SKU都有预设的安全库存阈值：
- 当库存数量低于安全库存时，系统自动提示需要补货
- 安全库存通常设置为历史平均销量的1.5倍
- 可以根据季节性调整安全库存

### 库存周转管理
库存周转率计算：
周转率 = 销售数量 / 平均库存数量

健康指标：
- 周转率 > 12: 库存周转良好
- 周转率 6-12: 库存周转正常
- 周转率 < 6: 库存积压，需要优化

### 库存预警机制
系统提供三级库存预警：
- 绿色：库存充足，数量 > 安全库存 * 1.5
- 黄色：库存偏低，数量在安全库存到安全库存*1.5之间
- 红色：库存危急，数量 < 安全库存

## 订单处理规则

### 订单优先级
订单按以下优先级处理：
1. 紧急订单（标注为优先发货）
2. 正常订单
3. 批量订单

### 订单审核流程
销售订单需要审核：
- 订单金额 < 5000元：自动审核通过
- 订单金额 5000-10000元：需要主管审核
- 订单金额 > 10000元：需要经理审核

### 订单时效要求
- 待处理订单应在24小时内开始处理
- 处理中的订单应在48小时内完成
- 超时未完成的订单自动提醒

## 入库管理规则

### 入库验收标准
入库前必须进行质量检查：
1. 核对数量与订单一致
2. 检查商品外观无损坏
3. 检查包装完好
4. 核对批次号和生产日期

### 入库上架规则
商品上架按照就近原则：
- 优先放置在同类商品附近
- 高频商品放置在低层货架
- 大件商品放置在底层

### 入库记录保存
入库记录必须包含：
- 入库单号
- 供应商信息
- 批次号
- 入库时间
- 上架位置
- 质检结果

## 库位管理规则

### 库位编码规则
库位编码采用三级编码：
- 第一级：区域（A-E）
- 第二级：排号（01-99）
- 第三级：层号（01-99）

完整编码示例：A-01-02

### 库位分配原则
不同类型商品分配不同区域：
- A区：电子元件类
- B区：包装材料类
- C区：办公用品类
- D区：家居用品类
- E区：食品原料类

### 库位利用率
目标库位利用率：
- 正常运行：70-80%
- 备货高峰：85-90%
- 避免超过90%，留出周转空间
"""
        documents.append(Document(
            page_content=rules_content,
            metadata={
                "source": "wms_rules",
                "title": "WMS业务规则",
                "category": "rules",
                "file_type": "markdown"
            }
        ))

        # 示例3：产品信息
        products_content = """
# WMS产品信息

## SKU001 - 商品A（电子元件）

### 基本信息
- SKU编码: SKU001
- 商品名称: 电子元件套装
- 商品类别: 电子元件
- 单位: 套
- 单位成本: 25.50元
- 销售价格: 30.00元

### 库存信息
- 安全库存: 100套
- 当前库存: 150套
- 可用库存: 120套
- 库存位置: A-01-02

### 供应商信息
- 主要供应商: 供应商A
- 供货周期: 7天
- 最小订货量: 50套

### 特殊说明
该商品为高频商品，建议保持较高库存水平。
包装规格：每套包含10个标准元件。

## SKU002 - 商品B（包装材料）

### 基本信息
- SKU编码: SKU002
- 商品名称: 标准包装箱
- 商品类别: 包装材料
- 单位: 个
- 单位成本: 15.00元
- 销售价格: 18.00元

### 库存信息
- 安全库存: 100个
- 当前库存: 80个
- 可用库存: 80个
- 库存位置: B-02-03

### 供应商信息
- 主要供应商: 供应商B
- 供货周期: 5天
- 最小订货量: 100个

### 特殊说明
当前库存低于安全库存，建议尽快补货。

## SKU003 - 商品C（办公用品）

### 基本信息
- SKU编码: SKU003
- 商品名称: 办公文具套装
- 商品类别: 办公用品
- 单位: 套
- 单位成本: 8.50元
- 销售价格: 10.00元

### 库存信息
- 安全库存: 150套
- 当前库存: 250套
- 可用库存: 200套
- 库存位置: C-01-01

### 特殊说明
库存充足，近期无需补货。
需求稳定，周转率约8次/年。

## SKU004 - 商品D（家居用品）

### 基本信息
- SKU编码: SKU004
- 商品名称: 家居收纳盒
- 商品类别: 家居用品
- 单位: 个
- 单位成本: 45.00元
- 销售价格: 55.00元

### 库存信息
- 安全库存: 50个
- 当前库存: 40个
- 可用库存: 40个
- 库存位置: D-03-02

### 特殊说明
库存危急，需要立即补货。
单品价值较高，需注意防盗。

## SKU005 - 商品E（食品原料）

### 基本信息
- SKU编码: SKU005
- 商品名称: 食品添加剂原料
- 商品类别: 食品原料
- 单位: 千克
- 单位成本: 12.00元
- 销售价格: 15.00元

### 库存信息
- 安全库存: 200千克
- 当前库存: 500千克
- 可用库存: 450千克
- 库存位置: E-01-03

### 特殊说明
库存充足，但需注意保质期管理。
平均保质期：12个月。
"""
        documents.append(Document(
            page_content=products_content,
            metadata={
                "source": "wms_products",
                "title": "WMS产品信息",
                "category": "products",
                "file_type": "markdown"
            }
        ))

        logger.success(f"创建了 {len(documents)} 个示例文档")
        return documents