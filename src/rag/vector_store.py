"""
向量数据库模块

功能:
1. 将文档转换为向量（embeddings）
2. 存储向量到数据库
3. 根据查询向量检索相似文档

核心概念:
- Embeddings（嵌入/向量化）: 将文本转换为数字向量，使计算机能理解语义
- 向量数据库: 专门存储和检索向量的数据库
- 相似度检索: 找到与查询向量最相似的文档向量

为什么需要向量化？
- 传统关键词搜索：匹配文字，无法理解语义
- 向量搜索：理解语义含义，找到语义相近的内容
  例如："如何查询库存" 和 "库存查询方法" 虽然文字不同，但语义相近，
  向量搜索能找到两者，关键词搜索可能找不到。
"""
from typing import List, Optional
from pathlib import Path
from loguru import logger

# LangChain相关导入
from langchain.schema import Document
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

# ChromaDB
import chromadb
from chromadb.config import Settings


class VectorStore:
    """
    向量数据库管理类

    使用ChromaDB作为向量数据库，OpenAI Embeddings作为向量化模型。

    使用流程:
        1. 初始化VectorStore
        2. 添加文档到数据库（add_documents）
        3. 检索相关文档（search）

    示例:
        store = VectorStore(persist_directory="./data/chroma_db")
        store.add_documents(documents)
        results = store.search("如何查询库存", k=3)
    """

    def __init__(
        self,
        persist_directory: str,
        openai_api_key: str,
        openai_api_base: str = "https://api.openai.com/v1",
        collection_name: str = "wms_knowledge_base"
    ):
        """
        初始化向量数据库

        Args:
            persist_directory: 数据库持久化目录
            openai_api_key: OpenAI API密钥（用于Embeddings）
            openai_api_base: OpenAI API基础URL
            collection_name: 数据库集合名称

        说明:
            ChromaDB:
            - 轻量级向量数据库，适合开发和中小型项目
            - 支持本地持久化，数据保存在磁盘上
            - 提供高效的向量检索功能

            OpenAI Embeddings:
            - 使用text-embedding-ada-002模型
            - 输入文本 → 输出1536维向量
            - 高质量的语义向量表示
        """
        self.persist_directory = persist_directory
        self.collection_name = collection_name

        logger.info(f"初始化向量数据库，目录: {persist_directory}")

        # 确保目录存在
        Path(persist_directory).mkdir(parents=True, exist_ok=True)

        # 初始化OpenAI Embeddings
        # Embeddings是将文本转换为向量的模型
        self.embeddings = OpenAIEmbeddings(
            openai_api_key=openai_api_key,
            openai_api_base=openai_api_base,
            model="text-embedding-ada-002"  # OpenAI的embedding模型
        )

        # 初始化ChromaDB客户端
        # ChromaDB是一个向量数据库，用于存储和检索向量
        self.client = chromadb.Client(
            Settings(
                persist_directory=persist_directory,
                anonymized_telemetry=False  # 禁用匿名遥测
            )
        )

        # 创建或获取集合
        # 集合类似于数据库中的"表"，用于存储相关数据
        try:
            self.collection = self.client.get_or_create_collection(
                name=collection_name,
                metadata={"description": "WMS知识库"}
            )
            logger.info(f"向量数据库集合 '{collection_name}' 已创建/获取")
        except Exception as e:
            logger.error(f"创建向量数据库集合失败: {str(e)}")
            raise

        # LangChain的Chroma包装器（提供更方便的接口）
        self.vectorstore: Optional[Chroma] = None

        logger.success("向量数据库初始化完成")

    def add_documents(self, documents: List[Document]) -> int:
        """
        将文档添加到向量数据库

        Args:
            documents: Document对象列表

        Returns:
            添加的文档数量

        处理流程:
            1. 提取文档内容
            2. 使用Embeddings模型将文本转换为向量
            3. 存储向量到ChromaDB
            4. 保存元数据（用于后续过滤和展示）

        说明:
            向量化过程:
            输入: "如何查询库存"（文本）
            处理: OpenAI Embeddings模型
            输出: [0.123, 0.456, ..., 0.789]（1536维向量）

            存储内容:
            - 向量（用于检索）
            - 原文本（用于展示结果）
            - 元数据（如来源、标题等）
        """
        logger.info(f"开始添加 {len(documents)} 个文档到向量数据库")

        if not documents:
            logger.warning("没有文档需要添加")
            return 0

        try:
            # 使用LangChain的Chroma.from_documents方法
            # 这个方法会自动：
            # 1. 将每个文档转换为向量
            # 2. 存储到ChromaDB
            # 3. 保存元数据
            self.vectorstore = Chroma.from_documents(
                documents=documents,
                embedding=self.embeddings,
                persist_directory=self.persist_directory,
                collection_name=self.collection_name
            )

            # 持久化到磁盘
            self.vectorstore.persist()

            logger.success(f"成功添加 {len(documents)} 个文档到向量数据库")
            return len(documents)

        except Exception as e:
            logger.error(f"添加文档失败: {str(e)}")
            raise

    def search(
        self,
        query: str,
        k: int = 3,
        filter_dict: Optional[dict] = None
    ) -> List[Document]:
        """
        搜索相似文档

        Args:
            query: 查询文本
            k: 返回的最相似文档数量
            filter_dict: 元数据过滤条件（可选）

        Returns:
            最相似的Document对象列表

        搜索流程:
            1. 将查询文本转换为向量
            2. 在向量数据库中查找最相似的向量
            3. 返回对应的文档

        相似度计算:
            使用余弦相似度（Cosine Similarity）
            值范围：-1到1
            值越大表示越相似

        示例:
            query: "如何查询库存"
            → 向量化: [0.123, 0.456, ...]
            → 搜索数据库找到最相似的向量
            → 返回: ["库存查询操作手册", "库存管理规则", ...]
        """
        logger.info(f"搜索查询: '{query}'，返回 {k} 个结果")

        if not self.vectorstore:
            logger.error("向量数据库尚未初始化，请先添加文档")
            raise ValueError("向量数据库尚未初始化")

        try:
            # similarity_search方法会：
            # 1. 将query转换为向量
            # 2. 计算与数据库中所有向量的相似度
            # 3. 返回最相似的k个文档

            if filter_dict:
                # 如果有过滤条件，按元数据筛选
                results = self.vectorstore.similarity_search(
                    query,
                    k=k,
                    filter=filter_dict
                )
            else:
                # 无过滤条件，直接搜索
                results = self.vectorstore.similarity_search(query, k=k)

            logger.success(f"搜索完成，找到 {len(results)} 个相关文档")
            return results

        except Exception as e:
            logger.error(f"搜索失败: {str(e)}")
            raise

    def search_with_scores(
        self,
        query: str,
        k: int = 3
    ) -> List[tuple]:
        """
        搜索并返回相似度分数

        Args:
            query: 查询文本
            k: 返回的数量

        Returns:
            (Document, score)元组列表，score表示相似度

        说明:
            score范围通常在0-1之间，越大越相似。
            可以根据score筛选高质量结果。
        """
        logger.info(f"带分数搜索: '{query}'")

        if not self.vectorstore:
            logger.error("向量数据库尚未初始化")
            raise ValueError("向量数据库尚未初始化")

        try:
            # similarity_search_with_score返回文档和相似度分数
            results = self.vectorstore.similarity_search_with_score(query, k=k)

            logger.success(f"搜索完成，返回 {len(results)} 个结果（带分数）")
            return results

        except Exception as e:
            logger.error(f"搜索失败: {str(e)}")
            raise

    def delete_collection(self):
        """
        删除整个集合

        说明:
            清空向量数据库，用于重新初始化或清理数据。
            谨慎使用，数据不可恢复。
        """
        logger.warning(f"删除向量数据库集合: {self.collection_name}")

        try:
            self.client.delete_collection(self.collection_name)
            self.vectorstore = None
            logger.success("集合已删除")
        except Exception as e:
            logger.error(f"删除集合失败: {str(e)}")
            raise

    def get_collection_info(self) -> dict:
        """
        获取集合信息

        Returns:
            集合信息字典，包括文档数量等
        """
        try:
            count = self.collection.count()
            info = {
                "collection_name": self.collection_name,
                "document_count": count,
                "persist_directory": self.persist_directory
            }
            return info
        except Exception as e:
            logger.error(f"获取集合信息失败: {str(e)}")
            return {}

    def update_documents(self, documents: List[Document]):
        """
        更新文档（先删除旧数据，再添加新数据）

        Args:
            documents: 新的Document列表

        说明:
            当知识库需要更新时使用：
            - 删除旧的数据
            - 添加新的数据

            例如：产品信息更新后，重新导入。
        """
        logger.info("更新向量数据库中的文档")

        # 删除旧集合
        self.delete_collection()

        # 重新创建并添加新文档
        self.add_documents(documents)

        logger.success("文档更新完成")


class RAGSystem:
    """
    RAG系统完整封装

    将文档处理和向量数据库整合在一起，提供完整的RAG功能。

    使用流程:
        1. 初始化RAGSystem
        2. 构建知识库（build_knowledge_base）
        3. 检索知识（retrieve）

    示例:
        rag = RAGSystem(config)
        rag.build_knowledge_base()
        results = rag.retrieve("如何查询库存")
    """

    def __init__(self, config):
        """
        初始化RAG系统

        Args:
            config: 配置对象，包含OpenAI和向量数据库配置
        """
        logger.info("初始化RAG系统")

        # 初始化文档处理器
        from .document_processor import DocumentProcessor
        self.doc_processor = DocumentProcessor(
            chunk_size=config.vector_db.chunk_size,
            chunk_overlap=config.vector_db.chunk_overlap
        )

        # 初始化向量数据库
        self.vector_store = VectorStore(
            persist_directory=config.vector_db.db_path,
            openai_api_key=config.openai.api_key,
            openai_api_base=config.openai.api_base
        )

        logger.success("RAG系统初始化完成")

    def build_knowledge_base(self, use_sample_data: bool = True):
        """
        构建知识库

        Args:
            use_sample_data: 是否使用示例数据（用于演示和学习）

        说明:
            知识库构建流程：
            1. 加载/创建文档
            2. 分割文档为小块
            3. 向量化并存入数据库

            当你有真实文档时：
            - 设置use_sample_data=False
            - 使用doc_processor.load_text_file等方法加载真实文档
        """
        logger.info("开始构建知识库")

        # 获取文档
        if use_sample_data:
            # 使用示例文档
            documents = self.doc_processor.create_sample_documents()
        else:
            # 加载真实文档（需要指定路径）
            # documents = self.doc_processor.process_directory("./data/documents")
            logger.warning("请指定真实文档路径，或使用示例数据")
            documents = []

        if not documents:
            logger.warning("没有文档，知识库构建中止")
            return

        # 分割文档
        logger.info(f"分割 {len(documents)} 个文档")
        chunks = self.doc_processor.split_documents(documents)

        # 添加到向量数据库
        logger.info(f"添加 {len(chunks)} 个文档块到向量数据库")
        self.vector_store.add_documents(chunks)

        logger.success("知识库构建完成")

    def retrieve(
        self,
        query: str,
        k: int = 3,
        filter_dict: Optional[dict] = None
    ) -> List[Document]:
        """
        检索相关知识

        Args:
            query: 查询文本
            k: 返回结果数量
            filter_dict: 元数据过滤条件

        Returns:
            相关Document列表
        """
        logger.info(f"检索知识: '{query}'")

        results = self.vector_store.search(query, k, filter_dict)

        # 打印检索结果（用于调试）
        for i, doc in enumerate(results):
            logger.info(f"结果 {i+1}:")
            logger.info(f"  来源: {doc.metadata.get('source', 'unknown')}")
            logger.info(f"  内容长度: {len(doc.page_content)}字符")

        return results

    def get_knowledge_summary(self) -> dict:
        """
        获取知识库摘要信息
        """
        return self.vector_store.get_collection_info()