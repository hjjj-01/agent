"""

功能:
  1. 将文档转换为向量（embeddings）
  2. 存储向量到数据库
  3. 根据查询向量检索相似文档

=============================================================================
LangChain 1.x 包结构变化：
=============================================================================

  LangChain 1.x 为了更好地管理依赖，把很多集成拆成了独立包：

  | 功能 | 0.3.x 路径 | 1.x 推荐路径 |
  |------|-----------|-------------|
  | Chroma 向量库 | langchain_community.vectorstores | langchain_chroma |
  | SentenceTransformer | langchain_community.embeddings | langchain_huggingface |
  | DashScope (阿里云) | langchain_community.embeddings | langchain_community.embeddings（未变）|
  | OpenAI Embeddings | langchain_openai | langchain_openai（未变） |

  注意：
    - langchain_community 中的旧路径仍然可用（向后兼容），但推荐使用新路径
    - 本项目同时展示新旧两种写法，帮助理解迁移过程
    - 如果安装 langchain_chroma 有问题，回退到 langchain_community 即可

=============================================================================
核心概念：
=============================================================================

  1. Embeddings（向量化）：
     将文本转换为数字向量，使计算机能理解语义。
     例如：
       输入: "如何查询库存"
       处理: OpenAI text-embedding-3-small 模型
       输出: [0.012, -0.345, 0.678, ..., 0.234]（1536维浮点数数组）

  2. 向量数据库：
     专门存储和检索向量的数据库。
     不是存储原始文本，而是存储文本对应的向量。

  3. 相似度检索：
     将查询文本向量化，然后在数据库中找"距离最近"的向量。

=============================================================================
为什么需要向量化（而不是用普通关键词搜索）？
=============================================================================

  传统关键词搜索的局限：
    搜索"库存查询方法"  →  只能匹配到包含这些字的文档
    搜索"怎么看库存多少"  →  字面不同但意思一样，可能搜不到

  向量搜索的优势：
    搜索"库存查询方法"  →  向量化为 [0.12, -0.34, ...]
    搜索"怎么看库存多少"  →  向量化为 [0.11, -0.36, ...]
    两个向量非常接近 → 能识别出语义相似

  类比：关键词搜索像"对暗号"（字必须一样），向量搜索像"理解意思"（语义相近就行）。

=============================================================================
ChromaDB 简介：
=============================================================================

  ChromaDB 是一个轻量级向量数据库，特点：
    - 开源免费
    - 支持本地持久化（数据存磁盘，重启不丢失）
    - Python 原生支持，无需额外服务
    - 适合开发和小型项目（百万级向量以内）

  对于大型项目，可以考虑：
    - Pinecone（云服务，托管）
    - Weaviate（开源，支持更大规模）
    - Milvus（开源，企业级）
    - FAISS（Facebook 开源，纯本地，最快）

  本学习项目使用 ChromaDB，因为最简单、无需额外配置。
=============================================================================
"""
from typing import List, Optional
from pathlib import Path
from loguru import logger
from langchain_core.documents import Document

# =============================================================================
# Embedding 模型导入
# =============================================================================

# OpenAI Embeddings：使用 OpenAI 的 text-embedding 模型
# 模型选项：text-embedding-3-small（性价比高）、text-embedding-3-large（精度高）
from langchain_openai import OpenAIEmbeddings

# SentenceTransformer Embeddings：本地 embedding 模型，无需联网
# LangChain 1.x 推荐从 langchain_huggingface 导入（需要 pip install langchain-huggingface）
# 如果没装 langchain-huggingface，回退到 langchain_community
try:
    from langchain_huggingface import HuggingFaceEmbeddings
    logger.info("使用 langchain_huggingface.HuggingFaceEmbeddings（推荐）")
except ImportError:
    # 回退到 langchain_community（向后兼容）
    from langchain_community.embeddings import HuggingFaceEmbeddings
    logger.info("回退到 langchain_community.embeddings.HuggingFaceEmbeddings")

# DashScope Embeddings：阿里云的 embedding 模型
# 支持中文，适合国内使用（仍在 langchain_community 中）
from langchain_community.embeddings import DashScopeEmbeddings

# =============================================================================
# Chroma 向量数据库导入
# =============================================================================

# LangChain 1.x 推荐从 langchain_chroma 导入（需要 pip install langchain-chroma）
# 如果没装，回退到 langchain_community
try:
    from langchain_chroma import Chroma
    logger.info("使用 langchain_chroma.Chroma（推荐）")
except ImportError:
    from langchain_community.vectorstores import Chroma
    logger.info("回退到 langchain_community.vectorstores.Chroma")

# ChromaDB 原生客户端（底层数据库操作）
import chromadb
from chromadb.config import Settings


# =============================================================================
# 为了向后兼容，保留旧的 SentenceTransformerEmbeddings 别名
# =============================================================================
# 如果你之前用 SentenceTransformerEmbeddings 这个类名，
# 在 LangChain 1.x 中它改名为 HuggingFaceEmbeddings。
# 下面这行确保旧名称仍可用：
SentenceTransformerEmbeddings = HuggingFaceEmbeddings


class VectorStore:
    """
    使用 ChromaDB 作为向量数据库，支持多种 embedding 模型。
    =============================================================================
    支持的 Embedding 模型：
    =============================================================================
      1. OpenAI text-embedding-3-small（默认，1536维，性价比高）
      2. OpenAI text-embedding-3-large（3072维，精度更高）
      3. 阿里云 DashScope embedding（中文友好）
      4. 本地 SentenceTransformer（离线可用，无需API密钥）
         推荐模型: all-MiniLM-L6-v2（384维，轻量快速）
                   bge-large-zh-v1.5（1024维，中文效果好）

    =============================================================================
    使用流程：
    =============================================================================
        1. 初始化 VectorStore（选择 embedding 模型）
        2. 添加文档到数据库（add_documents）
        3. 检索相关文档（search）
        4. [可选] 带分数的检索（search_with_scores）

    =============================================================================
    示例：
    =============================================================================
        store = VectorStore(
            persist_directory="./data/chroma_db",
            openai_api_key="sk-xxx",
        )
        store.add_documents(documents)
        results = store.search("如何查询库存", k=3)
    """

    def __init__(
        self,
        persist_directory: str,
        openai_api_key: str,
        openai_api_base: str = "https://api.openai.com/v1",
        collection_name: str = "wms_knowledge_base",
        embedding_model: str = "text-embedding-3-small",
        embedding_api_base: str = None,
        aliyun_api_key: str = None
    ):
        """
        初始化向量数据库

        Args:
            persist_directory: 数据库持久化目录（数据保存位置）
            openai_api_key: OpenAI API 密钥
            openai_api_base: OpenAI API 基础 URL（使用代理时需要修改）
            collection_name: 数据库集合名称（类似"表名"）
            embedding_model: embedding 模型名称
            embedding_api_base: embedding 的独立 API 地址（None 则复用 openai_api_base）
            aliyun_api_key: 阿里云 DashScope API 密钥

        =============================================================================
        persist_directory 说明：
        =============================================================================
          数据会保存在这个目录下，包括：
            - chroma.sqlite3：向量元数据
            - *.bin：向量数据文件
          重启程序后数据不会丢失。
        =============================================================================
        """
        self.persist_directory = persist_directory
        self.collection_name = collection_name

        logger.info(f"初始化向量数据库（LangChain 1.x），目录: {persist_directory}")

        # 确保目录存在
        Path(persist_directory).mkdir(parents=True, exist_ok=True)

        # =============================================================================
        # 初始化 Embedding 模型
        # =============================================================================
        # 根据配置选择不同的 embedding 模型：
        #   1. 本地模型（离线，无 API 费用）
        #   2. 阿里云 DashScope（中文场景）
        #   3. OpenAI（通用场景，推荐）
        if not embedding_api_base or embedding_model.startswith("local"):
            # 本地模型：无需 API，免费，但精度略低
            # all-MiniLM-L6-v2: 英文为主，384维，约 80MB
            # 如果需要中文，推荐 bge-large-zh-v1.5
            logger.info(f"使用本地 HuggingFace embedding 模型（离线）")
            self.embeddings = HuggingFaceEmbeddings(
                model_name="all-MiniLM-L6-v2"
            )
        elif "dashscope" in embedding_api_base:
            # 阿里云 DashScope：中文效果好，需要阿里云账号
            logger.info(f"使用阿里云 DashScope embedding 模型: {embedding_model}")
            self.embeddings = DashScopeEmbeddings(
                model=embedding_model,
                dashscope_api_key=aliyun_api_key
            )
        else:
            # OpenAI 兼容 API：最常用，精度最高
            # 支持任何 OpenAI 兼容的 API（包括国内代理）
            logger.info(f"使用 OpenAI 兼容 embedding 模型: {embedding_model}")
            self.embeddings = OpenAIEmbeddings(
                openai_api_key=openai_api_key,
                openai_api_base=embedding_api_base or openai_api_base,
                model=embedding_model
            )

        # =============================================================================
        # 初始化 ChromaDB 客户端
        # =============================================================================
        # ChromaDB 有两种模式：
        #   1. 内存模式：数据在内存中，程序退出后丢失（用于测试）
        #   2. 持久化模式：数据存磁盘，重启不丢失（本项目的选择）
        self.client = chromadb.Client(
            Settings(
                persist_directory=persist_directory,
                anonymized_telemetry=False  # 不上报使用数据，保护隐私
            )
        )

        # =============================================================================
        # 创建或获取集合（Collection）
        # =============================================================================
        # Collection 类似数据库中的"表"：
        #   - 一个 Collection 存储一类文档的向量
        #   - 可以用不同的 Collection 分类存储（如"操作手册"、"产品信息"）
        try:
            self.collection = self.client.get_or_create_collection(
                name=collection_name,
                metadata={"description": "WMS 知识库"}
            )
            logger.info(f"向量数据库集合 '{collection_name}' 已就绪")
            logger.info(f"  集合文档数: {self.collection.count()}")
        except Exception as e:
            logger.error(f"创建向量数据库集合失败: {str(e)}")
            raise

        # LangChain 的 Chroma 包装器（提供更方便的接口）
        # 初始为 None，在 add_documents 时创建
        self.vectorstore: Optional[Chroma] = None

        logger.success("向量数据库初始化完成")

    def add_documents(self, documents: List[Document]) -> int:
        """
        将文档添加到向量数据库

        Args:
            documents: Document 对象列表

        Returns:
            添加的文档数量

        =============================================================================
        处理流程：
        =============================================================================
          1. 提取文档的 page_content（文本内容）
          2. 使用 Embedding 模型将文本转为向量
             例如："如何查询库存" → [0.012, -0.345, 0.678, ..., 0.234]
          3. 将向量 + 原文本 + 元数据 存储到 ChromaDB
          4. 持久化到磁盘（防止重启丢失）

        =============================================================================
        为什么同时存储向量和原文本？
        =============================================================================
          - 向量用于检索（找相似的）
          - 原文本用于展示（给 LLM 阅读）
          - 元数据用于过滤（按来源、类型等筛选）

        =============================================================================
        性能说明：
        =============================================================================
          ChromaDB 第一次添加文档时，如果文档数量多（>1000），可能较慢。
          这是正常的——每条文档都需要调用 embedding API 进行向量化。
          后续添加少量文档会很快（只向量化新增部分）。
        =============================================================================
        """
        logger.info(f"开始添加 {len(documents)} 个文档到向量数据库")

        if not documents:
            logger.warning("没有文档需要添加")
            return 0

        try:
            # Chroma.from_documents 会自动：
            # 1. 调用 embeddings.embed_documents() 将每个文档转为向量
            # 2. 将向量 + 原文本 + 元数据 存入 ChromaDB
            # 3. 返回 Chroma 实例（后续用于检索）
            #
            # LangChain 1.x 中这个 API 没有变化，完全向后兼容
            self.vectorstore = Chroma.from_documents(
                documents=documents,
                embedding=self.embeddings,
                persist_directory=self.persist_directory,
                collection_name=self.collection_name
            )

            # ChromaDB 新版本（langchain-chroma 0.2+）自动持久化
            # persist() 方法已移除，数据在 from_documents() 时就写入磁盘了

            logger.success(
                f"成功添加 {len(documents)} 个文档到向量数据库"
            )
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
        搜索相似文档（语义检索）

        Args:
            query: 查询文本（自然语言）
            k: 返回的最相似文档数量（默认 3）
            filter_dict: 元数据过滤条件（可选）

        Returns:
            最相似的 Document 对象列表（按相似度降序排列）

        =============================================================================
        搜索流程：
        =============================================================================
          1. 将查询文本 query 向量化（调用 embedding API）
             query = "如何查询库存"
             → query_vector = [0.012, -0.345, ...]

          2. 在向量数据库中计算 query_vector 与所有文档向量的相似度
             使用余弦相似度（Cosine Similarity）

          3. 按相似度降序排列，返回 top-k 个最相似的文档

          4. [可选] 如果有 filter_dict，只搜索满足条件的文档
             例如：{"category": "manual"} 只搜索类别为"手册"的文档

        =============================================================================
        相似度计算的幕后原理：
        =============================================================================
          两个向量的余弦相似度 = 两个向量夹角的余弦值。
          值越接近 1，两个向量方向越一致，文档越相关。
          值越接近 -1，方向越相反。
          值接近 0，无关。

          在向量空间中，"意思相近"的文本会被 embedding 模型映射到
          相近的位置（向量方向接近），所以 cosine_similarity 高。

        =============================================================================
        示例：
        =============================================================================
          query: "如何查询库存"
            → 向量化: [0.12, -0.34, 0.67, ...]
            → 搜索数据库找到最相似的向量
            → 返回: ["WMS系统操作手册-库存查询", "WMS业务规则-库存管理", ...]
        =============================================================================
        """
        logger.info(f"搜索查询: '{query}'，返回 {k} 个结果")

        if not self.vectorstore:
            logger.error("向量数据库尚未初始化，请先调用 add_documents 添加文档")
            raise ValueError("向量数据库尚未初始化，请先添加文档")

        try:
            # similarity_search：LangChain 提供的简便检索方法
            # 内部自动完成：query → 向量化 → 相似度计算 → 排序 → 返回
            if filter_dict:
                # 按元数据过滤后搜索
                results = self.vectorstore.similarity_search(
                    query,
                    k=k,
                    filter=filter_dict
                )
            else:
                # 无过滤，全库搜索
                results = self.vectorstore.similarity_search(
                    query,
                    k=k
                )

            logger.success(
                f"搜索完成，找到 {len(results)} 个相关文档"
            )

            # 打印每个结果的来源信息（用于调试）
            for i, doc in enumerate(results):
                source = doc.metadata.get('source', 'unknown')
                logger.info(
                    f"  结果 {i+1}: 来源={source}, 内容长度={len(doc.page_content)}"
                )

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
      
        Args:
            query: 查询文本
            k: 返回的数量

        Returns:
            (Document, score) 元组列表
            score 范围通常在 0-1 之间（余弦相似度），越大越相似

        =============================================================================
        什么时候需要 score？
        =============================================================================
          - 调试：检查检索质量（score 太低说明知识库可能需要丰富）
          - 过滤：只使用 score > 0.8 的结果（提高回答质量）
          - 排序：多个查询结果合并后按 score 重新排序

        =============================================================================
        score 解读：
        =============================================================================
          > 0.9: 非常相关，基本是同一个意思
          0.7-0.9: 比较相关，主题接近
          0.5-0.7: 弱相关，可能有部分关联
          < 0.5: 不太相关，建议忽略
        =============================================================================
        """
        logger.info(f"带分数搜索: '{query}'")

        if not self.vectorstore:
            logger.error("向量数据库尚未初始化")
            raise ValueError("向量数据库尚未初始化")

        try:
            # similarity_search_with_score 返回 (Document, float) 元组
            results = self.vectorstore.similarity_search_with_score(
                query,
                k=k
            )

            logger.success(
                f"搜索完成，返回 {len(results)} 个结果（带分数）"
            )

            # 打印分数信息
            for i, (doc, score) in enumerate(results):
                logger.info(
                    f"  结果 {i+1}: 分数={score:.4f}, "
                    f"来源={doc.metadata.get('source', 'unknown')}"
                )

            return results

        except Exception as e:
            logger.error(f"搜索失败: {str(e)}")
            raise

    def delete_collection(self):
        """
        删除整个集合

        危险操作！数据不可恢复！

        使用场景：
          - 重新初始化知识库
          - 清理测试数据
        """
        logger.warning(f"⚠️ 删除向量数据库集合: {self.collection_name}")

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
            集合信息字典，包括：
              - collection_name: 集合名称
              - document_count: 文档总数
              - persist_directory: 存储路径
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
        更新文档（全量替换）

        先删除旧数据，再添加新数据。

        Args:
            documents: 新的 Document 列表

        使用场景：
          - 知识库内容需要更新（如产品信息变更）
          - 重新导入优化后的文档

        注意：
            这是全量替换，不是增量更新。
            如果只需要添加新文档，用 add_documents。
        """
        logger.info("更新向量数据库中的文档（全量替换）")

        # 删除旧数据
        self.delete_collection()

        # 重新创建集合并添加新文档
        self.add_documents(documents)

        logger.success("文档更新完成")


class RAGSystem:
    """
    RAG 系统完整封装（LangChain 1.x 版本）

    将文档处理和向量数据库整合在一起，提供完整的 RAG 功能。

    =============================================================================
    RAG = Retrieval-Augmented Generation（检索增强生成）

    工作原理：
      1. 离线阶段（构建知识库）：
         文档 → 分割 → 向量化 → 存入向量数据库

      2. 在线阶段（回答用户问题）：
         用户提问 → 向量化 → 检索相关文档 → 作为上下文给 LLM → 生成回答

    为什么需要 RAG？
      - LLM 的知识截止于训练日期，不知道最新的信息
      - LLM 可能"幻觉"（编造不存在的信息）
      - RAG 让 LLM 基于真实文档回答，减少幻觉
      - 不需要微调模型，只需更新知识库文档

    =============================================================================
    使用流程：
    =============================================================================
        1. 初始化 RAGSystem（配置 embedding 和数据库路径）
        2. 构建知识库（用示例数据或真实文档）
        3. 检索知识（用户提问时调用）

    =============================================================================
    示例：
    =============================================================================
        rag = RAGSystem(config)
        rag.build_knowledge_base()
        results = rag.retrieve("如何查询库存")
    =============================================================================
    """

    def __init__(self, config):
        """
        初始化 RAG 系统

        Args:
            config: 配置对象，包含 OpenAI 和向量数据库配置

        =============================================================================
        系统组成：
        =============================================================================
          1. DocumentProcessor（文档处理器）：
             - 负责加载各种格式的文档
             - 将长文档分割成小块（chunks）
             - 管理文档的元数据

          2. VectorStore（向量数据库）：
             - 负责将文档向量化
             - 存储向量到 ChromaDB
             - 提供语义检索功能
        =============================================================================
        """
        logger.info("初始化 RAG 系统（LangChain 1.x）")

        # 初始化文档处理器
        # chunk_size: 每个文档块的最大字符数（500 是比较合适的大小）
        # chunk_overlap: 相邻块之间的重叠字符数（避免语义断裂）
        from .document_processor import DocumentProcessor
        self.doc_processor = DocumentProcessor(
            chunk_size=config.vector_db.chunk_size,
            chunk_overlap=config.vector_db.chunk_overlap
        )

        # 初始化向量数据库 上面那个类 VectorStore
        self.vector_store = VectorStore(
            persist_directory=config.vector_db.db_path,
            openai_api_key=config.openai.api_key,
            openai_api_base=config.openai.api_base,
            embedding_model=config.openai.embedding_model,
            embedding_api_base=config.openai.embedding_api_base,
            aliyun_api_key=config.openai.aliyun_api_key
        )

        logger.success("RAG 系统初始化完成")

    def build_knowledge_base(self, use_sample_data: bool = True):
        """
        构建知识库

        Args:
            use_sample_data: 是否使用示例数据（用于演示和学习）

        =============================================================================
        知识库构建流程：
        =============================================================================
          1. 加载/创建文档
          2. 将文档分割成小块
          3. 将每个块向量化
          4. 存入 ChromaDB（持久化到磁盘）

        关于 use_sample_data：
          - True（默认）: 使用内置的示例文档（WMS操作手册、业务规则、产品信息）
          - False: 加载你指定的真实文档（需要设置文档路径）
        =============================================================================
        """
        logger.info("开始构建知识库")

        # 获取文档
        if use_sample_data:
            documents = self.doc_processor.create_sample_documents()
            logger.info(f"使用示例文档，共 {len(documents)} 个")
        else:
            # 加载真实文档
            # 取消下面注释并设置文档路径：
            # documents = self.doc_processor.process_directory("./data/documents")
            logger.warning("请指定真实文档路径，或使用示例数据（use_sample_data=True）")
            documents = []

        if not documents:
            logger.warning("没有文档，知识库构建中止")
            return

        # 分割文档为小块
        logger.info(f"分割 {len(documents)} 个文档")
        chunks = self.doc_processor.split_documents(documents)
        logger.info(f"分割完成，生成 {len(chunks)} 个文档块")

        # 向量化并存入 ChromaDB
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
            k: 返回结果数量（默认 3）
            filter_dict: 元数据过滤条件

        Returns:
            相关 Document 列表

        =============================================================================
        检索 vs 搜索：
        =============================================================================
          这里说的"检索"其实就是语义搜索——找到与查询最相关的文档。
          区别在于：
            - 传统搜索：基于关键词匹配
            - 语义检索：基于向量相似度，理解语义

        =============================================================================
        k 的选择：
        =============================================================================
          - k=1: 只要最相关的（精度优先）
          - k=3: 平衡选择（比较常用）
          - k=5: 更多上下文（可能引入噪音）
          - k 太大会超出 LLM 上下文窗口，导致信息丢失
        =============================================================================
        """
        logger.info(f"检索知识: '{query}'")

        results = self.vector_store.search(query, k, filter_dict)

        # 打印检索结果（用于调试和理解检索效果）
        for i, doc in enumerate(results):
            logger.info(f"  结果 {i+1}:")
            logger.info(f"    来源: {doc.metadata.get('source', 'unknown')}")
            logger.info(f"    标题: {doc.metadata.get('title', '未知')}")
            logger.info(f"    内容长度: {len(doc.page_content)} 字符")

        return results

    def get_knowledge_summary(self) -> dict:
        """
        获取知识库摘要信息

        Returns:
            包含文档总数、存储路径等信息的字典
        """
        return self.vector_store.get_collection_info()
