
from typing import List, Optional
from pathlib import Path
from loguru import logger
from langchain_core.documents import Document

from langchain_openai import OpenAIEmbeddings

try:
    from langchain_huggingface import HuggingFaceEmbeddings
    logger.info("使用 langchain_huggingface.HuggingFaceEmbeddings（推荐）")
except ImportError:
    # 回退到 langchain_community（向后兼容）
    from langchain_community.embeddings import HuggingFaceEmbeddings
    logger.info("回退到 langchain_community.embeddings.HuggingFaceEmbeddings")


from langchain_community.embeddings import DashScopeEmbeddings

try:
    from langchain_chroma import Chroma
    logger.info("使用 langchain_chroma.Chroma（推荐）")
except ImportError:
    from langchain_community.vectorstores import Chroma
    logger.info("回退到 langchain_community.vectorstores.Chroma")

# ChromaDB 原生客户端（底层数据库操作）
import chromadb
from chromadb.config import Settings

SentenceTransformerEmbeddings = HuggingFaceEmbeddings


class VectorStore:
   
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
       
        self.persist_directory = persist_directory
        self.collection_name = collection_name

        logger.info(f"初始化向量数据库（LangChain 1.x），目录: {persist_directory}")

        # 确保目录存在
        Path(persist_directory).mkdir(parents=True, exist_ok=True)

        if not embedding_api_base or embedding_model.startswith("local"):
          
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

        self.client = chromadb.Client(
            Settings(
                persist_directory=persist_directory,
                anonymized_telemetry=False  # 不上报使用数据，保护隐私
            )
        )

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


        self.vectorstore: Optional[Chroma] = None

        logger.success("向量数据库初始化完成")

    def add_documents(self, documents: List[Document]) -> int:
       
        logger.info(f"开始添加 {len(documents)} 个文档到向量数据库")

        if not documents:
            logger.warning("没有文档需要添加")
            return 0

        try:
          
            self.vectorstore = Chroma.from_documents(
                documents=documents,
                embedding=self.embeddings,
                persist_directory=self.persist_directory,
                collection_name=self.collection_name
            )

         
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
      
        logger.warning(f"⚠️ 删除向量数据库集合: {self.collection_name}")

        try:
            self.client.delete_collection(self.collection_name)
            self.vectorstore = None
            logger.success("集合已删除")
        except Exception as e:
            logger.error(f"删除集合失败: {str(e)}")
            raise

    def get_collection_info(self) -> dict:
       
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
       
        logger.info("更新向量数据库中的文档（全量替换）")

        # 删除旧数据
        self.delete_collection()

        # 重新创建集合并添加新文档
        self.add_documents(documents)

        logger.success("文档更新完成")


class RAGSystem:
    
    def __init__(self, config):
       
        logger.info("初始化 RAG 系统（LangChain 1.x）")

        # 初始化文档处理器
        # chunk_size: 每个文档块的最大字符数（500 是比较合适的大小）
        # chunk_overlap: 相邻块之间的重叠字符数（避免语义断裂）
        from .document_processor import DocumentProcessor
        self.doc_processor = DocumentProcessor(
            chunk_size=config.vector_db.chunk_size,
            chunk_overlap=config.vector_db.chunk_overlap
        )

        # 初始化向量数据库
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
        return self.vector_store.get_collection_info()
