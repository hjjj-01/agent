"""
文档处理器模块（LangChain 1.x 版本）

功能:
  1. 加载各种格式的文档（文本、Markdown、JSON 等）
  2. 将长文档分割成小块（chunks）
  3. 为每个文档块添加元数据

=============================================================================
为什么要分割文档（Chunking）？
=============================================================================

  三个核心原因：

  1. LLM 有上下文长度限制（Context Window）：
     - GPT-4o: 约 128K tokens
     - GPT-4o-mini: 约 128K tokens
     - DeepSeek: 约 128K tokens
     一个 WMS 操作手册可能有 10 万字（约 30K tokens），看起来能放下。
     但 LLM 还需要同时处理：对话历史 + 系统提示词 + 多个检索结果。
     所以必须把文档切小，只给 LLM 最相关的部分。

  2. 语义检索更精确：
     如果把整本手册作为一个大块，向量化后就是一个"平均向量"。
     用户问"如何查询库存"时，这个大块里"查询库存"的部分信号
     会被其他大量无关内容稀释，导致检索不精确。
     切小块后，每个块聚焦一个主题（如"库存查询操作"），
     检索时更容易找到精确定位的内容。

  3. 向量化效果更好：
     Embedding 模型对短文本的向量化效果优于长文本。
     因为短文本主题集中，向量能更准确地表达语义。

  类比：
     一本字典如果只有"所有词的解释"这一个条目，查找效率很低。
     分成每个词一个条目，就能快速定位。

=============================================================================
分割策略详解：
=============================================================================

  1. 按字符数分割（CharacterTextSplitter）：
     最简单的方式，按固定字符数切割。
     优点：简单、快速
     缺点：可能在句子中间切断，破坏语义完整性
     适用：结构简单的纯文本

  2. 递归分割（RecursiveCharacterTextSplitter）：
     按优先级尝试不同的分割符：
       "\\n\\n"（段落）→ "\\n"（行）→ " "（词）→ ""（字符）
     优先在"自然断点"（段落、句子）处分割。
     优点：保持语义完整性，适合大多数场景
     缺点：略慢
     适用：通用文本分割（本项目默认选择）

  3. 按标题分割（MarkdownHeaderTextSplitter）：
     根据 Markdown 的 #/##/### 标题结构分割。
     每个章节成为一个独立的块。
     优点：完美保持文档结构
     缺点：只适用于 Markdown，且各块大小可能差异很大
     适用：结构化好的 Markdown 文档

=============================================================================
chunk_overlap（块重叠）的重要性和原理：
=============================================================================

  假设 chunk_size=500, chunk_overlap=50：
    第1块: 字符 0-500
    第2块: 字符 450-950  ← 与第1块重叠50字符
    第3块: 字符 900-1400  ← 与第2块重叠50字符

  为什么需要重叠？
    考虑这样一个情况——关键信息恰好被切在两块边界上：
      第1块结尾: "...查询库存时需要先输入 SKU 编号"
      第2块开头: "，然后系统会返回库存信息..."

    如果没有重叠，两句话就被切断了，LLM 只能看到半句话。
    有重叠时，第2块会从 "SKU 编号" 之前开始，确保上下文连续。

  chunk_overlap 的推荐值：
    - 一般设为 chunk_size 的 10%-20%
    - 500 的块 → overlap 50-100
    - 不要太小（< 10）：容易丢失上下文
    - 不要太大（> chunk_size/2）：生成太多冗余块
=============================================================================
"""
from typing import List, Dict, Any
from pathlib import Path
from loguru import logger

# Document：LangChain 的标准文档类型
# 只有两个核心属性：
#   - page_content（str）：文档的文本内容
#   - metadata（dict）：元数据，如来源、标题、作者、创建时间等
from langchain_core.documents import Document

# langchain_text_splitters 是独立的包（LangChain 1.x 中保持不变）
# 所有的文本分割器都在这个包里
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,   # 递归分割器（最常用）
    CharacterTextSplitter,            # 简单按字符分割
    MarkdownHeaderTextSplitter        # 按 Markdown 标题分割
)


class DocumentProcessor:
    """
    文档处理器（LangChain 1.x 版本）

    用于加载和处理文档，将其分割成适合向量化的块。

    =============================================================================
    使用流程：
    =============================================================================
        1. 创建处理器：processor = DocumentProcessor(chunk_size=500, chunk_overlap=50)
        2. 加载文档：documents = processor.load_text_file("manual.txt")
        3. 分割文档：chunks = processor.split_documents(documents)
        4. 存入向量库：vector_store.add_documents(chunks)

    =============================================================================
    参数选择指南：
    =============================================================================
      chunk_size 的选择取决于文档类型和 embedding 模型：
        - 操作手册、规章制度：300-500（每段一个主题，块小精确）
        - 技术文档、长文章：500-1000（段落较长，需要更多上下文）
        - 小说、长文本：1000-2000（需要较多上下文理解情节）
        - 代码文档：200-400（代码块通常较短）

      embedding 模型的上下文限制：
        - OpenAI text-embedding-3-small: 8191 tokens
        - SentenceTransformer all-MiniLM-L6-v2: 256 tokens
        块的字符数不能超过模型能处理的最大长度。

    =============================================================================
    示例：
    =============================================================================
        processor = DocumentProcessor(chunk_size=500, chunk_overlap=50)
        documents = processor.load_text_file("manual.txt")
        chunks = processor.split_documents(documents)
    =============================================================================
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
            chunk_size: 每个文档块的最大字符数（默认 500）
            chunk_overlap: 相邻文档块之间的重叠字符数（默认 50）
            separators: 分割符列表，按优先级尝试（默认: 段落 → 行 → 词 → 字符）

        =============================================================================
        separators 的优先级设计：
        =============================================================================
          ["\\n\\n", "\\n", " ", ""]
          这个顺序表示：
            1. 先尝试按段落分割（双换行）
            2. 段落太长，就按行分割（单换行）
            3. 行太长，就按词分割（空格）
            4. 词太长，被迫按字符分割（英文场景很罕见）

          递归分割器会尝试每个分割符，直到找到能分成 ≤ chunk_size 的方案。
        =============================================================================
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # 默认分割符：优先按段落、句子分割，最后才按字符分割
        if separators is None:
            separators = ["\\n\\n", "\\n", " ", ""]

        # =============================================================================
        # 初始化递归分割器（最常用的分割器）
        # =============================================================================
        # RecursiveCharacterTextSplitter 是 LangChain 中最推荐的分割器。
        # "递归"的意思是：先尝试第一个分割符，如果分割后有些块仍大于 chunk_size，
        # 就对这些大块用第二个分割符继续分割，以此类推。
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=separators,
            # length_function: 用于计算文本长度的函数
            # 默认 len() 按字符数计算，中文一个汉字 = 1 个字符
            # 如果按 token 计算，可以用 tiktoken 库
            length_function=len
        )

        # =============================================================================
        # 初始化 Markdown 分割器（按标题结构分割）
        # =============================================================================
        # 这个分割器根据 Markdown 的标题层级（#, ##, ###）来分割文档。
        # 每个标题及其下属内容成为一个独立的块。
        # headers_to_split_on 定义了要按哪些级别的标题分割。
        self.markdown_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[
                ("#", "header1"),     # 一级标题 → metadata["header1"] = 标题文本
                ("##", "header2"),    # 二级标题 → metadata["header2"] = 标题文本
                ("###", "header3"),   # 三级标题 → metadata["header3"] = 标题文本
            ]
        )

        logger.info(
            f"文档处理器初始化完成，块大小: {chunk_size}，"
            f"重叠: {chunk_overlap}"
        )

    def load_text_file(
        self,
        file_path: str,
        metadata: Dict[str, Any] = None
    ) -> List[Document]:
        """
        加载文本文件

        Args:
            file_path: 文件路径（绝对路径或相对路径）
            metadata: 文档元数据（如标题、来源、作者等）

        Returns:
            Document 对象列表（List[Document]）

        =============================================================================
        Document 对象结构：
        =============================================================================
          Document(
              page_content="这是文档的文本内容...",  # 核心：文档正文
              metadata={                              # 元数据：描述性信息
                  "source": "/path/to/file.txt",      # 文件路径
                  "filename": "file.txt",             # 文件名
                  "title": "操作手册",                 # 标题（用于展示和过滤）
                  "category": "manual",               # 分类（用于过滤）
                  "author": "张三",                   # 作者
                  "created_date": "2024-01-15",       # 创建日期
              }
          )

        元数据的作用：
          - 展示：告诉用户检索结果来自哪里
          - 过滤：只搜索特定类型的文档（如 filter={"category": "manual"}）
          - 排序：按日期、作者等排序
        =============================================================================
        """
        logger.info(f"加载文本文件: {file_path}")

        path = Path(file_path)
        if not path.exists():
            logger.error(f"文件不存在: {file_path}")
            raise FileNotFoundError(f"文件不存在: {file_path}")

        # 读取文件内容
        # encoding="utf-8" 适用于绝大多数中文文档
        # 如果遇到 GBK 编码的文件，改为 encoding="gbk"
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        # 构建元数据（确保包含来源和文件名等基本信息）
        if metadata is None:
            metadata = {}

        metadata["source"] = file_path
        metadata["filename"] = path.name
        metadata["file_type"] = "text"

        # 创建 Document 对象
        document = Document(page_content=content, metadata=metadata)

        logger.success(f"文件加载成功，内容长度: {len(content)} 字符")
        return [document]

    def load_markdown_file(
        self,
        file_path: str,
        metadata: Dict[str, Any] = None
    ) -> List[Document]:
        """
        加载 Markdown 文件

        Args:
            file_path: 文件路径
            metadata: 文档元数据

        Returns:
            Document 对象列表

        =============================================================================
        Markdown 的特性处理：
        =============================================================================
          Markdown 文件通常有结构化的标题（h1, h2, h3...），
          直接整篇按字符分割会破坏这个结构。

          MarkdownHeaderTextSplitter 会根据标题层级分割文档：
            原始 Markdown:
              # 库存管理
              ## 查询库存
              ...内容...
              ## 库存预警
              ...内容...

            分割后:
              块1: metadata["header1"]="库存管理", metadata["header2"]="查询库存"
              块2: metadata["header1"]="库存管理", metadata["header2"]="库存预警"

          这样每个小块都保留了它在文档中的"位置信息"。
        =============================================================================
        """
        logger.info(f"加载 Markdown 文件: {file_path}")

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

        # 按标题分割 Markdown
        md_documents = self.markdown_splitter.split_text(content)

        # 为每个分割后的文档添加公共元数据
        for doc in md_documents:
            doc.metadata.update(metadata)

        logger.success(
            f"Markdown 文件加载成功，按标题分割为 {len(md_documents)} 个部分"
        )
        return md_documents

    def load_json_data(
        self,
        data: Dict[str, Any],
        source: str = "json_data"
    ) -> List[Document]:
        """
        从 JSON 数据创建 Document

        Args:
            data: JSON 数据字典
            source: 数据来源标识

        Returns:
            Document 对象列表

        =============================================================================
        使用场景：
        =============================================================================
          将结构化数据（数据库导出、API 响应等）转为文档。
          例如：
            - API 返回的产品列表 → 转为文档存入知识库
            - 数据库导出的业务规则表 → 转为文档供检索
            - 配置文件中的操作说明 → 转为文档
        =============================================================================
        """
        logger.info(f"从 JSON 数据创建 Document")

        documents = []

        # 遍历 JSON 数据，为每个条目创建 Document
        for key, value in data.items():
            if isinstance(value, dict):
                # 如果 value 是字典，将其转为格式化的文本
                content = self._dict_to_text(value)
                metadata = {
                    "source": source,
                    "key": key,
                    "file_type": "json"
                }
                doc = Document(page_content=content, metadata=metadata)
                documents.append(doc)

        logger.success(f"从 JSON 数据创建了 {len(documents)} 个 Document")
        return documents

    def _dict_to_text(self, data: Dict[str, Any]) -> str:
        """
        将字典转换为文本格式

        Args:
            data: 字典数据

        Returns:
            格式化的文本字符串

        格式示例：
            {"name": "商品A", "price": 100, "stock": 50}
            → "name: 商品A\\nprice: 100\\nstock: 50"
        """
        lines = []
        for key, value in data.items():
            lines.append(f"{key}: {value}")
        return "\\n".join(lines)

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        分割文档为小块

        Args:
            documents: 原始 Document 对象列表

        Returns:
            分割后的 Document 对象列表（每块 ≤ chunk_size 字符）

        =============================================================================
        分割原理（递归分割器的工作流程）：
        =============================================================================
          假设有一个 2000 字的文档，chunk_size=500，separators=["\\n\\n","\\n"," ",""]：

          第1轮 - 尝试按段落分割（"\\n\\n"）：
            第1段: 300字 ✓ （< 500，合格）
            第2段: 800字 ✗ （> 500，需要继续分割）
            第3段: 400字 ✓
            第4段: 500字 ✓

          第2轮 - 对800字的大段尝试按行分割（"\\n"）：
            行1: 350字 ✓
            行2: 250字 ✓
            行3: 200字 ✓

          最终结果：5 个块，每个 ≤ 500 字

          重叠处理：相邻块之间重叠 chunk_overlap 个字符。
        =============================================================================
        """
        logger.info(f"开始分割 {len(documents)} 个文档")

        # 使用递归分割器分割文档
        chunks = self.text_splitter.split_documents(documents)

        # 为每个块添加分割相关的元数据
        for i, chunk in enumerate(chunks):
            chunk.metadata["chunk_id"] = i            # 块编号（用于去重和追踪）
            chunk.metadata["chunk_size"] = len(chunk.page_content)  # 块的实际大小

        logger.success(
            f"文档分割完成，从 {len(documents)} 个文档 "
            f"分割为 {len(chunks)} 个块"
        )
        return chunks

    def process_directory(
        self,
        directory_path: str,
        file_types: List[str] = None
    ) -> List[Document]:
        """
        批量处理目录中的所有文件

        Args:
            directory_path: 目录路径
            file_types: 要处理的文件扩展名列表，如 [".txt", ".md"]

        Returns:
            所有文档分割后的块列表（可直接用于向量化）

        =============================================================================
        使用场景：
        =============================================================================
          当你有一个目录，里面放了所有需要导入知识库的文档时，
          用这个方法一次性处理整个目录。

          目录结构示例：
            data/documents/
            ├── 操作手册.md
            ├── 业务规则.txt
            ├── 产品信息.md
            └── FAQ.txt

          调用：
            chunks = processor.process_directory("data/documents/")
        =============================================================================
        """
        logger.info(f"处理目录: {directory_path}")

        if file_types is None:
            file_types = [".txt", ".md"]

        directory = Path(directory_path)
        if not directory.exists():
            logger.error(f"目录不存在: {directory_path}")
            raise FileNotFoundError(f"目录不存在: {directory_path}")

        all_chunks = []

        # 递归遍历目录中的所有文件（rglob = recursive glob）
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

        logger.success(
            f"目录处理完成，总共生成 {len(all_chunks)} 个文档块"
        )
        return all_chunks

    def create_sample_documents(self) -> List[Document]:
        """
        创建示例文档（用于演示和学习）

        Returns:
            示例 Document 对象列表

        =============================================================================
        当前知识库主题：员工ID和岗位ID映射表
        =============================================================================
          本知识库的核心作用是"ID映射"——因为 WMS API 只认数字ID，
          不认员工的文字姓名或岗位名。当用户说"张三的工作量"时，
          需要先查知识库找到张三的 worker_id，才能调用 WMS API。

          知识库包含：
            1. 岗位ID映射：岗位名称 → node_id（数字）
               例如："打包" → "1928371092732805121"
            2. 员工ID映射：员工姓名 → worker_id（数字）
               例如："LS-温明瑶" → "2072639244575629313"

          当你有真实文档时：
            1. 设置 use_sample_data=False
            2. 使用 process_directory() 或 load_text_file() 加载真实文档
        =============================================================================
        """
        logger.info("创建示例文档（用于演示和学习）")

        documents = []

        # =============================================================================
        # 示例1：id登记处（知识库核心数据——ID映射表）
        # =============================================================================
        # 本知识库用于 WMS API 的"ID映射"：
        #   WMS API 只认数字ID（如 1928371092732805121），不认文字（员工姓名、岗位名）。
        #   当用户说"张三的工作量"时，需要先查知识库找到张三的 worker_id，
        #   才能调用 WMS API 查询。
        #
        # 知识库内容：
        #   1. 岗位ID：岗位名称 → node_id（数字ID）
        #      例如："打包：1928371092732805121"
        #   2. 员工ID：员工姓名 → worker_id（数字ID）
        #      例如："LS-温明瑶    2072639244575629313"（Tab分隔）
        #
        # 工具查询方式：
        #   wms_query_employee_work 工具内部会自动：
        #     1. 用 worker_name/node_name 查知识库，找对应的数字ID
        #     2. 用数字ID调用 WMS API
        #     3. 如查具体员工，在返回结果中过滤该 worker_id 的数据
        #
        # RAG 检索示例（向量相似度匹配）：
        #   query="张三的ID" → 找到包含张三那行的内容 → 提取数字ID
        #   query="打包岗位的node_id" → 找到"打包：1928371092732805121"那行
        # =============================================================================
        id_register_content = """
    #一.岗位id
        1.压包：1928370811596996610
        2.打包/备货打包：1928371092732805121

    #二.员工id
        LS-温明瑶	2072639244575629313
        LS-梁含	2072634972781862913
        LS-曹佳雯	2072634879446568961
        LS-朱智	2072634734469898242
        孙悦轩	2072197399236304898
        LS-吴俊成	2071180364402339842
        LS-陈家良	2071180218159542273
        LS-肖语焉	2070403796444852226
        LS-曾楚旋	2070400886252187650
        LS-张汉铭	2070400763602845697
        LS-吴俊鸿	2070122604688519170
        LS-郑棋延	2069653411715772417
        LS-徐逸臣	2069653285558960130
        LS-郭安凤	2069294715412312065
        LS-岳喜宝	2069267104560738306
        LS-沈向源	2069267014349111297
        LS-肖语嫣	2069219784984510465
        LS-李辰傲	2068956389354483714
        LS-陈珍	2068939408542171137
        LS-蔡丽梅	2068859410037321730
        LS-吴登鑫	2067410731887120386
        陈可欣	2067054550719590401
        杨再华	2067054479265427458
        杨雪芳	2066423096843157505
        LS-修包1	2062138317519904769
        徐丹	2061612840262291458
        韩晓梅	2038547429002227714
        付甫凤	2035535075222667265
        吕诗婷	2034833957387714562
        喻万婷	2033365440440246274
        狐启琴	2033365252082442242
        吴瑶	2032697623604830209
        陈颖	2030829689240518657
        易庆美	2027188976443392002
        蔡永再	2009603824376827906
        易桂兰	2009506938114494466
        李外香	2009506937988665345
        王志评	2009506937762172930
        杨景松	2009506937544069121
        蔡明浪	2009506937451794433
        赵凤	2009506937246273538
        曾婉冰	2009506937170776065
        钟赛花	2009506937153998849
        许丽影	2009506936843620353
        LS-柴安妍	2071892982486249473
        林果	2071866371300913154
        LS-杜佳佳	2071759016441671682
        预包核验	2070095749529387009
        卯光龙	2070012914470666242
        邓仔豪	2068165691696988161
        艾少俊	2068165566610259970
        LS-杜圆圆	2067150835576942593
        葛铭浩	2067054096346443778
        韦天汉	2067054000225579010
        许为勇	2065380945468309506
        王猛猛	2065380874084921346
        钟贞菁	2064534850504146945
        黄运兴	2064534738491666433
        唐方为	2064178494014124033
        LS-王小冷	2063853276414554114
        LS-陈玉段	2063826235476619266
        LS-陆孝甜	2063802601211113473
        LS-何诗涵	2063802369707036673
        LS-李涛	2063801098702901250
        郑立雄	2062698043948892162
        刘永强	2062002023389126657
        LS-曾思瑶	2061987662000009217
        LS-李静煜	2061987523524472833
        陈忠果	2061613589121716225
        封箱回单测试	2039173450630619138
        齐志力	2036265982715957250
        齐自祥	2036265813270269953
        刘锦城	2034834477929660417
        刘智超	2033365589098962945
        谷仓&TK专用账号3	2021116337915830274
        谷仓&TK专用账号2	2021116284262846466
        谷仓&TK专用账号1	2021116208664158210
        柯建平	2013435879178559490
        打包不质检回单专用	2013188005665140737
        殷后毅	2009506938328403970
        郑州	2009506938265489410
        蔡振华	2009506938189991938
        黄诗团	2009506937837670401
        陈忠果	2009506937560846338
        陈奕彬	2009506937481154562
        潘高望	2009506937095278593
        蔡银单	2009506937061724162
        蔡家忍	2009506937044946946
        颜锦鹏	2009506936889757697
        吕小园	2009506936814260226
        林佳辉	2009506936734568449
"""
        documents.append(Document(
            page_content=id_register_content,
            metadata={
                "source": "id_register",
                "title": "id登记处（员工ID和岗位ID映射表）",
                "category": "id_mapping",
                "file_type": "markdown",
                "description": "WMS API ID映射：员工姓名→worker_id，岗位名→node_id"
            }
        ))

      


        logger.success(
            f"创建了 {len(documents)} 个示例文档 "
            f"（id登记处——员工ID和岗位ID映射表）"
        )
        return documents
