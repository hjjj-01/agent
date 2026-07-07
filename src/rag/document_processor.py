from typing import List, Dict, Any
from loguru import logger
from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter

class MarkdownDocumentProcessor:
    """
    专门处理 Markdown 格式文档，按 #、##、### 标题进行分割
    """

    def __init__(self):
        # 创建 Markdown 分割器，指定要分割的标题层级
        self.markdown_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[
                ("#", "header1"),    # 一级标题 → metadata["header1"]
                ("##", "header2"),   # 二级标题 → metadata["header2"]
                ("###", "header3"),  # 三级标题 → metadata["header3"]
            ]
        )
        logger.info("Markdown 文档处理器已初始化")

    # 输出文档列表
    def create_sample_documents(self) -> List[Document]:
        """
        创建示例文档（员工ID和岗位ID映射表）
        内容包含 # 一级标题，用于演示分割
        """
        logger.info("创建示例文档")

        # 这是你原来的示例内容（我保留完整）
        id_register_content = """
            # 一.岗位id
                1.压包：1928370811596996610
                2.打包/备货打包：1928371092732805121

            # 二.员工id
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
        # 创建一个 Document，包含内容和元数据
        doc = Document(
            page_content=id_register_content,
            metadata={
                "source": "id_register",
                "title": "id登记处（员工ID和岗位ID映射表）",
                "category": "id_mapping",
                "description": "WMS API ID映射：员工姓名→worker_id，岗位名→node_id"
            }
        )
        return [doc]

    def split_by_headers(self, documents: List[Document]) -> List[Document]:
        """
        对文档列表中的每个文档，按 Markdown 标题分割
        返回分割后的所有块（每个块是一个 Document）
        """
        logger.info(f"开始按 Markdown 标题分割 {len(documents)} 个文档")
        all_chunks = []

        for doc in documents:
            #    使用 Markdown 分割器拆分内容
            #    注意：split_text 返回的是 List[Document]，每个小块自带 metadata，
            #    其中包含了该块所属的标题路径（如 header1="一级标题", header2="二级标题"）
            splitted = self.markdown_splitter.split_text(doc.page_content)

            # 为每个分割后的块合并原始文档的元数据
            for chunk in splitted:
                # 先复制原始元数据（注意不要覆盖已有的 header1/header2/header3）
                base_meta = doc.metadata.copy()
                # 更新（合并）标题元数据，chunk.metadata 里已经包含了 header1 等
                base_meta.update(chunk.metadata)
                # 替换 chunk 的元数据为合并后的
                chunk.metadata = base_meta

            all_chunks.extend(splitted)

        # 给每个块添加编号和大小信息（方便后续使用），enumerate 是 Python 的内置函数，它会一边遍历列表，一边自动数数。
        for i, chunk in enumerate(all_chunks):
            chunk.metadata["chunk_id"] = i
            chunk.metadata["chunk_size"] = len(chunk.page.content)

        logger.success(f"分割完成，共生成 {len(all_chunks)} 个块")
        return all_chunks


# ==================== 使用示例 ====================
if __name__ == "__main__":
    # 1. 创建处理器
    processor = MarkdownDocumentProcessor()

    # 2. 生成示例文档
    docs = processor.create_sample_documents()

    # 3. 按标题分割
    chunks = processor.split_by_headers(docs)

    # 4. 打印结果
    print(f"\n总共得到 {len(chunks)} 个块\n")
    for chunk in chunks:
        print("=" * 50)
        print(f"块 ID: {chunk.metadata.get('chunk_id')}")
        print(f"标题层级: 一级='{chunk.metadata.get('header1')}', 二级='{chunk.metadata.get('header2')}', 三级='{chunk.metadata.get('header3')}'")
        print(f"内容预览（前80字符）: {chunk.page_content[:80].replace(chr(10), ' ')}...")
        print(f"完整元数据: {chunk.metadata}")
        print("=" * 50)