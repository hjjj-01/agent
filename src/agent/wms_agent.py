"""
WMS AI Agent主类

这是整个系统的核心，整合了所有模块，提供智能对话功能。

Agent的工作流程：
1. 接收用户问题
2. 使用LLM分析用户意图
3. 决定调用哪些工具
4. 执行工具并获取结果
5. 基于结果生成回答

Agent的能力：
- 查询库存、订单、入库等实时数据（通过WMS工具）
- 检索操作手册、业务规则等知识（通过RAG工具）
- 进行复杂的分析和推理（通过LLM）
- 提供个性化的回答和建议
"""
from typing import Optional, List
from loguru import logger

# LangChain相关导入
from langchain.agents import AgentExecutor, create_react_agent
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema import AgentAction, AgentFinish
from langchain.memory import ConversationBufferMemory

# 导入工具和配置
from .tools import create_all_tools
from ..wms import WMSClient
from ..rag import RAGSystem
from ..utils.config import Config


class WMSAgent:
    """
    WMS AI Agent

    一个智能的WMS仓库助手，能够：
    - 理解用户的自然语言问题
    - 自动选择合适的工具获取数据或知识
    - 基于获取的信息生成专业、准确的回答
    - 保持对话上下文，支持多轮对话

    使用方式：
        agent = WMSAgent(config)
        response = agent.chat("查询SKU001的库存")
    """

    def __init__(self, config: Config):
        """
        初始化Agent

        Args:
            config: 配置对象

        初始化流程：
            1. 创建LLM（大语言模型）
            2. 创建WMS客户端
            3. 创建RAG系统
            4. 创建工具集
            5. 创建Agent执行器

        说明：
            这个初始化过程将所有模块整合在一起，
            形成一个完整的智能对话系统。
        """
        logger.info("开始初始化WMS AI Agent")

        # 1. 创建LLM（大语言模型）
        # LLM是Agent的"大脑"，负责理解意图、决策、生成回答
        self.llm = ChatOpenAI(
            api_key=config.openai.api_key,
            base_url=config.openai.api_base,
            model=config.openai.model,
            temperature=0.7,  # 控制回答的创造性，0-1之间
            max_tokens=2000   # 限制回答长度
        )
        logger.info(f"LLM初始化完成，模型: {config.openai.model}")

        # 2. 创建WMS客户端
        # 用于获取实时的库存、订单等数据
        self.wms_client = WMSClient(
            api_base_url=config.wms.api_base_url,
            api_token=config.wms.api_token
        )
        logger.info("WMS客户端初始化完成")

        # 3. 创建RAG系统
        # 用于检索操作手册、业务规则等知识
        self.rag_system = RAGSystem(config)
        logger.info("RAG系统初始化完成")

        # 4. 构建知识库（使用示例数据）
        # 实际使用时可以加载真实文档
        self.rag_system.build_knowledge_base(use_sample_data=True)
        logger.info("知识库构建完成")

        # 5. 创建工具集
        # 将WMS和RAG功能包装成Agent可调用的工具
        self.tools = create_all_tools(self.wms_client, self.rag_system)
        logger.info(f"工具集创建完成，共 {len(self.tools)} 个工具")

        # 6. 创建对话记忆
        # 用于保持多轮对话的上下文
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        logger.info("对话记忆初始化完成")

        # 7. 创建Agent执行器
        # Agent执行器是LangChain的核心组件，协调LLM和工具的交互
        self.agent_executor = self._create_agent_executor()
        logger.info("Agent执行器创建完成")

        logger.success("WMS AI Agent初始化完成")

    def _create_prompt_template(self) -> PromptTemplate:
        """
        创建Agent的提示词模板

        提示词的作用：
            - 定义Agent的角色和行为
            - 告诉Agent如何使用工具
            - 定义输出格式

        ReAct提示词结构：
            Thought: 思考当前情况
            Action: 决定使用什么工具
            Action Input: 工具的输入参数
            Observation: 工具的输出结果
            ... （重复直到得出答案）
            Final Answer: 最终回答

        Returns:
            PromptTemplate对象
        """
        # ReAct格式的提示词模板
        # ReAct = Reasoning + Acting
        template = """
你是一个专业的WMS仓库管理助手，能够帮助用户查询库存、订单、入库等信息，
并提供仓库运营建议。你有丰富的仓库管理知识和经验。

你可以使用以下工具：
{tools}

工具名称: {tool_names}

使用工具时，请遵循以下格式：

Question: 用户的问题
Thought: 你应该思考要做什么
Action: 要使用的工具名称（必须是 [{tool_names}] 中的一个）
Action Input: 工具的输入参数
Observation: 工具的执行结果
... (这个 Thought/Action/Action Input/Observation 可以重复N次)
Thought: 我现在知道最终答案了
Final Answer: 对用户问题的最终回答

开始！

历史对话:
{chat_history}

Question: {input}
{agent_scratchpad}
"""

        prompt = PromptTemplate.from_template(template)
        return prompt

    def _create_agent_executor(self) -> AgentExecutor:
        """
        创建Agent执行器

        Agent执行器的作用：
            - 协调LLM和工具的交互
            - 执行推理循环（Thought → Action → Observation）
            - 处理错误和异常
            - 返回最终答案

        Returns:
            AgentExecutor对象
        """
        logger.info("创建Agent执行器")

        # 创建提示词模板
        prompt = self._create_prompt_template()

        # 创建ReAct Agent
        # ReAct Agent是一种强大的Agent类型，能够推理和行动
        agent = create_react_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=prompt
        )

        # 创建Agent执行器
        # Agent执行器管理Agent的运行，包括错误处理、迭代限制等
        agent_executor = AgentExecutor(
            agent=agent,
            tools=self.tools,
            memory=self.memory,
            verbose=True,  # 打印详细日志（调试用）
            max_iterations=10,  # 最大迭代次数，防止无限循环
            handle_parsing_errors=True,  # 自动处理解析错误
            return_intermediate_steps=False  # 不返回中间步骤（简化输出）
        )

        return agent_executor

    def chat(self, user_input: str) -> str:
        """
        与Agent对话

        Args:
            user_input: 用户输入的问题

        Returns:
            Agent的回答

        对话流程：
            1. 接收用户问题
            2. Agent分析问题，决定使用哪些工具
            3. 执行工具，获取结果
            4. 基于结果生成回答
            5. 返回回答给用户

        说明：
            这是一个智能的过程，Agent会：
            - 自动识别问题类型（库存查询、知识咨询等）
            - 自动选择合适的工具
            - 可能多次调用工具获取完整信息
            - 基于获取的信息进行推理和分析
            - 生成专业、准确、个性化的回答
        """
        logger.info(f"用户问题: {user_input}")

        try:
            # 调用Agent执行器处理问题
            # Agent会自动进行推理、调用工具、生成回答
            response = self.agent_executor.invoke({
                "input": user_input
            })

            # 获取最终回答
            answer = response.get("output", "抱歉，我无法理解您的问题")

            logger.success(f"Agent回答: {answer[:100]}...")  # 只显示前100字符
            return answer

        except Exception as e:
            logger.error(f"Agent对话失败: {str(e)}")
            # 返回友好的错误提示
            return f"抱歉，处理您的请求时出现错误：{str(e)}。请稍后再试或联系管理员。"

    def clear_memory(self):
        """
        清空对话记忆

        说明:
            开始新的对话时可以清空之前的记忆。
        """
        logger.info("清空对话记忆")
        self.memory.clear()
        logger.success("对话记忆已清空")

    def get_available_tools(self) -> List[str]:
        """
        获取可用的工具列表

        Returns:
            工具名称列表

        说明:
            用于了解Agent有哪些能力。
        """
        return [tool.name for tool in self.tools]

    def test_agent(self):
        """
        测试Agent功能

        说明:
            执行一系列测试对话，验证Agent是否正常工作。
            用于开发和调试阶段。
        """
        logger.info("开始Agent功能测试")

        test_cases = [
            "查询SKU001的库存",
            "查询所有库存",
            "查询所有待处理订单",
            "如何处理退货订单？",
            "检查低库存商品",
            "仓库概览"
        ]

        for question in test_cases:
            logger.info(f"\n测试问题: {question}")
            response = self.chat(question)
            logger.info(f"回答: {response}\n")
            logger.info("-" * 60)

        logger.success("Agent功能测试完成")


def create_wms_agent(config: Config) -> WMSAgent:
    """
    创建WMS Agent的便捷函数

    Args:
        config: 配置对象

    Returns:
        WMSAgent实例

    说明:
        这是创建Agent的推荐方式，封装了初始化过程。
    """
    return WMSAgent(config)