"""
WMS AI Agent主类（LangChain 1.x 版本）

这是一个库位库存查询智能助手，整合了 WMS 实时库存数据。

=============================================================================
当前项目功能（精简版）：
=============================================================================
  - WMS 工具（1个）：查询库位库存（按库位 / 按SKU / 按SPU，含低库存预警）

=============================================================================
LangChain 1.x 重大变化（相比 0.3.x）：
=============================================================================
  1. create_agent() 新 API：
     - 一站式创建 Agent，替代了 create_tool_calling_agent + AgentExecutor
     - 内部基于 LangGraph 构建，原生支持工具调用循环
     - 你不再需要手动管理 Thought → Action → Observation 循环

  2. 消息格式统一：
     - 输入输出都使用 {"messages": [消息列表]} 格式
     - 消息是 langchain_core.messages 中的标准消息类型

  3. 不再需要 agent_scratchpad：
     - 0.3.x 中需要手动在 prompt 里添加 MessagesPlaceholder("agent_scratchpad")
     - 1.x 中 create_agent 自动管理中间步骤的记录

  4. 不再需要 AgentExecutor：
     - 0.3.x: agent + AgentExecutor 两步创建
     - 1.x: create_agent 一步到位，返回的 CompiledStateGraph 可直接 invoke

  5. 流式支持内置：
     - 原生支持 .stream() 和 .astream()，无需额外配置

旧的写法（0.3.x）—— 仅供参考，已不再使用：
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="messages"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),  # 1.x不再需要
    ])
    agent = create_tool_calling_agent(llm, tools, prompt)       # 已废弃
    executor = AgentExecutor(agent, tools)                       # 已废弃
    result = executor.invoke({"input": "..."})

新的写法（1.x）：
    agent = create_agent(model=llm, tools=tools, system_prompt="...")
    result = agent.invoke({
        "messages": [HumanMessage(content="...")]
    })
    # create_agent 自动循环调用工具直到得到最终回答
    # result["messages"] 包含了完整的对话历史
    # 最后一条 AIMessage 就是最终回答
=============================================================================

Agent的工作流程：
  1. 接收用户问题（自然语言）
  2. LLM 分析意图，决定是否需要调用工具
  3. 如果需要，LLM 生成工具调用（tool_calls），系统自动执行
  4. 工具返回结果，LLM 继续思考（可能需要更多工具调用）
  5. LLM 认为信息足够时，生成最终回答
  6. 整个循环由 create_agent 内置的 LangGraph 引擎自动管理

Agent的能力：
  - 查询库位库存、SKU库存、商品库存等实时数据（通过WMS工具）
  - 进行库存分析和低库存预警（通过LLM）
  - 保持对话上下文，支持多轮对话
  - 自动决定调用哪些工具、调用几次
"""
from typing import Optional, List
from loguru import logger

# =============================================================================
# LangChain 1.x 导入
# =============================================================================

# create_agent：LangChain 1.0 的新 API，一站式创建 Agent
# 返回 CompiledStateGraph（LangGraph 编译后的状态图）
# 这个函数替代了 0.3.x 中的 create_tool_calling_agent + AgentExecutor
from langchain.agents import create_agent

# ChatOpenAI：OpenAI 兼容的大语言模型接口
# 支持任何 OpenAI 兼容的 API（包括国内代理、本地模型等）
from langchain_openai import ChatOpenAI

# 消息类型：LangChain 的标准化消息格式
# HumanMessage：用户消息
# SystemMessage：系统指令（在 create_agent 中通过 system_prompt 参数传递）
# AIMessage：AI 回复消息（可能包含 tool_calls）
# ToolMessage：工具执行结果消息
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage

# =============================================================================
# 项目内部导入
# =============================================================================
from .tools import create_all_tools
from ..wms import WMSClient
from ..utils.config import Config


class WMSAgent:
    """
    WMS AI Agent（LangChain 1.x 版）

    一个智能的库位库存查询助手，基于 LangChain 1.x 的 create_agent 构建。

    核心能力：
      - 理解用户的自然语言问题
      - 自动选择合适的工具获取实时库存数据
      - 基于获取的信息生成专业、准确的回答
      - 保持对话上下文，支持多轮对话

    使用方式：
        agent = WMSAgent(config)
        response = agent.chat("查询库位 W3-C1-01-01 的库存")

    LangChain 1.x 架构说明：
      create_agent 内部使用 LangGraph 的状态图（StateGraph）来编排
      LLM 和工具的交互。每次 invoke 时，LangGraph 会自动：
        1. 调用 LLM 分析消息
        2. 如果 LLM 返回 tool_calls，自动执行工具
        3. 将工具结果反馈给 LLM 继续思考
        4. 重复直到 LLM 给出最终回答（没有 tool_calls）
      这个循环对开发者而言是透明的——你只需要 invoke，不需要手动管理循环。
    """

    def __init__(self, config: Config):
        """
        初始化 Agent（LangChain 1.x 方式）

        Args:
            config: 配置对象，包含 OpenAI API密钥、WMS接口地址等

        初始化流程（5步）：
            1. 创建LLM（大语言模型）——Agent的"大脑"
            2. 创建WMS客户端——获取实时库存数据
            3. 创建工具集——将库存查询功能包装成Agent可调用的工具
            4. 初始化对话消息历史——保持多轮对话上下文
            5. 创建Agent（create_agent）——LangChain 1.x 一站式创建
        """
        logger.info("开始初始化 WMS AI Agent（LangChain 1.x）")

        # =====================================================================
        # 第1步：创建LLM（大语言模型）
        # =====================================================================
        # LLM 是 Agent 的"大脑"，负责：
        #   - 理解用户意图
        #   - 决定是否调用工具、调用哪个工具
        #   - 基于工具返回的数据生成最终回答
        #
        # ChatOpenAI 的参数说明：
        #   - api_key: API密钥（支持 OpenAI 官方、国内代理等）
        #   - base_url: API 基础URL（使用代理时需要修改）
        #   - model: 模型名称（如 gpt-4o, gpt-4o-mini, deepseek-chat 等）
        #   - temperature: 温度参数（0=精确，1=创意），0.7 是比较平衡的选择
        #   - max_tokens: 单次回复的最大 token 数
        self.llm = ChatOpenAI(
            api_key=config.openai.api_key,
            base_url=config.openai.api_base,
            model=config.openai.model,
            temperature=0.7,
            max_tokens=2000
        )
        logger.info(f"LLM 初始化完成，模型: {config.openai.model}")

        # =====================================================================
        # 第2步：创建WMS客户端
        # =====================================================================
        # WMS 客户端用于获取实时的库位库存数据
        # 这些数据是动态变化的——每次查询都从WMS系统获取最新数据
        self.wms_client = WMSClient(
            api_base_url=config.wms.api_base_url,
            api_token=config.wms.api_token
        )
        logger.info("WMS 客户端初始化完成")

        # =====================================================================
        # 第3步：创建工具集
        # =====================================================================
        # 将 WMS 客户端的库存查询功能包装成 LangChain 工具
        #
        # 当前只有 1 个工具：
        #   wms_query_inventory：查询库位库存（按库位 / 按SKU / 按SPU）
        #
        # 每个工具都有 name（名称）和 description（描述），
        # LLM 通过阅读 description 来判断何时使用哪个工具。
        self.tools = create_all_tools(self.wms_client)
        logger.info(f"工具集创建完成，共 {len(self.tools)} 个工具")
        # 打印工具列表，方便调试和理解
        for tool in self.tools:
            logger.info(f"  - {tool.name}: {tool.description[:50]}...")

        # =====================================================================
        # 第4步：初始化对话消息历史
        # =====================================================================
        # messages 是一个列表，存储完整的对话历史：
        #   [
        #       HumanMessage("查询SKU001的库存"),      # 第1轮用户问题
        #       AIMessage("SKU001当前库存150件..."),    # 第1轮AI回答
        #       HumanMessage("那SKU002呢？"),           # 第2轮用户问题
        #       AIMessage("SKU002当前库存80件..."),     # 第2轮AI回答
        #   ]
        #
        # 为什么要保存历史？
        #   - LLM 本身是无状态的，每次调用不记得之前说了什么
        #   - 把历史消息一起传给它，就能实现"多轮对话"的效果
        #   - create_agent 会自动管理消息历史，你只需要在调用时传入
        self.messages: list = []
        logger.info("对话消息历史初始化完成")

        # =====================================================================
        # 第7步：创建Agent（LangChain 1.x 的 create_agent）
        # =====================================================================
        # create_agent 是 LangChain 1.x 的核心 API，它：
        #   1. 接收 LLM、工具列表、系统提示词
        #   2. 内部使用 LangGraph 构建一个状态图（StateGraph）
        #   3. 返回 CompiledStateGraph 对象（编译后的可执行图）
        #
        # 状态图的工作流程（自动，无需手动控制）：
        #   ┌─────────┐    tool_calls?    ┌──────────┐
        #   │  LLM    │ ───────────────→  │ 执行工具  │
        #   │ 思考    │                   │          │
        #   │         │ ←───────────────  │          │
        #   └────┬────┘   工具结果         └──────────┘
        #        │ no tool_calls（最终回答）
        #        ▼
        #   ┌─────────┐
        #   │ 结束     │
        #   └─────────┘
        #
        # 对比 0.3.x 方式（已废弃）：
        #   0.3.x: 需要手动创建 prompt、agent、executor，手动管理工具调用循环
        #   1.x:   create_agent 一步搞定，工具调用循环全自动
        self.agent_executor = self._create_agent()
        logger.info("Agent 创建完成（create_agent）")

        logger.success("WMS AI Agent 初始化完成（LangChain 1.x）")

    def _create_system_prompt(self) -> str:
        """
        创建 Agent 的系统提示词

        系统提示词定义了 Agent 的：
          - 角色定位（我是一个专业的WMS仓库管理助手）
          - 行为准则（如何回复用户）
          - 工具使用指导（什么时候用什么工具）

        Returns:
            系统提示词字符串

        说明：
            在 LangChain 1.x 中，系统提示词直接传给 create_agent 的
            system_prompt 参数，不再需要手动构建 ChatPromptTemplate。
            create_agent 内部会自动处理系统消息的格式。
        """
        system_prompt = """
你是一个专业的WMS仓库库位库存查询助手，能够帮助用户查询库位库存、SKU库存、
商品库存等实时数据，并提供低库存预警分析。

你可以使用以下工具来完成用户的任务：
  - wms_query_inventory：查询库位库存数据
    · 支持三种查询维度：
      · 按库位查：传 location_code（库位编码，如 "W3-C1-01-01"）
      · 按SKU查：传 sku_code（SKU编码，如 "PT236DBKM"）
      · 按商品查：传 spu_code（商品编码，如 "PT236D"）
    · 也可以组合（如"W3-C1-01-01 库位上 PT236DBKM 的库存"）
    · 可选 filter_non_zero="Y" 过滤零库存；传空字符串 "" 不过滤
    · 可选 low_stock_threshold 设置低库存预警阈值，总库存<=阈值时标记⚠️

回复规则：
  1. 用户问"某个库位有什么库存"：传 location_code
  2. 用户问"某个SKU有多少库存 / 在哪个库位"：传 sku_code
  3. 用户问"哪个库位库存不足"：结合 low_stock_threshold 查询并标记低库存
  4. 如果查询结果为空，告诉用户"未查询到符合条件的库存记录（可能该库位/商品当前没有库存）"
  5. 回答要专业、清晰，使用中文，适当使用表格或分段展示数据
"""
        return system_prompt

    def _create_agent(self):
        """
        创建 Agent（LangChain 1.x 的 create_agent 方式）

        这是 LangChain 1.x 最核心的变化——从两步创建变成一步创建。

        Returns:
            CompiledStateGraph：LangGraph 编译后的状态图对象

        =========================================================================
        创建流程说明：
        =========================================================================

        1. get_system_prompt() → 获取系统提示词
        2. create_agent(
               model=llm,         # 大语言模型
               tools=tools,       # 工具列表
               system_prompt=prompt  # 系统提示词
           )
        3. 返回 CompiledStateGraph（可以直接 invoke、stream、ainvoke）

        =========================================================================
        create_agent 内部做了什么？
        =========================================================================

        create_agent 内部使用 LangGraph 的 prebuilt.create_react_agent：
          - 构建一个状态图（StateGraph），节点包括：LLM节点、工具执行节点
          - 边定义：LLM → 工具（如果有 tool_calls）→ LLM → ...
          - 循环直到 LLM 不再产生 tool_calls（即给出最终回答）
          - 编译成 CompiledStateGraph，提供 invoke/stream/ainvoke 接口

        这个编译后的图是 LangGraph 的核心概念——有向图 + 状态管理。
        每次 invoke，消息列表作为"状态"在图中的节点间流动。

        =========================================================================
        对比旧版（0.3.x）架构：
        =========================================================================

        旧版代码（已不再使用，仅供参考）：
            prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                MessagesPlaceholder("messages"),
                MessagesPlaceholder("agent_scratchpad"),
            ])
            agent = create_tool_calling_agent(llm, tools, prompt)
            executor = AgentExecutor(agent, tools, verbose=True)
            result = executor.invoke({"input": "..."})

        新版代码（1.x）：
            agent = create_agent(
                model=llm,
                tools=tools,
                system_prompt=system_prompt,
            )
            result = agent.invoke({
                "messages": [HumanMessage(content="...")]
            })
        =========================================================================
        """
        logger.info("创建 Agent（create_agent）")

        system_prompt = self._create_system_prompt()

        # create_agent：LangChain 1.x 的新 API
        # 参数说明：
        #   - model: 大语言模型实例（必须是 LangChain chat model）
        #   - tools: 工具列表（StructuredTool 或 @tool 装饰的函数）
        #   - system_prompt: 系统提示词字符串
        #   - response_format: 可选的输出格式（如 Pydantic 模型）
        #
        # 返回值 CompiledStateGraph 的方法：
        #   - invoke(input): 同步执行，返回最终状态
        #   - stream(input): 流式执行，逐块返回中间状态
        #   - ainvoke(input): 异步执行
        #   - astream(input): 异步流式执行
        agent = create_agent(
            model=self.llm,
            tools=self.tools,
            system_prompt=system_prompt,
        )

        logger.success("Agent 创建成功（CompiledStateGraph）")
        return agent

    def _sanitize_text(self, text: str) -> str:
        """
        清理文本，移除无法在 Windows 控制台显示的字符

        为什么需要这个？
          - Windows 控制台默认使用 GBK 编码
          - 有些特殊 Unicode 字符（如 emoji、特殊符号）在 GBK 中不存在
          - 如果不清理，print() 时会报 UnicodeEncodeError

        Args:
            text: 原始文本

        Returns:
            清理后的文本（移除了 GBK 不支持的字符）
        """
        if not text:
            return text
        try:
            # 尝试用 GBK 编码，如果成功说明所有字符都兼容
            text.encode('gbk')
            return text
        except UnicodeEncodeError:
            # 逐个字符检查，移除不兼容的字符
            cleaned = []
            for char in text:
                try:
                    char.encode('gbk')
                    cleaned.append(char)
                except UnicodeEncodeError:
                    # 用空格替代不兼容的字符（避免破坏排版）
                    cleaned.append(' ')
            return ''.join(cleaned)

    def _sanitize_messages(self, messages: list) -> list:
        """
        清理消息列表中的文本

        遍历消息列表，将每条消息的 content 通过 _sanitize_text 清理。

        Args:
            messages: 消息列表（HumanMessage, AIMessage 等）

        Returns:
            清理后的消息列表（原地修改）
        """
        for msg in messages:
            if hasattr(msg, 'content') and msg.content:
                msg.content = self._sanitize_text(msg.content)
        return messages

    def chat(self, user_input: str) -> str:
        """
        与 Agent 对话（LangChain 1.x 方式）

        Args:
            user_input: 用户输入的自然语言问题，例如：
                       "查询SKU001的库存"
                       "检查哪些商品需要补货"
                       "仓库现在的整体情况怎么样？"

        Returns:
            Agent 的回答（自然语言文本）

        =========================================================================
        对话流程（LangChain 1.x 自动化，无需手动管理循环）：
        =========================================================================

        1. 接收用户问题 → 包装成 HumanMessage
        2. 将消息列表传入 agent.invoke({"messages": [...]})
        3. create_agent 内部的 LangGraph 引擎自动执行：
           ┌─────────────────────────────────────────────────┐
           │  LLM 分析消息 → 判断是否需要工具                 │
           │     ↓ 需要                                      │
           │  生成 tool_calls → 自动执行工具 →               │
           │  工具结果以 ToolMessage 形式加入消息列表          │
           │     ↓                                           │
           │  回到 LLM 继续思考                               │
           │     ↓ 不需要（给出最终回答）                      │
           │  返回完整的消息列表，最后一条 AIMessage 就是答案   │
           └─────────────────────────────────────────────────┘
        4. 从返回的消息列表中提取最终回答
        5. 更新 self.messages 为完整的对话历史（用于下一轮）
        6. 返回回答给用户

        =========================================================================
        对比 0.3.x 方式（已废弃，仅供参考）：
        =========================================================================

        旧代码中需要手动管理工具调用循环：
            while tool_call_count < max_tool_calls:
                response = executor.invoke({"messages": messages})
                last_message = response["messages"][-1]
                if last_message.tool_calls:
                    for tc in last_message.tool_calls:
                        result = find_and_execute_tool(tc)
                        messages.append(ToolMessage(result, ...))
                    tool_call_count += 1
                else:
                    return last_message.content

        新代码中 create_agent 内部自动完成了这一切。
        这个变化大大简化了代码，减少了出错的可能。
        =========================================================================
        """
        logger.info(f"用户问题: {user_input}")

        try:
            # 步骤1：将用户问题包装成 HumanMessage
            # HumanMessage 是 LangChain 的标准消息格式
            # content 是纯文本，role 自动为 "user"
            self.messages.append(HumanMessage(content=user_input))

            # 步骤2：调用 agent.invoke()
            # create_agent 内部会：
            #   a. 将 self.messages 传入 LangGraph 状态图
            #   b. LLM 节点分析消息，判断是否需要工具
            #   c. 如果需要，自动调用工具，结果加入消息列表
            #   d. 循环直到 LLM 给出最终回答
            #   e. 返回包含完整对话历史的 {"messages": [...]}
            #
            # 输入格式固定为 {"messages": [消息列表]}
            # 所有消息都是 langchain_core.messages 中的类型：
            #   HumanMessage(用户消息), AIMessage(AI回复),
            #   ToolMessage(工具结果), SystemMessage(系统指令)
            response = self.agent_executor.invoke({
                "messages": self.messages
            })

            # 步骤3：获取返回的消息列表
            # response["messages"] 包含了完整的对话历史：
            #   之前的历史消息 + 当前轮的 HumanMessage
            #   + 可能的 AIMessage(含 tool_calls) + ToolMessage(工具结果)
            #   + 最终的 AIMessage(最终回答)
            response_messages = response.get("messages", [])

            if not response_messages:
                return "抱歉，系统未返回有效回复。请稍后再试。"

            # 清理消息中的特殊字符（Windows GBK 兼容）
            self._sanitize_messages(response_messages)

            # 步骤4：提取最终回答
            # 在 LangGraph 返回的消息列表中：
            #   - 中间可能有 AIMessage(tool_calls=[...]) 和 ToolMessage
            #   - 最后一条 AIMessage（没有 tool_calls）就是最终回答
            last_message = response_messages[-1]

            # 获取最终回答的文本内容
            answer = getattr(last_message, "content", None)

            if answer is None:
                # 兜底：如果最后一条消息没有 content（不太可能，但以防万一）
                answer = "抱歉，我无法理解您的问题。请换个方式重新提问。"

            # 步骤5：更新对话历史
            # 将返回的完整消息历史保存到 self.messages
            # 这样下一轮对话时，Agent 就能"记住"之前说了什么
            self.messages = response_messages

            # 步骤6：清理并返回
            answer = self._sanitize_text(answer)
            logger.success(f"Agent 回答: {answer[:100]}...")
            return answer

        except Exception as e:
            # 异常处理：确保在出错时也能给出友好的提示
            error_str = self._sanitize_text(str(e))
            logger.error(f"Agent 对话失败: {error_str}")
            return f"抱歉，处理您的请求时出现错误：{error_str}。请稍后再试或联系管理员。"

    def stream_chat(self, user_input: str):
        """
        流式对话（LangChain 1.x 新特性）

        与 chat() 功能相同，但是通过 .stream() 逐块返回结果，
        适合在 Web 界面上实现"打字机效果"。

        Args:
            user_input: 用户输入

        Yields:
            每个流式输出块（通常是 AIMessageChunk）

        说明：
            LangChain 1.x 的 create_agent 原生支持流式输出。
            这是 0.3.x 做起来比较麻烦的功能，1.x 开箱即用。
        """
        logger.info(f"流式对话: {user_input}")

        self.messages.append(HumanMessage(content=user_input))

        try:
            # stream() 返回一个生成器，逐块输出
            final_messages = None
            for chunk in self.agent_executor.stream({
                "messages": self.messages
            }):
                final_messages = chunk
                yield chunk

            # 流式完成后更新消息历史
            if final_messages and "messages" in final_messages:
                self.messages = final_messages["messages"]

        except Exception as e:
            error_str = self._sanitize_text(str(e))
            logger.error(f"流式对话失败: {error_str}")

    async def async_chat(self, user_input: str) -> str:
        """
        异步对话（LangChain 1.x 新特性）

        与 chat() 功能相同，但是异步执行，
        适合在 FastAPI/飞书机器人等异步框架中使用。

        Args:
            user_input: 用户输入

        Returns:
            Agent 的回答

        说明：
            LangChain 1.x 的 create_agent 原生支持异步调用（.ainvoke()），
            在异步 Web 框架中不会阻塞事件循环。
        """
        logger.info(f"异步对话: {user_input}")

        self.messages.append(HumanMessage(content=user_input))

        try:
            response = await self.agent_executor.ainvoke({
                "messages": self.messages
            })

            response_messages = response.get("messages", [])
            if not response_messages:
                return "抱歉，系统未返回有效回复。"

            self._sanitize_messages(response_messages)
            answer = getattr(response_messages[-1], "content", "抱歉，无法理解您的问题")
            self.messages = response_messages
            answer = self._sanitize_text(answer)

            logger.success(f"异步回答: {answer[:100]}...")
            return answer

        except Exception as e:
            error_str = self._sanitize_text(str(e))
            logger.error(f"异步对话失败: {error_str}")
            return f"抱歉，处理您的请求时出现错误：{error_str}。"

    def clear_memory(self):
        """
        清空对话记忆

        使用场景：
          - 开始全新的对话话题
          - 之前的对话上下文不再需要
          - 释放内存（对话历史过长时）

        说明：
            清空后 Agent 将不记得之前的任何对话内容，
            就像"重启"了一样。
        """
        logger.info("清空对话记忆")
        previous_count = len(self.messages)
        self.messages = []
        logger.success(f"对话记忆已清空（清除了 {previous_count} 条消息）")

    def get_available_tools(self) -> List[str]:
        """
        获取所有可用工具的名称列表

        Returns:
            工具名称列表，例如：
            ['wms_query_inventory', 'wms_query_orders', ...]

        说明：
            用于调试和展示。在界面上列出可用功能。
        """
        return [tool.name for tool in self.tools]

    def get_chat_history(self) -> List[dict]:
        """
        获取对话历史（用于展示或调试）

        Returns:
            消息列表，每条消息包含 role 和 content

        说明：
            将 LangChain 的消息对象转为简单的字典格式，
            方便在 Web 界面展示。
        """
        history = []
        for msg in self.messages:
            msg_type = type(msg).__name__
            if msg_type == "HumanMessage":
                role = "用户"
            elif msg_type == "AIMessage":
                role = "AI"
            elif msg_type == "ToolMessage":
                role = "工具"
            elif msg_type == "SystemMessage":
                role = "系统"
            else:
                role = msg_type

            content = getattr(msg, "content", "")
            content = self._sanitize_text(str(content))

            # 限制每条消息的显示长度
            if len(content) > 200:
                content = content[:200] + "..."

            history.append({"role": role, "content": content})

        return history

    def test_agent(self):
        """
        测试 Agent 功能

        执行一系列预设的测试问题，验证 Agent 是否正常工作。
        用于开发和调试阶段。

        测试覆盖的场景：
          - 按库位查询
          - 按 SKU 查询
          - 按库位+SKU 组合查询
          - 低库存检查
          - 综合查询
        """
        logger.info("=" * 60)
        logger.info("开始 Agent 功能测试（LangChain 1.x）")
        logger.info("=" * 60)

        test_cases = [
            ("按库位查询", "查询库位 W3-C1-01-01 的库存"),
            ("按SKU查询", "查询 SKU PT236DBKM 的库存分布"),
            ("组合查询", "查询 W3-C1-01-01 库位上 PT236DBKM 的库存"),
            ("低库存检查", "哪些 SKU PT236DBKM 的库存低于 50？"),
            ("综合查询", "仓库整体库存情况怎么样？"),
        ]

        for category, question in test_cases:
            logger.info(f"\n{'='*60}")
            logger.info(f"【{category}】测试问题: {question}")
            logger.info(f"{'='*60}")
            response = self.chat(question)
            logger.info(f"回答: {response}")
            logger.info("-" * 60)

        logger.info("\n" + "=" * 60)
        logger.success("Agent 功能测试完成")
        logger.info("=" * 60)


def create_wms_agent(config: Config) -> WMSAgent:
    """
    创建 WMS Agent 的便捷函数

    Args:
        config: 配置对象（包含所有必要的配置信息）

    Returns:
        WMSAgent 实例（已完全初始化，可直接使用）

    说明：
        这是创建 Agent 的推荐方式，封装了初始化过程。
        调用这个函数后，Agent 就可以直接用了：

            from src.utils import load_config
            from src.agent.wms_agent import create_wms_agent

            config = load_config()
            agent = create_wms_agent(config)
            response = agent.chat("查询库存")
    """
    return WMSAgent(config)
