# WMS AI Agent 使用指南

## 目录
1. [项目概述](#项目概述)
2. [快速开始](#快速开始)
3. [配置说明](#配置说明)
4. [运行方式](#运行方式)
5. [功能说明](#功能说明)
6. [常见问题](#常见问题)
7. [进阶使用](#进阶使用)

---

## 项目概述

### 项目简介
WMS AI Agent是一个智能的仓库管理助手，能够：
- 查询库存、订单、入库等实时数据
- 检索操作手册、业务规则等知识信息
- 提供智能分析和运营建议
- 通过飞书机器人与用户对话

### 系统架构
```
飞书用户 → 飞书机器人 → Web服务器 → Agent → WMS客户端/RAG系统 → 数据/知识
```

### 核心技术
- **LangChain**: AI Agent框架
- **RAG**: 检索增强生成（知识库）
- **OpenAI GPT**: 大语言模型
- **ChromaDB**: 向量数据库
- **FastAPI**: Web框架
- **飞书SDK**: 消息收发

---

## 快速开始

### 1. 安装依赖
```bash
pip install -r requirements.txt
```

### 2. 配置API密钥
复制 `.env.example` 为 `.env`，并填写必要的配置：
```bash
copy .env.example .env
```

编辑 `.env` 文件，填写：
```
# 必填：OpenAI API密钥
OPENAI_API_KEY=sk-your-key-here

# 可选：飞书配置（如果要使用飞书机器人）
FEISHU_APP_ID=cli_xxxxxx
FEISHU_APP_SECRET=your-secret-here
```

### 3. 启动服务
```bash
python main.py
```

### 4. 测试Agent
访问 http://localhost:8080/test 或使用命令行测试：
```bash
python main.py --test
```

---

## 配置说明

### OpenAI配置（必填）
```env
OPENAI_API_KEY=sk-your-api-key-here
OPENAI_API_BASE=https://api.openai.com/v1  # 或自定义端点
OPENAI_MODEL=gpt-4-turbo-preview  # 推荐使用GPT-4
```

### WMS系统配置（可选）
```env
WMS_API_BASE_URL=http://your-wms-api.com/api
WMS_API_TOKEN=your-wms-token-here
```
目前使用模拟数据，未来可替换为真实接口。

### 飞书机器人配置（可选）
```env
FEISHU_APP_ID=cli_xxxxxx
FEISHU_APP_SECRET=your-app-secret-here
FEISHU_VERIFICATION_TOKEN=your-verification-token
FEISHU_ENCRYPT_KEY=your-encrypt-key  # 可选
```

### 向量数据库配置
```env
VECTOR_DB_PATH=./data/chroma_db
CHUNK_SIZE=500  # 文档块大小
CHUNK_OVERLAP=50  # 文档块重叠
```

### 服务配置
```env
SERVER_PORT=8080  # Web服务端口
LOG_LEVEL=INFO  # 日志级别
```

---

## 运行方式

### 方式1：完整服务模式（推荐）
启动完整的Web服务器，支持飞书机器人：
```bash
python main.py
```

服务启动后：
- 访问 http://localhost:8080/docs 查看API文档
- 访问 http://localhost:8080/health 检查服务状态
- 访问 http://localhost:8080 测试Agent

### 方式2：测试模式
直接测试Agent功能，无需启动服务器：
```bash
python main.py --test
```

### 方式3：使用测试接口
通过HTTP接口测试：
```bash
curl -X POST http://localhost:8080/test \
  -H "Content-Type: application/json" \
  -d "{\"message\": \"查询SKU001的库存\"}"
```

### 方式4：飞书机器人模式
配置飞书开放平台：
1. 登录飞书开放平台：https://open.feishu.cn
2. 创建应用，获取APP_ID和APP_SECRET
3. 配置事件订阅URL：`http://your-server-ip:8080/webhook`
4. 启用消息接收权限
5. 在飞书中与机器人对话

---

## 功能说明

### Agent能力
Agent可以回答以下类型的问题：

#### 1. 库存查询
- "查询SKU001的库存"
- "查询所有库存"
- "库存数量是多少"
- "SKU002在哪里"

#### 2. 订单查询
- "查询订单ORD20240115001"
- "查询所有待处理订单"
- "查询采购订单"
- "订单统计"

#### 3. 入库查询
- "查询入库单IN20240114001"
- "查询SKU001的入库记录"
- "最近的入库"

#### 4. 库存检查
- "检查低库存"
- "哪些商品需要补货"
- "库存预警"

#### 5. 综合信息
- "仓库概览"
- "运营状况"
- "整体情况"

#### 6. 知识咨询
- "如何查询库存"
- "退货订单处理流程"
- "SKU001的产品信息"
- "业务规则"

### Agent工具列表
- `wms_query_inventory`: 库存查询
- `wms_query_orders`: 订单查询
- `wms_query_inbounds`: 入库查询
- `wms_get_statistics`: 统计信息
- `wms_check_low_stock`: 低库存检查
- `wms_get_summary`: 仓库概览
- `rag_search_knowledge`: 知识检索

---

## 常见问题

### Q1: Agent初始化失败
**问题**: 提示"Agent创建失败"
**解决**:
- 检查OPENAI_API_KEY是否正确
- 检查API_BASE是否可访问
- 查看日志文件logs/app.log

### Q2: 飞书消息无法接收
**问题**: 飞书机器人不回复消息
**解决**:
- 检查飞书配置是否正确
- 确认事件订阅URL配置正确
- 确认服务器可以被飞书访问（公网IP）
- 查看飞书开放平台的请求日志

### Q3: RAG检索无结果
**问题**: 知识检索返回空
**解决**:
- 检查向量数据库路径是否正确
- 确认知识库已构建（首次启动时会自动构建）
- 查看日志确认向量化过程是否成功

### Q4: 响应速度慢
**问题**: Agent回复慢
**原因**:
- OpenAI API调用需要时间
- 向量检索需要时间
- 多轮工具调用会累积时间
**优化建议**:
- 使用更快的模型（如gpt-3.5-turbo）
- 减少chunk_size以加快检索
- 优化提示词减少工具调用次数

---

## 进阶使用

### 1. 自定义WMS接口
替换模拟数据为真实API：
- 编辑 `src/wms/wms_client.py`
- 修改 `_init_mock_data()` 方法为真实API调用
- 使用 `requests` 或 `httpx` 调用WMS API

示例：
```python
def query_inventory(self, sku: Optional[str] = None):
    # 调用真实API
    response = requests.get(
        f"{self.api_base_url}/inventory",
        headers={"Authorization": f"Bearer {self.api_token}"},
        params={"sku": sku} if sku else {}
    )
    return InventoryQueryResult(**response.json())
```

### 2. 自定义知识库
添加真实文档：
- 将文档放在 `data/documents/` 目录
- 支持.txt和.md格式
- 编辑 `src/rag/document_processor.py`

示例：
```python
# 加载真实文档
documents = processor.process_directory("./data/documents")
```

### 3. 自定义Agent行为
修改提示词：
- 编辑 `src/agent/wms_agent.py`
- 修改 `_create_prompt_template()` 方法
- 调整Agent的角色和回答风格

### 4. 添加新工具
扩展Agent能力：
- 编辑 `src/agent/tools.py`
- 添加新的Tool对象
- 定义name、description和func

示例：
```python
Tool(
    name="wms_new_feature",
    description="新功能的描述...",
    func=self.new_feature_impl
)
```

### 5. 部署到生产环境
推荐部署方案：
- 使用云服务器（阿里云、腾讯云等）
- 配置nginx反向代理
- 使用supervisor管理进程
- 配置日志监控

---

## 项目结构

```
wms-agent/
├── src/
│   ├── wms/          # WMS接口模块
│   ├── rag/          # RAG知识库模块
│   ├── feishu/       # 飞书机器人模块
│   ├── agent/        # Agent核心模块
│   └── utils/        # 工具模块
├── data/             # 数据存储
├── tests/            # 测试文件
├── logs/             # 日志文件
├── docs/             # 文档
├── main.py           # 主入口
├── requirements.txt  # 依赖包
├── .env.example      # 配置示例
└── .gitignore
```

---

## 下一步建议

1. **理解代码**: 阅读各模块的代码和注释
2. **运行测试**: 使用测试模式验证功能
3. **配置飞书**: 配置飞书机器人进行实际对话
4. **替换数据**: 替换模拟数据为真实WMS接口
5. **优化性能**: 根据实际情况调整配置
6. **扩展功能**: 根据需求添加新工具和知识

---

## 技术支持

遇到问题可以：
- 查看日志文件: `logs/app.log`
- 查看代码注释: 每个文件都有详细说明
- 参考概念指南: `docs/concepts_guide.md`
- 使用测试接口调试: `/test`

---

**祝你使用愉快！这是一个完整的AI Agent应用，希望你能通过它学会Agent开发的全流程。**