# WMS AI Agent - 仓库管理智能助手

一个基于LangChain和RAG的智能仓库管理AI Agent，通过飞书机器人实现对话交互。

## 🎯 项目特点

- **智能对话**: 理解自然语言，自动调用工具获取数据
- **知识检索**: RAG技术检索操作手册和业务规则
- **数据分析**: 实时库存、订单、入库数据查询和分析
- **飞书集成**: 通过飞书机器人便捷交互
- **详细注释**: 完整的教学代码，适合学习Agent开发

## 📦 功能列表

- 库存查询（单个SKU/全部库存）
- 订单查询（按订单号/状态/类型）
- 入库记录查询
- 低库存检查和补货建议
- 仓库运营概览
- 知识库检索（操作流程/业务规则）

## 🚀 快速开始

### 1. 安装依赖
```bash
pip install -r requirements.txt
```

### 2. 配置环境变量
复制 `.env.example` 为 `.env`，填写OpenAI API密钥：
```bash
copy .env.example .env
# 编辑.env文件，填写OPENAI_API_KEY
```

### 3. 启动服务
```bash
python main.py
```

### 4. 测试Agent
访问 http://localhost:8080/test 或：
```bash
python main.py --test
```

## 📚 学习资源

- [概念讲解](docs/concepts_guide.md) - LangChain、RAG、Agent原理
- [使用指南](docs/USAGE_GUIDE.md) - 详细的使用和配置说明
- 代码注释 - 每个文件都有详细的中文注释

## 🏗️ 项目结构

```
wms-agent/
├── src/
│   ├── wms/          # WMS数据获取模块
│   ├── rag/          # RAG知识库模块
│   ├── feishu/       # 飞书机器人模块
│   ├── agent/        # Agent核心模块
│   └── utils/        # 工具函数
├── main.py           # 主入口
├── requirements.txt  # 依赖包
└── .env.example      # 配置示例
```

## 🔧 配置说明

必填配置：
- `OPENAI_API_KEY`: OpenAI API密钥

可选配置：
- `FEISHU_APP_ID`: 飞书应用ID（使用飞书机器人时需要）
- `WMS_API_BASE_URL`: WMS系统API地址（替换模拟数据时需要）

## 💡 核心技术

- **LangChain**: AI Agent框架
- **RAG**: 检索增强生成
- **OpenAI GPT**: 大语言模型
- **ChromaDB**: 向量数据库
- **FastAPI**: Web框架
- **飞书SDK**: 机器人交互

## 📖 代码讲解

项目采用教学式开发，每个文件都包含：
- 详细的功能说明
- 实现原理讲解
- 关键代码注释
- 使用示例

适合：
- 学习Agent应用开发
- 理解LangChain框架
- 掌握RAG技术
- 实战飞书机器人开发

## ⚙️ 自定义扩展

- 替换WMS模拟数据为真实API
- 添加自定义知识库文档
- 扩展Agent工具能力
- 修改Agent回答风格

详见 [使用指南](docs/USAGE_GUIDE.md)

## 🤝 适合人群

- Agent应用开发初学者
- LangChain框架学习者
- 企业AI应用开发者
- 对RAG技术感兴趣的开发者

## 📝 开发进度

- ✅ 环境搭建和配置
- ✅ WMS接口模块
- ✅ RAG知识库模块
- ✅ Agent核心模块
- ✅ 飞书机器人集成
- ✅ 测试和文档

## 🎓 学习建议

1. 先阅读概念讲解，理解原理
2. 运行WMS客户端测试，理解数据获取
3. 运行Agent测试，理解对话流程
4. 配置飞书机器人，实际交互体验
5. 自定义扩展，深入学习

---

**这是一个完整的AI Agent教学项目，带你从零开始学会Agent应用开发！**