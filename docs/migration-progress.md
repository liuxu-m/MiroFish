# Zep → Graphiti + Neo4j 迁移进度

> 最后更新: 2026-04-04

## 完成状态: 14/14 (100%)

---

## ✅ 已完成

| Task | 状态 | 提交Hash | 说明 |
|------|------|---------|------|
| Task 1 | ✅ | 5310b8e | 安装 graphiti-core, neo4j，添加 Neo4j 配置 |
| Task 2 | ✅ | 40a261c | graphiti_service.py + MiniMax 客户端 |
| Task 3 | ✅ | 788a4bc | graphiti_tools.py |
| Task 4 | ✅ | 2fa6399 | entity_reader.py |
| Task 5 | ✅ | 235cd7b | memory_updater.py |
| Task 6 | ✅ | 33a2207 | 重写 graph_builder.py |
| Task 7 | ✅ | f6c6993 | 更新 __init__.py 导出 |
| Task 8 | ✅ | 7ceb0fb | 适配 simulation_manager |
| Task 9 | ✅ | 678beae | 适配 report_agent |
| Task 10 | ✅ | b98832d | 移除 oasis_profile_generator 中的 Zep |
| Task 11 | ✅ | 9fd1c46 | 适配 API 端点 |
| Task 12 | ✅ | 523b228 | 前端日志提示 |
| Task 13 | ✅ | - | 图谱展示无需修改 |
| Task 14 | ✅ | - | 后端模块导入测试通过 |

---

## 环境状态

- **Neo4j**: 需手动启动 (`docker start neo4j`)
- **依赖**: graphiti-core 0.28.2, neo4j 6.1.0 已安装
- **后端**: 模块导入测试通过

---

## ⚠️ 待配置 (.env)

在项目根目录 `.env` 文件中添加：

```
# MiniMax (LLM)
MINIMAX_API_KEY=your_api_key_here

# OpenAI (Embedding - 后续替换)
OPENAI_API_KEY=your_api_key_here
```

---

## 使用说明

1. 启动 Neo4j:
```bash
docker start neo4j
```

2. 启动后端:
```bash
cd backend
uvicorn app.main:app --reload
```

3. 启动前端:
```bash
cd frontend
npm run dev
```

---

## 新增文件

```
backend/app/services/
├── graphiti_service.py      # Graphiti 核心服务 + MiniMax 客户端
├── graphiti_tools.py     # 图谱检索工具
├── entity_reader.py    # 实体读取服务
└── memory_updater.py  # 动态记忆更新
```

## 修改文件

```
backend/app/
├── config.py                    # Neo4j 配置
├── services/
│   ├── __init__.py             # 服务导出
│   ├── graph_builder.py         # 重写为 Graphiti
│   ├── simulation_manager.py    # 适配
│   ├── report_agent.py        # 适配
│   └── oasis_profile_generator.py # 移除 Zep
├── api/
│   ├── graph.py               # 适配
│   ├── simulation.py         # 适配
│   └── report.py             # 适配
frontend/src/components/
└── Step2EnvSetup.vue          # 日志提示
locales/
├── zh.json
└── en.json
```