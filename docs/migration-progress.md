# Zep → Graphiti + Neo4j 迁移进度

> 最后更新: 2026-04-04

## 完成状态: 5/14 (36%)

---

## ✅ 已完成

| Task | 状态 | 提交Hash | 说明 |
|------|------|---------|------|
| Task 1: 安装依赖 | ✅ | 5310b8e | 安装 graphiti-core, neo4j，添加 Neo4j 配置 |
| Task 2: Graphiti 服务基类 | ✅ | 40a261c | graphiti_service.py + MiniMax 客户端 |
| Task 3: 图谱检索工具 | ✅ | 788a4bc | graphiti_tools.py |
| Task 4: 实体读取服务 | ✅ | 2fa6399 | entity_reader.py |
| Task 5: 动态记忆更新 | ✅ | 235cd7b | memory_updater.py |

---

## ⏸️ 待完成

| Task | 描述 | 文件 |
|------|------|------|
| Task 6 | 重写图谱构建服务 | graph_builder.py |
| Task 7 | 更新服务导出 | __init__.py |
| Task 8 | 适配 simulation_manager | simulation_manager.py |
| Task 9 | 适配 report_agent | report_agent.py |
| Task 10 | 移除 Zep 调用 | oasis_profile_generator.py |
| Task 11 | 适配 API 端点 | graph.py, simulation.py, report.py |
| Task 12 | 前端日志提示 | Step2EnvSetup.vue |
| Task 13 | 前端图谱展示 | GraphPanel.vue |
| Task 14 | 端到端测试 | - |

---

## 环境状态

- **Neo4j**: 运行中 (Docker container: neo4j)
- **依赖**: graphiti-core 0.28.2, neo4j 6.1.0 已安装
- **配置**: NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD 已添加到 config.py

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

## 新建文件列表

```
backend/app/services/
├── graphiti_service.py      ✅ (Task 2)
├── graphiti_tools.py    ✅ (Task 3)
├── entity_reader.py    ✅ (Task 4)
└── memory_updater.py ✅ (Task 5)
```

---

## 修改文件列表

```
backend/app/
├── config.py              ✅ (Task 1 - 添加 Neo4j 配置)
├── services/__init__.py   ⏸️ (待更新)
├── services/graph_builder.py ⏸️ (待重写)
├── services/simulation_manager.py ⏸️ (待适配)
├── services/report_agent.py ⏸️ (待适配)
├── services/oasis_profile_generator.py ⏸️ (待修改)
├── api/graph.py        ⏸️ (待适配)
├── api/simulation.py ⏸️ (待适配)
└── api/report.py    ⏸️ (待适配)

frontend/src/components/
├── Step2EnvSetup.vue ⏸️
└── GraphPanel.vue ⏸️
```

---

## 下次继续

从 **Task 6: 重写图谱构建服务** 开始

命令：
```bash
# 确保 Neo4j 运行
docker start neo4j

# 检查当前状态
git log --oneline -5
```