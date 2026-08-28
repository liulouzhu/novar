---
name: compile
description: 解析论文并构建知识图谱
---
请解析以下论文：$ARGUMENTS

执行步骤：
1. 如果输入是关键词，先用 paper_search 搜索相关论文（英文关键词，limit 设为 10）
2. 浏览返回结果，选择最相关的论文用 paper_parse 解析（会自动完成：下载 PDF → 分块 → 向量化 → 摘要实体提取 → 知识图谱构建）
3. 解析完成后，提取以下信息：
   - 研究问题（Research Question）
   - 方法（Methodology）
   - 关键发现（Key Findings）
   - 局限性（Limitations）
   - 主要贡献（Contributions）
4. 用 knowledge_graph(action="query") 查看自动提取的实体和关系
5. 如果自动提取遗漏了重要实体，用 knowledge_graph(action="extract_from_abstract", paper_id="...", entities=[...]) 手动补充
6. 识别论文的 **主要贡献（Contribution）** 和 **局限性（Limitation）**，用 knowledge_graph(action="extract_from_abstract", paper_id="...", entities=[{"name":"贡献描述","type":"Contribution"},{"name":"局限描述","type":"Limitation"}]) 写入图谱
7. 用中文输出结构化摘要，附带知识图谱中的概念关系
