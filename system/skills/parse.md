---
name: parse
description: 解析论文 PDF 并提取关键信息
---
请解析以下论文：$ARGUMENTS

执行步骤：
1. 如果输入是关键词，先用 paper_search 找到论文
2. 如果输入是论文 ID 或 URL，直接用 paper_parse 解析（解析完成后会自动从摘要提取方法、数据集、任务等实体到知识图谱）
3. 解析完成后，提取以下信息：
   - 研究问题（Research Question）
   - 方法（Methodology）
   - 关键发现（Key Findings）
   - 局限性（Limitations）
   - 主要贡献（Contributions）
4. 用 knowledge_graph(action="query") 查看自动提取的实体和关系
5. 如果摘要自动提取遗漏了重要实体，用 knowledge_graph(action="extract_from_abstract") 手动补充
6. 识别论文的 **主要贡献（Contribution）** 和 **局限性（Limitation）**，用 knowledge_graph(action="extract_from_abstract", paper_id="...", entities=[{"name":"贡献描述","type":"Contribution"},{"name":"局限描述","type":"Limitation"}]) 写入图谱
7. 用中文输出结构化摘要
