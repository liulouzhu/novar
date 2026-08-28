---
name: research
description: 搜索论文并生成综述
---
请帮我调研以下主题：$ARGUMENTS

执行步骤：
1. 用 paper_search 搜索相关论文（使用英文关键词，limit 设为 10）
2. 浏览返回结果，选择最相关的 2-3 篇用 paper_parse 解析全文（解析完成后会自动从摘要提取实体到知识图谱）
3. 用 rag_query 针对关键问题在已解析论文中检索细节
4. 用 knowledge_graph(action="query") 查看已提取的实体和关系，了解研究脉络
5. 对每篇已解析论文，提取 **主要贡献（Contribution）** 和 **局限性（Limitation）**，用 knowledge_graph(action="extract_from_abstract", paper_id="...", entities=[{"name":"...","type":"Contribution"},{"name":"...","type":"Limitation"}]) 写入图谱
6. 用中文撰写结构化综述：
   - 研究背景与动机
   - 主要方法与技术路线
   - 关键发现与结论
   - 现有局限与未来方向
6. 每个论点引用具体论文（标题、作者、年份）
