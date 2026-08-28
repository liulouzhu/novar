"""科研智能体 MCP Server - stdio 传输"""

import asyncio
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# 加载项目根目录的 .env
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from mcp.server import Server, InitializationOptions, NotificationOptions
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

# 预加载重型依赖（避免在 handler 中首次 import 阻塞事件循环）
import numpy  # noqa: F401

# 日志配置（输出到 stderr，不干扰 stdio 通信）
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger("research-server")

# 数据目录
DATA_DIR = os.environ.get("RESEARCH_DATA_DIR", "./data")
PAPERS_DIR = os.path.join(DATA_DIR, "papers")
KG_PATH = os.path.join(DATA_DIR, "knowledge_graph.json")

# 确保目录存在
os.makedirs(PAPERS_DIR, exist_ok=True)

server = Server("research")


# ── 工具注册 ──────────────────────────────────────────────────────────────

@server.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="echo",
            description="Echo 回显工具，用于测试 MCP 连接是否正常。",
            inputSchema={
                "type": "object",
                "properties": {
                    "message": {
                        "type": "string",
                        "description": "要回显的消息",
                    }
                },
                "required": ["message"],
            },
        ),
        Tool(
            name="paper_search",
            description="搜索学术论文。同时查询 Semantic Scholar 和 arXiv，返回论文列表（标题、作者、摘要、引用数、PDF 链接）。",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "搜索关键词或自然语言描述"},
                    "year_from": {"type": "integer", "description": "起始年份（可选）"},
                    "year_to": {"type": "integer", "description": "结束年份（可选）"},
                    "limit": {"type": "integer", "description": "返回数量，默认 10，最大 20"},
                },
                "required": ["query"],
            },
        ),
        Tool(
            name="paper_parse",
            description="解析论文 PDF，提取结构化内容（各章节文本、参考文献），自动建立 RAG 索引。",
            inputSchema={
                "type": "object",
                "properties": {
                    "paper_id": {"type": "string", "description": "论文 ID（Semantic Scholar ID 或 arXiv ID）"},
                    "pdf_url": {"type": "string", "description": "直接提供 PDF URL（与 paper_id 二选一）"},
                    "upload_id": {"type": "string", "description": "Web 上传返回的 upload_id（推荐，自动校验当前用户权限并去重）"},
                    "file_path": {"type": "string", "description": "兼容 CLI 的本地 PDF 路径；Web 上传请使用 upload_id"},
                },
            },
        ),
        Tool(
            name="rag_query",
            description="在已解析的论文库中进行语义检索，返回最相关的文本片段。需要先用 paper_parse 解析论文。",
            inputSchema={
                "type": "object",
                "properties": {
                    "question": {"type": "string", "description": "自然语言问题"},
                    "top_k": {"type": "integer", "description": "返回结果数量（1-50），默认 5"},
                    "paper_id": {
                        "type": "string",
                        "description": "限定在指定论文中检索（与 paper_ids 二选一）",
                    },
                    "paper_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "限定在指定论文列表中检索（与 paper_id 二选一）",
                    },
                },
                "required": ["question"],
            },
        ),
        Tool(
            name="knowledge_graph",
            description="查询或更新当前用户的论文知识图谱（存储在 workspace/{user_id}/knowledge_graph.json）。支持添加论文/概念/关系，查询子图，查找路径。",
            inputSchema={
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["add_paper", "add_concept", "add_relation", "extract_from_abstract", "normalize", "query", "find_path", "stats"],
                    },
                    "paper_id": {"type": "string"},
                    "name": {"type": "string"},
                    "type": {"type": "string", "description": "概念类型（可选）"},
                    "description": {"type": "string"},
                    "subject": {"type": "string"},
                    "predicate": {"type": "string"},
                    "object": {"type": "string"},
                    "target": {"type": "string"},
                    "entities": {
                        "type": "array",
                        "description": "手动提供的实体列表（可选），每项含 name 和 type（Method/Dataset/Task/Metric/Contribution/Limitation）",
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string"},
                                "type": {"type": "string", "enum": ["Method", "Dataset", "Task", "Metric", "Contribution", "Limitation", "Concept"]},
                            },
                            "required": ["name"],
                        },
                    },
                },
                "required": ["action"],
            },
        ),
        Tool(
            name="code_execute",
            description="在本地执行 Python 代码。适用于数据分析、统计计算、可视化。预装 numpy、pandas、matplotlib、scipy。",
            inputSchema={
                "type": "object",
                "properties": {
                    "code": {"type": "string", "description": "要执行的 Python 代码"},
                    "timeout": {"type": "integer", "description": "超时秒数，默认 30"},
                },
                "required": ["code"],
            },
        ),
        Tool(
            name="innovation_search",
            description=(
                "创新点研究的文献搜索工具。支持两种模式：\n"
                "- landscape: 对研究主题进行多源文献扫描（Semantic Scholar + arXiv），返回按相关性排序的论文列表，用于了解研究全貌\n"
                "- novelty_search: 针对特定创新点候选的关键词，搜索相关论文以评估新颖性"
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["landscape", "novelty_search"],
                        "description": "操作类型：landscape（文献景观扫描）或 novelty_search（新颖性搜索）",
                    },
                    "topic": {"type": "string", "description": "研究主题或问题描述"},
                    "keywords": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "搜索关键词列表（novelty_search 模式使用）",
                    },
                    "max_per_source": {
                        "type": "integer",
                        "description": "每个数据源最大返回数量，默认 8",
                    },
                },
                "required": ["action"],
            },
        ),
        Tool(
            name="tavily_search",
            description=(
                "通用网页搜索工具，使用 Tavily API 搜索互联网内容。"
                "适用于搜索最新资讯、技术文档、博客文章、官方文档等非学术内容。"
                "支持基础搜索和高级搜索深度，可按主题过滤（通用、新闻、金融）。"
                "返回搜索结果包含 AI 生成的摘要答案。"
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "搜索查询关键词或自然语言描述",
                    },
                    "search_depth": {
                        "type": "string",
                        "enum": ["basic", "advanced"],
                        "description": "搜索深度：basic（快速搜索）或 advanced（深入搜索，更全面但较慢），默认 basic",
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "最大返回结果数量，默认 5，最大 20",
                    },
                    "include_answer": {
                        "type": "boolean",
                        "description": "是否包含 AI 生成的答案摘要，默认 true",
                    },
                    "topic": {
                        "type": "string",
                        "enum": ["general", "news", "finance"],
                        "description": "搜索主题类别：general（通用）、news（新闻）、finance（金融），默认 general",
                    },
                },
                "required": ["query"],
            },
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    logger.info("Tool call: %s(%s)", name, arguments)

    # Extract user_id context injected by the web backend
    user_id = arguments.pop("_user_id", None)

    if name == "echo":
        return [TextContent(type="text", text=arguments.get("message", ""))]

    if name == "paper_search":
        from tools.paper_search import handle_paper_search
        result = await handle_paper_search(arguments, user_id=user_id)
        return [TextContent(type="text", text=result)]

    if name == "paper_parse":
        from tools.paper_parse import handle_paper_parse
        result = await handle_paper_parse(arguments, user_id=user_id)
        return [TextContent(type="text", text=result)]

    if name == "rag_query":
        from tools.rag_query import handle_rag_query
        result = await handle_rag_query(arguments, user_id=user_id)
        return [TextContent(type="text", text=result)]

    if name == "knowledge_graph":
        from tools.knowledge_graph import handle_knowledge_graph
        result = await handle_knowledge_graph(arguments, user_id=user_id)
        return [TextContent(type="text", text=result)]

    if name == "code_execute":
        from tools.code_execute import handle_code_execute
        result = await handle_code_execute(arguments, user_id=user_id)
        return [TextContent(type="text", text=result)]

    if name == "innovation_search":
        from tools.innovation_search import handle_innovation_search
        result = await handle_innovation_search(arguments, user_id=user_id)
        return [TextContent(type="text", text=result)]

    if name == "tavily_search":
        from tools.tavily_search import handle_tavily_search
        result = await handle_tavily_search(arguments, user_id=user_id)
        return [TextContent(type="text", text=result)]

    return [TextContent(type="text", text=f"Unknown tool: {name}")]


# ── 入口 ──────────────────────────────────────────────────────────────────

async def main():
    logger.info("Research Agent MCP Server starting (data_dir=%s)", DATA_DIR)
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream, write_stream,
            InitializationOptions(
                server_name="research",
                server_version="0.1.0",
                capabilities=server.get_capabilities(
                    NotificationOptions(),
                    {},
                ),
            ),
        )


if __name__ == "__main__":
    asyncio.run(main())
