"""novare/graph/builder.py — 构建主图

图结构：

    START → bootstrap → call_model
    call_model →(route_after_model)→ execute_tools | verify_answer | finalize
    execute_tools →(route_after_tools)→ call_model | max_iterations | finalize
    verify_answer → finalize
    max_iterations → finalize
    finalize → END

依赖注入方式：每个 run_turn 用闭包捕获 RunContext 构图（构图成本为
内存操作，微秒级）。MVP 不接入 checkpointer，因此不需要复用编译产物；
阶段 2 引入 checkpoint 时再切换为"静态图 + configurable"。
"""

from __future__ import annotations

import logging

from langgraph.graph import END, START, StateGraph

from novare.graph.context import RunContext
from novare.graph.nodes.finalize import (
    bootstrap,
    finalize,
    route_after_model,
    verify_answer,
)
from novare.graph.nodes.model import call_model
from novare.graph.nodes.tools import execute_tools, route_after_tools
from novare.graph.state import GraphState

logger = logging.getLogger("novare.graph.builder")


async def _max_iterations_node(state: GraphState, ctx: RunContext) -> dict:
    """达到最大迭代：标记状态，具体提示在 finalize 输出。"""
    return {"run_status": "max_iterations"}


def build_graph(ctx: RunContext):
    """按 RunContext 构建并编译主图。"""

    # LangGraph 通过 iscoroutinefunction 判断节点类型，必须注册
    # async 闭包函数（lambda 返回 coroutine 会被当作同步返回值）
    async def bootstrap_node(state: GraphState) -> dict:
        return await bootstrap(state, ctx)

    async def call_model_node(state: GraphState) -> dict:
        return await call_model(state, ctx)

    async def execute_tools_node(state: GraphState) -> dict:
        return await execute_tools(state, ctx)

    async def verify_node(state: GraphState) -> dict:
        return await verify_answer(state, ctx)

    async def max_iter_node(state: GraphState) -> dict:
        return {"run_status": "max_iterations"}

    async def finalize_node(state: GraphState) -> dict:
        return await finalize(state, ctx)

    g = StateGraph(GraphState)
    g.add_node("bootstrap", bootstrap_node)
    g.add_node("call_model", call_model_node)
    g.add_node("execute_tools", execute_tools_node)
    g.add_node("verify_answer", verify_node)
    g.add_node("max_iterations", max_iter_node)
    g.add_node("finalize", finalize_node)

    g.add_edge(START, "bootstrap")
    g.add_edge("bootstrap", "call_model")

    g.add_conditional_edges(
        "call_model",
        lambda state: route_after_model(state, ctx),
        {
            "execute_tools": "execute_tools",
            "verify_answer": "verify_answer",
            "finalize": "finalize",
        },
    )
    g.add_conditional_edges(
        "execute_tools",
        lambda state: route_after_tools(state, ctx),
        {
            "call_model": "call_model",
            "max_iterations": "max_iterations",
            "finalize": "finalize",
        },
    )
    g.add_edge("verify_answer", "finalize")
    g.add_edge("max_iterations", "finalize")
    g.add_edge("finalize", END)

    return g.compile()
