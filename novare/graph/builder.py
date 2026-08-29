"""novare/graph/builder.py — 构建主图

图结构：

    START → bootstrap → call_model
    call_model →(route_after_model)→ execute_tools | verify_answer | finalize
    execute_tools →(route_after_tools)→ call_model | max_iterations | finalize
    verify_answer → finalize
    max_iterations → finalize
    finalize → END

阶段 2 改造：图为静态编译产物（GraphRunner 构造时 build 一次并挂载
checkpointer），per-turn 依赖（RunContext）经 config["configurable"]["run_ctx"]
注入节点；条件边只读 state（iteration_limit / verify_enabled 在 turn 开始时
写入 state），保证 checkpoint 恢复后的路由决策与首次执行一致。

checkpointer 只序列化 GraphState（全 dict/int/str/bool），RunContext 中的
session、model port、回调等不可序列化对象不进入 checkpoint。
"""

from __future__ import annotations

import logging
from typing import Any

from langgraph.graph import END, START, StateGraph

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


def _ctx(config) -> Any:
    """从 RunnableConfig 提取 turn-scoped RunContext（不可序列化依赖）。"""
    return config["configurable"]["run_ctx"]


async def bootstrap_node(state: GraphState, config) -> dict:
    return await bootstrap(state, _ctx(config))


async def call_model_node(state: GraphState, config) -> dict:
    return await call_model(state, _ctx(config))


async def execute_tools_node(state: GraphState, config) -> dict:
    return await execute_tools(state, _ctx(config))


async def verify_node(state: GraphState, config) -> dict:
    return await verify_answer(state, _ctx(config))


async def max_iterations_node(state: GraphState) -> dict:
    """达到最大迭代：标记状态，具体提示在 finalize 输出。"""
    return {"run_status": "max_iterations"}


async def finalize_node(state: GraphState, config) -> dict:
    return await finalize(state, _ctx(config))


def build_graph(checkpointer=None):
    """构建并编译主图。checkpointer 由 GraphRunner 注入（见 checkpointer.py）。"""
    g = StateGraph(GraphState)

    # LangGraph 通过 iscoroutinefunction 判断节点类型；第二参数 config
    # 由框架按签名自动传入
    g.add_node("bootstrap", bootstrap_node)
    g.add_node("call_model", call_model_node)
    g.add_node("execute_tools", execute_tools_node)
    g.add_node("verify_answer", verify_node)
    g.add_node("max_iterations", max_iterations_node)
    g.add_node("finalize", finalize_node)

    g.add_edge(START, "bootstrap")
    g.add_edge("bootstrap", "call_model")

    g.add_conditional_edges(
        "call_model",
        route_after_model,
        {
            "execute_tools": "execute_tools",
            "verify_answer": "verify_answer",
            "finalize": "finalize",
        },
    )
    g.add_conditional_edges(
        "execute_tools",
        route_after_tools,
        {
            "call_model": "call_model",
            "max_iterations": "max_iterations",
            "finalize": "finalize",
        },
    )
    g.add_edge("verify_answer", "finalize")
    g.add_edge("max_iterations", "finalize")
    g.add_edge("finalize", END)

    return g.compile(checkpointer=checkpointer) if checkpointer else g.compile()
