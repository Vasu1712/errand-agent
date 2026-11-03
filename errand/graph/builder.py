"""Assembles the workflow graph and its SQLite checkpointer."""

import sqlite3
from pathlib import Path

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from errand.graph import nodes
from errand.graph.deps import Deps
from errand.graph.edges import ROUTES, router
from errand.graph.state import AgentState

NODES = {name: getattr(nodes, name) for name in [*ROUTES, "finish"]}


def open_checkpointer(db_path: Path) -> SqliteSaver:
    """A checkpointer on the agent DB. The caller closes ``saver.conn`` when done."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    # SqliteSaver serializes access to the connection with its own lock.
    saver = SqliteSaver(sqlite3.connect(db_path, check_same_thread=False))
    saver.setup()
    return saver


def build_graph(checkpointer: BaseCheckpointSaver | None = None) -> CompiledStateGraph:
    graph = StateGraph(AgentState, context_schema=Deps)
    for name, run in NODES.items():
        graph.add_node(name, run)
    graph.add_edge(START, "load_context")
    for source, targets in ROUTES.items():
        destinations = {target: target for target in (*targets, "finish")}
        graph.add_conditional_edges(source, router(source), destinations)
    graph.add_edge("finish", END)
    return graph.compile(checkpointer=checkpointer)
