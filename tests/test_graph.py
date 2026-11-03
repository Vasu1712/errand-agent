from errand.graph import build_graph
from errand.graph.edges import ROUTES
from errand.graph.nodes import load_context


def test_the_compiled_graph_has_exactly_the_designed_edges():
    graph = build_graph().get_graph()
    edges = {(edge.source, edge.target) for edge in graph.edges}
    designed = {
        (source, target) for source, targets in ROUTES.items() for target in (*targets, "finish")
    }
    designed |= {("__start__", "load_context"), ("finish", "__end__")}
    assert edges == designed


def test_side_effects_are_only_reachable_through_a_clear_yes():
    def into(node):
        return {source for source, targets in ROUTES.items() if node in targets}

    assert into("hold_slot") == {"approval_check"}  # nothing is held before a yes
    assert into("execute") == {"approval_check", "hold_slot"}
    assert into("approval_check") == {"await_approval", "route_intent"}
    assert ROUTES["build_summary"] == ("await_approval",)


def test_nodes_take_langgraphs_runtime_argument():
    # Regression for docs/bugs.md #2: a wrapper that copied __wrapped__ hid this parameter.
    import inspect

    assert list(inspect.signature(load_context).parameters) == ["state", "runtime"]
