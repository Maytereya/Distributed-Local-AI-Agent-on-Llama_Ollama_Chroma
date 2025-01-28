# Do not work properly (

from graphviz import Digraph
from agent_main import Agent  # Импортируем агента

agent = Agent()


def visualize_graph(agent_in):
    dot = Digraph()
    for node_name, node in agent_in.graph.nodes.items():
        dot.node(node_name)
        for neighbor in node.successors:
            dot.edge(node_name, neighbor)
    dot.render("agent_graph", format="png", view=True)  # Откроет граф и сохранит его как PNG


visualize_graph(agent)
