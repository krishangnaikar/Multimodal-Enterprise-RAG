from __future__ import annotations
from pyvis.network import Network
import tempfile, streamlit as st

def render_pyvis_graph(nx_graph, nodes, height = "520px"):
    nodes = list(nodes) if nodes is not None else []
    if not nodes:
        st.info("No graph nodes to render yet. Try a different entity or increase neighbor depth.")
        return

    sub = nx_graph.subgraph(nodes).copy()

    net = Network(height=height, width="100%", directed=True, notebook=False)
    net.barnes_hut(gravity=-30000, central_gravity=0.2, spring_length=130, spring_strength=0.01)

    for nid, data in sub.nodes(data=True):
        label = data.get("name") or str(nid)
        group = data.get("label") or "Entity"
        title = f"{group}: {label}"
        net.add_node(str(nid), label=label, title=title, group=group)

    for u, v, edata in sub.edges(data=True):
        etype = edata.get("type") or "rel"
        evid = edata.get("evidence") or ""
        net.add_edge(str(u), str(v), label=etype, title=evid, arrows="to")

    with tempfile.NamedTemporaryFile("w", delete=False, suffix=".html", encoding="utf-8") as f:
        net.write_html(f.name, open_browser=False, notebook=False)
        html = f.read() if f.closed else open(f.name, "r", encoding="utf-8").read()

    st.components.v1.html(html, height=560, scrolling=True)
