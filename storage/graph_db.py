from __future__ import annotations
import uuid
import networkx as nx

class GraphStore:
    def __init__(self):
        self.G = nx.MultiDiGraph()

    def _safe_entity_id(self, raw_id):
        if isinstance(raw_id, str) and raw_id.strip():
            return raw_id.strip()
        return f"e_{uuid.uuid4().hex[:8]}"

    def add_entities_relations(self, entities, relations):
        entities = entities or []
        relations = relations or []

        for ent in entities:
            if not isinstance(ent, dict):
                continue
            eid = self._safe_entity_id(ent.get("id"))
            name = ent.get("name") or eid
            label = ent.get("label") or "Entity"

            attrs = {"name": str(name), "label": str(label)}
            for k, v in ent.items():
                if k == "id" or v is None:
                    continue
                attrs[k] = v
            if self.G.has_node(eid):
                self.G.nodes[eid].update(attrs)
            else:
                self.G.add_node(eid, **attrs)

        for rel in relations:
            if not isinstance(rel, dict):
                continue
            src = rel.get("source")
            tgt = rel.get("target")

            if not isinstance(src, str) or not src.strip():
                continue
            if not isinstance(tgt, str) or not tgt.strip():
                continue
            if not (self.G.has_node(src) and self.G.has_node(tgt)):
                continue

            rtype = rel.get("type") or "related_to"
            evidence = rel.get("evidence") or ""

            eattrs = {"type": rtype, "evidence": evidence}
            for k, v in rel.items():
                if k in ("source", "target") or v is None:
                    continue
                eattrs[k] = v

            self.G.add_edge(src, tgt, **eattrs)

    def search_entities_by_name(self, query, limit = 10):
        if not query:
            return []
        q = query.lower()
        out = []
        for nid, data in self.G.nodes(data=True):
            name = (data.get("name") or "").lower()
            if q in name:
                out.append({"id": nid, "name": data.get("name"), "label": data.get("label")})
                if len(out) >= limit:
                    break
        return out

    def neighbors_of(self, node_id, depth = 1):
        if not self.G.has_node(node_id):
            return []
        seen = {node_id}
        layer = {node_id}
        for _ in range(max(1, int(depth))):
            nxt = set()
            for u in layer:
                nxt.update(self.G.successors(u))
                nxt.update(self.G.predecessors(u))
            layer = nxt - seen
            seen.update(layer)
        return [{"id": n, **self.G.nodes[n]} for n in seen if n != node_id]
