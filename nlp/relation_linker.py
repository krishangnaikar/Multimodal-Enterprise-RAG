
def link_cross_modal(entities_by_doc):
    name_to_canon = {}
    id_map = {}
    for doc in entities_by_doc:
        for e in doc.get("entities", []):
            key = e.get("name", "").strip().lower()
            if not key:
                continue
            if key not in name_to_canon:
                name_to_canon[key] = e["id"]
            id_map[e["id"]] = name_to_canon[key]
    return id_map
