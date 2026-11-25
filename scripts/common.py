"""Shared helpers for decision trees, prompts, and data files."""
import json
import re
from collections import Counter, deque

PROMPT_INTRO = ("The following JSON is the decision tree of a clinical practice "
                "guideline flowchart.\n\n")
CASE_HEAD = "A patient presents with the following case:\n\n"
CASE_TAIL = (
    "\n\nUsing the decision tree you extracted above, follow it step by step for "
    "this patient. Do NOT skip decision points.\n\n"
    "Respond with ONLY a JSON object:\n"
    "{\n"
    '  "traversal": [\n'
    '    {"decision_point":"the question at this node", '
    '"patient_data":"relevant patient information", '
    '"answer":"which branch you follow", "next_step":"where you go next"}\n'
    "  ],\n"
    '  "recommendation":"The final action/recommendation you reached",\n'
    '  "confidence":"high/medium/low"\n'
    "}"
)

# Asks whether a vignette selects the intended branch at one decision.
CHECK_PROMPT = """One criterion from a clinical decision flowchart, its answer options, and a patient case.

Criterion: {criterion}
Options: {options}

Case:
{vignette}

Based only on the case, which single option is correct? Reply with exactly one option, copied verbatim from the list. Reply with the option text only."""


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(rows, path):
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def pair_key(meta):
    return "|".join(str(meta[k]) for k in ("pmcid", "fig_id", "node", "seed"))


def first_json_object(text):
    """Return the first complete JSON object in a string, or None."""
    if not text:
        return None
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    for j in range(start, len(text)):
        if text[j] == "{":
            depth += 1
        elif text[j] == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start:j + 1])
                except json.JSONDecodeError:
                    return None
    return None


def tree_from_prompt(text):
    return first_json_object(text)


def vignette_from_prompt(text):
    start = text.find(CASE_HEAD) + len(CASE_HEAD)
    end = text.find("\n\nUsing the decision tree", start)
    return text[start:end].strip()


def make_example(tree, vignette, steps, recommendation, meta):
    """One fine-tuning or evaluation row in LLaMA-Factory sharegpt format."""
    user = PROMPT_INTRO + json.dumps(tree) + "\n\n" + CASE_HEAD + "\n" + vignette + CASE_TAIL
    target = json.dumps({"traversal": steps, "recommendation": recommendation,
                         "confidence": "high"})
    return {"messages": [{"role": "user", "content": user},
                         {"role": "assistant", "content": target}],
            "images": [], "meta": meta}


# Tree structure

def edges_of(node):
    return [e for e in node.get("edges", []) if e.get("target")]


def is_decision(node):
    return len([e for e in edges_of(node) if e.get("label")]) >= 2


def build_graph(tree):
    nodes = {n.get("id"): n for n in tree.get("nodes", []) if n.get("id")}
    root = tree.get("root") or tree.get("nodes", [{}])[0].get("id")
    return nodes, root


def find_root(tree):
    """Declared root if present, otherwise the first node with no incoming edge."""
    nodes = {n["id"]: n for n in tree["nodes"]}
    incoming = Counter(e["target"] for n in tree["nodes"] for e in n.get("edges", []))
    roots = [n["id"] for n in tree["nodes"] if incoming[n["id"]] == 0]
    if tree.get("root") in nodes:
        return tree["root"]
    return roots[0] if roots else tree["nodes"][0]["id"]


def path_to(nodes, root, target_id):
    """Shortest list of node ids from the root to a target node."""
    queue = deque([[root]])
    seen = {root}
    while queue:
        path = queue.popleft()
        if path[-1] == target_id:
            return path
        node = nodes.get(path[-1])
        if not node:
            continue
        for e in edges_of(node):
            t = e["target"]
            if t not in seen and t in nodes:
                seen.add(t)
                queue.append(path + [t])
    return None


def path_to_leaf(nodes, start_id):
    """Path from a node to a leaf with the fewest further decisions, then fewest nodes."""
    best = None
    queue = deque([([start_id], 0)])
    seen = {}
    while queue:
        path, cost = queue.popleft()
        node = nodes.get(path[-1])
        if node is None:
            continue
        out = edges_of(node)
        if not out:
            rank = (cost, len(path))
            if best is None or rank < best[0]:
                best = (rank, path)
            continue
        if len(path) > 14:
            continue
        for e in out:
            t = e["target"]
            nxt = nodes.get(t)
            c = cost + (1 if nxt and is_decision(nxt) else 0)
            if t in seen and seen[t] <= c:
                continue
            seen[t] = c
            queue.append((path + [t], c))
    return best[1] if best else None


def steps_for(nodes, id_path):
    """Target traversal for a path of node ids.

    Returns (steps, decisions, recommendation), where decisions lists
    (criterion, options, answer) for each decision node on the path.
    """
    steps, decisions = [], []
    for a, b in zip(id_path, id_path[1:]):
        node = nodes[a]
        out = edges_of(node)
        edge = next((e for e in out if e["target"] == b), None)
        if edge is None:
            return None
        if is_decision(node):
            label = edge.get("label", "")
            options = [e.get("label", "") for e in out if e.get("label")]
            decisions.append((node.get("text", ""), options, label))
        else:
            label = "next"
        steps.append({"decision_point": node.get("text", ""), "patient_data": "",
                      "answer": label, "next_step": nodes[b].get("text", "")})
    return steps, decisions, nodes[id_path[-1]].get("text", "")


def walk(tree, traversal):
    """Follow a target traversal through the tree by its next_step fields.

    Yields (node, step) for each step. Ends with ("end", last_node), or with
    ("fail", step) when a step does not lead to a child of the current node.
    """
    nodes = {n["id"]: n for n in tree["nodes"]}
    cur = find_root(tree)
    for step in traversal:
        node = nodes[cur]
        yield node, step
        nxt = [e for e in node.get("edges", []) if nodes[e["target"]]["text"] == step.get("next_step")]
        if not nxt:
            yield "fail", step
            return
        cur = nxt[0]["target"]
    yield "end", nodes[cur]


# Answer matching for the verification step

def norm(s):
    s = (s or "").lower()
    s = s.replace("≥", ">=").replace("≤", "<=").replace("⩾", ">=").replace("⩽", "<=")
    s = re.sub(r"[^a-z0-9<>= ]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def strict_match(reply, options, wanted):
    """True if the reply names the wanted option and no other option."""
    r, w = norm(reply), norm(wanted)
    opts = [norm(o) for o in options]
    if not r or not w:
        return False
    if r == w:
        return True
    if r in opts:
        return False
    contained = [o for o in opts if o and o in r]
    contained = [o for o in contained if not any(o != p and o in p for p in contained)]
    return len(contained) == 1 and contained[0] == w
