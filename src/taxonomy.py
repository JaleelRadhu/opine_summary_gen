"""Parse taxonomy_tree.txt and provide node lookup utilities."""
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

TAXONOMY_FILE = Path(__file__).parent.parent / "taxonomy_tree.txt"

_NEG_PHRASES = [" not ", "nothing specifically", "but not", "in general not"]


def _is_negative_sibling(name: str) -> bool:
    n = name.lower()
    return any(p in n for p in _NEG_PHRASES)


def load_taxonomy(filepath: Path = TAXONOMY_FILE) -> Dict[tuple, dict]:
    """
    Parse the taxonomy tree file and return a dict keyed by path tuples.

    Each value has keys:
      name, level, path (list), children (list of path tuples),
      parent (path tuple or None), is_leaf, is_negative_sibling_style
    """
    nodes: Dict[tuple, dict] = {}
    stack: List[Tuple[int, tuple]] = []  # (level, path_tuple)

    with open(filepath) as f:
        lines = f.readlines()

    for raw in lines:
        line = raw.rstrip("\n")
        if not line.strip():
            continue

        if "├── " in line or "└── " in line:
            b1 = line.rfind("├── ")
            b2 = line.rfind("└── ")
            branch_pos = max(b1, b2)
            name = line[branch_pos + 4:].strip()
            level = branch_pos // 4 + 1
        else:
            name = line.strip()
            level = 0

        while stack and stack[-1][0] >= level:
            stack.pop()

        parent_path: Optional[tuple] = stack[-1][1] if stack else None
        current_path = (parent_path or ()) + (name,)

        nodes[current_path] = {
            "name": name,
            "level": level,
            "path": list(current_path),
            "children": [],
            "parent": parent_path,
            "is_leaf": True,
            "is_negative_sibling_style": _is_negative_sibling(name),
        }

        if parent_path is not None and parent_path in nodes:
            nodes[parent_path]["children"].append(current_path)
            nodes[parent_path]["is_leaf"] = False

        stack.append((level, current_path))

    return nodes


def path_to_key(path: list) -> tuple:
    return tuple(path)


def get_leaf_nodes(nodes: Dict[tuple, dict]) -> List[dict]:
    return [n for n in nodes.values() if n["is_leaf"]]


def get_processing_order(nodes: Dict[tuple, dict]) -> List[tuple]:
    """Internal node paths sorted deepest-first (bottom-up processing order)."""
    internals = [
        (path, n["level"])
        for path, n in nodes.items()
        if not n["is_leaf"]
    ]
    internals.sort(key=lambda x: -x[1])
    return [p for p, _ in internals]
