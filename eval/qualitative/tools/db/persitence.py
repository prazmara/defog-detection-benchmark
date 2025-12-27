# persistence.py
from __future__ import annotations
import csv, json, os, tempfile, shutil
from typing import Dict, Iterable, List, Tuple

DEFAULT_FIELDS = [
    "model", "split", "city", "basename",
    "foggy_path", "cand_path", "gt_path",
    "clarity", "fidelity", "artifacts", "structure_consistency", "realism", "total",
    "explanation"
]

def ensure_csv(path: str, fields: List[str] = None) -> None:
    """Create CSV with header if it does not exist."""
    fields = fields or DEFAULT_FIELDS
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=fields).writeheader()

def load_existing_keys(path: str, key_fields: Iterable[str]) -> set:
    """Build a set of composite keys to avoid duplicate appends."""
    keys = set()
    if not os.path.exists(path):
        return keys
    with open(path, "r", newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            keys.add(tuple(row.get(k, "") for k in key_fields))
    return keys

def _atomic_append(path: str, rows: List[Dict], fields: List[str]) -> None:
    """Append rows atomically to tolerate crashes."""
    dirname = os.path.dirname(path) or "."
    os.makedirs(dirname, exist_ok=True)
    tmp_fd, tmp_path = tempfile.mkstemp(prefix=".tmp_", dir=dirname)
    os.close(tmp_fd)
    try:
        # Copy existing (or create header if missing)
        if os.path.exists(path):
            shutil.copyfile(path, tmp_path)
        else:
            with open(tmp_path, "w", newline="", encoding="utf-8") as f:
                csv.DictWriter(f, fieldnames=fields).writeheader()
        # Append
        with open(tmp_path, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            for r in rows:
                w.writerow({k: r.get(k, "") for k in fields})
        # Replace atomically
        os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            try: os.remove(tmp_path)
            except OSError: pass


def load_existing_pairs(csv_path):
    """Return a set of (basename, model_name) pairs already in CSV."""
    if not os.path.exists(csv_path):
        return set()
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        return {
            (r["basename"], r["model_name"])
            for r in reader
            if "basename" in r and "model_name" in r
        }


def check_existing(
    csv_path: str, 
    model: str, 
    city: str, 
    basename: str
) -> bool:
    """
    Check if a (model, city, basename) composite key exists in the CSV.
    Returns True if found, False otherwise.
    """
    existing = load_existing_keys(csv_path, ["model", "city", "basename"])
    key = (model, city, basename)
    return key in existing


def append_unique(
    csv_path: str,
    row: Dict,
    key_fields: Iterable[str] = ("model","split","city","basename"),
    fields: List[str] = None,
    jsonl_path: str | None = None
) -> bool:
    """
    Append a single row if composite key not present.
    Returns True if written, False if skipped (duplicate).
    """
    fields = fields or DEFAULT_FIELDS
    ensure_csv(csv_path, fields)
    existing = load_existing_keys(csv_path, key_fields)
    key = tuple(row.get(k, "") for k in key_fields)
    if key in existing:
        return False
    _atomic_append(csv_path, [row], fields)
    if jsonl_path:
        os.makedirs(os.path.dirname(jsonl_path) or ".", exist_ok=True)
        with open(jsonl_path, "a", encoding="utf-8") as jf:
            jf.write(json.dumps(row, ensure_ascii=False) + "\n")
    return True
