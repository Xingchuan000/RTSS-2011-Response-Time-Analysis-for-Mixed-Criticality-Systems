from __future__ import annotations

import json
from pathlib import Path

from formal_toolchain.core.hashing import sha256_object

ROOT = Path(__file__).resolve().parents[1]
STATEMENTS = ROOT / "formal_toolchain" / "theory" / "statements"
OUTPUT = ROOT / "formal_toolchain" / "theory" / "hashes.json"


def main() -> None:
    result: dict[str, dict[str, object]] = {}
    for path in sorted(STATEMENTS.glob("*.json")):
        row = json.loads(path.read_text(encoding="utf-8"))
        theorem_id = str(row["theorem_id"])
        row["statement_hash"] = sha256_object({
            "theorem_id": theorem_id,
            "exact_statement": row["exact_statement"],
            "conclusion": row["conclusion"],
            "source_reference": row["source_reference"],
            "assurance_level": row["assurance_level"],
            "version": row["version"],
        })
        row["assumption_hash"] = sha256_object({
            "theorem_id": theorem_id,
            "assumptions": row.get("assumptions", []),
            "premise_obligation_ids": row.get("premise_obligation_ids", []),
            "version": row["version"],
        })
        path.write_text(json.dumps(row, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        result[theorem_id] = {
            "statement_hash": row["statement_hash"],
            "assumption_hash": row["assumption_hash"],
        }

    OUTPUT.write_text(
        json.dumps(
            {
                "schema_version": "theory_hashes_v3",
                "statements": result,
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
