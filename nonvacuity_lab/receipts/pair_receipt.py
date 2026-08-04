from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

from ..canonical import canonical_json_hash, file_hash, tree_hash


@dataclass(frozen=True)
class PairReceipt:
    schema_version: str
    producer_mutation_id: str
    seed: int
    tree_variant: str
    leaf_id: int
    action_id: int
    base_tree_sha256: str
    mutated_tree_sha256: str
    mutated_tree_file_sha256: str
    activation_witness_sha256: str
    mutated_seed_snapshot_sha256: str

    def to_json(self) -> dict:
        return asdict(self)


class PairContractError(ValueError):
    pass


def write_pair_receipt(path: Path, receipt: PairReceipt) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(receipt.to_json(), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def consume_pair_receipt(
    receipt_path: Path,
    *,
    expected_producer: str,
    seed: int,
    variant: str,
    copied_tree_path: Path,
    copied_seed_dir: Path,
) -> dict:
    try:
        data = json.loads(receipt_path.read_text(encoding="utf-8"))
        if data["producer_mutation_id"] != expected_producer:
            raise PairContractError("wrong producer mutation")
        if int(data["seed"]) != int(seed) or data["tree_variant"] != variant:
            raise PairContractError("seed/variant mismatch")
        if file_hash(copied_tree_path) != data["mutated_tree_file_sha256"]:
            raise PairContractError("mutated tree does not match pair receipt")
        tree_data = json.loads(copied_tree_path.read_text(encoding="utf-8"))
        if canonical_json_hash(tree_data) != data["mutated_tree_sha256"]:
            raise PairContractError("mutated tree semantic hash does not match pair receipt")
        if tree_hash(copied_seed_dir) != data["mutated_seed_snapshot_sha256"]:
            raise PairContractError("mutated seed snapshot does not match pair receipt")
        witness_path = receipt_path.parent / "activation" / "activation_result.json"
        if not witness_path.is_file():
            raise PairContractError("producer activation witness is missing")
        if file_hash(witness_path) != data["activation_witness_sha256"]:
            raise PairContractError("producer activation witness does not match pair receipt")
    except (OSError, KeyError, TypeError, ValueError) as exc:
        if isinstance(exc, PairContractError):
            raise
        raise PairContractError(str(exc)) from exc
    return data
