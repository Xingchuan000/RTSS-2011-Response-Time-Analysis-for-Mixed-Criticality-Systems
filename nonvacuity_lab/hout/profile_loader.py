from __future__ import annotations

import json
from pathlib import Path

from .schema import HoutProfile


def load_hout_profiles(path: Path) -> dict[str, HoutProfile]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    profiles = raw.get("hout_profiles", raw)
    if not isinstance(profiles, dict):
        raise ValueError("hout_profiles must be an object")
    result = {}
    for profile_id, value in profiles.items():
        data = dict(value)
        data.setdefault("profile_id", profile_id)
        result[str(profile_id)] = HoutProfile.from_mapping(data)
    return result
