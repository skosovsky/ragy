"""Independent JSON Schema check. Requires jsonschema in the host QA environment."""
import copy
import json
from pathlib import Path

from jsonschema import Draft202012Validator

root = Path(__file__).resolve().parent
schema = json.loads((root / "schemas/read_coverage.schema.json").read_text())
fixture = json.loads((root / "fixtures/read_coverage.json").read_text())
Draft202012Validator.check_schema(schema)
validator = Draft202012Validator(schema)
validator.validate(fixture)

for state in ("unobserved", "complete", "unrestricted"):
    for skipped in (None, []):
        valid = dict(fixture, state=state, skipped_branches=skipped)
        validator.validate(valid)

invalid_reports = [
    dict(fixture, schema="incompatible"),
    dict(fixture, state="unsupported"),
    dict(fixture, state=""),
    dict(fixture, skipped_branches=[]),
    dict(fixture, skipped_branches=None),
    dict(fixture, skipped_branches=["a", "a"]),
    dict(fixture, skipped_branches=["private\nidentifier"]),
    dict(fixture, state="complete"),
    dict(fixture, private_ids=["secret"]),
]
for key in ("schema", "state", "skipped_branches"):
    invalid = copy.deepcopy(fixture)
    del invalid[key]
    invalid_reports.append(invalid)
for invalid in invalid_reports:
    if validator.is_valid(invalid):
        raise AssertionError(f"Invalid admission coverage accepted: {invalid}")

print(f"coverage schema: 7 valid reports and {len(invalid_reports)} negative reports passed")
