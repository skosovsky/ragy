"""Independent wire-shape validation; semantic references are checked in Go."""
import copy
import json
import re
from datetime import datetime
from pathlib import Path
from jsonschema import Draft202012Validator, FormatChecker

root = Path(__file__).resolve().parent
schema = json.loads((root / 'schemas/lifecycle.schema.json').read_text())
fixture = json.loads((root / 'fixtures/lifecycle_snapshot.json').read_text())
Draft202012Validator.check_schema(schema)
checker = FormatChecker()

@checker.checks('date-time')
def valid_timestamp(value):
    if not isinstance(value, str):
        return True
    if not re.fullmatch(r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})', value):
        return False
    try:
        datetime.fromisoformat(value.replace('Z', '+00:00'))
        return True
    except ValueError:
        return False

validator = Draft202012Validator(schema, format_checker=checker)
validator.validate(fixture)
invalid = []
for key, value in [('schema', 'unknown'), ('generation', -1), ('generation', 2**64), ('namespace', '')]:
    item = copy.deepcopy(fixture)
    item[key] = value
    invalid.append(item)
for field, value in [('state', 'unknown'), ('state', 'surprise'), ('checkpoint', 'staging'), ('targets', None), ('targets', [])]:
    item = copy.deepcopy(fixture)
    item['manifests'][0][field] = value
    invalid.append(item)
item = copy.deepcopy(fixture)
item['manifests'][0]['targets'][0]['state'] = 'failed'
invalid.append(item)
item = copy.deepcopy(fixture)
item['manifests'][0]['targets'][0]['artifacts'][0]['supports'] = []
invalid.append(item)
item = copy.deepcopy(fixture)
item['private_payload'] = 'not a contract field'
invalid.append(item)
for item in invalid:
    if validator.is_valid(item):
        raise AssertionError('Invalid lifecycle wire snapshot accepted')
partial = copy.deepcopy(fixture)
partial['manifests'][0]['partial'] = True
partial['manifests'][0]['targets'].append({'name':'tensor', 'required':True, 'state':'failed', 'revision':'', 'artifacts':[]})
validator.validate(partial)
cleanup = json.loads((root / 'fixtures/lifecycle_cleanup.json').read_text())
validator.validate(cleanup)
invalid_cleanup = copy.deepcopy(cleanup)
invalid_cleanup['cleanups'][0]['items'][0]['next_at'] = 'invalid-date'
if validator.is_valid(invalid_cleanup):
    raise AssertionError('Invalid cleanup timestamp accepted')
receipt = json.loads((root / 'fixtures/lifecycle_inventory_receipt.json').read_text())
validator.validate(receipt)
print(f'lifecycle schema: 4 positive and {len(invalid)+1} negative fixtures passed')
