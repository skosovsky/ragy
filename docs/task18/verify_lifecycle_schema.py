"""Validate authored v2 wire examples; Go checks semantic reference matching.

The JSON files are contract examples copied and extended from task12, not
generated serializer output. Derived examples below exercise structural rules.
"""
import copy
import hashlib
import json
import re
from datetime import datetime
from pathlib import Path

from jsonschema import Draft202012Validator, FormatChecker

root = Path(__file__).resolve().parent
schema = json.loads((root / 'schemas/lifecycle.schema.json').read_text())
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
positive = {}
for name in ['snapshot', 'cleanup', 'inventory_receipt']:
    positive[name] = json.loads((root / f'fixtures/lifecycle_{name}.json').read_text())
fixture = positive['snapshot']

partial = copy.deepcopy(fixture)
partial['manifests'][0]['partial'] = True
partial['manifests'][0]['targets'].append({
    'name': 'tensor', 'required': True, 'state': 'failed',
    'revision': '', 'artifacts': [],
})
positive['partial'] = partial

pin = {
    'id': 'reader-1', 'publication': 'publication-1',
    'requested_targets': ['lexical'],
    'targets': [{
        'target': 'lexical', 'namespace': 'n', 'source': 'policy',
        'revision': 'r1', 'transformation': 'chunk', 'access_fingerprint': 'acl',
    }],
    'released': False,
}
live = copy.deepcopy(fixture)
live['pins'] = [pin]
positive['live_pin'] = live
released = copy.deepcopy(live)
released['pins'][0]['released'] = True
positive['released_pin'] = released
empty_pin = copy.deepcopy(live)
empty_pin['pins'][0]['targets'] = None
positive['complete_empty_pin'] = empty_pin

retired = copy.deepcopy(positive['cleanup'])
retired['cleanups'][0]['complete'] = True
retired['cleanups'][0]['items'][0]['state'] = 'complete'
old = retired['manifests'][0]
reference = old['targets'][0]['artifacts'][0]['reference']
digest = hashlib.sha256(json.dumps(reference, separators=(',', ':')).encode()).hexdigest()
old['retired'] = True
old['artifact_fences'] = [{'target': 'lexical', 'digest': digest}]
old['targets'][0]['artifacts'] = None
retired['pins'] = [copy.deepcopy(released['pins'][0])]
positive['retired_skeleton_released_pin'] = retired

for name, example in positive.items():
    try:
        validator.validate(example)
    except Exception as error:
        raise AssertionError(f'Positive fixture rejected: {name}') from error

negative = {}
for key, value in [('schema', 'unknown'), ('schema', 'ragy.lifecycle'),
                   ('generation', -1), ('generation', 2**64), ('namespace', '')]:
    item = copy.deepcopy(fixture)
    item[key] = value
    negative[f'{key}={value}'] = item
for field, value in [('state', 'unknown'), ('state', 'surprise'),
                     ('checkpoint', 'staging'), ('targets', None), ('targets', [])]:
    item = copy.deepcopy(fixture)
    item['manifests'][0][field] = value
    negative[f'manifest.{field}={value}'] = item
item = copy.deepcopy(fixture)
item['manifests'][0]['targets'][0]['state'] = 'failed'
negative['required_target_failed'] = item
item = copy.deepcopy(fixture)
item['manifests'][0]['targets'][0]['artifacts'][0]['supports'] = []
negative['empty_supports'] = item
item = copy.deepcopy(fixture)
item['private_payload'] = 'not a contract field'
negative['unknown_snapshot_field'] = item
item = copy.deepcopy(positive['cleanup'])
item['cleanups'][0]['items'][0]['next_at'] = 'invalid-date'
negative['invalid_cleanup_timestamp'] = item

for value in ['a' * 63, 'a' * 65, 'A' * 64, 'g' * 64, '', 123]:
    item = copy.deepcopy(retired)
    item['manifests'][0]['artifact_fences'][0]['digest'] = value
    negative[f'bad_fence_digest={value}'] = item
item = copy.deepcopy(retired)
item['manifests'][0]['artifact_fences'] *= 2
negative['duplicate_fence'] = item
item = copy.deepcopy(fixture)
item['manifests'][0]['artifact_fences'] = copy.deepcopy(old['artifact_fences'])
negative['live_manifest_with_fences'] = item
item = copy.deepcopy(retired)
item['manifests'][0]['targets'][0]['artifacts'] = copy.deepcopy(fixture['manifests'][0]['targets'][0]['artifacts'])
negative['retired_with_artifacts'] = item
item = copy.deepcopy(retired)
item['manifests'][0]['state'] = 'unknown'
item['manifests'][0]['checkpoint'] = 'published'
negative['retired_unknown_state'] = item
item = copy.deepcopy(retired)
item['manifests'][0]['partial'] = True
item['manifests'][0]['targets'][0]['state'] = 'unknown'
negative['retired_unknown_target_state'] = item

for field in ['retired', 'artifact_fences']:
    item = copy.deepcopy(fixture)
    del item['manifests'][0][field]
    negative[f'missing_manifest_{field}'] = item
item = copy.deepcopy(fixture)
del item['pins']
negative['missing_snapshot_pins'] = item
for field in ['id', 'publication', 'requested_targets', 'targets', 'released']:
    item = copy.deepcopy(live)
    del item['pins'][0][field]
    negative[f'missing_pin_{field}'] = item
for value in [None, [], ['lexical', 'lexical'], ['']]:
    item = copy.deepcopy(live)
    item['pins'][0]['requested_targets'] = value
    negative[f'invalid_requested_targets={value}'] = item
item = copy.deepcopy(live)
item['pins'][0]['publication'] = 'current'
negative['pin_current_profile'] = item
for name, path in [
    ('manifest', ['manifests', 0]),
    ('pin', ['pins', 0]),
    ('target_revision', ['pins', 0, 'targets', 0]),
    ('fence', ['manifests', 0, 'artifact_fences', 0]),
]:
    item = copy.deepcopy(retired)
    row = item
    for key in path:
        row = row[key]
    row['private_payload'] = True
    negative[f'unknown_{name}_field'] = item

for name, example in negative.items():
    if validator.is_valid(example):
        raise AssertionError(f'Invalid lifecycle wire example accepted: {name}')
print(f'lifecycle v2 schema: {len(positive)} positive and {len(negative)} negative examples passed')
