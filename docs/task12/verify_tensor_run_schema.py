"""Validate actual consumer envelope and root record; Go checks cross-field admission."""
import copy
import json
from pathlib import Path
from jsonschema import Draft202012Validator
from referencing import Registry, Resource

root = Path(__file__).resolve().parent
schema = json.loads((root / 'schemas/tensor-run.schema.json').read_text())
evidence = json.loads((root / 'schemas/evidence.schema.json').read_text())
registry = Registry().with_resource(evidence['$id'], Resource.from_contents(evidence))
Draft202012Validator.check_schema(schema)
validator = Draft202012Validator(schema, registry=registry)
fixture = json.loads((root / 'results/tensor-recorded-run.json').read_text())
validator.validate(fixture)
assert fixture['record'] == json.loads((root / 'results/tensor-record.json').read_text())
assert fixture['candidate_budget'] == 100
assert len(fixture['candidate_ids']) == len(fixture['candidate_document_ids']) == 3
record = fixture['record']
for key, record_key in [('configuration', 'recipe'), ('scope', 'scope'), ('publication', 'publication')]:
    assert record[record_key] == {'state': 'observed', 'value': fixture[key]}
assert fixture['candidate_document_ids'] == [hit['id']['value'] for hit in record['stages'][0]['hits']]
assert fixture['candidate_ids'] == [hit['id']['value'] for hit in record['stages'][1]['hits']]
assert [stage['name']['value'] for stage in record['stages']] == ['dense-candidates', 'tensor-candidate-observations', 'maxsim']
assert all(stage['status'] == 'observed' and stage['hits_state'] == 'observed' for stage in record['stages'])
by_id = {hit['id']['value']: hit for hit in record['stages'][1]['hits']}
final_ids = [hit['id']['value'] for hit in record['stages'][2]['hits']]
assert len(set(final_ids)) == len(final_ids)
assert all(by_id[hit['id']['value']] == hit for hit in record['stages'][2]['hits'])
assert [hit['score']['value'] for hit in record['stages'][2]['hits']] == [2, 1, -1]
invalid = []
for key in schema['required']:
    changed = copy.deepcopy(fixture)
    del changed[key]
    invalid.append(changed)
for key, value in [('candidate_budget', 0), ('candidate_ids', []), ('candidate_document_ids', []),
                   ('configuration', 'unqualified'), ('scope', ''), ('publication', ''),
                   ('unknown', True)]:
    changed = copy.deepcopy(fixture)
    changed[key] = value
    invalid.append(changed)
changed = copy.deepcopy(fixture)
changed['candidate_ids'][1] = changed['candidate_ids'][0]
invalid.append(changed)
changed = copy.deepcopy(fixture)
changed['record']['unknown'] = True
invalid.append(changed)
for item in invalid:
    assert not validator.is_valid(item), 'invalid consumer artifact accepted'
print(f'tensor run: 1 actual positive, {len(invalid)} negatives; cross-field association PASS')
