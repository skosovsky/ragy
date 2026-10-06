"""Independent JSON Schema validation of the same corpus used by Go tests.
Install jsonschema outside the repository, e.g. pip --target /tmp/task17-jsonschema jsonschema.
"""
import json
from pathlib import Path
from jsonschema import Draft202012Validator
root = Path(__file__).resolve().parent
schema = json.loads((root / 'schemas/evidence.schema.json').read_text())
Draft202012Validator.check_schema(schema)
validator = Draft202012Validator(schema)
def semantic_valid(record):
    decision = record['decision']
    if decision['state'] != 'observed':
        return True
    queries = decision['queries'] or []
    for index, query in enumerate(queries):
        if query['index'] != index:
            return False
    for index, selected in enumerate(decision['selected'] or []):
        if selected['index'] != index:
            return False
        for contributor in selected['contributors']:
            ordinal = contributor['query_index']
            if ordinal >= len(queries) or not queries[ordinal]['selected']:
                return False
            if selected['delivered'] and not selected['uncertain'] and not queries[ordinal]['delivered']:
                return False
    return True

for case in json.loads((root / 'fixtures/evidence-v2.json').read_text()):
    accepted = validator.is_valid(case['record']) and semantic_valid(case['record'])
    assert accepted == case['valid'], case['name']
    print(case['name'], 'accept' if accepted else 'reject')
print('JSON Schema + independent relational validator agreement corpus passed. Relational constraints require executable validation; pure JSON Schema cannot express index/reference equality. Neither authenticates provenance; raw byte/depth limits are transport constraints.')
