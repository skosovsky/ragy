#!/usr/bin/env python3
"""Independent pinned wire-schema audit of TASK15 raw Go test fixtures.

No Go adapter implementation or Go wire structs are imported. Request assertions
are audited as explicit source markers; response literals are parsed and checked.
The Go runner separately exercises HTTP behavior. This is not live verification.
"""
import hashlib
import json
import math
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / 'docs/task19/provider-wire-contracts.json'


def validate(body, contract, count):
    value = json.loads(body, parse_constant=lambda _: (_ for _ in ()).throw(ValueError('nonfinite')))
    if 'model' in value and value['model'] != contract['model']:
        raise ValueError('model')
    if contract.get('kind') == 'rerank':
        rows = value['results']
        if len(rows) != count or sorted(row['index'] for row in rows) != list(range(count)):
            raise ValueError('cardinality/indices')
        if any(not isinstance(row['relevance_score'], (int, float)) or not math.isfinite(row['relevance_score']) for row in rows):
            raise ValueError('score')
        usage = value.get('meta', {}).get('billed_units', {})
        if 'search_units' in usage and (not isinstance(usage['search_units'], (int, float)) or not math.isfinite(usage['search_units']) or usage['search_units'] < 0):
            raise ValueError('billed search units')
        return
    rows = value[contract['rows']]
    if len(rows) != count:
        raise ValueError('cardinality')
    if contract['indexed'] and sorted(row['index'] for row in rows) != list(range(count)):
        raise ValueError('indices')
    for row in rows:
        if contract['rows'] == 'embeddings' and set(row) != {contract['vector']}:
            raise ValueError('unexpected embedding fields')
        vectors = row[contract['vector']]
        if not contract['matrix']:
            vectors = [vectors]
        if not vectors:
            raise ValueError('empty matrix')
        for vector in vectors:
            if len(vector) != contract['dimension'] or not any(vector):
                raise ValueError('dimensions or zero vector')
            if any(not isinstance(x, (int, float)) or not math.isfinite(x) or abs(x) > 3.4028235e38 for x in vector):
                raise ValueError('finite float32')
    usage = value.get(contract['usage_object'], {})
    if contract['usage_field'] in usage and (not isinstance(usage[contract['usage_field']], int) or usage[contract['usage_field']] < 0):
        raise ValueError('usage')


def verify(root=ROOT):
    contracts = json.loads(CONTRACT.read_text())
    outcomes = []
    for c in contracts['profiles']:
        source = (root / c['fixture_source']).read_text()
        errors = []
        for marker in c['required_assertions']:
            if marker not in source:
                errors.append('missing request/negative assertion: ' + marker)
        literals = re.findall(r'`([^`]+)`', source)
        positive = [body for body in literals if body.startswith('{') and c['positive_marker'] in body]
        if len(positive) != 1:
            errors.append('expected exactly one pinned positive fixture')
        else:
            try:
                validate(positive[0], c, 2)
            except (ValueError, KeyError, TypeError) as error:
                errors.append('positive fixture: ' + str(error))
        negatives = 0
        # The adversarial-response table uses one submitted input. Check all its
        # raw JSON literals independently, not only those rejected by Go today.
        negative_source = (root / c.get('negative_source', c['fixture_source'])).read_text()
        match = re.search(r'func ' + c['negative_function'] + r'\(.*?(?=\nfunc |\Z)', negative_source, re.S)
        if not match:
            errors.append('missing negative fixture function')
        else:
            for body in re.findall(r'`([^`]+)`', match.group()):
                if not body.startswith('{') or body in c.get('valid_adversarial', []):
                    continue
                try:
                    validate(body, c, c.get('negative_count', 1))
                    errors.append('negative fixture accepted: ' + body[:80])
                except (ValueError, KeyError, TypeError, OverflowError):
                    negatives += 1
        if negatives < c['minimum_negatives']:
            errors.append('negative coverage reduced')
        outcomes.append(dict(profile=c['id'], status='failed' if errors else 'passed',
                             errors=errors, negative_fixtures=negatives,
                             fixture_sha256=hashlib.sha256(source.encode()).hexdigest(),
                             negative_source_sha256=hashlib.sha256(negative_source.encode()).hexdigest(),
                             sources=c['official_sources']))
    return dict(schema=contracts['schema'], checked_at=contracts['verified_date'],
                evidence='independent local fixture schema audit; no live provider request',
                contract_sha256=hashlib.sha256(CONTRACT.read_bytes()).hexdigest(), outcomes=outcomes,
                status='failed' if any(x['status'] == 'failed' for x in outcomes) else 'passed')

if __name__ == '__main__':
    report = verify()
    print(json.dumps(report, indent=2))
    sys.exit(report['status'] != 'passed')
