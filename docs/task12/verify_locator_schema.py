"""Independent locator envelope shape validation; retained byte/geometry bounds stay in Go."""
import copy
import json
from pathlib import Path
from jsonschema import Draft202012Validator

root = Path(__file__).resolve().parent
schema = json.loads((root / 'schemas/locator.schema.json').read_text())
Draft202012Validator.check_schema(schema)
validator = Draft202012Validator(schema)
fixture = json.loads((root / 'fixtures/locator.json').read_text())
valid = [fixture]
for kind in ('document', 'page', 'region', 'table_cell', 'image_region'):
    value = copy.deepcopy(fixture)
    loc = value['location']
    loc['kind'] = kind
    loc['span'] = {'start': 0, 'end': 0}
    if kind != 'document':
        loc['page'] = {'physical_index': 1, 'printed_label': '1', 'width_pt': 600, 'height_pt': 800, 'rotation_deg': 90}
    if kind in ('region', 'image_region'):
        loc['region'] = {'left': 100, 'top': 200, 'right': 300, 'bottom': 400}
    if kind == 'table_cell':
        loc['cell'] = {'table': 't1', 'element': 'c1', 'row': 0, 'column': 0, 'row_span': 1, 'column_span': 2}
    valid.append(value)
for value in valid:
    validator.validate(value)
negative = []
for field in ('schema', 'location'):
    value = copy.deepcopy(fixture)
    del value[field]
    negative.append(value)
for field in fixture['location']:
    value = copy.deepcopy(fixture)
    del value['location'][field]
    negative.append(value)
for field in fixture['location']['reference']:
    value = copy.deepcopy(fixture)
    del value['location']['reference'][field]
    negative.append(value)
for field, replacement in (('schema', 'unknown'), ('unknown', 1)):
    value = copy.deepcopy(fixture)
    value[field] = replacement
    negative.append(value)
for field, replacement in (('kind', 'guessed'), ('page', None), ('span', {'start': 0, 'end': 0})):
    value = copy.deepcopy(fixture)
    value['location'][field] = replacement
    negative.append(value)
value = copy.deepcopy(valid[2])
value['location']['page']['rotation_deg'] = 45
negative.append(value)
value = copy.deepcopy(valid[4])
value['location']['cell']['column_span'] = 0
negative.append(value)
for value in negative:
    if not list(validator.iter_errors(value)):
        raise AssertionError('invalid locator envelope accepted')
print(f'locator schema: {len(valid)} positive and {len(negative)} negative fixtures passed')
