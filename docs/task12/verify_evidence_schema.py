"""Independent wire validation; semantic admission/ownership are tested in Go."""
import copy
import json
from pathlib import Path
from jsonschema import Draft202012Validator

root = Path(__file__).resolve().parent
schema = json.loads((root / 'schemas/evidence.schema.json').read_text())
fixture = json.loads((root / 'fixtures/evidence_record.json').read_text())
Draft202012Validator.check_schema(schema)
validator = Draft202012Validator(schema)
valid = [fixture]
for score in (-1, 2):
    changed = copy.deepcopy(fixture)
    changed['stages'][0]['hits'][0]['score']['state'] = 'native'
    changed['stages'][0]['hits'][0]['score']['value'] = score
    valid.append(changed)
for status in ('not_run', 'unsupported', 'missing_observation', 'unavailable'):
    changed = copy.deepcopy(fixture)
    changed['stages'][1]['status'] = status
    valid.append(changed)
changed = copy.deepcopy(fixture)
changed['outcome'] = 'complete_empty'
changed['stages'][0]['hits'] = None
valid.append(changed)
zero_span={'start':0,'end':0}
zero_page={'physical_index':0,'printed_label':'','width_pt':0,'height_pt':0,'rotation_deg':0}
zero_region={'left':0,'top':0,'right':0,'bottom':0}
zero_cell={'table':'','element':'','row':0,'column':0,'row_span':0,'column_span':0}
location_cases=[]
for kind in ('document','text','page','region','table_cell','image_region'):
    changed=copy.deepcopy(fixture)
    hit=changed['stages'][0]['hits'][0]
    loc={'source':copy.deepcopy(hit['sources'][0]),'kind':kind,'span':copy.deepcopy(zero_span),'page':copy.deepcopy(zero_page),'region':copy.deepcopy(zero_region),'cell':copy.deepcopy(zero_cell)}
    if kind=='text':loc['span']={'start':6,'end':10}
    if kind in ('page','region','table_cell','image_region'):loc['page']={'physical_index':0,'printed_label':'i','width_pt':600,'height_pt':800,'rotation_deg':90}
    if kind in ('region','image_region'):loc['region']={'left':60,'top':80,'right':100,'bottom':100}
    if kind=='table_cell':loc['cell']={'table':'t1','element':'c1','row':0,'column':0,'row_span':1,'column_span':2}
    hit.update(locations_state='observed',locations=[loc])
    valid.append(changed)
    location_cases.append(changed)
contributed=copy.deepcopy(location_cases[1])
hit=contributed['stages'][0]['hits'][0]
hit.update(contributions_state='observed',contributions=[{'query_index':1,'document_id':{'state':'observed','value':'p3'},'rank':1,'locations_state':'observed','locations':copy.deepcopy(hit['locations'])}])
valid.append(contributed)
contributed_redacted=copy.deepcopy(contributed)
contributed_redacted['stages'][0]['hits'][0]['contributions'][0].update(locations_state='omitted',locations=None)
valid.append(contributed_redacted)
for item in valid:
    validator.validate(item)

invalid = []
def reject(change):
    changed = copy.deepcopy(fixture)
    change(changed)
    invalid.append(changed)
reject(lambda x: x.update(schema='incompatible'))
reject(lambda x: x.update(auth='TOKEN'))
reject(lambda x: x.pop('reason'))
reject(lambda x: x.update(outcome='partial', reason='none'))
reject(lambda x: x.update(outcome='complete_empty'))
reject(lambda x: x['stages'][0]['hits'][0]['score'].update(state='normalized', value=2))
reject(lambda x: x['stages'][0]['hits'][0]['score'].update(state='unavailable', value=0))
reject(lambda x: x['stages'][0]['hits'][0]['score']['semantics'].update(state='unavailable', value=None))
reject(lambda x: x['stages'][0]['hits'][0]['rank'].update(value=1.5))
reject(lambda x: x['stages'][0]['hits'][0]['judgment']['grade'].update(state='observed', value=0))
reject(lambda x: x['stages'][0]['hits'][0].update(sources=None))
reject(lambda x: x['stages'][0]['hits'][0]['id'].update(state='omitted', value=None))
reject(lambda x: x['stages'][0]['hits'][0]['snippet'].pop('value'))
reject(lambda x: x['stages'][0].update(status='not_run'))
reject(lambda x: x.update(diagnostics=[{'kind':'raw_auth','number':{'state':'observed','value':1}}]))
reject(lambda x: x.update(diagnostics=[{'kind':'cost_units','number':{'state':'observed','value':1.5}}]))
for change in (
    lambda h: h.update(locations_state='observed',locations=None),
    lambda h: h.pop('locations_state'),
    lambda h: h.update(locations_state='omitted',locations=location_cases[0]['stages'][0]['hits'][0]['locations']),
):
    changed=copy.deepcopy(fixture);change(changed['stages'][0]['hits'][0]);invalid.append(changed)
for index,change in (
    (0,lambda l:l['span'].update(end=10)),
    (1,lambda l:l['span'].update(end=0)),
    (2,lambda l:l['page'].update(rotation_deg=45)),
    (4,lambda l:l['cell'].update(column_span=0)),
    (5,lambda l:l.update(access_fingerprint='TOKEN')),
    (0,lambda l:l['source']['id'].update(state='omitted',value=None)),
):
    changed=copy.deepcopy(location_cases[index]);change(changed['stages'][0]['hits'][0]['locations'][0]);invalid.append(changed)
for change in (
    lambda h:h.pop('contributions_state'),
    lambda h:h.update(contributions_state='observed',contributions=None),
    lambda h:h.update(contributions_state='omitted'),
    lambda h:h['contributions'][0].update(query_index=-1),
    lambda h:h['contributions'][0].update(rank=0),
    lambda h:h['contributions'][0]['document_id'].update(state='omitted',value=None),
    lambda h:h['contributions'][0].update(raw_query='secret'),
):
    changed=copy.deepcopy(contributed);change(changed['stages'][0]['hits'][0]);invalid.append(changed)
for item in invalid:
    if validator.is_valid(item):
        raise AssertionError('invalid evidence accepted')
print(f'evidence schema: {len(valid)} positive and {len(invalid)} negative fixtures passed')
