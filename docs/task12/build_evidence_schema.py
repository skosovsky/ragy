import json
from pathlib import Path
root=Path(__file__).resolve().parent
def obj(props, extra=None):
    result={'type':'object','additionalProperties':False,'required':list(props),'properties':props}
    if extra: result.update(extra)
    return result
def ref(name): return {'$ref':'#/$defs/'+name}
def array(name): return {'type':['array','null'],'items':ref(name)}
def captured(value):
    return obj({'state':{'enum':['observed','omitted','unavailable','unsupported']},'value':{}}, {'oneOf':[
        {'properties':{'state':{'const':'observed'},'value':value}},
        {'properties':{'state':{'enum':['omitted','unavailable','unsupported']},'value':{'type':'null'}}}]})
def textstate(state): return obj({'state':{'const':state},'value':{'type':'null'}})
def observed_or_omitted(base): return {'allOf':[ref(base),{'properties':{'state':{'enum':['observed','omitted']}}}]}
text=captured({'type':'string','minLength':1})
number=captured({'type':'number'})
fields=['namespace','id','revision','transformation','artifact','representation']
source=obj({key:ref('text') for key in fields})
captured_source=obj({key:observed_or_omitted('text') for key in fields})
missing_source=obj({key:textstate('unavailable') for key in fields})
score=obj({'state':{'enum':['native','normalized','absent','omitted','unavailable','unsupported']},'value':{},'semantics':ref('text')},{'oneOf':[
 {'properties':{'state':{'const':'native'},'value':{'type':'number'},'semantics':observed_or_omitted('text')}},
 {'properties':{'state':{'const':'normalized'},'value':{'type':'number','minimum':0,'maximum':1},'semantics':observed_or_omitted('text')}},
 {'properties':{'state':{'const':'absent'},'value':{'type':'null'},'semantics':textstate('unavailable')}},
 {'properties':{'state':{'enum':['omitted','unavailable','unsupported']},'value':{'type':'null'}}}]})
label=obj({'state':{'enum':['observed','ungradable']},'query':ref('text'),'source':ref('source'),'grade':ref('number'),'rubric':ref('text')},{'oneOf':[
 {'properties':{'state':{'const':'observed'},'query':observed_or_omitted('text'),'source':ref('captured_source'),'grade':observed_or_omitted('number'),'rubric':observed_or_omitted('text')}},
 {'properties':{'state':{'const':'ungradable'},'query':textstate('unavailable'),'source':ref('missing_source'),'grade':textstate('unavailable'),'rubric':textstate('unavailable')}}]})
observed_source=obj({key:{'allOf':[ref('text'),{'properties':{'state':{'const':'observed'}}}]} for key in fields})
span=obj({'start':{'type':'integer','minimum':0},'end':{'type':'integer','minimum':0}})
page=obj({'physical_index':{'type':'integer','minimum':0},'printed_label':{'type':'string'},'width_pt':{'type':'number','minimum':0},'height_pt':{'type':'number','minimum':0},'rotation_deg':{'enum':[0,90,180,270]}})
region=obj({key:{'type':'number','minimum':0} for key in ['left','top','right','bottom']})
cell=obj({'table':{'type':'string'},'element':{'type':'string'},'row':{'type':'integer','minimum':0},'column':{'type':'integer','minimum':0},'row_span':{'type':'integer','minimum':0},'column_span':{'type':'integer','minimum':0}})
zero_span={'start':0,'end':0}
zero_page={'physical_index':0,'printed_label':'','width_pt':0,'height_pt':0,'rotation_deg':0}
zero_region={key:0 for key in ['left','top','right','bottom']}
zero_cell={'table':'','element':'','row':0,'column':0,'row_span':0,'column_span':0}
variants=[]
for kind,active in [('document',[]),('text',['span']),('page',['page']),('region',['page','region']),('table_cell',['page','cell']),('image_region',['page','region'])]:
    props={'kind':{'const':kind}}
    for key,value in [('span',zero_span),('page',zero_page),('region',zero_region),('cell',zero_cell)]:
        if key not in active: props[key]={'const':value}
    if 'span' in active: props['span']={'properties':{'end':{'minimum':1}}}
    if 'page' in active: props['page']={'properties':{'width_pt':{'exclusiveMinimum':0},'height_pt':{'exclusiveMinimum':0}}}
    if 'cell' in active: props['cell']={'properties':{'table':{'minLength':1},'element':{'minLength':1},'row_span':{'minimum':1},'column_span':{'minimum':1}}}
    variants.append({'properties':props})
location=obj({'source':ref('observed_source'),'kind':{'enum':['document','text','page','region','table_cell','image_region']},'span':ref('span'),'page':ref('page'),'region':ref('region'),'cell':ref('cell')},{'oneOf':variants})
contribution=obj({'query_index':{'type':'integer','minimum':0},'document_id':{'allOf':[ref('text'),{'properties':{'state':{'const':'observed'}}}]},'rank':{'type':'integer','minimum':1},'locations_state':{'enum':['observed','omitted','unavailable','unsupported']},'locations':array('location')},{'allOf':[
 {'if':{'properties':{'locations_state':{'const':'observed'}}},'then':{'properties':{'locations':{'type':'array','minItems':1}}},'else':{'properties':{'locations':{'maxItems':0}}}}]})
hit=obj({'id':{'allOf':[ref('text'),{'properties':{'state':{'const':'observed'}}}]},'rank':ref('number'),'score':ref('score'),'snippet':ref('text'),'sources_state':{'enum':['observed','unavailable','unsupported']},'sources':array('captured_source'),'judgment':ref('label'),'locations_state':{'enum':['observed','omitted','unavailable','unsupported']},'locations':array('location'),'contributions_state':{'enum':['observed','omitted','unavailable','unsupported']},'contributions':array('contribution')},{'allOf':[
 {'if':{'properties':{'contributions_state':{'const':'observed'}}},'then':{'properties':{'contributions':{'type':'array','minItems':1}}},'else':{'properties':{'contributions':{'maxItems':0}}}},
 {'if':{'properties':{'locations_state':{'const':'observed'}}},'then':{'properties':{'locations':{'type':'array','minItems':1}}},'else':{'properties':{'locations':{'maxItems':0}}}},
 {'if':{'properties':{'rank':{'properties':{'state':{'const':'observed'}}}}},'then':{'properties':{'rank':{'properties':{'value':{'type':'integer','minimum':1}}}}}},
 {'if':{'properties':{'sources_state':{'const':'observed'}}},'then':{'properties':{'sources':{'type':'array','minItems':1}}},'else':{'properties':{'sources':{'maxItems':0}}}}]})
stage=obj({'index':{'type':'integer','minimum':0},'name':{'allOf':[ref('text'),{'properties':{'state':{'enum':['observed','omitted','unsupported']}}}]},'status':{'enum':['observed','not_run','unsupported','missing_observation','unavailable']},'hits_state':{'enum':['observed','omitted','unavailable']},'hits':array('hit')},{'allOf':[
 {'if':{'properties':{'status':{'const':'observed'}}},'then':{'properties':{'hits_state':{'enum':['observed','omitted']}}},'else':{'properties':{'hits_state':{'const':'unavailable'},'hits':{'maxItems':0}}}},
 {'if':{'properties':{'hits_state':{'enum':['omitted','unavailable']}}},'then':{'properties':{'hits':{'maxItems':0}}}}]})
diag=obj({'kind':{'enum':['model_calls','retrieval_calls','input_tokens','output_tokens','cost_units','latency_ms']},'number':ref('number')},{'allOf':[
 {'if':{'properties':{'number':{'properties':{'state':{'const':'observed'}}}}},'then':{'properties':{'number':{'properties':{'value':{'minimum':0}}}}}},
 {'if':{'properties':{'kind':{'not':{'const':'latency_ms'}},'number':{'properties':{'state':{'const':'observed'}}}}},'then':{'properties':{'number':{'properties':{'value':{'type':'integer'}}}}}}]})
coverage=json.loads((root/'schemas/read_coverage.schema.json').read_text()); coverage.pop('$id');coverage.pop('$schema')
schema=obj({'schema':{'const':'ragy.retrieval-evidence'},'retrieval_id':ref('text'),'scope':ref('text'),'publication':ref('text'),'recipe':ref('text'),'query':ref('text'),'outcome':{'enum':['complete','complete_empty','partial','insufficient','failed']},'reason':{'enum':['none','budget','deadline','missing_evidence','partial_targets','target_failure']},'coverage':ref('coverage'),'stages':array('stage'),'diagnostics':array('diagnostic')},{'allOf':[
 {'if':{'properties':{'outcome':{'enum':['complete','complete_empty']}}},'then':{'properties':{'reason':{'const':'none'}}}},
 {'if':{'properties':{'outcome':{'const':'partial'}}},'then':{'properties':{'reason':{'not':{'const':'none'}}}}},
 {'if':{'properties':{'outcome':{'const':'complete_empty'}}},'then':{'properties':{'stages':{'items':{'properties':{'hits':{'maxItems':0}}}}}}}]})
schema.update({'$schema':'https://json-schema.org/draft/2020-12/schema','$id':'https://ragy.invalid/schemas/retrieval-evidence','$defs':{'text':text,'number':number,'source':source,'captured_source':captured_source,'missing_source':missing_source,'score':score,'label':label,'hit':hit,'stage':stage,'diagnostic':diag,'coverage':coverage,'observed_source':observed_source,'span':span,'page':page,'region':region,'cell':cell,'location':location,'contribution':contribution}})
(root/'schemas/evidence.schema.json').write_text(json.dumps(schema,ensure_ascii=False,indent=2)+'\n')
