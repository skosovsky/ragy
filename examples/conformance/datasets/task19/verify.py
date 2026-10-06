#!/usr/bin/env python3
"""Offline stdlib schema-subset and cross-reference validation. No strategy execution."""
import argparse, collections, hashlib, json, pathlib
CATEGORIES = {'lookup','multihop','no_answer','harmful_rewrite','duplicates','conflicting_stale','limited_scope','multilingual','malicious_instructions'}
def check(value, schema, root, path='$'):
    if '$ref' in schema:
        s=root
        for part in schema['$ref'][2:].split('/'): s=s[part]
        return check(value,s,root,path)
    if 'oneOf' in schema:
        passed=0
        for s in schema['oneOf']:
            try: check(value,s,root,path); passed+=1
            except AssertionError: pass
        assert passed==1, f'{path}: oneOf matches {passed}'
    if 'const' in schema: assert value==schema['const'], f'{path}: const'
    if 'enum' in schema: assert value in schema['enum'], f'{path}: enum'
    typ=schema.get('type')
    if typ:
        valid={'object':isinstance(value,dict),'array':isinstance(value,list),'string':isinstance(value,str),'boolean':isinstance(value,bool),'integer':isinstance(value,int) and not isinstance(value,bool)}
        assert valid[typ], f'{path}: expected {typ}'
    if typ=='object':
        assert set(schema['required'])<=value.keys(), f'{path}: required'
        assert not schema.get('additionalProperties') and value.keys()<=schema['properties'].keys(), f'{path}: unexpected field'
        for k,v in value.items(): check(v,schema['properties'][k],root,f'{path}.{k}')
    if typ=='array':
        for i,v in enumerate(value): check(v,schema['items'],root,f'{path}[{i}]')
    if typ=='string': assert len(value)>=schema.get('minLength',0), f'{path}: minLength'
    if typ=='integer': assert schema.get('minimum',value)<=value<=schema.get('maximum',value), f'{path}: range'
def unique(rows,key):
    vals=[r[key] for r in rows]; assert len(vals)==len(set(vals)), f'duplicate {key}'
def verify(base,holdout=None):
    schema=json.loads((base/'schema.json').read_text()); corpus=json.loads((base/'corpus.json').read_text())
    check(corpus,schema,schema); docs=corpus['documents']; unique(docs,'id'); ids={d['id']:d for d in docs}
    assert len(docs)>=30
    assert len({(d['source_id'],d['revision']) for d in docs})==len(docs), 'source revision identity collision'
    for d in docs:
        for r in d['relations']: assert r['target_id'] in ids, 'dangling relation'
    splits=[json.loads((base/'dev.json').read_text())]
    hp=holdout or base/'holdout.json'
    if hp.exists(): splits.append(json.loads(hp.read_text()))
    allqueries=[]
    for split in splits:
        check(split,schema,schema); rows=split['queries']; unique(rows,'id'); unique(rows,'case_id')
        counts=collections.Counter(q['category'] for q in rows)
        assert set(counts)==CATEGORIES and all(n>=2 for n in counts.values()), 'category coverage'
        for q in rows:
            assert q['id'].startswith(split['split']+'-') and q['case_id'].startswith(split['split']+'-'), 'split identity'
            assert q['answerable']==bool(q['qrels']), 'answerability mismatch'
            unique(q['qrels'],'document_id')
            for rel in q['qrels']:
                assert rel['document_id'] in ids, 'unknown qrel document'
                d=ids[rel['document_id']]
                assert d['current'] and d['scope'] in ('public',q['scope']), 'ineligible gold'
        allqueries+=rows
    unique(allqueries,'id'); unique(allqueries,'case_id'); unique(allqueries,'text')
    if len(splits)==2: assert len(allqueries)>=24
    manifest=json.loads((base/'manifest.json').read_text()) if (base/'manifest.json').exists() else None
    if manifest:
        for name,digest in manifest['sha256'].items():
            path=hp if name=='holdout.json' else base/name
            if path.exists(): assert hashlib.sha256(path.read_bytes()).hexdigest()==digest, f'digest mismatch: {name}'
    print(json.dumps({'documents':len(docs),'splits':{s['split']:len(s['queries']) for s in splits},'validation':'passed'},sort_keys=True))
if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--holdout',type=pathlib.Path); a=p.parse_args()
    verify(pathlib.Path(__file__).resolve().parent,a.holdout)
