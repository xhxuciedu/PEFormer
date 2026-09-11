"""Independent checks of completed runs, checkpoint choices and fingerprints."""
import argparse
import json
from v31 import *
from run_stage import tag

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--expected',type=int,default=32);args=ap.parse_args()
    files=sorted((OUT/'runs').glob('*/results.json'))
    assert len(files)==args.expected,(len(files),args.expected)
    jobs=[]
    for stage in ('controls','factorial'):
        p=OUT/f'jobs_{stage}.json'
        if p.exists():jobs.extend(json.loads(p.read_text()))
    assert {p.parent.name for p in files}=={tag(j) for j in jobs}
    hashes={}; checks=[]
    index=pd.read_parquet(CACHE/'index.parquet')
    assert index.groupby('component').surface.nunique().max()==1
    for p in files:
        r=json.loads(p.read_text());prov=json.loads((p.parent/'provenance.json').read_text())
        h=r['history']; assert len(h)==11 and h[0]['step']==0 and h[-1]['step']==r['args']['steps']
        assert sum(x['groups'] for x in prov['label_budget'].values())==1001
        for policy in ('target','constrained'):
            candidates=h if policy=='target' else [x for x in h if x['feasible']]
            best=max(candidates,key=lambda x:x['inner'])
            assert best['step']==r['views'][policy]['step']
            for entry in h: assert entry['feasible']==feasible(entry['source_domains'],prov['initial_source_domains'])
        cp=pd.read_parquet(p.parent/'checkpoint_predictions.parquet')
        for step,g in cp.loc[cp.surface=='inner_val'].groupby('step'):
            entry=next(x for x in h if x['step']==step)
            actual=table(index.iloc[g.row.to_numpy()],g.score.to_numpy()).achieved.mean()
            assert abs(actual-entry['inner'])<1e-12
        for view in ('target','constrained','last'):
            for surface in ('target_outer_val','source_val','source_audit'):
                f=pd.read_parquet(p.parent/f'{view}_{surface}.parquet')
                expected=index.loc[index.surface==surface]
                assert len(f)==len(expected)
                assert f[['group_id','design_key']].reset_index(drop=True).equals(expected[['group_id','design_key']].reset_index(drop=True))
                actual=table(f,f.selection,source=surface.startswith('source')).achieved.mean()
                assert abs(actual-r['views'][view]['surfaces'][surface]['achieved'])<1e-12
        for ledger in (prov['inputs_sha256'],r['outputs_sha256']):
            for name,digest in ledger.items():
                if name in hashes: assert hashes[name]==digest,(name,'conflicting hash')
                hashes[name]=digest
        checks.append({'tag':r['tag'],'checkpoint_policy_verified':True,'endpoints_recomputed':9})
    for n,(name,digest) in enumerate(hashes.items()):
        assert sha256(ROOT/name)==digest,(name,'changed hash')
        if n%50==0:print('verified fingerprints',n,len(hashes),flush=True)
    # Selection code is independent of exposed audit utility by construction.
    from analyze import choose
    controls=[json.loads(p.read_text()) for p in files if json.loads(p.read_text())['args']['source_loss']=='none']
    selection=json.loads((OUT/'control_selection.json').read_text())
    for arm,policies in selection['recipes'].items():
        for policy,recipe in policies.items():
            assert choose([r for r in controls if r['args']['arm']==arm],policy)['tag']==recipe['tag']
    write_json(OUT/'verification.json',{'fits_verified':len(files),'distinct_fingerprints_verified':len(hashes),
        'checkpoint_prediction_tables_verified':len(files),'selected_and_last_endpoint_checks':len(files)*9,
        'test_scoring_present':False,'checks':checks,'analysis_selection_verified':True})
    print('VERIFIED',len(files),'fits',len(hashes),'fingerprints',flush=True)

if __name__=='__main__':main()
