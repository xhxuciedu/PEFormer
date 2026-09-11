"""Verify all completed v3 fits and every recorded run input/output fingerprint."""
import json
import numpy as np
from common import ROOT,OUT,sha256,write_json

def main():
    files=sorted((OUT/'runs').glob('*/results.json'))
    assert len(files)==34,len(files)
    expected={(a,b,20260910) for a in list('ABCDEFG')+['C_frozen','F_replay'] for b in (200,1000)}
    expected.update((a,b,s) for a in 'ADEF' for b in (200,1000) for s in (20260911,20260912))
    seen=set()
    fingerprints={}
    for path in files:
        r=json.loads(path.read_text())
        p=json.loads((path.parent/'provenance.json').read_text())
        key=(r['arm'],r['budget'],r['seed'])
        assert key in expected and key not in seen
        seen.add(key)
        assert r['subset_seed']==20260910 and r['target_updates']==100
        assert p['checkpoint_selection'].startswith('budget inner validation only')
        for role in ('train','inner_val'):
            assert p['label_budget'][role]['groups']>0
        assert sum(x['groups'] for x in p['label_budget'].values())=={200:202,1000:1001}[r['budget']]
        assert sum(x['candidates'] for x in p['label_budget'].values())=={200:788,1000:3848}[r['budget']]
        best=max(h['inner']['selection']['achieved_at_1'] for h in r['history'])
        chosen=next(h for h in r['history'] if h['step']==r['best_step'])
        assert abs(chosen['inner']['selection']['achieved_at_1']-best)<1e-12
        assert len(r['history'])==11
        assert not any('test' in name for name in r['surfaces'])
        for source in (p['inputs_sha256'],r['outputs_sha256']):
            for name,digest in source.items():
                if name in fingerprints:
                    assert fingerprints[name]==digest,(name,'conflicting fingerprints')
                fingerprints[name]=digest
    assert seen==expected
    for name,digest in fingerprints.items():
        assert sha256(ROOT/name)==digest,(name,'fingerprint mismatch')
    rep=json.loads((OUT/'replication_results.json').read_text())
    for r in rep['runs']:
        for surface in ('target_outer_val','source_val','source_audit'):
            mean=np.mean([s[surface] for s in r['seeds']])
            assert abs(mean-r['surfaces'][surface]['achieved'])<1e-12
    deploy=json.loads((OUT/'deployment_check_E_b1000_sub20260910_s20260910.json').read_text())
    assert all(not any(c['changed_choices']) for c in deploy['checks'].values())
    result={'fits_verified':len(files),'distinct_fingerprints_verified':len(fingerprints),
            'replicated_aggregate_rows_verified':len(rep['runs']),
            'test_scoring_present':False,'deployment_choices_match':True}
    write_json(OUT/'verification.json',result)
    print(result,flush=True)

if __name__=='__main__':
    main()
