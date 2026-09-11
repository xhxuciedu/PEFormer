"""Paired mean-outcome replication across acquisitions, never score ensembling."""
import json
from v31 import *
from e27_adaptation_test import reference_scores
from e29_summarize_audit import paired_interval
from scipy.stats import spearmanr
from secondary_analysis import deeper
from error_strata import within_spearman

def interval(a,b):
    assert a.index.equals(b.index)
    return paired_interval(a.achieved-b.achieved,a.component)

def main():
    manifest=json.loads((OUT/'jobs_replication.json').read_text()); jobs=manifest['jobs']; assert len(jobs)==48
    frame=pd.read_parquet(CACHE/'index.parquet')
    data=torch.load(CACHE/'prepared.pt',map_location='cpu',weights_only=False);q0=data['q0'].numpy()
    base={}
    for surface in ('target_outer_val','source_val','source_audit'):
        sub=frame.loc[frame.surface==surface]
        base[surface]=table(sub,q0[sub.row.to_numpy()],surface.startswith('source'))
        if surface=='target_outer_val':
            op=sub.merge(reference_scores(),left_on=['group_id','design_key'],right_on=['edit_key','design_key'],validate='1:1',how='left')
            assert op.op.notna().all();base['optiprime']=table(op,op.op)
    tables={};runs=[];hashes={}
    for j in jobs:
        folder=OUT/'runs'/j['tag'];r=json.loads((folder/'results.json').read_text())
        assert r['args']==j['args']
        prov=json.loads((folder/'provenance.json').read_text())
        for src in (prov['inputs_sha256'],r['outputs_sha256']):
            for name,digest in src.items():
                if name in hashes:assert hashes[name]==digest
                hashes[name]=digest
        row={'label':j['label'],'tag':j['tag'],'args':r['args'],'views':{},'label_budget':prov['label_budget'],
             'total_seconds':r['total_seconds'],'training_seconds':r['training_seconds'],
             'reused_screen_fit':j['tag'] in manifest['reused_at_launch']}
        for policy in ('target','constrained'):
            row['views'][policy]={'step':r['views'][policy]['step'],'inner':r['views'][policy]['inner'],'surfaces':{}}
            for surface in ('target_outer_val','source_val','source_audit'):
                f=pd.read_parquet(folder/f'{policy}_{surface}.parquet')
                t=table(f,f.selection,surface.startswith('source'))
                assert t.index.equals(base[surface].index)
                assert abs(t.achieved.mean()-r['views'][policy]['surfaces'][surface]['achieved'])<1e-12
                tables[(j['tag'],policy,surface)]=t
                row['views'][policy]['surfaces'][surface]=float(t.achieved.mean())
                if surface == 'target_outer_val':
                    row['views'][policy]['secondary'] = {
                        'pooled_spearman': float(spearmanr(f.selection, f.y).statistic),
                        'regret': float(t.regret.mean()),
                        **within_spearman(f),
                        'depth_endpoints': [deeper(f,k) for k in (1,3,5)]}
        runs.append(row)
    labels=list(dict.fromkeys(j['label'] for j in jobs)); aggregate=[]; means={}
    for budget in (200,1000):
        for label in labels:
            group=[r for r in runs if r['label']==label and r['args']['budget']==budget]
            assert len(group)==6
            for policy in ('target','constrained'):
                row={'label':label,'budget':budget,'policy':policy,'n_runs':6,'surfaces':{},'contrasts':{},
                     'step_zero_fallbacks':sum(r['views'][policy]['step']==0 for r in group),'acquisitions':[]}
                for surface in ('target_outer_val','source_val','source_audit'):
                    arr=pd.concat([tables[(r['tag'],policy,surface)].achieved.rename(r['tag']) for r in group],axis=1)
                    assert not arr.isna().any().any()
                    t=base[surface].copy();t['achieved']=arr.mean(axis=1);means[(label,budget,policy,surface)]=t
                    values=[r['views'][policy]['surfaces'][surface] for r in group]
                    row['surfaces'][surface]={'achieved':float(t.achieved.mean()),'run_sd':float(np.std(values,ddof=1)),
                                             'run_min':min(values),'run_max':max(values),'groups':len(t),
                                             'domains':{str(k):float(g.achieved.mean()) for k,g in t.groupby('study')}}
                    row['contrasts'][surface+'_minus_start']=interval(t,base[surface])
                    if surface=='target_outer_val':row['contrasts']['target_minus_optiprime']=interval(t,base['optiprime'])
                for seed in (20260910,20260911,20260912):
                    rr=[r for r in group if r['args']['subset_seed']==seed]
                    row['acquisitions'].append({'subset_seed':seed,'optimizer_runs':2,
                       'target_mean':float(np.mean([r['views'][policy]['surfaces']['target_outer_val'] for r in rr])),
                       'source_audit_mean':float(np.mean([r['views'][policy]['surfaces']['source_audit'] for r in rr])),
                       'target_optimizer_sd':float(np.std([r['views'][policy]['surfaces']['target_outer_val'] for r in rr],ddof=1))})
                row['secondary_mean_over_runs'] = {
                    key: float(np.mean([r['views'][policy]['secondary'][key] for r in group]))
                    for key in ('pooled_spearman','regret','mean_defined_group_spearman')}
                row['secondary_mean_over_runs']['depth_endpoints'] = [
                    {'k': k, 'eligible_groups': group[0]['views'][policy]['secondary']['depth_endpoints'][i]['eligible_groups'],
                     'best_measured_outcome_among_top_k': float(np.mean([
                         r['views'][policy]['secondary']['depth_endpoints'][i]['best_measured_outcome_among_top_k'] for r in group]))}
                    for i,k in enumerate((1,3,5))]
                excluded=json.loads((OUT/'core_overlap_audit.json').read_text())['target_components']
                full=means[(label,budget,policy,'target_outer_val')]
                keep=~full.component.isin(excluded)
                row['core19_excluded_target_minus_optiprime']=interval(full.loc[keep],base['optiprime'].loc[keep])
                aggregate.append(row)
    direct=[]
    for budget in (200,1000):
        for label in labels:
            if label=='M':continue
            for policy in ('target','constrained'):
                for surface in ('target_outer_val','source_val','source_audit'):
                    direct.append({'budget':budget,'left':'M','left_policy':'constrained','right':label,'right_policy':policy,
                        'surface':surface,**interval(means[('M',budget,'constrained',surface)],means[(label,budget,policy,surface)])})
    control_contrasts=[]
    for budget in (200,1000):
        for left,right in [('E','A'),('E','C_frozen'),('M','E')]:
            for surface in ('target_outer_val','source_val','source_audit'):
                control_contrasts.append({'budget':budget,'left':left,'right':right,'policy':'target','surface':surface,
                    **interval(means[(left,budget,'target',surface)],means[(right,budget,'target',surface)])})
    for i,(name,digest) in enumerate(hashes.items()):
        assert sha256(ROOT/name)==digest,(name,'changed fingerprint')
        if i%100==0:print('verified replication fingerprints',i,len(hashes),flush=True)
    result={'status':'Retrospective acquisition/optimizer replication; mean achieved outcomes, not score ensemble.',
            'uncertainty':'Paired component intervals conditional on six fitted runs; acquisition and optimizer variability reported separately.',
            'baselines':{k:float(v.achieved.mean()) for k,v in base.items()},'aggregates':aggregate,'direct_contrasts':direct,
            'control_contrasts':control_contrasts,'runs':runs,
            'optiprime_secondary': {'pooled_spearman':float(spearmanr(op.op,op.y).statistic),
                'regret':float(base['optiprime'].regret.mean()),**within_spearman(op.assign(selection=op.op)),
                'depth_endpoints':[deeper(op.assign(selection=op.op),k) for k in (1,3,5)]},
            'configurations':len(jobs),'new_fits':sum(not r['reused_screen_fit'] for r in runs),
            'reused_fits':sum(r['reused_screen_fit'] for r in runs),
            'new_process_hours':sum(r['total_seconds'] for r in runs if not r['reused_screen_fit'])/3600,
            'fingerprints_verified':len(hashes),'inputs_sha256':provenance([OUT/'jobs_replication.json',OUT/'REPLICATION_PROTOCOL.md',Path(__file__)])}
    write_json(OUT/'replication_results.json',result)
    lines=['# V3.1 acquisition replication: completed results','',result['status'],result['uncertainty'],'',
           'Three acquisition subsets from one library, two optimizer seeds each. Shared outer validation',
           'is not six independent biological datasets. Labels include inner validation; original source',
           'retention validation was seen during pretraining. No new target-test or independent-study scoring.','',
           '| Budget | Method | Policy | Target @1 mean | Run SD | Delta vs OptiPrime [95% component CI] | Source @1 mean | Source change [95% CI] | Fallbacks / 6 |',
           '|---:|---|---|---:|---:|---|---:|---|---:|']
    for r in aggregate:
        d=r['contrasts']['target_minus_optiprime'];lo,hi=d['ci95'];s=r['contrasts']['source_audit_minus_start'];sl,sh=s['ci95']
        lines.append(f"| {r['budget']} | {r['label']} | {r['policy']} | {r['surfaces']['target_outer_val']['achieved']:.6f} | {r['surfaces']['target_outer_val']['run_sd']:.6f} | {d['observed']:+.6f} [{lo:+.6f}, {hi:+.6f}] | {r['surfaces']['source_audit']['achieved']:.6f} | {s['observed']:+.6f} [{sl:+.6f}, {sh:+.6f}] | {r['step_zero_fallbacks']} |")
    lines+=['','## Promoted constrained model versus target-selected controls','',
            '| Budget | Comparator | Surface | Difference | 95% component CI |','|---:|---|---|---:|---|']
    for d in direct:
        if d['right_policy']!='target':continue
        lo,hi=d['ci95'];lines.append(f"| {d['budget']} | {d['right']} | {d['surface']} | {d['observed']:+.6f} | [{lo:+.6f}, {hi:+.6f}] |")
    lines+=['','## Target-selected procedure contrasts','',
            '| Budget | Contrast | Surface | Difference | 95% component CI |','|---:|---|---|---:|---|']
    for d in control_contrasts:
        lo,hi=d['ci95'];lines.append(f"| {d['budget']} | {d['left']} minus {d['right']} | {d['surface']} | {d['observed']:+.6f} | [{lo:+.6f}, {hi:+.6f}] |")
    lines+=['','## Secondary target endpoints','',
            'Means of per-run metrics, not metrics of an averaged score. @3 and @5 use only',
            '4,014 and 2,056 groups with sufficient depth; they are not directly comparable to all-group @1.',
            'Within-group correlations exclude groups with constant measured outcomes or model scores.','',
            '| Budget | Method | Policy | Pooled Spearman | Mean defined within-group Spearman | Regret | @3 | @5 |',
            '|---:|---|---|---:|---:|---:|---:|---:|']
    for r in aggregate:
        s=r['secondary_mean_over_runs'];d=s['depth_endpoints']
        lines.append(f"| {r['budget']} | {r['label']} | {r['policy']} | {s['pooled_spearman']:.6f} | {s['mean_defined_group_spearman']:.6f} | {s['regret']:.6f} | {d[1]['best_measured_outcome_among_top_k']:.6f} | {d[2]['best_measured_outcome_among_top_k']:.6f} |")
    s=result['optiprime_secondary'];d=s['depth_endpoints']
    lines.append(f"| — | OptiPrime | released | {s['pooled_spearman']:.6f} | {s['mean_defined_group_spearman']:.6f} | {s['regret']:.6f} | {d[1]['best_measured_outcome_among_top_k']:.6f} | {d[2]['best_measured_outcome_among_top_k']:.6f} |")
    lines+=['','## All acquisition/optimizer outcomes','',
            '| Budget | Method | Subset | Optimizer | Policy | Selected step | Target @1 | Source audit @1 |','|---:|---|---:|---:|---|---:|---:|---:|']
    for r in runs:
        for policy,v in r['views'].items():
            a=r['args'];lines.append(f"| {a['budget']} | {r['label']} | {a['subset_seed']} | {a['seed']} | {policy} | {v['step']} | {v['surfaces']['target_outer_val']:.6f} | {v['surfaces']['source_audit']:.6f} |")
    lines+=['',f"{result['new_fits']} new fits + {result['reused_fits']} exact screen reuses; {result['new_process_hours']:.3f} new summed process-hours.",
            'All checkpoint/prediction fingerprints verified. Statistical noninferiority or superiority on',
            'these exposed development populations is not a substitute for independent confirmation.']
    with (OUT/'REPLICATION_RESULTS.md').open('x') as f:f.write('\n'.join(lines)+'\n')
    print(json.dumps({'aggregates':aggregate,'new_process_hours':result['new_process_hours']},indent=2),flush=True)

if __name__=='__main__':main()
