"""Predeclared inner-only selection, followed by exploratory paired reporting."""
import argparse
import json
from v31 import *
from e27_adaptation_test import reference_scores
from e29_summarize_audit import paired_interval

def choose(records,policy):
    return sorted(records,key=lambda r:(-r['views'][policy]['inner'],r['args']['steps'],
                                        r['args']['lr_mult'],r['args']['weight'],r['tag']))[0]

def contrast(a,b):
    if not a.index.equals(b.index): raise ValueError('Mismatched groups')
    return paired_interval(a.achieved-b.achieved,a.component)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--stage',choices=['controls','factorial'],required=True)
    args=ap.parse_args()
    files=sorted((OUT/'runs').glob('*/results.json'))
    runs=[json.loads(p.read_text()) for p in files]
    controls=[r for r in runs if r['args']['source_loss']=='none']
    if len(controls)!=20: raise RuntimeError(f'Controls incomplete: {len(controls)}')
    if args.stage=='controls':
        recipes={a:{policy:{'tag':choose([r for r in controls if r['args']['arm']==a],policy)['tag'],
                                   'args':choose([r for r in controls if r['args']['arm']==a],policy)['args'],
                                   'inner':choose([r for r in controls if r['args']['arm']==a],policy)['views'][policy]['inner']}
                       for policy in ('target','constrained')} for a in ('A','B','D','C_frozen','E')}
        conventional={p:max(('B','D','C_frozen'),key=lambda a:recipes[a][p]['inner']) for p in ('target','constrained')}
        write_json(OUT/'control_selection.json',{'selection':'Inner target only; per-study feasibility for constrained policy.',
            'recipes':recipes,'strongest_other_conventional':conventional,
            'inputs_sha256':provenance(files+[Path(__file__)])})
        selected=[(a,p,recipes[a][p]['tag']) for a in recipes for p in ('target','constrained')]
        stage_runs=controls
    else:
        factorial=[r for r in runs if r['args']['source_loss']!='none']
        if len(factorial)!=12: raise RuntimeError(f'Factorial incomplete: {len(factorial)}')
        c=json.loads((OUT/'control_selection.json').read_text())
        winner=choose(factorial,'constrained')
        delta=winner['views']['constrained']['inner']-c['recipes']['E']['constrained']['inner']
        write_json(OUT/'factorial_selection.json',{'selection':'Constrained target inner only',
            'candidate':winner['tag'],'args':winner['args'],'inner_delta_vs_constrained_E':delta,
            'promoted':bool(delta>=.0005),'threshold':.0005,
            'inputs_sha256':provenance(files+[OUT/'control_selection.json',Path(__file__)])})
        selected=[(r['tag'],p,r['tag']) for r in factorial for p in ('target','constrained')]
        stage_runs=factorial
    index=pd.read_parquet(CACHE/'index.parquet')
    q=torch.load(CACHE/'prepared.pt',map_location='cpu',weights_only=False)['q0'].numpy()
    base={}
    for surface in ('target_outer_val','source_val','source_audit'):
        f=index.loc[index.surface==surface]
        base[surface]=table(f,q[f.row.to_numpy()],surface.startswith('source'))
        if surface=='target_outer_val':
            op=f.merge(reference_scores(),left_on=['group_id','design_key'],right_on=['edit_key','design_key'],how='left',validate='1:1')
            if op.op.isna().any(): raise ValueError('Missing comparator')
            base['optiprime']=table(op,op.op)
    lookup={r['tag']:r for r in runs}; records=[]; tables={}
    for label,policy,tag in selected:
        r=lookup[tag]; row={'label':label,'policy':policy,'tag':tag,'args':r['args'],
                           'step':r['views'][policy]['step'],'inner':r['views'][policy]['inner'],
                           'surfaces':{},'contrasts':{}}
        for surface in ('target_outer_val','source_val','source_audit'):
            f=pd.read_parquet(OUT/'runs'/tag/f'{policy}_{surface}.parquet')
            t=table(f,f.selection,surface.startswith('source')); tables[(label,policy,surface)]=t
            row['surfaces'][surface]={'achieved':float(t.achieved.mean()),'groups':len(t)}
            row['contrasts'][surface+'_minus_start']=contrast(t,base[surface])
            if surface=='target_outer_val': row['contrasts']['target_minus_optiprime']=contrast(t,base['optiprime'])
        records.append(row)
    direct=[]
    if args.stage=='controls':
        for p in ('target','constrained'):
            for left,right in [('E','A'),('E',conventional[p])]:
                for surface in ('target_outer_val','source_val','source_audit'):
                    direct.append({'policy':p,'left':left,'right':right,'surface':surface,
                        **contrast(tables[(left,p,surface)],tables[(right,p,surface)])})
    else:
        c=json.loads((OUT/'control_selection.json').read_text())
        for p in ('target','constrained'):
            reference=c['recipes']['E'][p]['tag']
            for surface in ('target_outer_val','source_val','source_audit'):
                f=pd.read_parquet(OUT/'runs'/reference/f'{p}_{surface}.parquet')
                t=table(f,f.selection,surface.startswith('source'))
                direct.append({'policy':p,'left':winner['tag'],'right':'E','surface':surface,
                    **contrast(tables[(winner['tag'],p,surface)],t)})
    result={'stage':args.stage,'status':'One-subset, one-seed retrospective development; unadjusted paired component intervals.',
            'baselines':{k:{'achieved':float(t.achieved.mean()),'groups':len(t)} for k,t in base.items()},
            'selected':records,'direct_contrasts':direct,
            'all_runs':[{k:r[k] for k in ('tag','args','views','target_pass_equivalents','source_rows_processed',
                                       'training_seconds','total_seconds','peak_gpu_bytes')} for r in stage_runs],
            'stage_process_hours':sum(r['total_seconds'] for r in stage_runs)/3600,
            'inputs_sha256':provenance(files+[Path(__file__),OUT/'v31.py'])}
    write_json(OUT/f'{args.stage}_results.json',result)
    lines=[f'# V3.1 {args.stage}: completed results','',result['status'],'',
           'All configurations and checkpoints are selected using inner data only. Source-constrained',
           'selection uses the two prespecified source-validation studies, not source fold 0.',
           'Each fit has 1001 total labelled target groups, including 162 inner-validation groups.',
           '',f"Released OptiPrime target @1: {base['optiprime'].achieved.mean():.6f}; original source @1: {base['source_audit'].achieved.mean():.6f}.",'',
           '| Method | Policy | Horizon / LR multiplier | Selected step | Inner @1 | Target @1 | Delta vs OptiPrime [95% CI] | Source audit @1 |',
           '|---|---|---|---:|---:|---:|---|---:|']
    for r in records:
        d=r['contrasts']['target_minus_optiprime'];lo,hi=d['ci95']; a=r['args']
        label=r['label'] if args.stage=='controls' else f"{a['source_loss']} / {a['sampling']} / w={a['weight']:g}"
        lines.append(f"| {label} | {r['policy']} | {a['steps']} / {a['lr_mult']} | {r['step']} | {r['inner']:.6f} | {r['surfaces']['target_outer_val']['achieved']:.6f} | {d['observed']:+.6f} [{lo:+.6f}, {hi:+.6f}] | {r['surfaces']['source_audit']['achieved']:.6f} |")
    lines+=['','## Direct contrasts','','| Policy | Contrast | Surface | Difference | 95% CI |','|---|---|---|---:|---|']
    for d in direct:
        lo,hi=d['ci95']; lines.append(f"| {d['policy']} | {d['left']} minus {d['right']} | {d['surface']} | {d['observed']:+.6f} | [{lo:+.6f}, {hi:+.6f}] |")
    lines+=['','## All fit configurations and endpoint trajectories','',
            '| Arm / source loss / sampling / weight | Horizon | LR | Target-selected step | Target-selected outer @1 | Final inner @1 | Final outer @1 | Target passes |',
            '|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in stage_runs:
        a=r['args']; v=r['views']; label=f"{a['arm']} / {a['source_loss']} / {a['sampling']} / {a['weight']:g}"
        lines.append(f"| {label} | {a['steps']} | {a['lr_mult']} | {v['target']['step']} | {v['target']['surfaces']['target_outer_val']['achieved']:.6f} | {v['last']['inner']:.6f} | {v['last']['surfaces']['target_outer_val']['achieved']:.6f} | {r['target_pass_equivalents']:.2f} |")
    lines+=['',f"{len(stage_runs)} fits; {result['stage_process_hours']:.3f} summed process-hours including evaluation/I/O.",
            'The point-estimate source feasibility filter is not formal noninferiority.',
            'These are single-model outcomes, not seed means or score ensembles. No independent confirmation is implied.']
    with (OUT/f'{args.stage.upper()}_RESULTS.md').open('x') as f: f.write('\n'.join(lines)+'\n')
    print(json.dumps({'stage':args.stage,'selected':[{k:r[k] for k in ('label','policy','tag','step','inner','surfaces')} for r in records]},indent=2),flush=True)

if __name__=='__main__': main()
