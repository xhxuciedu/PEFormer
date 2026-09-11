"""Matched frozen-head controls and prespecified uniform own-model ensembles."""
import json
import numpy as np
import pandas as pd
import torch
from common import OUT,CACHE,provenance,write_json
from summarize import group_table,contrast
from e27_adaptation_test import reference_scores

SURFACES=("target_outer_val","source_val","source_audit")

def read_predictions(folder,surface):
    return pd.read_parquet(folder/f"{surface}.parquet").sort_values(["edit_key","design_key"]).reset_index(drop=True)

def main():
    torch.set_num_threads(2)
    index=pd.read_parquet(CACHE/"index.parquet")
    data=torch.load(CACHE/"prepared.pt",map_location="cpu",weights_only=False)
    base={}
    for surface in SURFACES:
        f=index.loc[index.surface==surface,["row","group_id","design_key","component","y"]].rename(columns={"group_id":"edit_key"}).copy()
        f["selection"]=data["q0"].numpy()[f.row.to_numpy()]
        base[surface]=group_table(f,source=surface.startswith("source"))
        if surface=="target_outer_val":
            op=f.drop(columns="selection").merge(reference_scores(),on=["edit_key","design_key"],validate="1:1")
            base['optiprime']=group_table(op.assign(selection=op.op))
    controls,ensembles,paths=[],[],[]
    for budget in (200,1000):
        for arm,refarm in (("C_frozen","E"),("F_replay","F")):
            folder=OUT/"runs"/f"{arm}_b{budget}_sub20260910_s20260910"
            reference=OUT/"runs"/f"{refarm}_b{budget}_sub20260910_s20260910"
            # Completion marker first: do not read partial fits.
            json.loads((folder/"results.json").read_text())
            row={'budget':budget,'arm':arm,'reference':refarm,'surfaces':{},'contrasts':{}}
            for surface in SURFACES:
                f=read_predictions(folder,surface)
                r=read_predictions(reference,surface)
                a=group_table(f,source=surface.startswith('source'))
                b=group_table(r,source=surface.startswith('source'))
                row['surfaces'][surface]={'achieved':float(a.achieved.mean()),'regret':float(a.regret.mean())}
                row['contrasts'][surface+'_minus_'+refarm]=contrast(a,b)
                row['contrasts'][surface+'_minus_start']=contrast(a,base[surface])
                if surface=='target_outer_val':
                    row['contrasts']['target_minus_optiprime']=contrast(a,base['optiprime'])
                paths.extend([folder/f'{surface}.parquet',reference/f'{surface}.parquet'])
            controls.append(row)
        for arm in ('A','D','E','F'):
            row={'budget':budget,'arm':arm,'surfaces':{},'contrasts':{}}
            for surface in SURFACES:
                frames=[]
                for seed in (20260910,20260911,20260912):
                    folder=OUT/'runs'/f'{arm}_b{budget}_sub20260910_s{seed}'
                    frames.append(read_predictions(folder,surface))
                    paths.append(folder/f'{surface}.parquet')
                first=frames[0].copy()
                for f in frames[1:]:
                    if not first[['edit_key','design_key','component','y']].equals(f[['edit_key','design_key','component','y']]):
                        raise ValueError('Ensemble input/order mismatch')
                first['selection']=np.mean([f.selection.to_numpy(dtype=np.float64) for f in frames],axis=0)
                first['prediction']=np.mean([f.prediction.to_numpy(dtype=np.float64) for f in frames],axis=0)
                a=group_table(first,source=surface.startswith('source'))
                row['surfaces'][surface]={'achieved':float(a.achieved.mean()),'regret':float(a.regret.mean())}
                row['contrasts'][surface+'_minus_start']=contrast(a,base[surface])
                if surface=='target_outer_val':
                    row['contrasts']['target_minus_optiprime']=contrast(a,base['optiprime'])
                dest=OUT/'cache'/f'ensemble_{arm}_b{budget}_{surface}.parquet'
                if dest.exists():
                    raise FileExistsError(dest)
                first.to_parquet(dest,index=False)
            ensembles.append(row)
    result={'note':'Exploratory post-replication checks. Ensembles use all three seeds with equal score weights, not OptiPrime. No weight tuning.',
            'controls':controls,'ensembles':ensembles,'inputs_sha256':provenance(paths+[OUT/'CONTROL_DECISION.md',__file__])}
    write_json(OUT/'control_results.json',result)
    lines=['# V3 matched controls and uniform ensembles','',result['note'],'',
        '## Matched frozen-backbone controls','',
        '| Budget | Control | Reference | Target @1 | Target difference [95% CI] | Source @1 | Source difference [95% CI] |',
        '|---:|---|---|---:|---|---:|---|']
    for r in controls:
        t=r['contrasts']['target_outer_val_minus_'+r['reference']]
        s=r['contrasts']['source_audit_minus_'+r['reference']]
        lines.append(f"| {r['budget']} | {r['arm']} | {r['reference']} | {r['surfaces']['target_outer_val']['achieved']:.6f} | {t['observed']:+.6f} [{t['ci95'][0]:+.6f}, {t['ci95'][1]:+.6f}] | {r['surfaces']['source_audit']['achieved']:.6f} | {s['observed']:+.6f} [{s['ci95'][0]:+.6f}, {s['ci95'][1]:+.6f}] |")
    lines += ['', 'Controls have one seed. C_frozen removes encoder-update/capacity differences from E,',
        'but not initialization or initial score-scale differences. Source-label replay and soft-teacher',
        'KL have different gradient scales; equal weights do not establish mechanistic equivalence.',
        '', '## Equal-weight three-seed score ensembles', '',
        '| Budget | Arm | Target @1 | Difference vs OptiPrime [95% CI] | Source @1 | Source change [95% CI] |',
        '|---:|---|---:|---|---:|---|']
    for r in ensembles:
        t=r['contrasts']['target_minus_optiprime'];s=r['contrasts']['source_audit_minus_start']
        lines.append(f"| {r['budget']} | {r['arm']} | {r['surfaces']['target_outer_val']['achieved']:.6f} | {t['observed']:+.6f} [{t['ci95'][0]:+.6f}, {t['ci95'][1]:+.6f}] | {r['surfaces']['source_audit']['achieved']:.6f} | {s['observed']:+.6f} [{s['ci95'][0]:+.6f}, {s['ci95'][1]:+.6f}] |")
    lines += ['', 'These are actual score ensembles, distinct from mean seed outcomes. A/D need three',
        'adapted backbones; E/F can share one frozen backbone with three scalar heads. No fitted',
        'ensemble weights and no additional target labels. Intervals are exploratory, unadjusted',
        'component bootstrap intervals on the same exposed development evaluation.']
    (OUT/'CONTROL_RESULTS.md').write_text('\n'.join(lines)+'\n')
    print('Control and ensemble results written',flush=True)

if __name__=='__main__':
    main()
