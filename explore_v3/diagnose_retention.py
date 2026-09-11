"""Post-fit diagnosis: does soft teacher agreement imply top-choice retention?"""
import json
import numpy as np
import pandas as pd
import torch
from scipy.special import expit
from common import OUT,CACHE,write_json,provenance
from model import Selector

def quantiles(x):
    return {str(q):float(np.quantile(x,q)) for q in (.1,.5,.9)}

def main():
    torch.set_num_threads(2)
    f=pd.read_parquet(CACHE/"index.parquet")
    data=torch.load(CACHE/"prepared.pt",map_location="cpu",weights_only=False)
    rows=[]
    input_paths=[CACHE/"prepared.json",__file__]
    for arm in ("E","F"):
        run=OUT/"runs"/f"{arm}_b1000_sub20260910_s20260910"
        meta=json.loads((run/"provenance.json").read_text())
        ck=torch.load(run/"best.pt",map_location="cpu",weights_only=False)
        input_paths.extend([run/"best.pt",run/"provenance.json"])
        head=Selector(data['h'].shape[1],residual=True).eval()
        if ck['best_step']:
            head.load_state_dict(ck['state']['selector'])
        q=(data['q0']-meta['q_mean'])/meta['q_sd']
        with torch.no_grad():
            s=torch.cat([head(data['h'][a:a+2048],q[a:a+2048]) for a in range(0,len(f),2048)]).numpy()
        teacher=q.numpy()
        for surface in ('source_replay','source_val','source_audit','target_outer_val'):
            records=[]
            sub=f.loc[f.surface==surface]
            for gid,g in sub.groupby('gid',sort=False):
                if len(g)<2:
                    continue
                ix=g.row.to_numpy()
                yy=g.y.to_numpy()
                tt,ss=teacher[ix],s[ix]
                i,j=np.triu_indices(len(g),1)
                td,sd=tt[i]-tt[j],ss[i]-ss[j]
                p=expit(td.astype(np.float64))
                kl=np.logaddexp(0,sd)-p*sd-(np.logaddexp(0,td)-p*td)
                a,b=int(np.argmax(tt)),int(np.argmax(ss))
                gap=float(np.sort(tt)[-1]-np.sort(tt)[-2])
                records.append({'gid':gid,'study':g.source_study.iloc[0],
                    'informative':bool(yy.max()>yy.min()),'teacher_top_gap':gap,
                    'teacher_top_pair_probability':float(expit(gap)),
                    'mean_pair_kl':max(0.,float(kl.mean())), 'changed_choice':a!=b,
                    'utility_change':float(yy[b]-yy[a])})
            r=pd.DataFrame(records)
            for study in ('all',*sorted(r.study.unique())):
                z=r if study=='all' else r.loc[r.study==study]
                inf=z.loc[z.informative]
                rows.append({'arm':arm,'surface':surface,'study':study,'groups':len(z),
                    'informative_groups':len(inf),'teacher_top_gap_quantiles':quantiles(z.teacher_top_gap),
                    'teacher_top_pair_probability_median':float(z.teacher_top_pair_probability.median()),
                    'teacher_top_pair_probability_below_0p55':float((z.teacher_top_pair_probability<.55).mean()),
                    'mean_pair_kl':float(z.mean_pair_kl.mean()),
                    'choice_change_fraction':float(z.changed_choice.mean()),
                    'informative_utility_change':float(inf.utility_change.mean()) if len(inf) else None})
    write_json(OUT/'retention_diagnostic.json',{'note':'Post-fit, first-seed diagnostic only; not causal proof or a new hyperparameter selection rule.',
        'rows':rows,'inputs_sha256':provenance(input_paths)})
    print(json.dumps([r for r in rows if r['study']=='all'],indent=2),flush=True)

if __name__=='__main__':
    main()
