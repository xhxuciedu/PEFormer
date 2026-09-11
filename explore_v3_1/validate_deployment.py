"""Compare saved deployed models with cached evaluation, including source fallback."""
import argparse
import json
from v31 import *
from e26_adaptation_pilot import load_base

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--tag',required=True);ap.add_argument('--policy',choices=['target','constrained'],required=True);args=ap.parse_args()
    if not torch.cuda.is_available(): raise RuntimeError('GPU validation requested')
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    run=OUT/'runs'/args.tag
    ck=torch.load(run/f'{args.policy}.pt',map_location='cpu',weights_only=False)
    prov=ck['provenance']; a=prov['args'];base=load_base('cuda').eval(); holder={}
    base.head.register_forward_pre_hook(lambda module,x:holder.update(h=x[0]))
    if ck['state'] is not None and ck['state']['base'] is not None:base.load_state_dict(ck['state']['base'])
    head=None
    if ck['state'] is not None and ck['state']['selector'] is not None:
        head=Selector(768,residual=a['arm']=='E').cuda().eval();head.load_state_dict(ck['state']['selector'])
    data=torch.load(CACHE/'prepared.pt',map_location='cpu',weights_only=False)
    index=pd.read_parquet(CACHE/'index.parquet'); checks={}
    for surface in ('target_outer_val','source_audit'):
        saved=pd.read_parquet(run/f'{args.policy}_{surface}.parquet')
        depths=saved.groupby('group_id').size(); groups=sorted(depths[depths>=2].index)[:128]
        f=saved.loc[saved.group_id.isin(groups)];ix=f.row.to_numpy(); outputs=[]
        for batch_size in (31,128):
            values=[]
            with torch.no_grad():
                for start in range(0,len(ix),batch_size):
                    rows=ix[start:start+batch_size]
                    batch={k:v[rows].cuda() for k,v in data['inputs'].items()}
                    out=base(batch);q=(base.efficiency_from_output(out)-prov['q_mean'])/prov['q_sd']
                    if head is not None:q=head(holder['h'],q)
                    values.extend(q.cpu().numpy().tolist())
            outputs.append(np.asarray(values))
        baseline=table(f,f.selection)
        changed=[int((table(f,s).winner!=baseline.winner).sum()) for s in outputs]
        errors=[float(np.max(np.abs(s-f.selection.to_numpy()))) for s in outputs]
        assert max(errors)<1e-4 and changed==[0,0],(surface,errors,changed)
        checks[surface]={'groups':len(groups),'candidates':len(ix),'changed_choices':changed,'max_score_error':errors}
    write_json(OUT/f'deployment_{args.tag}_{args.policy}.json',{'checks':checks,'tag':args.tag,'policy':args.policy,
        'step':ck['step'],'inputs_sha256':provenance([run/f'{args.policy}.pt',Path(__file__)])})
    print(json.dumps(checks,indent=2),flush=True)

if __name__=='__main__':main()
