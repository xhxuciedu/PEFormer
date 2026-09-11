"""Saved-model forward latency on identical pretokenized inputs; no model fitting."""
import argparse
import json
import subprocess
import time
from v31 import *
from e26_adaptation_pilot import load_base


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--physical-gpu',type=int,required=True); args=ap.parse_args()
    status=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
    usage={int(r.split(',')[0]):[int(x) for x in r.split(',')[1:]] for r in status.strip().splitlines()}
    assert usage[args.physical_gpu][0]<=100 and usage[args.physical_gpu][1]<=5, usage
    assert torch.cuda.is_available() and torch.cuda.device_count()==1
    torch.set_num_threads(2); torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    control=json.loads((OUT/'control_selection.json').read_text()); fact=json.loads((OUT/'factorial_selection.json').read_text())
    picks=[('A',control['recipes']['A']['target']['tag'],'target'),
           ('E',control['recipes']['E']['target']['tag'],'target'),('M',fact['candidate'],'constrained')]
    index=pd.read_parquet(CACHE/'index.parquet')
    data=torch.load(CACHE/'prepared.pt',map_location='cpu',weights_only=False)
    ix=index.loc[index.surface=='target_outer_val','row'].to_numpy()[:1024].copy()
    inputs={k:v[ix].cuda() for k,v in data['inputs'].items()}
    records=[]
    for label,tag,policy in picks:
        ck=torch.load(OUT/'runs'/tag/f'{policy}.pt',map_location='cpu',weights_only=False)
        prov=ck['provenance']; base=load_base('cuda').eval(); holder={}
        hook=base.head.register_forward_pre_hook(lambda module,x:holder.update(h=x[0]))
        state=ck['state']; assert state is not None
        if state['base'] is not None: base.load_state_dict(state['base'])
        head=None
        if state['selector'] is not None:
            head=Selector(768,residual=prov['args']['arm']=='E').cuda().eval(); head.load_state_dict(state['selector'])
        def forward():
            result=[]
            for i in range(0,len(ix),128):
                out=base({k:v[i:i+128] for k,v in inputs.items()})
                q=(base.efficiency_from_output(out)-prov['q_mean'])/prov['q_sd']
                result.append(q if head is None else head(holder['h'],q))
            return torch.cat(result)
        with torch.inference_mode():
            for _ in range(2): forward()
            torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
            times=[]
            for _ in range(5):
                start=time.perf_counter(); prediction=forward(); torch.cuda.synchronize()
                times.append(time.perf_counter()-start)
        peak=torch.cuda.max_memory_allocated()
        saved=pd.read_parquet(OUT/'runs'/tag/f'{policy}_target_outer_val.parquet').set_index('row').loc[ix,'selection'].to_numpy()
        error=float(np.max(np.abs(prediction.cpu().numpy()-saved))); assert error<1e-4
        params=sum(p.numel() for p in base.parameters())+(sum(p.numel() for p in head.parameters()) if head is not None else 0)
        records.append({'label':label,'tag':tag,'policy':policy,'selected_step':ck['step'],
            'candidates':len(ix),'batch_size':128,'warmups':2,'repeats':5,'seconds':times,
            'median_ms_per_candidate':float(np.median(times)*1000/len(ix)),
            'median_candidates_per_second':float(len(ix)/np.median(times)),
            'deployed_parameters':params,'peak_allocated_gpu_bytes':peak,'max_score_error':error})
        print(records[-1],flush=True)
        hook.remove(); del base,head,prediction; holder.clear(); torch.cuda.empty_cache()
    write_json(OUT/'inference_cost.json',{'device':torch.cuda.get_device_name(),'precision':'FP32, TF32 disabled',
        'timing_scope':'Full encoder plus deployed readout, synchronized GPU forward on pretokenized GPU-resident inputs; excludes tokenization, input transfer, loading and sorting.',
        'not_claimed':'No matched OptiPrime latency benchmark; hardware timings do not establish a comparator speed advantage.',
        'records':records,'inputs_sha256':provenance([Path(__file__),OUT/'control_selection.json',OUT/'factorial_selection.json'])})


if __name__=='__main__': main()
