"""Launch frozen job manifests only on explicitly idle GPUs."""
import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from v31 import OUT, ROOT, write_json

def tag(a):
    return f"{a['arm']}_h{a['steps']}_lr{a['lr_mult']}_{a.get('source_loss','none')}_{a.get('sampling','mixture')}_w{a.get('weight',1.):g}_b{a.get('budget',1000)}_sub{a.get('subset_seed',20260910)}_s{a.get('seed',20260910)}"

def worker(gpu,jobs):
    results=[]
    tempdir=OUT/'cache'/'tmp'; tempdir.mkdir(parents=True,exist_ok=True)
    for a in jobs:
        name=tag(a)
        if (OUT/'runs'/name/'results.json').exists():
            print('SKIP completed',name,flush=True); continue
        log=OUT/'logs'/f'{name}.log'; log.parent.mkdir(exist_ok=True)
        cmd=[sys.executable,'-u',str(OUT/'train_followup.py')]
        for k,v in a.items(): cmd += ['--'+k.replace('_','-'),str(v)]
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',TMPDIR=str(tempdir))
        with log.open('x') as stream:
            p=subprocess.run(cmd,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT)
        results.append({'tag':name,'gpu':gpu,'returncode':p.returncode})
        print(results[-1],flush=True)
        if p.returncode: raise RuntimeError(f'Failed {name}; inspect log, no overwrite/retry')
    return results

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--stage',choices=['controls','factorial'],required=True)
    ap.add_argument('--gpu',type=int,required=True)
    ap.add_argument('--frozen-gpu',type=int)
    args=ap.parse_args()
    status=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
    usage={int(r.split(',')[0]):[int(v) for v in r.split(',')[1:]] for r in status.strip().splitlines()}
    devices={args.gpu} if args.frozen_gpu is None else {args.gpu,args.frozen_gpu}
    if any(usage[g][0]>100 or usage[g][1]>5 for g in devices): raise RuntimeError(f'GPU not idle: {usage}')
    if args.stage=='controls':
        jobs=[{'arm':a,'steps':h,'lr_mult':lr} for h in (100,500) for lr in (1,3) for a in ('A','B','D','C_frozen','E')]
    else:
        if (OUT/'e_recipe.json').exists():
            recipe=json.loads((OUT/'e_recipe.json').read_text())['args']
        else:
            decision=json.loads((OUT/'control_selection.json').read_text())
            recipe=decision['recipes']['E']['target']['args']
        jobs=[{'arm':'E','steps':recipe['steps'],'lr_mult':recipe['lr_mult'],
               'source_loss':loss,'sampling':sampling,'weight':w}
              for loss in ('kl','margin','utility') for sampling in ('mixture','balanced') for w in (.1,1.)]
    path=OUT/f'jobs_{args.stage}.json'
    if path.exists():
        if json.loads(path.read_text())!=jobs: raise ValueError('Job manifest changed')
    else: write_json(path,jobs)
    if args.frozen_gpu is None or args.frozen_gpu==args.gpu:
        result=worker(args.gpu,jobs)
    else:
        frozen=[j for j in jobs if j['arm'] in ('E','C_frozen')]
        full=[j for j in jobs if j['arm'] not in ('E','C_frozen')]
        with ThreadPoolExecutor(max_workers=2) as pool:
            fut=[pool.submit(worker,args.gpu,full),pool.submit(worker,args.frozen_gpu,frozen)]
            result=[r for f in fut for r in f.result()]
    write_json(OUT/f'execution_{args.stage}.json',result)

if __name__=='__main__': main()
