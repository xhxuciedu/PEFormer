"""Execute the frozen acquisition/optimizer matrix; reuse exact screen fits."""
import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from queue import Queue, Empty
from v31 import *
from run_stage import tag

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--gpus',nargs='+',type=int,required=True);args=ap.parse_args()
    assert len(args.gpus)==len(set(args.gpus))
    decision=json.loads((OUT/'REPLICATION_DECISION.json').read_text())
    if not decision['proceed']:raise RuntimeError('Replication gate not approved')
    verify=json.loads((OUT/'verification.json').read_text());assert verify['fits_verified']==32
    control=json.loads((OUT/'control_selection.json').read_text())
    fact=json.loads((OUT/'factorial_selection.json').read_text());assert fact['promoted']
    other=control['strongest_other_conventional']['target']
    recipes={a:control['recipes'][a]['target']['args'] for a in ('A',other,'E')}
    recipes['M']=fact['args']
    jobs=[]
    for budget in (200,1000):
        for subset in (20260910,20260911,20260912):
            for seed in (20260910,20260911):
                for label,recipe in recipes.items():
                    a={**recipe,'budget':budget,'subset_seed':subset,'seed':seed}
                    jobs.append({'label':label,'args':a,'tag':tag(a)})
    assert len(jobs)==48 and len({j['tag'] for j in jobs})==48
    reused=[];pending=[]
    for j in jobs:
        p=OUT/'runs'/j['tag']/'results.json'
        if p.exists():
            r=json.loads(p.read_text()); assert r['args']==j['args']
            prov=json.loads((p.parent/'provenance.json').read_text())
            for file in ('explore_v3_1/train_followup.py','explore_v3_1/v31.py','explore_v3_1/EXECUTION_PROTOCOL.md'):
                assert sha256(ROOT/file)==prov['inputs_sha256'][file]
            reused.append(j['tag'])
        else:pending.append(j)
    manifest=OUT/'jobs_replication.json'
    if manifest.exists():assert json.loads(manifest.read_text())['jobs']==jobs
    else:write_json(manifest,{'jobs':jobs,'reused_at_launch':reused,'new_at_launch':len(pending),
         'inputs_sha256':provenance([OUT/'REPLICATION_PROTOCOL.md',OUT/'REPLICATION_DECISION.json',
                    OUT/'control_selection.json',OUT/'factorial_selection.json',Path(__file__)])})
    status=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
    usage={int(r.split(',')[0]):[int(v) for v in r.split(',')[1:]] for r in status.strip().splitlines()}
    if any(usage[g][0]>100 or usage[g][1]>5 for g in args.gpus):raise RuntimeError(f'GPU not idle: {usage}')
    q=Queue()
    for job in pending:q.put(job)
    def worker(gpu):
        records=[]
        while True:
            try:j=q.get_nowait()
            except Empty:break
            cmd=[sys.executable,'-u',str(OUT/'train_followup.py')]
            for k,v in j['args'].items():cmd+=['--'+k.replace('_','-'),str(v)]
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',TMPDIR=str(OUT/'cache/tmp'))
            with (OUT/'logs'/f"{j['tag']}.log").open('x') as f:
                p=subprocess.run(cmd,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT)
            record={'tag':j['tag'],'label':j['label'],'gpu':gpu,'returncode':p.returncode};records.append(record)
            print(record,flush=True)
            if p.returncode:raise RuntimeError(f"Failed {j['tag']}; no automatic overwrite")
        return records
    with ThreadPoolExecutor(max_workers=len(args.gpus)) as pool:
        futures=[pool.submit(worker,g) for g in args.gpus]
        records=[r for f in futures for r in f.result()]
    write_json(OUT/'execution_replication.json',{'completed_new':records,'reused':reused})

if __name__=='__main__':main()
