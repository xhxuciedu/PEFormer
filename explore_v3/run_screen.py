"""Run the prespecified bounded screen on explicitly selected idle GPUs."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import os
import subprocess
import sys
from common import ROOT, OUT, write_json

def worker(gpu, jobs):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS="4", MKL_NUM_THREADS="4")
    results = []
    logdir = OUT / "logs"
    logdir.mkdir(exist_ok=True)
    for arm, budget, seed in jobs:
        tag = f"{arm}_b{budget}_sub20260910_s{seed}"
        if (OUT / "runs" / tag / "results.json").exists():
            print("already complete", tag, flush=True)
            continue
        log = logdir / f"{tag}.log"
        with log.open("x") as stream:
            p = subprocess.run([sys.executable,"-u",str(OUT / "train.py"),"--arm",arm,
                                "--budget",str(budget),"--seed",str(seed)],
                               cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
        results.append({"tag":tag,"gpu":gpu,"returncode":p.returncode})
        print(results[-1], flush=True)
        if p.returncode:
            raise RuntimeError(f"Failed {tag}; inspect {log}, no automatic overwrite")
    return results

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpus", nargs="+", type=int, default=[2,7])
    ap.add_argument("--arms", nargs="+", default=list("ABCDEFG"))
    ap.add_argument("--seeds", nargs="+", type=int, default=[20260910])
    ap.add_argument("--label", default="initial")
    args = ap.parse_args()
    status = subprocess.check_output(["nvidia-smi","--query-gpu=index,memory.used,utilization.gpu",
                                      "--format=csv,noheader,nounits"],text=True)
    usage = {int(row.split(",")[0]): [int(v) for v in row.split(",")[1:]]
             for row in status.strip().splitlines()}
    if any(usage[g][0] > 100 or usage[g][1] > 5 for g in args.gpus):
        raise RuntimeError(f"Requested GPU not idle: {usage}")
    jobs = [(arm,b,seed) for b in (200,1000) for seed in args.seeds for arm in args.arms]
    with ThreadPoolExecutor(max_workers=len(args.gpus)) as pool:
        fut = [pool.submit(worker,g,jobs[i::len(args.gpus)]) for i,g in enumerate(args.gpus)]
        result = [row for f in fut for row in f.result()]
    write_json(OUT / f"execution_{args.label}.json", result)

if __name__ == "__main__":
    main()
