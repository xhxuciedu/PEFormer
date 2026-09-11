"""Check cached/head training against end-to-end, batched deployed predictions."""
import argparse
import numpy as np
import pandas as pd
import torch
from common import OUT,CACHE,write_json,provenance
from inference import Deployed

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--run",default="E_b1000_sub20260910_s20260910")
    args=ap.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Use an idle GPU for the end-to-end deployment check")
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    run=OUT/"runs"/args.run
    model=Deployed(run,"cuda")
    index=pd.read_parquet(CACHE/"index.parquet")
    data=torch.load(CACHE/"prepared.pt",map_location="cpu",weights_only=False)
    checks={}
    for surface in ("target_outer_val","source_audit"):
        sub=index.loc[index.surface==surface].copy()
        # Outcome-independent first 128 complete groups in deterministic group order.
        depth=sub.groupby("group_id",sort=True).size()
        groups=depth.loc[depth>=2].index[:128]
        sub=sub.loc[sub.group_id.isin(groups)]
        ix=sub.row.to_numpy(copy=True)
        cached=pd.read_parquet(run/f"{surface}.parquet")
        aligned=sub.merge(cached[["edit_key","design_key","selection"]],left_on=["group_id","design_key"],right_on=["edit_key","design_key"],validate="1:1")
        values=[]
        for bs in (31,128):
            score=[]
            with torch.no_grad():
                for a in range(0,len(ix),bs):
                    b=ix[a:a+bs]
                    batch={k:v[b].cuda() for k,v in data["inputs"].items()}
                    score.append(model(batch).cpu().numpy())
            values.append(np.concatenate(score))
        err=float(np.max(np.abs(values[0]-aligned.selection.to_numpy())))
        batcherr=float(np.max(np.abs(values[0]-values[1])))
        assert err<1e-5 and batcherr<1e-5,(surface,err,batcherr)
        winners=[]
        for score in (aligned.selection.to_numpy(),*values):
            f=sub.assign(score=score).sort_values(["group_id","score","design_key"],ascending=[True,False,True])
            winners.append(f.drop_duplicates("group_id").design_key.to_numpy())
        changes=[int((winners[0]!=w).sum()) for w in winners[1:]]
        assert not any(changes),(surface,changes)
        checks[surface]={"candidates":len(ix),"groups":len(groups),"max_cached_vs_end_to_end_error":err,
                         "max_batch_size_error":batcherr,"changed_choices":changes}
    write_json(OUT/f"deployment_check_{args.run}.json",{"checks":checks,"inputs_sha256":provenance([
        run/"best.pt",run/"provenance.json",CACHE/"prepared.json",OUT/"inference.py",__file__])})
    print(checks,flush=True)

if __name__=="__main__":
    main()
