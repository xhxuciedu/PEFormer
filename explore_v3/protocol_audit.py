"""Audit precision migration, source isolation and independent-panel feasibility."""
import json
import numpy as np
import pandas as pd
import torch
from common import ROOT, OUT, CACHE, write_json, provenance
from e29_summarize_audit import metrics, paired_interval
from e27_adaptation_test import reference_scores

def main():
    torch.set_num_threads(2)
    index = pd.read_parquet(CACHE / "index.parquet")
    data = torch.load(CACHE / "prepared.pt", map_location="cpu", weights_only=False)
    oldpath = ROOT / "explore_v2/cache/adapt_followup_predictions/start_checkpoint__target_val.parquet"
    old = pd.read_parquet(oldpath)
    f = index.loc[index.surface == "target_outer_val", ["row","group_id","design_key","component","y"]].rename(columns={"group_id":"edit_key"})
    f["selection"] = data["q0"][f.row.to_numpy()].numpy()
    f["prediction"] = f.selection
    new_m, new_t = metrics(f)
    old_m, old_t = metrics(old)
    aligned = f.merge(old[["edit_key","design_key","selection"]],on=["edit_key","design_key"],validate="1:1",suffixes=("_new","_old"))
    paired = new_t.merge(old_t,on="edit_key",suffixes=("_new","_old"),validate="1:1")
    op = f.drop(columns=["prediction","selection"]).merge(reference_scores(),on=["edit_key","design_key"],validate="1:1")
    op_m, op_t = metrics(op.assign(prediction=op.op,selection=op.op))
    baseline_pair = new_t.merge(op_t,on="edit_key",suffixes=("_new","_op"),validate="1:1")
    diff = baseline_pair.selection_at_1_new-baseline_pair.selection_at_1_op
    # Illustrative independent-group approximation; not a confirmatory power calculation.
    sigma = float(diff.std(ddof=1))
    runmeta = pd.read_csv(OUT/"data/raw/oped_run_metadata.tsv",sep="\t")
    result = {"inputs_sha256":provenance([CACHE/"prepared.json",oldpath,OUT/"data/raw/oped_run_metadata.tsv",__file__]),
        "precision":{"new_fp32":new_m,"historical_bf16":old_m,
            "candidate_max_abs_delta":float((aligned.selection_new-aligned.selection_old).abs().max()),
            "achieved_delta":float(new_m["selection"]["achieved_at_1"]-old_m["selection"]["achieved_at_1"]),
            "groups_with_changed_selected_outcome":int((paired.selection_at_1_new!=paired.selection_at_1_old).sum())},
        "released_optiprime_outer_validation":op_m,
        "start_minus_op":paired_interval(diff,baseline_pair.site_new),
        "feasibility":{"kim_development_paired_delta_sd":sigma,
            "illustrative_independent_groups_for_delta_0p001_80pct_power":int(np.ceil((1.96+.84)**2*sigma*sigma/.001**2)),
            "illustrative_95pct_halfwidth_at_15_groups":1.96*sigma/np.sqrt(15),
            "warning":"Normal independent-group approximation using Kim development variance; not K562 study-level power; clustering/domain heterogeneity may worsen precision"},
        "oped_metadata":{"runs":len(runmeta),"compressed_fastq_bytes":int(runmeta.fastq_bytes.str.split(";").explode().astype(np.int64).sum()),
            "sample_titles":runmeta.sample_title.tolist(),"raw_reads_downloaded":False,
            "missing":"Verified mapping from multiplexed runs/read barcodes to pegRNA, PE2 condition and biological replicate"}}
    write_json(OUT/"protocol_audit.json",result)
    print(json.dumps(result,indent=2),flush=True)

if __name__ == "__main__":
    main()
