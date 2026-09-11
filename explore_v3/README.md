# V3 computational research

Start with [the completed-stage research report](RESEARCH_REPORT.md),
[the next-step plan](NEXT_STEPS.md), [the original research plan](RESEARCH_PLAN.md), the frozen initial
[execution protocol](EXECUTION_PROTOCOL.md), and [data eligibility audit](DATA_AUDIT.md).
This directory is separate from historical v2 artifacts. Large datasets,
representations and checkpoints are local reproducibility products, not part of
the public manuscript package. In particular, do not redistribute the privately
supplied source corpus or sequence-level derivatives without permission.

## Reproduction

From the repository root, using the existing environment:

```bash
.venv/bin/python explore_v3/acquire_data.py
.venv/bin/python explore_v3/audit_data.py
CUDA_VISIBLE_DEVICES=2 .venv/bin/python explore_v3/prepare.py
.venv/bin/python -m pytest explore_v3/test_protocol.py explore_v2/test_adaptation_followup.py explore_v2/test_endpoints.py -q
.venv/bin/python explore_v3/run_screen.py --gpus 2 7
.venv/bin/python explore_v3/protocol_audit.py
.venv/bin/python explore_v3/summarize.py
.venv/bin/python explore_v3/run_screen.py --gpus 2 7 --arms A D E F --seeds 20260911 20260912 --label replication
.venv/bin/python explore_v3/summarize_replication.py
.venv/bin/python explore_v3/run_screen.py --gpus 2 7 --arms C_frozen F_replay --label controls
.venv/bin/python explore_v3/analyze_controls.py
.venv/bin/python explore_v3/diagnose_retention.py
CUDA_VISIBLE_DEVICES=2 .venv/bin/python explore_v3/validate_deployment.py
.venv/bin/python explore_v3/verify_artifacts.py
```

Network and GPU access may require environment approval. The launcher checks the
requested devices are idle. It never stops unrelated processes. Existing run
directories and preparation outputs are not overwritten; inspect failures before
resuming. Completed runs are skipped by the launcher. Training logs are in
`logs/`; checkpoint selection uses only total-budget inner validation.

The initial experiment matrix has seven arms at each of two budgets, one paired
training seed. Each run records all 100 target updates, validation checkpoints,
actual labelled-candidate costs, additional source replay, hardware, timing and
input/output hashes. Step zero is deployment of the original source predictor,
including for arms whose newly initialized head is not itself a useful model.

Completed execution adds 16 replication fits and four matched controls (34 total).
Run `summarize.py` immediately after the initial 14 fits to reproduce the historical
screen snapshot. Its immutable output is not overwritten by later analyses.

The data audit does not yet establish the planned two independent deep-selection
studies. Initial Kim results remain development evidence, not new independent
confirmation. No new target-test scoring occurs in these commands.
