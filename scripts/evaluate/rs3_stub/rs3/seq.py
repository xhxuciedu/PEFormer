"""Stub: the real rs3 package needs scikit-learn<=1.0.2, incompatible with this
environment's numpy/torch stack. All RuleSet3Score hashes needed for this run were
precomputed with an isolated rs3 environment and written to the OptiPrime disk cache
(see scripts/evaluate/precompute_ruleset3_cache.py), so this function is never actually
called -- it only needs to exist so the top-level `from rs3.seq import predict_seq`
import in scripts/pe/pe_inputs.py succeeds."""


def predict_seq(*args, **kwargs):
    raise RuntimeError(
        "rs3.seq.predict_seq stub called directly -- this means a spacer_hash was "
        "missing from the precomputed RuleSet3Score disk cache. Run "
        "scripts/evaluate/precompute_ruleset3_cache.py (in the isolated rs3 env) first."
    )
