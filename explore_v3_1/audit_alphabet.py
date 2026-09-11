"""Resolve RNA/DNA identity and legacy token conventions without changing fits."""
import json
from v31 import *
from pe_rankformer.data.tokenizer import NUC_TOKEN_TO_ID, encode_pegrna
from audit_external import min_hamming

def dna(s): return s.upper().replace('U','T')

def main():
    torch.set_num_threads(2)
    source=pd.read_parquet(ROOT/'explore_v2/cache/corpus_canonical_v2.parquet',columns=['spacer','pbs','rtt','source_study'])
    meta=[]
    for study,g in source.groupby('source_study'):
        for field in ('spacer','pbs','rtt'):
            col=g[field].str.upper()
            meta.append({'study':study,'field':field,'rows':len(g),'contains_U':int(col.str.contains('U').sum()),
                         'contains_T':int(col.str.contains('T').sum()),'contains_N':int(col.str.contains('N').sum()),
                         'characters':sorted(set(''.join(col.unique())))})
    data=torch.load(CACHE/'prepared.pt',map_location='cpu',weights_only=False)
    index=pd.read_parquet(CACHE/'index.parquet'); tokens=data['inputs']['peg_nuc_ids'].numpy()
    token_counts={surface:{base:int((tokens[f.row.to_numpy()]==NUC_TOKEN_TO_ID[base]).sum()) for base in ('T','N')}
                  for surface,f in index.groupby('surface')}
    ep=pd.read_parquet(OUT/'data/epridict_input_audit.parquet')
    ref=source.spacer.map(dna).str[-19:]
    distance,n=min_hamming(ep.spacer.map(dna).str[-19:].tolist(),ref.unique(),19)
    ep['source_core19_min_hamming']=distance
    ep['local_homology_clear']=(ep.source_core19_min_hamming>2)&(ep.source_flanks24_min_hamming>2)
    ep.to_parquet(OUT/'data/epridict_input_audit_alphabet_corrected.parquet',index=False)
    populations={}
    for cell in ('K562','HEK293T'):
        sub=ep.loc[ep.token_safe&ep.local_homology_clear&ep[f'has_{cell}_outcome']]
        depth=sub.groupby('edit_key').guide.nunique()
        populations[cell]={'rows':len(sub),'alleles':len(depth),'groups_ge2':int((depth>=2).sum()),
                           'max_depth':int(depth.max()) if len(depth) else 0}
    # Check cross-surface sequence isolation after alphabet normalization, not raw strings.
    index['dna_core']=index.spacer.map(dna).str[-19:]
    replay=index.loc[index.surface.isin(['source_replay','source_val'])]
    isolation={}
    for surface in ('target_budget_pool','target_outer_val','source_audit'):
        other=index.loc[index.surface==surface]
        isolation[surface]={'overlapping_normalized_core19':len(set(other.dna_core)&set(replay.dna_core)),
                            'note':'Core19 matches are more conservative than original full-spacer identity.'}
    result={'source_alphabets':meta,'cached_peg_token_counts':token_counts,
            'token_example':{s:encode_pegrna(s,'','',6).nuc_ids for s in ('ACGU','ACGT','ACGN')},
            'normalized_source_unique_cores':n,'external_exact_core19_matches':int((ep.source_core19_min_hamming==0).sum()),
            'external_local_homology_flagged':int((~ep.local_homology_clear).sum()),'corrected_populations':populations,
            'cross_surface_isolation':isolation,
            'interpretation':'Legacy tokenizer maps RNA U to N, not DNA T. This is an injective encoding for unambiguous ACGU inputs, but DNA input is not token-equivalent; preserve checkpoint convention or retrain under a versioned normalization. No current training inputs changed.',
            'supersedes':'data_feasibility.json epridict source-core homology counts and combined eligibility only; reconstruction and flank checks unchanged.',
            'inputs_sha256':provenance([Path(__file__),OUT/'audit_external.py',OUT/'data_feasibility.json',CACHE/'prepared.json'])}
    write_json(OUT/'alphabet_audit.json',result)
    print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__':main()
