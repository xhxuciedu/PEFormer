"""Quantify the newly discovered core19 sensitivity without changing primary splits."""
import json
from v31 import *
from audit_alphabet import dna

def main():
    f=pd.read_parquet(CACHE/'index.parquet');f['dna_core']=f.spacer.map(dna).str[-19:]
    source=f.loc[f.surface.isin(['source_replay','source_val'])]
    target=f.loc[f.surface=='target_outer_val']
    shared=set(source.dna_core)&set(target.dna_core)
    linked=target.loc[target.dna_core.isin(shared)]
    # Exclude whole target components, rather than individual candidate rows.
    components=set(linked.component)
    affected=target.loc[target.component.isin(components)]
    sp=source.loc[source.dna_core.isin(shared)]
    out={'shared_cores':len(shared),'direct_target_groups':int(linked.group_id.nunique()),
         'whole_target_components':len(components),'excluded_target_groups':int(affected.group_id.nunique()),
         'excluded_target_candidates':len(affected),'target_components':sorted(components),
         'source_rows':len(sp),'source_groups':int(sp.group_id.nunique()),
         'source_surfaces':sp.groupby('surface').agg(rows=('gid','size'),groups=('gid','nunique')).reset_index().to_dict('records'),
         'exact_full_spacer_match':bool(set(sp.spacer)&set(linked.spacer)),
         'exact_normalized_full_spacer_match':bool(set(sp.spacer.map(dna))&set(linked.spacer.map(dna))),
         'exact_allele_match':bool(set(sp.edit_key)&set(linked.edit_key)),
         'interpretation':'Local core19 match despite original full-spacer/allele isolation. Exclude whole connected evaluation components in a secondary sensitivity, keeping the frozen primary population unchanged.',
         'inputs_sha256':provenance([Path(__file__),OUT/'alphabet_audit.json',CACHE/'prepared.json'])}
    write_json(OUT/'core_overlap_audit.json',out)
    print(json.dumps(out,indent=2),flush=True)

if __name__=='__main__':main()
