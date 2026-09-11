"""Label accounting and overlap of the six locked acquisition manifests."""
import json
from itertools import combinations
from v31 import *
from train import budget_indices
from prepare import BUDGETS


def main():
    frame=pd.read_parquet(CACHE/'index.parquet'); budgets=pd.read_parquet(BUDGETS)
    rows=[]; groups={}; all_rows=set()
    for budget in (200,1000):
        for seed in (20260910,20260911,20260912):
            roles=budget_indices(frame,budgets,budget,seed)
            ix=np.sort(np.concatenate(list(roles.values()))); all_rows.update(ix.tolist())
            groups[(budget,seed)]=set(frame.iloc[ix].gid)
            row={'nominal_groups':budget,'subset_seed':seed,'roles':{}}
            for role,part in {**roles,'total':ix}.items():
                f=frame.iloc[part]
                row['roles'][role]={'groups':int(f.gid.nunique()),'candidates':len(f),
                    'measurements':int(f.n_meas.sum()),'components':int(f.component.nunique())}
            rows.append(row)
    overlaps=[]
    for budget in (200,1000):
        for a,b in combinations((20260910,20260911,20260912),2):
            x,y=groups[(budget,a)],groups[(budget,b)]
            overlaps.append({'budget':budget,'subset_a':a,'subset_b':b,'shared_groups':len(x&y),
                             'group_jaccard':len(x&y)/len(x|y)})
    union=frame.iloc[sorted(all_rows)]
    assert set(union.surface)=={'target_budget_pool'}
    val=frame.loc[frame.surface=='target_outer_val']
    assert not set(union.component)&set(val.component)
    result={'acquisitions':rows,'overlap':overlaps,
        'union_across_budgets_and_subsets':{'groups':int(union.gid.nunique()),'candidates':len(union),
            'measurements':int(union.n_meas.sum()),'components':int(union.component.nunique())},
        'additional_exposed_outer_validation':{'groups':int(val.gid.nunique()),'candidates':len(val)},
        'note':'Union counts do not include source/pretraining labels or the rest of historical target development. Repeated optimizer fits reuse the same labels.',
        'inputs_sha256':provenance([Path(__file__),BUDGETS,CACHE/'prepared.json'])}
    write_json(OUT/'acquisition_audit.json',result)
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
