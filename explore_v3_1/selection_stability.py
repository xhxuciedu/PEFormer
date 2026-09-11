"""Post-screen validation-selection diagnostics; do not change frozen recipes."""
import json
from v31 import *
from e29_summarize_audit import paired_interval

def main():
    torch.set_num_threads(2)
    selection=json.loads((OUT/'control_selection.json').read_text())
    factorial=json.loads((OUT/'factorial_selection.json').read_text())
    manifest=json.loads((OUT/'jobs_controls.json').read_text())
    from run_stage import tag
    index=pd.read_parquet(CACHE/'index.parquet')
    q0=torch.load(CACHE/'prepared.pt',map_location='cpu',weights_only=False)['q0'].numpy()
    stability=[]
    for arm in ('A','B','D','C_frozen','E'):
        candidates=[]; names=[];components=None;groupindex=None
        for job in sorted([j for j in manifest if j['arm']==arm],key=lambda j:(j['steps'],j['lr_mult'])):
            name=tag(job); f=pd.read_parquet(OUT/'runs'/name/'checkpoint_predictions.parquet')
            for step,g in f.loc[f.surface=='inner_val'].groupby('step',sort=True):
                t=table(index.iloc[g.row.to_numpy()],g.score.to_numpy())
                if groupindex is None:groupindex=t.index;components=t.component.to_numpy()
                else:assert t.index.equals(groupindex)
                candidates.append(t.achieved.to_numpy());names.append(f'{name}:step{step}')
        values=np.stack(candidates)
        unique,code=np.unique(components,return_inverse=True);counts=np.bincount(code)
        sums=np.stack([np.bincount(code,weights=v) for v in values])
        observed=int(np.argmax(values.mean(axis=1)));rng=np.random.default_rng(20260911)
        chosen=[]
        for _ in range(2000):
            draw=rng.integers(0,len(unique),size=len(unique))
            chosen.append(int(np.argmax(sums[:,draw].sum(axis=1)/counts[draw].sum())))
        counts_choice=np.bincount(chosen,minlength=len(names))
        stability.append({'arm':arm,'observed_winner':names[observed],'states_including_duplicates':len(names),
            'same_exact_state_fraction':float((np.asarray(chosen)==observed).mean()),
            'most_frequent_bootstrap_states':[{'state':names[i],'fraction':int(counts_choice[i])/2000}
                 for i in np.argsort(-counts_choice)[:5]],
            'note':'Bootstrap selection stability on fixed inner predictions, not retrained or independent validation.'})
    candidate=factorial['candidate']; f=pd.read_parquet(OUT/'runs'/candidate/'constrained_source_val.parquet')
    a,b=table(f,f.selection,True),table(f,q0[f.row.to_numpy()],True)
    domains={}
    for study,g in a.groupby('study'):
        domains[str(study)]=paired_interval(g.achieved-b.loc[g.index,'achieved'],g.component)
    write_json(OUT/'selection_stability.json',{'controls':stability,'promoted_candidate':candidate,
         'source_validation_change_by_study':domains,
         'caveat':'All intervals are post-selection descriptive. They do not validate the .001 feasibility filter or alter the locked recipe.',
         'inputs_sha256':provenance([Path(__file__),OUT/'control_selection.json',OUT/'factorial_selection.json'])})
    print(json.dumps({'controls':stability,'source_validation_change_by_study':domains},indent=2),flush=True)

if __name__=='__main__':main()
