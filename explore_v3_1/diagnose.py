"""Read-only historical diagnostics; generated results stay in v3.1."""
import json
from v31 import *

def main():
    torch.set_num_threads(2)
    f = pd.read_parquet(CACHE/'index.parquet')
    q = torch.load(CACHE/'prepared.pt', map_location='cpu', weights_only=False)['q0'].numpy()
    mix = []
    for (surface, study), block in f.groupby(['surface','source_study']):
        t = table(block, q[block.row.to_numpy()])
        mix.append({'surface':surface, 'study':study, 'candidates':len(block),
                    'groups':len(t), 'informative_groups':int(((t.depth>=2)&(t.oracle>t.minimum)).sum()),
                    'depth_counts':{str(k):int(v) for k,v in t.depth.value_counts().items()}})
    histories, flips = [], []
    paths = sorted((ROOT/'explore_v3/runs').glob('*/results.json'))
    for p in paths:
        r = json.loads(p.read_text()); h = r['history']
        vals = [x['inner']['selection']['achieved_at_1'] for x in h]
        histories.append({'run':p.parent.name,'arm':r['arm'],'budget':r['budget'],
                          'best_step':r['best_step'], 'initial_inner':vals[0],
                          'last_inner':vals[-1], 'max_inner':max(vals),
                          'last_target_loss':h[-1].get('target_loss'),
                          'last_gradient_norm':h[-1].get('gradient_norm')})
        if r['budget'] != 1000 or r['seed'] != 20260910:
            continue
        for surface in ('target_outer_val','source_val','source_audit'):
            sub = f.loc[f.surface==surface].copy()
            pred = pd.read_parquet(p.parent/f'{surface}.parquet')
            sub = sub.merge(pred[['edit_key','design_key','selection']],
                            left_on=['group_id','design_key'],right_on=['edit_key','design_key'],validate='1:1')
            base = table(sub, q[sub.row.to_numpy()], source=surface.startswith('source'))
            new = table(sub, sub.selection, source=surface.startswith('source'))
            new['delta'] = new.achieved-base.achieved
            new['flip'] = new.winner!=base.winner
            new['depth_bin'] = pd.cut(new.depth,[0,2,4,8,np.inf],labels=['1-2','3-4','5-8','9+'])
            for field in ('study','cell','editor','depth_bin'):
                for name,g in new.groupby(field, observed=True):
                    flips.append({'arm':r['arm'],'surface':surface,'stratum':field,'value':str(name),
                        'groups':len(g),'flips':int(g.flip.sum()), 'harmful':int((g.delta<0).sum()),
                        'beneficial':int((g.delta>0).sum()),'equal_outcome':int((g.delta==0).sum()),
                        'mean_delta':float(g.delta.mean())})
    pools = replay_groups(f)
    result = {'historical_fits':len(paths),'histories':histories,'source_mix':mix,'flips':flips,
              'replay_eligible_groups':{k:len(v) for k,v in pools.items()},
              'selected_step_zero':sum(r['best_step']==0 for r in histories),
              'selected_final_step':sum(r['best_step']==100 for r in histories),
              'limitations':['Unsaved intermediate candidate scores cannot be reconstructed.',
                            'No new model selection using source audit.',
                            'Edit-type and outcome/score-gap analyses deferred; present strata are descriptive.'],
              'inputs_sha256':provenance(paths+[CACHE/'prepared.json', Path(__file__), OUT/'v31.py'])}
    write_json(OUT/'diagnostic.json',result)
    print(json.dumps({k:result[k] for k in ('historical_fits','selected_step_zero','selected_final_step','replay_eligible_groups','source_mix')},indent=2))

if __name__=='__main__': main()
