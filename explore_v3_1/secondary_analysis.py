"""Post-screen descriptive mechanisms, depth endpoints and overlap sensitivity."""
import json
from scipy.stats import spearmanr
from v31 import *
from e27_adaptation_test import reference_scores
from e29_summarize_audit import paired_interval

def deeper(f,k):
    d=f.groupby('group_id').size(); eligible=d[d>=k].index
    sub=f.loc[f.group_id.isin(eligible)].sort_values(['group_id','selection','design_key'],ascending=[True,False,True])
    top=sub.groupby('group_id',sort=False).head(k).groupby('group_id').y.max()
    return {'k':k,'eligible_groups':len(top),'best_measured_outcome_among_top_k':float(top.mean()) if len(top) else None}

def main():
    controls=json.loads((OUT/'controls_results.json').read_text())
    factorial=json.loads((OUT/'factorial_results.json').read_text())
    selection=json.loads((OUT/'control_selection.json').read_text())
    fs=json.loads((OUT/'factorial_selection.json').read_text())
    candidate=fs['candidate']
    picks=[(a,selection['recipes'][a]['target']['tag'],'target') for a in ('A','B','D','C_frozen','E')]
    picks += [('factorial_inner_candidate',candidate,p) for p in ('target','constrained')]
    overlap=json.loads((OUT/'core_overlap_audit.json').read_text())['target_components']
    core_rows=[]; metrics=[]
    for label,tag,policy in picks:
        f=pd.read_parquet(OUT/'runs'/tag/f'{policy}_target_outer_val.parquet')
        op=f.merge(reference_scores(),left_on=['group_id','design_key'],right_on=['edit_key','design_key'],validate='1:1')
        for pop,block in [('primary',op),('core19_component_excluded',op.loc[~op.component.isin(overlap)])]:
            a,b=table(block,block.selection),table(block,block.op)
            core_rows.append({'label':label,'tag':tag,'policy':policy,'population':pop,
                             **paired_interval(a.achieved-b.achieved,a.component)})
        for method,score in [(label,f.selection),('OptiPrime',op.op)]:
            block=f.assign(selection=score.to_numpy())
            metrics.append({'label':method,'policy':policy,'tag':tag,
                'pooled_spearman':float(spearmanr(block.selection,block.y).statistic),
                'depth_endpoints':[deeper(block,k) for k in (1,3,5)]})
    trajectories=[]; sampling=[]; scale=[]
    from run_stage import tag as run_tag
    initial_jobs=sum([json.loads((OUT/f'jobs_{stage}.json').read_text()) for stage in ('controls','factorial')],[])
    for p in sorted(OUT/'runs'/run_tag(job)/'results.json' for job in initial_jobs):
        r=json.loads(p.read_text());prov=json.loads((p.parent/'provenance.json').read_text())
        initial=prov['initial_source_domains']
        for h in r['history']:
            trajectories.append({'tag':r['tag'],'arm':r['args']['arm'],'source_loss':r['args']['source_loss'],
                'step':h['step'],'inner':h['inner'],'required_point_margin':max(0.,max(initial[k]['achieved']-h['source_domains'][k]['achieved'] for k in initial)),
                'deepprime_change':h['source_domains']['deepprime']['achieved']-initial['deepprime']['achieved'],
                'pridict_change':h['source_domains']['pridict_pridict2']['achieved']-initial['pridict_pridict2']['achieved'],
                'feasible':h['feasible']})
        if r['args']['source_loss']!='none':
            counts=r['source_groups_by_study']; rows=r['source_rows_by_study']
            sampling.append({'tag':r['tag'],'sampling':r['args']['sampling'],'groups':counts,'rows':rows,
                             'deepprime_group_fraction':counts['deepprime']/sum(counts.values())})
            trace=pd.read_parquet(p.parent/'training_trace.parquet').dropna(subset=['weighted_source_gradient_norm'])
            ratio=trace.weighted_source_gradient_norm/trace.target_gradient_norm.clip(lower=1e-12)
            scale.append({'tag':r['tag'],'weight':r['args']['weight'],'source_loss':r['args']['source_loss'],
                          'median_weighted_source_to_target_gradient':float(ratio.median()),
                          'min_ratio':float(ratio.min()),'max_ratio':float(ratio.max())})
    frame=pd.read_parquet(CACHE/'index.parquet')
    # The shared core affects singleton source groups only if none of its rows
    # occurs in the explicitly eligible replay pool.
    from audit_alphabet import dna
    frame['dna_core']=frame.spacer.map(dna).str[-19:]
    cores=set(frame.loc[frame.component.isin(overlap),'dna_core'])
    eligible=set(np.concatenate([g for pool in replay_groups(frame).values() for g in pool]).tolist())
    linked=set(frame.loc[(frame.surface=='source_replay')&frame.dna_core.isin(cores),'row'])
    result={'overlap_sensitivity':core_rows,'secondary_target_metrics':metrics,'checkpoint_trajectories':trajectories,
            'realized_sampling':sampling,'gradient_scale':scale,'matched_core_source_rows_in_eligible_replay':len(linked&eligible),
            'nonzero_feasible_checkpoints':sum(t['feasible'] and t['step']>0 for t in trajectories),
            'evaluated_nonzero_checkpoints':sum(t['step']>0 for t in trajectories),
            'note':'Post-screen diagnostics, not new model selection or confirmation. @k uses only groups with depth >=k.',
            'inputs_sha256':provenance([OUT/'controls_results.json',OUT/'factorial_results.json',OUT/'core_overlap_audit.json',Path(__file__)])}
    write_json(OUT/'secondary_results.json',result)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4.5),sharex=True,sharey=True)
    colors={'kl':'#4477aa','margin':'#228833','utility':'#cc6677'}
    for ax,data,title in zip(axes,[controls,factorial],['Inner-selected optimization controls','Preservation/sampling factorial']):
        for r in data['selected']:
            if r['policy']!='target':continue
            a=r['args']; label=r['label'] if data['stage']=='controls' else f"{a['source_loss'][0].upper()}{'b' if a['sampling']=='balanced' else 'm'}{a['weight']:g}"
            ax.scatter(r['contrasts']['source_audit_minus_start']['observed']*100,
                       r['contrasts']['target_minus_optiprime']['observed']*100,
                       color=colors.get(a['source_loss'],'#333333'),s=32)
            ax.annotate(label,(r['contrasts']['source_audit_minus_start']['observed']*100,
                              r['contrasts']['target_minus_optiprime']['observed']*100),xytext=(3,4),textcoords='offset points',fontsize=7)
        ax.axhline(0,color='gray',lw=.8);ax.axvline(-.1,color='gray',lw=.8,ls='--')
        ax.set_title(title,fontsize=10);ax.set_xlabel('Source audit change vs initialization (percentage points)')
    axes[0].set_ylabel('Target @1 change vs OptiPrime (percentage points)')
    fig.suptitle('V3.1 development point estimates: not confidence-based superiority or retention',fontsize=10)
    fig.tight_layout();fig.savefig(OUT/'tradeoff.svg');plt.close(fig)
    print(json.dumps({'nonzero_feasible_checkpoints':result['nonzero_feasible_checkpoints'],
                      'evaluated_nonzero_checkpoints':result['evaluated_nonzero_checkpoints'],
                      'matched_core_source_rows_in_eligible_replay':result['matched_core_source_rows_in_eligible_replay'],
                      'overlap_sensitivity':core_rows},indent=2),flush=True)

if __name__=='__main__':main()
