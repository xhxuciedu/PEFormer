import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from v31 import *
from train import budget_indices

def test_margin_identity_and_harmful_flip():
    t=torch.tensor([.2,.1,-.4],requires_grad=True)
    s=t.detach().clone().requires_grad_()
    loss=top_margin(s,t,[slice(0,3)],.3)
    loss.backward()
    assert loss.item()==0 and t.grad is None
    s=torch.tensor([.0,.3,-.4],requires_grad=True)
    loss=top_margin(s,t,[slice(0,3)],.3); loss.backward()
    assert loss>0 and s.grad[0]<0 and s.grad[1]>0

def test_teacher_tie_has_no_positive_margin():
    s=torch.zeros(2,requires_grad=True)
    assert top_margin(s,torch.zeros(2),[slice(0,2)],1.).item()==0
    value=top_margin(s,torch.zeros(2),[slice(0,1),slice(1,2)],1.)
    value.backward(); assert torch.equal(s.grad,torch.zeros(2))

def test_feasibility_is_per_domain():
    start={'a':{'achieved':.1},'b':{'achieved':.2}}
    assert feasible(start,start)
    assert not feasible({'a':{'achieved':.098},'b':{'achieved':.3}},start)
    assert feasible({'a':{'achieved':.099},'b':{'achieved':.2}},start)

def test_endpoint_parity_and_tie_rule():
    from e29_summarize_audit import metrics
    f=pd.DataFrame({'group_id':['a','a','b','b'],'design_key':['z','a','a','b'],
                    'component':['x','x','y','y'],'source_study':['s']*4,'cell_type':['c']*4,
                    'pe_type':['p']*4,'y':[.1,.2,0,0]})
    s=np.array([1,1,2,1])
    for source in (False,True):
        t=table(f,s,source)
        ref,_=metrics(f.rename(columns={'group_id':'edit_key'}).assign(selection=s,prediction=s),source)
        assert t.achieved.mean()==ref['selection']['achieved_at_1']
        assert t.loc['a','winner']=='a'

def test_actual_source_isolation_and_sampling():
    f=pd.read_parquet(CACHE/'index.parquet'); pool=replay_groups(f)
    group_sizes=f.groupby('gid').size().to_dict()
    assert {k:len(v) for k,v in pool.items()}=={'deepprime':486,'pridict_pridict2':2461}
    assert f.groupby('component').surface.nunique().max()==1
    rng=np.random.default_rng(12)
    counts={k:0 for k in pool}
    for _ in range(100):
        ix=source_batch(pool,rng,True)
        assert 1<len(ix)<=256 and len(set(ix))==len(ix)
        block=f.iloc[ix]
        assert set(block.surface)=={'source_replay'}
        for gid,g in block.groupby('gid'):
            assert len(g)==group_sizes[gid] and g.y.max()>g.y.min()
            counts[str(g.source_study.iloc[0])]+=1
    share=counts['deepprime']/sum(counts.values())
    assert .45<share<.55

def test_all_acquisition_manifests_covered():
    f=pd.read_parquet(CACHE/'index.parquet')
    b=pd.read_parquet(ROOT/'explore_v2/cache/adapt_followup_budget_components.parquet')
    part=pd.read_parquet(ROOT/'explore_v2/cache/adaptation_partition_v2.parquet')
    for seed in (20260910,20260911,20260912):
        for budget in (200,1000):
            roles=budget_indices(f,b,budget,seed)
            for role,ix in roles.items():
                expected=b.loc[(b.seed==seed)&(b.nominal_groups==budget)&(b.total_budget_role==role),'component']
                assert set(f.iloc[ix].component_numeric)==set(expected)
                assert len(ix)==len(part.loc[(part.split=='train')&part.component.isin(expected)])

def test_recipe_choice_ignores_outer_scores():
    from analyze import choose
    records=[{'tag':str(i),'args':{'steps':100,'lr_mult':1,'weight':1},
              'views':{'target':{'inner':inner,'outer':outer}}}
             for i,inner,outer in [(0,.04,.01),(1,.03,.99)]]
    assert choose(records,'target')['tag']=='0'

def test_external_reconstruction_preserves_strand_and_indels():
    from audit_external import reconstruct
    from canon import revcomp
    spacer='ACGTCAGTACGATCGTACGA'
    w='TGTACGTA'+spacer+'AGGTCAGTCAGTCTAGCATGACGTACGATCGATCGTA'
    nick=8+17
    for e in (w[:nick+6]+'A'+w[nick+7:],w[:nick+6]+'GC'+w[nick+6:],w[:nick+6]+w[nick+8:]):
        extension=revcomp(e[nick:nick+23])+revcomp(w[nick-13:nick])
        for wt,ed in ((w,e),(revcomp(w),revcomp(e))):
            r=reconstruct(spacer,extension,wt,ed)
            assert len(r)==1 and r[0]['edit_preserved'] and r[0]['aligned_tokens']<=100
            assert len(r[0]['pbs'])==13 and len(r[0]['rtt'])==23

def test_legacy_alphabet_is_not_silently_changed():
    from pe_rankformer.data.tokenizer import encode_pegrna
    from audit_alphabet import dna
    assert dna('acgu')=='ACGT'
    assert encode_pegrna('ACGU','','',6).nuc_ids!=encode_pegrna('ACGT','','',6).nuc_ids

def test_within_group_spearman_excludes_constants_and_preserves_index():
    from error_strata import within_spearman
    f=pd.DataFrame({'group_id':['a']*3+['b']*3+['c']*2,
                    'selection':[1,2,3,3,2,1,2,1],
                    'y':[1,3,2,1,2,3,0,0]},index=[2,5,9,11,12,20,25,31])
    result=within_spearman(f)
    assert result['groups_with_defined_spearman']==2
    assert abs(result['mean_defined_group_spearman']-(-.25))<1e-12

def test_depth_endpoint_excludes_shallow_groups():
    from secondary_analysis import deeper
    f=pd.DataFrame({'group_id':['a']*3+['b']*2,'design_key':['c','a','b','x','y'],
                    'selection':[1,1,0,1,0],'y':[.1,.2,.3,1.,1.]})
    result=deeper(f,3)
    assert result['eligible_groups']==1
    assert result['best_measured_outcome_among_top_k']==.3
    assert deeper(f,5)['best_measured_outcome_among_top_k'] is None
