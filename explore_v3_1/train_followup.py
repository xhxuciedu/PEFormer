"""V3.1 controlled optimization and source-preservation factorial."""
import argparse
import json
import time
from collections import Counter
from v31 import *
from train import budget_indices
from prepare import BUDGETS, START
import adapt
import adapt_data as AD
from e26_adaptation_pilot import load_base
from pe_rankformer.training.losses import ordinal_loss
from torch.nn import functional as F

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arm', choices=['A','B','D','C_frozen','E'], required=True)
    ap.add_argument('--steps', type=int, choices=[100,500], required=True)
    ap.add_argument('--lr-mult', type=int, choices=[1,3], required=True)
    ap.add_argument('--source-loss', choices=['none','kl','margin','utility'], default='none')
    ap.add_argument('--sampling', choices=['mixture','balanced'], default='mixture')
    ap.add_argument('--weight', type=float, choices=[.1,1.], default=1.)
    ap.add_argument('--budget', type=int, choices=[200,1000], default=1000)
    ap.add_argument('--subset-seed', type=int, default=20260910)
    ap.add_argument('--seed', type=int, default=20260910)
    args = ap.parse_args()
    if args.source_loss != 'none' and args.arm != 'E':
        raise ValueError('Factorial uses E only')
    if not torch.cuda.is_available():
        raise RuntimeError('Declared GPU execution requires GPU access')
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    rng, srng = np.random.default_rng(args.seed), np.random.default_rng(args.seed+100)
    tag = f'{args.arm}_h{args.steps}_lr{args.lr_mult}_{args.source_loss}_{args.sampling}_w{args.weight:g}_b{args.budget}_sub{args.subset_seed}_s{args.seed}'
    run = OUT/'runs'/tag; run.mkdir(parents=True, exist_ok=False)
    t0 = time.time()
    meta = json.loads((CACHE/'prepared.json').read_text())
    for p, digest in meta['outputs_sha256'].items():
        if sha256(ROOT/p) != digest: raise ValueError(f'Changed cache: {p}')
    frame = pd.read_parquet(CACHE/'index.parquet')
    data = torch.load(CACHE/'prepared.pt', map_location='cpu', weights_only=False)
    roles = budget_indices(frame,pd.read_parquet(BUDGETS),args.budget,args.subset_seed)
    tr, va = roles['train'], roles['inner_val']
    sv = np.flatnonzero((frame.surface=='source_val').to_numpy())
    gid = frame.gid.to_numpy()
    y,q0,h = (data[k].cuda() for k in ('y','q0','h'))
    qmu,qsd = q0[tr].mean().detach(),q0[tr].std().clamp_min(1e-6).detach()
    ymu,ysd = y[tr].mean().detach(),y[tr].std().clamp_min(1e-6).detach()
    q = (q0-qmu)/qsd
    frozen = args.arm in ('E','C_frozen')
    base = None if frozen else load_base('cuda')
    holder = {}; opened = []
    if base is not None:
        opened = adapt.unfreeze(base,'final_block')
        base.head.register_forward_pre_hook(lambda module,x: holder.update(h=x[0]))
    head = None if args.arm=='A' else Selector(h.shape[1], residual=args.arm=='E').cuda()
    params = []
    if base is not None:
        params.append({'params':[p for p in base.parameters() if p.requires_grad], 'lr':3e-5*args.lr_mult})
    if head is not None: params.append({'params':list(head.parameters()), 'lr':.001*args.lr_mult})
    trainable = [p for group in params for p in group['params']]
    opt = torch.optim.AdamW(params,weight_decay=.01)
    thresholds = None if base is None else torch.tensor(base.config.ordinal_thresholds,device='cuda')
    pool = replay_groups(frame)
    source_ix = np.sort(np.concatenate([g for groups in pool.values() for g in groups]))
    source_spans = frame.iloc[source_ix].groupby('gid').y.agg(lambda v:v.max()-v.min())
    source_scale = max(float(source_spans.mean()),1e-6)
    gaps = []
    qt = q.detach().cpu().numpy()
    for groups in pool.values():
        for ix in groups:
            z=qt[ix]; best=int(np.argmax(z)); gaps.extend((z[best]-np.delete(z,best)).tolist())
    cap = float(np.quantile(gaps,.9))

    def forward(ix):
        if base is None: return head(h[ix],q[ix]),None
        batch = {k:v[ix].cuda() for k,v in data['inputs'].items()}
        out = base(batch)
        return ((base.efficiency_from_output(out)-qmu)/qsd if head is None else head(holder['h'],q[ix])),out

    def scores(ix, original=False):
        if base is not None: base.eval()
        if head is not None: head.eval()
        if original: return qt[ix]
        with torch.no_grad():
            return torch.cat([forward(ix[a:a+512])[0].cpu() for a in range(0,len(ix),512)]).numpy()

    def state():
        return {'base':None if base is None else {k:v.detach().cpu().clone() for k,v in base.state_dict().items()},
                'selector':None if head is None else {k:v.detach().cpu().clone() for k,v in head.state_dict().items()}}

    def restore(s):
        if s is not None:
            if base is not None: base.load_state_dict(s['base'])
            if head is not None: head.load_state_dict(s['selector'])

    initial_domains = source_domains(frame.iloc[sv],qt[sv])
    initial_inner = float(table(frame.iloc[va],qt[va]).achieved.mean())
    best = {p:{'step':0,'inner':initial_inner,'state':None} for p in ('target','constrained')}
    hist = []; checkpoint_predictions = []; trace = []
    def checkpoint(step):
        si,ss = scores(va,step==0),scores(sv,step==0)
        value = float(table(frame.iloc[va],si).achieved.mean())
        domains = source_domains(frame.iloc[sv],ss)
        ok = feasible(domains,initial_domains)
        entry = {'step':step,'inner':value,'source_domains':domains,'feasible':ok}
        hist.append(entry)
        for surface,ix,s in [('inner_val',va,si),('source_val',sv,ss)]:
            checkpoint_predictions.append(pd.DataFrame({'step':step,'surface':surface,'row':ix,'score':s}))
        snapshot = None
        for policy in best:
            if (policy=='target' or ok) and value > best[policy]['inner']+1e-12:
                if snapshot is None: snapshot = state()
                best[policy] = {'step':step,'inner':value,'state':snapshot}
        print(tag,entry,flush=True)
    checkpoint(0)
    prov = {'args':vars(args),'tag':tag,'device':torch.cuda.get_device_name(),
            'opened':opened,'trainable_parameters':sum(p.numel() for p in trainable),
            'q_mean':float(qmu),'q_sd':float(qsd),'y_mean':float(ymu),'y_sd':float(ysd),
            'source_utility_scale':source_scale,'source_margin_cap':cap,
            'initial_source_domains':initial_domains,
            'label_budget':{k:{'groups':int(frame.iloc[ix].gid.nunique()),'candidates':len(ix),
                'measurements':int(frame.iloc[ix].n_meas.sum()),'components':int(frame.iloc[ix].component.nunique())}
                for k,ix in roles.items()},
            'inputs_sha256':provenance([START,BUDGETS,CACHE/'prepared.json',OUT/'EXECUTION_PROTOCOL.md',
                OUT/'train_followup.py',OUT/'v31.py',ROOT/'explore_v3/model.py',ROOT/'explore_v3/train.py']),
            'precision':'FP32, TF32 disabled', 'source_validation_seen_in_pretraining':True}
    write_json(run/'provenance.json',prov)
    batches=[]; cursor=0; target_rows=0; source_rows=0; source_groups_count=Counter(); source_rows_count=Counter()
    active=0
    for step in range(1,args.steps+1):
        if cursor==len(batches):
            batches=[tr[b] for b in AD.group_batches(gid[tr],256,rng)]; cursor=0
        ix=batches[cursor]; cursor+=1
        if base is not None: base.train()
        if head is not None: head.train()
        opt.zero_grad(set_to_none=True)
        s,out=forward(ix); groups=slices(gid[ix])
        if args.arm=='A': target=ordinal_loss(out[:,:len(thresholds)],y[ix],thresholds)
        elif args.arm=='B': target=F.huber_loss(s,(y[ix]-ymu)/ysd,delta=1.)
        else:
            target=pairwise(s,y[ix],groups)
            if args.arm=='D': target=target+F.huber_loss(s,(y[ix]-ymu)/ysd,delta=1.)
        ret=s.sum()*0
        if args.source_loss!='none':
            sx=source_batch(pool,srng,args.sampling=='balanced'); ss,_=forward(sx); sg=slices(gid[sx])
            if args.source_loss=='kl': ret=preserve(ss,q[sx],sg)
            elif args.source_loss=='margin': ret=top_margin(ss,q[sx],sg,cap)
            else: ret=utility(ss,y[sx],sg,source_scale)
            source_rows+=len(sx)
            for study,g in frame.iloc[sx].groupby('source_study'):
                source_rows_count[str(study)]+=len(g); source_groups_count[str(study)]+=int(g.gid.nunique())
        loss=target+args.weight*ret
        diagnostic=step % (args.steps//10)==0
        norms={}
        if diagnostic:
            norms['target_gradient_norm']=norm(torch.autograd.grad(target,trainable,retain_graph=True,allow_unused=True))
            if args.source_loss!='none':
                norms['weighted_source_gradient_norm']=norm(torch.autograd.grad(args.weight*ret,trainable,retain_graph=True,allow_unused=True))
        if not torch.isfinite(loss): raise ValueError('Nonfinite training loss')
        loss.backward()
        active=max(active,sum(p.numel() for p in trainable if p.grad is not None))
        grad=torch.nn.utils.clip_grad_norm_(trainable,1.)
        opt.step(); target_rows+=len(ix)
        trace.append({'step':step,'target_loss':float(target.detach()),'source_loss':float(ret.detach()),
                      'gradient_norm':float(grad),'target_rows':len(ix),**norms})
        if diagnostic: checkpoint(step)
    training_seconds=time.time()-t0
    final=state()
    views={}
    for policy,choice in {**best,'last':{'step':args.steps,'state':final,'inner':hist[-1]['inner']}}.items():
        restore(choice['state'])
        torch.save({'state':choice['state'],'step':choice['step'],'provenance':prov},run/f'{policy}.pt')
        views[policy]={'step':choice['step'],'inner':choice['inner'],'surfaces':{}}
        for surface in ('target_outer_val','source_val','source_audit'):
            ix=np.flatnonzero((frame.surface==surface).to_numpy()); score=scores(ix,choice['step']==0)
            sub=frame.iloc[ix].copy(); sub['selection']=score
            sub.to_parquet(run/f'{policy}_{surface}.parquet',index=False)
            t=table(sub,score,source=surface.startswith('source'))
            views[policy]['surfaces'][surface]={'achieved':float(t.achieved.mean()),'groups':len(t)}
            if surface.startswith('source'): views[policy]['surfaces'][surface]['domains']=source_domains(sub,score)
    pd.concat(checkpoint_predictions,ignore_index=True).to_parquet(run/'checkpoint_predictions.parquet',index=False)
    pd.DataFrame(trace).to_parquet(run/'training_trace.parquet',index=False)
    result={'args':vars(args),'tag':tag,'history':hist,'views':views,'target_rows_processed':target_rows,
            'target_pass_equivalents':target_rows/len(tr),'source_rows_processed':source_rows,
            'source_groups_by_study':dict(source_groups_count),'source_rows_by_study':dict(source_rows_count),
            'active_gradient_parameters':active,'training_seconds':training_seconds,'total_seconds':time.time()-t0,
            'peak_gpu_bytes':torch.cuda.max_memory_allocated(),
            'outputs_sha256':provenance(sorted(run.glob('*.pt'))+sorted(run.glob('*.parquet')))}
    write_json(run/'results.json',result)
    print('COMPLETE',tag,json.dumps(views),flush=True)

if __name__=='__main__': main()
