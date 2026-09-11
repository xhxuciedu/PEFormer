"""Check added historical-v3 and optimization-screen prose against numerical ledgers."""
import json
import re
from pathlib import Path

ROOT=Path(__file__).resolve().parent.parent
OUT=ROOT/'explore_v3_1'


def main():
    text=(ROOT/'reports/paper/sections/adaptation_followup.tex').read_text()
    text=text.replace('{','').replace('}','').replace('$','')
    numbers=set(x.lstrip('+') for x in re.findall(r'[+-]?\d+\.\d+',text))
    checks=[]
    def check(name,value,absolute=False):
        token=f'{abs(value) if absolute else value:.6f}'
        assert token in numbers,(name,token,'missing or different prose value')
        checks.append({'claim':name,'expected':token})
    def contrast(name,d,absolute=False):
        check(name,d['observed'],absolute)
        for i,value in enumerate(d['ci95']):check(name+f' CI {i}',value)
    old=json.loads((ROOT/'explore_v3/replication_results.json').read_text())
    old_runs={r['arm']:r for r in old['runs'] if r['budget']==1000}
    for a in ('A','D','E','F'):check('v3 target '+a,old_runs[a]['surfaces']['target_outer_val']['achieved'])
    for a in ('D','E'):check('v3 source '+a,old_runs[a]['surfaces']['source_audit']['achieved'])
    contrast('v3 E versus OptiPrime',old_runs['E']['contrasts']['target_minus_optiprime'])
    contrast('v3 E source change',old_runs['E']['contrasts']['source_audit_minus_start'])
    for left,right,surface in [('E','A','target_outer_val'),('F','E','source_audit')]:
        d=next(d for d in old['direct_contrasts'] if d['budget']==1000 and d['left']==left and d['right']==right and d['surface']==surface)
        contrast('v3 '+left+' versus '+right+' '+surface,d,absolute=left=='F')
    screen=json.loads((OUT/'controls_results.json').read_text())
    for a in ('A','E','C_frozen'):
        r=next(r for r in screen['selected'] if r['label']==a and r['policy']=='target')
        check('v3.1 screened target '+a,r['surfaces']['target_outer_val']['achieved'])
    for surface in ('target_outer_val','source_audit'):
        d=next(d for d in screen['direct_contrasts'] if d['policy']=='target' and d['left']=='E' and d['right']=='A' and d['surface']==surface)
        contrast('v3.1 screened E versus A '+surface,d)
    rep=json.loads((OUT/'replication_results.json').read_text())
    rr={(r['label'],r['policy']):r for r in rep['aggregates'] if r['budget']==1000}
    for a in ('A','E','M'):
        check('v3.1 replicated target '+a,rr[(a,'target')]['surfaces']['target_outer_val']['achieved'])
        if a!='M':
            check('v3.1 replicated source '+a,rr[(a,'target')]['surfaces']['source_audit']['achieved'])
    for a,surface in [('E','target_outer_val'),('M','target_outer_val'),('M','source_audit')]:
        right='A' if a=='E' else 'E'
        d=next(d for d in rep['control_contrasts'] if d['budget']==1000 and d['left']==a and d['right']==right and d['surface']==surface)
        contrast('v3.1 replicated '+a+' versus '+right+' '+surface,d)
    for field in ('source_audit_minus_start','target_minus_optiprime'):
        contrast('v3.1 M target-selected '+field,rr[('M','target')]['contrasts'][field])
    for a in ('A','M'):check('v3.1 pooled correlation '+a,rr[(a,'target')]['secondary_mean_over_runs']['pooled_spearman'])
    check('v3.1 OptiPrime pooled correlation',rep['optiprime_secondary']['pooled_spearman'])
    for surface in ('target_outer_val','source_audit'):
        check('v3.1 M constrained '+surface,rr[('M','constrained')]['surfaces'][surface]['achieved'])
    contrast('v3.1 M constrained source change',rr[('M','constrained')]['contrasts']['source_audit_minus_start'])
    contrast('v3.1 M constrained target versus OptiPrime',rr[('M','constrained')]['contrasts']['target_minus_optiprime'],absolute=True)
    d=next(d for d in rep['direct_contrasts'] if d['budget']==1000 and d['right']=='A' and d['right_policy']=='target' and d['surface']=='target_outer_val')
    contrast('v3.1 M constrained target versus A',d,absolute=True)
    for r in rr[('E','target')]['acquisitions']:
        check('v3.1 E acquisition source mean '+str(r['subset_seed']),r['source_audit_mean'])
    print(f'PASS: {len(checks)} added-prose numerical checks against v3/v3.1 ledgers')


if __name__=='__main__':main()
