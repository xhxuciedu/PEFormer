"""Render the complete replication table directly from verified numerical results."""
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent.parent
OUT=ROOT/'explore_v3_1'
TABLE=ROOT/'reports/paper/tables/tab_v31_replication.tex'


def render(r):
    names={'A':'Ordinal A','C_frozen':'Fresh frozen','E':'Anchored E','M':'Margin'}
    lines=[r'\begin{table}[htbp]',r'\centering',r'\small',
           r'\caption{Locked-recipe acquisition replication on the exposed external-library validation population. Each entry averages achieved outcomes over three acquisition subsets and two optimizer seeds, not averaged scores. All efficiencies are fractions. Target-only (T) and per-study source-constrained (C) checkpoint policies are distinct procedures. Fallback counts indicate deployment of the unchanged starting model.}',
           r'\label{tab:v31-replication}',r'\begin{tabular}{rlcrrrr}',r'\toprule',
           r'Budget & Method & Policy & Target @1 & $\Delta$ OptiPrime & Source @1 & Fallbacks \\',r'\midrule']
    rows=[]
    for budget in (200,1000):
        if budget==1000: lines.append(r'\midrule')
        for label,policy in [('A','target'),('C_frozen','target'),('E','target'),('M','target'),('M','constrained')]:
            a=next(x for x in r['aggregates'] if x['budget']==budget and x['label']==label and x['policy']==policy)
            target=a['surfaces']['target_outer_val']['achieved']; source=a['surfaces']['source_audit']['achieved']
            delta=a['contrasts']['target_minus_optiprime']['observed']; fallback=a['step_zero_fallbacks']
            rows.append({'budget':budget,'label':label,'policy':policy,'target':target,'source':source,'delta_optiprime':delta,'fallbacks':fallback})
            lines.append(f"{budget:,} & {names[label]} & {'T' if policy=='target' else 'C'} & {target:.6f} & ${delta:+.6f}$ & {source:.6f} & {fallback}/6 " + r'\\')
    lines += [r'\bottomrule',r'\end{tabular}',r'\par\smallskip',r'\begin{minipage}{0.98\textwidth}\footnotesize',
              f"Released OptiPrime target @1 is {r['baselines']['optiprime']:.6f}; starting-model target/source @1 are {r['baselines']['target_outer_val']:.6f}/{r['baselines']['source_audit']:.6f}.",
              r'Target evaluation retains all 5,561 groups; source audit uses 1,463 informative groups. All checkpoint policies, run-level outcomes, conditional component intervals and acquisition-specific results are reported in the v3.1 replication ledger. Released OptiPrime receives no target adaptation labels.',
              r'\end{minipage}',r'\end{table}']
    return '\n'.join(lines)+'\n',rows


def main():
    import argparse
    ap=argparse.ArgumentParser();ap.add_argument('--check',action='store_true');args=ap.parse_args()
    verified=json.loads((OUT/'replication_verification.json').read_text())
    assert verified['configurations_verified']==48
    r=json.loads((OUT/'replication_results.json').read_text()); assert r['new_fits']==44 and r['reused_fits']==4
    content,rows=render(r)
    if args.check:
        assert TABLE.read_text()==content
        assert json.loads((OUT/'manuscript_table_numbers.json').read_text())==rows
        print('PASS: 10 manuscript rows match verified replication results exactly')
    else:
        with TABLE.open('x') as f:f.write(content)
        with (OUT/'manuscript_table_numbers.json').open('x') as f:json.dump(rows,f,indent=2)
        print(TABLE)


if __name__=='__main__':main()
