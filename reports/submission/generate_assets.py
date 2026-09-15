"""Generate submission tables/figures from frozen aggregate records; no training.

Run from any directory with the repository Python environment. The resulting
LaTeX package builds independently of the repository and private data.
"""
from pathlib import Path
import hashlib
import json
import statistics
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
INPUTS = {}
VALUES = {}

def read(path):
    raw = (ROOT / path).read_bytes()
    INPUTS[path] = hashlib.sha256(raw).hexdigest()
    return json.loads(raw)

def table(name, caption, label, headers, rows, note, widths=None):
    columns = widths or ('l' + 'r' * (len(headers)-1))
    lines = [r'\begin{table}[htbp]', r'\centering\small',
             r'\caption{' + caption + '}', r'\label{' + label + '}',
             r'\begin{tabular}{' + columns + '}', r'\toprule',
             ' & '.join(headers) + r' \\', r'\midrule']
    lines += [' & '.join(map(str, row)) + r' \\' for row in rows]
    lines += [r'\bottomrule', r'\end{tabular}', r'\par\smallskip',
              r'\begin{minipage}{\linewidth}\footnotesize ' + note + r'\end{minipage}',
              r'\end{table}']
    (HERE / 'tables' / (name + '.tex')).write_text('\n'.join(lines) + '\n')
    VALUES[name] = {'headers': headers, 'rows': rows}

def savefig(name):
    plt.savefig(HERE / 'figures' / (name + '.pdf'), bbox_inches='tight')
    plt.savefig(HERE / 'figures' / (name + '.png'), dpi=180, bbox_inches='tight')
    plt.close()

def main():
    for folder in ('tables', 'figures'):
        (HERE / folder).mkdir(exist_ok=True)
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False,
                         'axes.spines.right': False, 'pdf.fonttype': 42})
    utility = read('explore_v2/utility_table_numbers.json')['panel_B_fold0_allele']
    ext = read('explore_v2/e16_corrected_endpoints.json')
    cor = read('explore_v2/e21_correlation_surfaces.json')
    adapt = read('explore_v2/e27_adaptation_test.json')
    audit = read('explore_v2/e29_head_audit.json')
    rep = read('explore_v3_1/replication_results.json')
    source = read('results/round4/heldout/metrics_round4_final.json')
    member_names={'ordSSM':'Ordinal + S4D','ssm':'Simplex + S4D','ordA':'Ordinal + layerwise FiLM',
                  'ordC':'Ordinal + features','familyA':'Simplex + layerwise FiLM'}
    table('source_members', 'Post-hoc source-benchmark family scores.', 'tab:sourcemembers',
          ['Family', 'Checkpoints', 'Pooled Spearman'],
          [[name,5,f"{source['member_only_global'][key]['spearman']:.4f}"] for key,name in member_names.items()],
          'Each family is itself a five-fold-checkpoint ensemble on all 20,509 held-out rows. The frozen heterogeneous combination uses twenty-five checkpoints and reaches 0.9079. These post-hoc scores did not replace the frozen ensemble. A five-checkpoint family is not an individual checkpoint or the subsequently adapted single backbone.')

    rows = []
    rows.append(['Source, informative', '1,463', f"{utility['op_at_1']:.5f}",
                 f"{utility['ours_at_1']:.5f}", f"{utility['delta_at_1']['observed']:+.5f}"])
    for s, title in zip(ext['strata'], ['External, all', 'External, informative',
                                       'External, depth $\\geq8$', 'External, high worth']):
        rows.append([title, f"{s['groups']:,}", f"{s['arms']['op']['achieved_at_1']:.5f}",
                     f"{s['arms']['ours']['achieved_at_1']:.5f}",
                     f"{s['paired']['ours_minus_op_at_1']['observed']:+.5f}"])
    table('decision', 'Source and zero-shot external fixed-edit selection.', 'tab:decision',
          ['Population', 'Groups', 'OptiPrime', r'\model{}', 'Difference'], rows,
          'Mean measured efficiency of the top-scored distinct candidate. Source uses the heterogeneous ensemble; external uses the five-fold ordinal--S4D ensemble. High worth means oracle minus random utility at least 0.05. Source and informative external subsets exclude constant-outcome groups. External all retains them. These are different populations, not a learning curve.')

    means = audit['source_means_by_arm']
    rows = []
    for key, title in [('P','Ordinal prediction'), ('S','Utility selector'),
                       ('S_pairwise','Pairwise selector'), ('Z','Frozen encoder')]:
        x = means[key]
        rows.append([title, x['n_seeds'], f"{x['prediction']['achieved_at_1']:.5f}",
                     f"{x['selection']['achieved_at_1']:.5f}"])
    table('head_audit', 'Retention depends on the score that is deployed.', 'tab:heads',
          ['Historical adaptation arm', 'Seeds', 'Prediction head', 'Deployed score'], rows,
          'Source audit: 1,463 informative groups. Matched initialization is 0.13350 for both scores. An unchanged or accurate unused prediction head does not protect the choices made by a newly trained selector. Pairwise has three source audits but only one historical target-test seed.')

    test = adapt['strata'][0]
    rows = []
    for prefix, title in [('P','Ordinal P'), ('S','Utility selector'), ('M','Multi-task'),
                          ('shared','Shared score'), ('S_pairwise','Pairwise selector'),
                          ('S_listnet','Listwise selector'), ('Z','Frozen encoder'),
                          ('Sgeom','Utility + geometry'), ('G','Geometry only')]:
        keys = [k for k in test['arms'] if k.startswith(prefix + '_s')]
        if not keys:
            continue
        value = statistics.mean(test['arms'][k]['achieved_at_1'] for k in keys)
        rho = statistics.mean(adapt['pooled_spearman'][k] for k in keys)
        rows.append([title, len(keys), f'{value:.5f}', f'{rho:.4f}',
                     f"{value-test['arms']['op']['achieved_at_1']:+.5f}"])
    table('adaptation', 'Historical full-budget target-library adaptation.', 'tab:adaptation',
          ['Method', 'Test seeds', 'Target $U_1$', r'Prediction $\rho_{\mathrm S}$', r'$\Delta$ OptiPrime'], rows,
          'E25 retrospective test: 5,557 groups and 23,044 candidates. Full-budget adaptation uses 19,357 training groups and 5,561 validation groups. OptiPrime receives no target labels: its test utility is 0.04020 and Spearman is 0.6922. Utility evaluates the deployed score; the historical correlation column evaluates the prediction output, which differs from the deployed selector for separate-head arms. Entries average outcomes/metrics across the available seeds, not scores. Single-seed arms cannot establish a replicated advantage over ordinal P.')

    names = {'A':'Ordinal', 'C_frozen':'Fresh frozen', 'E':'Anchored', 'M':'Margin'}
    selected = []
    rows = []
    for budget in (200, 1000):
        for key, policy in [('A','target'), ('C_frozen','target'), ('E','target'),
                            ('M','target'), ('M','constrained')]:
            a = next(a for a in rep['aggregates'] if (a['budget'],a['label'],a['policy']) == (budget,key,policy))
            selected.append(a)
            rows.append([f'{budget:,}', names[key], 'T' if policy=='target' else 'C',
                         f"{a['surfaces']['target_outer_val']['achieved']:.6f}",
                         f"{a['surfaces']['source_audit']['achieved']:.6f}",
                         f"{a['step_zero_fallbacks']}/6"])
    table('replication', 'Acquisition and optimizer replication with locked recipes.', 'tab:replication',
          ['Budget', 'Method', 'Policy', 'Target $U_1$', 'Source $U_1$', 'Fallbacks'], rows,
          'Six runs per entry: three label acquisitions and two optimizer seeds. T maximizes inner target utility; C imposes an additional per-study source-validation point-loss limit. Target: exposed E25 validation, 5,561 groups; source: 1,463 informative audit groups. Initialization target/source is 0.039627/0.133635; target OptiPrime is 0.041409. All constrained ordinal/fresh/anchored runs fall back to initialization at both budgets.')

    rows = []
    for a in selected[5:]:
        sec = a['secondary_mean_over_runs']
        vals = {x['k']: x['best_measured_outcome_among_top_k'] for x in sec['depth_endpoints']}
        rows.append([names[a['label']] + (' / C' if a['policy']=='constrained' else ' / T'),
                     f"{sec['pooled_spearman']:.4f}", f"{sec['mean_defined_group_spearman']:.4f}",
                     f"{sec['regret']:.5f}", f"{vals[3]:.5f}", f"{vals[5]:.5f}"])
    sec = rep['optiprime_secondary']
    vals = {x['k']: x['best_measured_outcome_among_top_k'] for x in sec['depth_endpoints']}
    rows.append(['OptiPrime', f"{sec['pooled_spearman']:.4f}", f"{sec['mean_defined_group_spearman']:.4f}",
                 f"{sec['regret']:.5f}", f"{vals[3]:.5f}", f"{vals[5]:.5f}"])
    table('secondary', 'Secondary target outcomes at the nominal 1,000-group budget.', 'tab:secondary',
          ['Method', 'Pooled $\\rho$', 'Group $\\rho$', 'Regret', '$U_3$', '$U_5$'], rows,
          'Group correlation averages 4,656 groups with defined Spearman correlation. Regret uses all 5,561 groups; top-three/top-five outcomes use 4,014/2,056 depth-eligible groups, respectively. These metrics do not have a common denominator across columns.')

    factorial = read('explore_v3_1/factorial_results.json')
    controls = read('explore_v3_1/controls_results.json')
    rows = []
    for a in factorial['selected']:
        if a['policy'] != 'target':
            continue
        args = a['args']
        constrained = next(x for x in factorial['selected'] if x['tag']==a['tag'] and x['policy']=='constrained')
        rows.append([args['source_loss'].upper() if args['source_loss']=='kl' else args['source_loss'].capitalize(),
                     args['sampling'].capitalize(), f"{args['weight']:g}", a['step'],
                     f"{a['surfaces']['target_outer_val']['achieved']:.6f}",
                     f"{a['surfaces']['source_audit']['achieved']:.6f}", constrained['step']])
    table('factorial', 'Complete source-preservation factorial: twelve fits.', 'tab:preservationgrid',
          ['Loss', 'Sampling', 'Weight', 'T step', 'Target $U_1$', 'Source $U_1$', 'C step'], rows,
          'One acquisition and optimizer seed; nominal 1,000-group budget. Utilities evaluate the target-selected (T) checkpoint. C step reports the source-constrained choice; zero means unchanged initialization. All fits use 100 updates and learning-rate multiplier one. Only balanced margin at weight one selects a nonzero constrained checkpoint.')

    # Preserve full aggregate ledgers, not private per-candidate sequences or outcomes.
    detailed = {'controls': controls, 'factorial': factorial, 'replication': rep}
    (HERE/'adaptation_source_data.json').write_text(json.dumps(detailed,indent=2)+'\n')
    with (HERE/'replication_summary.csv').open('w',newline='') as f:
        writer=csv.writer(f)
        writer.writerow(['budget','method','policy','n_runs','target_u1','source_u1','target_run_sd','source_run_sd','fallbacks'])
        for a in rep['aggregates']:
            writer.writerow([a['budget'],a['label'],a['policy'],a['n_runs'],
                             a['surfaces']['target_outer_val']['achieved'],a['surfaces']['source_audit']['achieved'],
                             a['surfaces']['target_outer_val']['run_sd'],a['surfaces']['source_audit']['run_sd'],a['step_zero_fallbacks']])

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    for ax, surface, title in zip(axes, ['held-out fold 0','reserved panel'], ['Source benchmark','External zero-shot library']):
        m = cor['surfaces'][surface]['models']
        vals = [m['OptiPrime']['spearman'],m['PE-RankFormer, raw score']['spearman']]
        ax.bar(['OptiPrime','PE-RankFormer'], vals, color=['#666666','#247ba0'], width=.55)
        ax.set_ylim(0,1); ax.set_title(title); ax.set_ylabel('Pooled Spearman correlation')
        for i,v in enumerate(vals): ax.text(i,v+.02,f'{v:.4f}',ha='center')
    fig.tight_layout(); savefig('prediction')

    fig, axes = plt.subplots(1,2,figsize=(9,3.5))
    for ax, vals, title in [(axes[0],[utility['rand_at_1'],utility['op_at_1'],utility['ours_at_1']], 'Source: 1,463 informative decisions'),
                            (axes[1],[ext['strata'][0]['arms'][x]['achieved_at_1'] for x in ['rand','op','ours']], 'External: 30,475 eligible decisions')]:
        ax.bar(['Random','OptiPrime','PE-RankFormer'],[v*100 for v in vals],color=['#bbbbbb','#666666','#247ba0'])
        ax.set_title(title,fontsize=10);ax.set_ylabel('Top-choice efficiency (%)')
        ax.set_ylim(0,max(vals)*125)
        ax.tick_params(axis='x',labelsize=9)
        for i,v in enumerate(vals): ax.text(i,v*100+max(vals)*4,f'{v*100:.3f}',ha='center',fontsize=9)
    fig.tight_layout();savefig('decision')

    fig, axes = plt.subplots(1,2,figsize=(9,3.6))
    colors = ['#247ba0','#aaaaaa','#ef8354','#57a773','#9163ad']
    for ax, surface, title in zip(axes,['target_outer_val','source_audit'],['Target adaptation','Source retention']):
        for i,a in enumerate(selected[5:]):
            y=a['surfaces'][surface]['achieved']*100
            contrast=a['contrasts'][surface+'_minus_start']
            base=rep['baselines'][surface]
            lo,hi=[(base+x)*100 for x in contrast['ci95']]
            ax.errorbar(i,y,yerr=[[y-lo],[hi-y]],fmt='o',color=colors[i],capsize=3)
        ax.axhline(rep['baselines'][surface]*100,color='#888888',ls=':',label='Initialization')
        if surface=='target_outer_val': ax.axhline(rep['baselines']['optiprime']*100,color='black',ls='--',label='OptiPrime')
        ax.set_xticks(range(5),['Ordinal\nT','Fresh\nT','Anchored\nT','Margin\nT','Margin\nC'])
        ax.set_ylabel('Top-choice efficiency (%)');ax.set_title(title);ax.legend(fontsize=8)
    fig.tight_layout();savefig('replication')

    fig, ax=plt.subplots(figsize=(7,3.4))
    labels=[]
    for i,(key,title) in enumerate([('P','Ordinal'),('S','Utility'),('S_pairwise','Pairwise'),('Z','Frozen encoder')]):
        labels.append(title)
        ax.bar(i-.17,means[key]['prediction']['achieved_at_1']*100,.34,color='#247ba0',label='Prediction head' if i==0 else None)
        ax.bar(i+.17,means[key]['selection']['achieved_at_1']*100,.34,color='#ef8354',label='Deployed score' if i==0 else None)
    ax.set_xticks(range(4),labels);ax.set_ylabel('Source top-choice efficiency (%)')
    ax.set_ylim(0,16);ax.legend(ncol=2);fig.tight_layout();savefig('head_audit')
    figure_data={'prediction':cor['surfaces'], 'source_decision':utility,
                 'external_decision':ext['strata'], 'head_audit':means,
                 'replication_baselines':rep['baselines'],
                 'replication_points':selected[5:]}
    (HERE/'source_data.json').write_text(json.dumps({'inputs_sha256':INPUTS,'tables':VALUES,
                                                   'figure_data':figure_data},indent=2)+'\n')
    print(f'Generated {len(VALUES)} tables and 4 figures from {len(INPUTS)} aggregate records.')

if __name__ == '__main__':
    main()
