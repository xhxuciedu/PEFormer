"""Descriptive edit/gap/flip analysis of frozen screen selections; no retuning."""
import json
from v31 import *
from canon import edit_type


def edit_class(key):
    ref, alt = key.split('|')[1].split('>')
    return edit_type(ref, alt)


def within_spearman(f):
    ranked = f.groupby('group_id')[['selection', 'y']].rank(method='average')
    centered = ranked - ranked.groupby(f.group_id).transform('mean')
    xy = (centered.selection * centered.y).groupby(f.group_id).sum()
    xx = centered.selection.pow(2).groupby(f.group_id).sum()
    yy = centered.y.pow(2).groupby(f.group_id).sum()
    corr = xy / np.sqrt(xx*yy).replace(0, np.nan)
    return {'groups_with_defined_spearman': int(corr.notna().sum()),
            'mean_defined_group_spearman': float(corr.mean())}


def main():
    control = json.loads((OUT/'control_selection.json').read_text())
    fact = json.loads((OUT/'factorial_selection.json').read_text())
    index = pd.read_parquet(CACHE/'index.parquet')
    q0 = torch.load(CACHE/'prepared.pt', map_location='cpu', weights_only=False)['q0'].numpy()
    picks = [(a, control['recipes'][a]['target']['tag'], 'target') for a in ('A', 'B', 'D', 'C_frozen', 'E')]
    picks.append(('M', fact['candidate'], 'constrained'))
    records, endpoints = [], []
    for label, tag, policy in picks:
        for surface in ('target_outer_val', 'source_val', 'source_audit'):
            f = pd.read_parquet(OUT/'runs'/tag/f'{policy}_{surface}.parquet')
            base = table(f, q0[f.row.to_numpy()], source=surface.startswith('source'))
            new = table(f, f.selection, source=surface.startswith('source'))
            new['edit_type'] = f.groupby('group_id').edit_key.first().map(edit_class)
            new['delta'] = new.achieved - base.achieved
            new['flip'] = new.winner != base.winner
            for field, score in [('teacher_gap', q0[f.row.to_numpy()]), ('outcome_gap', f.y.to_numpy())]:
                ordered = f.assign(value=score).sort_values(['group_id', 'value', 'design_key'], ascending=[True, False, True])
                top_two = ordered.groupby('group_id').head(2)
                gaps = top_two.groupby('group_id').value.agg(lambda x: x.iloc[0]-x.iloc[1] if len(x)>1 else np.nan)
                new[field] = gaps
                # Descriptive quartiles on this evaluation surface, never training rules.
                positive = gaps.loc[gaps > 0]
                cuts = sorted(set(positive.quantile([.25, .5, .75]).tolist()))
                bins = pd.cut(gaps, [-np.inf, 0.] + cuts + [np.inf], duplicates='drop')
                new[field+'_bin'] = bins.astype(str)
            for field in ('edit_type', 'teacher_gap_bin', 'outcome_gap_bin'):
                for value, g in new.groupby(field):
                    records.append({'label': label, 'policy': policy, 'surface': surface,
                        'stratum': field, 'value': str(value), 'groups': len(g),
                        'flips': int(g.flip.sum()), 'beneficial_flips': int((g.delta>0).sum()),
                        'harmful_flips': int((g.delta<0).sum()),
                        'equal_outcome_flips': int((g.flip & (g.delta==0)).sum()),
                        'unchanged_choices': int((~g.flip).sum()),
                        'mean_utility_change': float(g.delta.mean()),
                        'total_utility_change': float(g.delta.sum())})
            eligible = f.loc[f.group_id.isin(new.index)]
            endpoints.append({'label': label, 'surface': surface, 'groups': len(new),
                'achieved': float(new.achieved.mean()), 'regret': float(new.regret.mean()),
                'oracle': float(new.oracle.mean()), 'flips': int(new.flip.sum()),
                'beneficial_flips': int((new.delta>0).sum()), 'harmful_flips': int((new.delta<0).sum()),
                'equal_outcome_flips': int((new.flip & (new.delta==0)).sum()),
                **within_spearman(eligible)})
    write_json(OUT/'error_strata.json', {
        'status': 'Exploratory screen diagnostics, one subset/seed; no subgroup selection or confirmation.',
        'gap_definition': 'Winner-minus-runner-up, including tied maxima as zero; raw frozen source score for teacher gap.',
        'measurement_uncertainty': 'Measured outcome gap is not uncertainty; no replicate variance is inferred from measurement counts.',
        'endpoints': endpoints, 'strata': records,
        'inputs_sha256': provenance([Path(__file__), OUT/'control_selection.json', OUT/'factorial_selection.json', CACHE/'prepared.json'])})
    print(json.dumps(endpoints, indent=2), flush=True)


if __name__ == '__main__':
    main()
