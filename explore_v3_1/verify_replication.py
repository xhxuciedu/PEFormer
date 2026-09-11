"""Verify the locked replication matrix and independently recompute selection."""
import json
from v31 import *
from train import budget_indices
from prepare import BUDGETS


def main():
    manifest = json.loads((OUT/'jobs_replication.json').read_text())
    jobs = manifest['jobs']
    assert len(jobs) == 48 and len({j['tag'] for j in jobs}) == 48
    assert len(manifest['reused_at_launch']) == 4 and manifest['new_at_launch'] == 44
    index = pd.read_parquet(CACHE/'index.parquet')
    budgets = pd.read_parquet(BUDGETS)
    assert index.groupby('component').surface.nunique().max() == 1
    hashes = dict(manifest['inputs_sha256'])
    checks = []
    for j in jobs:
        folder = OUT/'runs'/j['tag']
        r = json.loads((folder/'results.json').read_text())
        prov = json.loads((folder/'provenance.json').read_text())
        assert j['args'] == r['args'] == prov['args']
        roles = budget_indices(index, budgets, r['args']['budget'], r['args']['subset_seed'])
        assert not set(index.iloc[roles['train']].component) & set(index.iloc[roles['inner_val']].component)
        for role, ix in roles.items():
            f = index.iloc[ix]
            actual = {'groups': int(f.gid.nunique()), 'candidates': len(f),
                      'measurements': int(f.n_meas.sum()), 'components': int(f.component.nunique())}
            assert actual == prov['label_budget'][role]
        h = r['history']
        assert [x['step'] for x in h] == list(range(0, r['args']['steps']+1, r['args']['steps']//10))
        for policy in ('target', 'constrained'):
            # Reproduce the declared tolerance and earliest-tie rule explicitly.
            best = h[0]
            for x in h[1:]:
                if (policy == 'target' or feasible(x['source_domains'], prov['initial_source_domains'])) and x['inner'] > best['inner'] + 1e-12:
                    best = x
            assert best['step'] == r['views'][policy]['step']
        cp = pd.read_parquet(folder/'checkpoint_predictions.parquet')
        for (step, surface), g in cp.groupby(['step', 'surface']):
            expected_rows = roles['inner_val'] if surface == 'inner_val' else index.loc[index.surface == 'source_val', 'row'].to_numpy()
            assert np.array_equal(g.row.to_numpy(), expected_rows)
            entry = next(x for x in h if x['step'] == step)
            f = index.iloc[g.row.to_numpy()]
            if surface == 'inner_val':
                assert abs(table(f, g.score.to_numpy()).achieved.mean() - entry['inner']) < 1e-12
            else:
                domains = source_domains(f, g.score.to_numpy())
                assert domains.keys() == entry['source_domains'].keys()
                for study, d in domains.items():
                    assert d['groups'] == entry['source_domains'][study]['groups']
                    assert abs(d['achieved'] - entry['source_domains'][study]['achieved']) < 1e-12
                assert feasible(domains, prov['initial_source_domains']) == entry['feasible']
        assert len(cp.groupby(['step', 'surface'])) == 22
        for policy in ('target', 'constrained', 'last'):
            for surface in ('target_outer_val', 'source_val', 'source_audit'):
                f = pd.read_parquet(folder/f'{policy}_{surface}.parquet')
                expected = index.loc[index.surface == surface].reset_index(drop=True)
                assert f[['row', 'group_id', 'design_key', 'y']].equals(expected[['row', 'group_id', 'design_key', 'y']])
                t = table(f, f.selection, source=surface.startswith('source'))
                endpoint = r['views'][policy]['surfaces'][surface]
                assert len(t) == endpoint['groups']
                assert abs(t.achieved.mean() - endpoint['achieved']) < 1e-12
        for ledger in (prov['inputs_sha256'], r['outputs_sha256']):
            for name, digest in ledger.items():
                if name in hashes:
                    assert hashes[name] == digest, (name, 'conflicting fingerprint')
                hashes[name] = digest
        checks.append({'tag': j['tag'], 'budget_manifest_verified': True,
                       'checkpoint_surfaces_recomputed': 22, 'endpoint_checks': 9})
        print('verified replication fit', len(checks), j['tag'], flush=True)
    for name, digest in hashes.items():
        assert sha256(ROOT/name) == digest, (name, 'changed fingerprint')
    write_json(OUT/'replication_verification.json', {
        'configurations_verified': 48, 'new_fits': 44, 'screen_reuses': 4,
        'checkpoint_surfaces_recomputed': 48*22, 'endpoint_checks': 48*9,
        'distinct_fingerprints_verified': len(hashes), 'checks': checks,
        'test_scoring_present': False,
        'inputs_sha256': provenance([Path(__file__), OUT/'jobs_replication.json'])})
    print('VERIFIED replication: 48 configurations, 44 new fits', flush=True)


if __name__ == '__main__':
    main()
