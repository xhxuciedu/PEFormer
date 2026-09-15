"""Check submission artifacts and selected numerical claims; no fitting/scoring."""
from pathlib import Path
import argparse
import hashlib
import json
import re
import statistics

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
CHECKS=[]

def check(condition, name):
    if not condition:
        raise AssertionError(name)
    CHECKS.append(name)

def read(path):
    return json.loads((ROOT/path).read_text())

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--portable',action='store_true')
    args=parser.parse_args()
    tex='\n'.join(p.read_text() for p in [HERE/'main.tex',*sorted((HERE/'sections').glob('*.tex')),*sorted((HERE/'tables').glob('*.tex'))])
    results=(HERE/'sections/results.tex').read_text()
    labels=re.findall(r'\\label\{([^}]+)\}',tex)
    refs=re.findall(r'\\(?:ref|eqref)\{([^}]+)\}',tex)
    check(len(labels)==len(set(labels)), 'Unique LaTeX labels')
    check(not(set(refs)-set(labels)), 'Every internal reference resolves')
    bibkeys=set(re.findall(r'@\w+\{([^,]+),',(HERE/'references.bib').read_text()))
    citations=set(k.strip() for block in re.findall(r'\\cite\w*\{([^}]+)\}',tex) for k in block.split(','))
    check(not(citations-bibkeys),'Every citation has a bibliography record')
    for path in re.findall(r'\\input\{([^}]+)\}',tex):
        check((HERE/(path+'.tex')).exists(),'Input exists: '+path)
    for path in re.findall(r'\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}',tex):
        check((HERE/path).is_file(),'Local figure exists: '+path)
        check('..' not in Path(path).parts,'Portable figure: '+path)
    ledger=json.loads((HERE/'source_data.json').read_text())
    for name,table in ledger['tables'].items():
        text=(HERE/'tables'/f'{name}.tex').read_text()
        for i,row in enumerate(table['rows']):
            check(' & '.join(map(str,row))+r' \\' in text,f'Table {name} row {i+1}')
    archive=json.loads((HERE/'adaptation_source_data.json').read_text())
    rep=archive['replication']
    check(rep['configurations']==48 and rep['new_fits']==44 and rep['reused_fits']==4,'48 configurations include 4 reuses')
    for a in rep['aggregates']:
        if a['label'] in ('A','C_frozen','E') and a['policy']=='constrained':
            check(a['step_zero_fallbacks']==6,f"Fallback accounting {a['label']} {a['budget']}")
    for a in rep['aggregates']:
        if a['budget']!=1000 or a['label']=='C_frozen' or (a['policy']=='constrained' and a['label']!='M'):
            continue
        for surface in ('source_audit','target_outer_val'):
            token=f"{a['surfaces'][surface]['achieved']:.6f}"
            check(token in tex,f"Replicated value {a['label']} {a['policy']} {surface}: {token}")
    def contrast(left,lp,right,rp,surface):
        return next(x for x in rep['direct_contrasts']+rep['control_contrasts'] if (x['budget'],x['left'],x.get('left_policy','target'),x['right'],x.get('right_policy','target'),x['surface'])==(1000,left,lp,right,rp,surface))
    for spec in [('M','target','E','target','source_audit'),('M','constrained','A','target','target_outer_val'),('E','target','A','target','source_audit')]:
        c=contrast(*spec)
        for x in [c['observed'],*c['ci95']]:
            check(f'{abs(x):.6f}' in results, f'Paired contrast {spec}: {x:.6f}')
    mc=next(x for x in rep['aggregates'] if (x['budget'],x['label'],x['policy'])==(1000,'M','constrained'))
    check(mc['contrasts']['source_audit_minus_start']['ci95'][0]>-.001,'Conditional 1k margin source interval clears working threshold')
    check(mc['contrasts']['target_minus_optiprime']['ci95'][1]<0,'Constrained margin below OptiPrime on target')
    check('Prediction $\\rho_{\\mathrm S}$' in tex and 'correlation column evaluates the prediction output' in tex,'Historical prediction/selection head distinction explicit')
    check('Utility + geometry' in tex and 'Geometry only' in tex,'Sequence-plus-geometry and geometry-only arms distinguished')
    check('including sentinels' in tex,'PegRNA token budget includes sentinels')
    check('conditional' in tex and 'not multiplicity-adjusted' in tex,'Uncertainty qualifications retained')

    if not args.portable:
        for path,digest in ledger['inputs_sha256'].items():
            check(hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==digest,'Aggregate fingerprint '+path)
        source=read('results/round4/final_bootstrap.json')
        for x in source['observed'].values():
            check(f'{x:.4f}' in tex,'Source benchmark/generation '+f'{x:.4f}')
        for x in source['round4_vs_optiprime']['ci95']:
            check(f'{x:.4f}' in results,'Original source paired CI '+f'{x:.4f}')
        utility=read('explore_v2/utility_table_numbers.json')['panel_B_fold0_allele']
        for key in ['op_at_1','ours_at_1']:
            check(f'{utility[key]:.5f}' in results,'Source fixed-edit '+key)
        ext=read('explore_v2/e16_corrected_endpoints.json')
        for stratum in ext['strata']:
            d=stratum['paired']['ours_minus_op_at_1']
            for x in [d['observed'],*d['ci95']]:
                check(f'{abs(x):.5f}' in results,'External corrected contrast '+stratum['stratum']+' '+f'{x:.5f}')
        old=read('explore_v2/e27_adaptation_test.json')
        pkeys=[k for k in old['strata'][0]['arms'] if k.startswith('P_s')]
        mean=statistics.mean(old['strata'][0]['arms'][k]['achieved_at_1'] for k in pkeys)
        check(f'{mean:.5f}' in results and len(pkeys)==3,'Historical ordinal three-seed utility')
        check(all(old['strata'][0]['paired'][k+'_minus_op']['ci95'][0]>0 for k in pkeys),'Each historical ordinal seed CI above comparator')
        acquisition=read('explore_v3_1/acquisition_audit.json')
        for a in acquisition['acquisitions']:
            total=a['roles']['total']
            row=[f"{a['nominal_groups']:,}",str(a['subset_seed']%1000),str(a['roles']['train']['groups']),str(a['roles']['inner_val']['groups']),f"{total['groups']:,}",f"{total['candidates']:,}",f"{total['measurements']:,}"]
            check(' & '.join(row) in tex,'Exact acquisition row '+str(a['subset_seed'])+' '+str(a['nominal_groups']))

    log=(HERE/'main.log').read_text() if (HERE/'main.log').exists() else ''
    if log:
        check('Warning' not in log and 'Overfull' not in log,'Clean final LaTeX compilation')
    import pymupdf
    pdf=pymupdf.open(HERE/'main.pdf')
    text='\n'.join(p.get_text() for p in pdf)
    check('??' not in text,'No unresolved PDF references')
    check(len(pdf)>=25,'Full-length manuscript and supplement compiled')
    check(all(len(p.get_text().strip())>30 for p in pdf),'No empty PDF pages')
    check('0.9079' in text and '0.04178' in text and '0.040652' in text,'Headline values present in compiled PDF')
    report={'status':'PASS','checks':len(CHECKS),'pages':len(pdf),'portable':args.portable,
            'pdf_sha256':hashlib.sha256((HERE/'main.pdf').read_bytes()).hexdigest(),
            'check_names':CHECKS,
            'scope':'Artifact, selected numerical, cross-reference and PDF checks; not scientific independence or exhaustive semantic verification.'}
    (HERE/'verification.json').write_text(json.dumps(report,indent=2)+'\n')
    print(f"PASS: {len(CHECKS)} checks; {len(pdf)} pages. No training or outcome scoring performed.")

if __name__=='__main__':
    main()
