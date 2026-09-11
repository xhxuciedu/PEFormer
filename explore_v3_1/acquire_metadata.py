"""Bounded public metadata acquisition, no sequencing-read download."""
import json
from concurrent.futures import ThreadPoolExecutor
import requests
from v31 import OUT, ROOT, write_json, sha256, pd

def main():
    dest=OUT/'data'/'raw'; dest.mkdir(parents=True,exist_ok=True)
    runmeta=pd.read_csv(ROOT/'explore_v3/data/raw/oped_run_metadata.tsv',sep='\t')
    jobs=[('oped_sequencing_primers.xlsx','https://media.springernature.com/original/springer-static/esm/art%3A10.1038%2Fs42256-023-00739-w/MediaObjects/42256_2023_739_MOESM9_ESM.xlsx'),
          ('oped_repo_tree.json','https://api.github.com/repos/wenjiegroup/OPED/git/trees/master?recursive=1')]
    jobs += [(f'{s}.xml',f'https://www.ebi.ac.uk/ena/browser/api/xml/{s}') for s in sorted(runmeta.experiment_accession.unique())]
    def fetch(job):
        name,url=job; p=dest/name; side=p.with_suffix(p.suffix+'.provenance.json')
        if p.exists():
            meta=json.loads(side.read_text()); assert sha256(p)==meta['sha256']; return meta
        try:
            r=requests.get(url,timeout=30); r.raise_for_status()
            if len(r.content)>10_000_000: raise ValueError('Metadata exceeds bounded size')
            if name.endswith('.xlsx') and not r.content.startswith(b'PK'): raise ValueError('Not XLSX')
            with p.open('xb') as f: f.write(r.content)
            meta={'file':name,'url':url,'final_url':r.url,'bytes':len(r.content),'sha256':sha256(p),'date':'2026-09-11'}
            write_json(side,meta); print('acquired',name,len(r.content),flush=True); return meta
        except Exception as e:
            return {'file':name,'url':url,'error':str(e)}
    with ThreadPoolExecutor(max_workers=4) as pool: records=list(pool.map(fetch,jobs))
    write_json(OUT/'metadata_acquisition.json',{'files':records,'raw_fastq_downloaded':False})

if __name__=='__main__': main()
