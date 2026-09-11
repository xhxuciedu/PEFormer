"""Freeze E's recipe after its four controls; no outer/audit-based selection."""
import json
from v31 import *
from analyze import choose

def main():
    files=[];records=[]
    for p in sorted((OUT/'runs').glob('*/results.json')):
        r=json.loads(p.read_text())
        if r['args']['arm']=='E' and r['args']['source_loss']=='none':files.append(p);records.append(r)
    assert len(records)==4
    winner=choose(records,'target')
    write_json(OUT/'e_recipe.json',{'tag':winner['tag'],'args':winner['args'],
        'selection':'Target inner only among all four E controls; other controls may still be running.',
        'inputs_sha256':provenance(files+[Path(__file__),OUT/'analyze.py'])})
    print(json.dumps({'tag':winner['tag'],'args':winner['args'],'inner':winner['views']['target']['inner']},indent=2))

if __name__=='__main__':main()
