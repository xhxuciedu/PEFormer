"""Additional external input/provenance checks, never model-outcome comparisons."""
import json
import re
import xml.etree.ElementTree as ET
from v31 import *
from canon import revcomp, minimal_edit, canonical_keys
from audit_data import SCAFFOLD, shortname
from pe_rankformer.data.seqops import align_pair

def occurrences(text,needle):
    return [m.start() for m in re.finditer('(?='+re.escape(needle)+')',text)]

def reconstruct(spacer,extension,wt,ed):
    answers={}
    for orientation,w,e in [('+',wt,ed),('-',revcomp(wt),revcomp(ed))]:
        pos,ref,alt=minimal_edit(w,e)
        for start in occurrences(w,spacer[-19:]):
            nick=start+16
            if pos<nick: continue
            for plen in range(1,len(extension)):
                pbs,rtt=extension[-plen:],extension[:-plen]
                if nick-plen<0 or revcomp(pbs)!=w[nick-plen:nick]: continue
                if nick+len(rtt)>len(e) or revcomp(rtt)!=e[nick:nick+len(rtt)]: continue
                if pos+len(alt)>nick+len(rtt): continue
                lo=max(0,min(start-1,nick-plen)-10)
                hi=min(len(w),max(start+19+3,nick+len(rtt)+len(ref)-len(alt),pos+len(ref))+10)
                end_ed=hi-len(ref)+len(alt)
                cw,ce=w[lo:hi],e[lo:end_ed]
                a,b=align_pair(cw,ce)
                key=(pbs,rtt,orientation,start)
                answers[key]={'pbs':pbs,'rtt':rtt,'orientation':orientation,'nick':nick,
                    'full_unedited':cw,'full_edited':ce,'aligned_tokens':len(a),'peg_tokens':len(spacer)+len(extension)+2,
                    'edit_preserved':minimal_edit(cw,ce)[1:]==(ref,alt),
                    'full_amplicon_tokens':max(len(w),len(e))}
    return list(answers.values())

def min_hamming(query,reference,length):
    refs=sorted({s for s in reference if isinstance(s,str) and len(s)==length and set(s)<=set('ACGT')})
    if not refs: raise ValueError('No homology reference')
    arr=np.frombuffer(''.join(refs).encode(),dtype=np.uint8).reshape(-1,length)
    result=[]
    for q in query:
        if not isinstance(q,str) or len(q)!=length or not set(q)<=set('ACGT'):
            result.append(None);continue
        best=length
        for oriented in (q,revcomp(q)):
            code=np.frombuffer(oriented.encode(),dtype=np.uint8)
            for offset in range(0,len(arr),20000):
                best=min(best,int((arr[offset:offset+20000]!=code).sum(axis=1).min()))
        result.append(best)
    return result,len(refs)

def main():
    raw=OUT/'data/raw'; oldraw=ROOT/'explore_v3/data/raw'
    experiments=[]
    for p in sorted(raw.glob('SRX*.xml')):
        tree=ET.parse(p); text=tree.findtext('.//DESIGN_DESCRIPTION') or ''
        rep=re.search(r'\brep(\d+)\b',text)
        condition=re.search(r'using (.*?) editing',text)
        experiments.append({'experiment':p.stem,'title':tree.findtext('.//TITLE'),
            'description':text,'replicate':rep.group(1) if rep else None,
            'conditions':condition.group(1) if condition else 'untreated' if 'untreated condition' in text else None,
            'explicit_barcode_map':False})
    primers=pd.read_excel(raw/'oped_sequencing_primers.xlsx',header=2)
    eligibility=pd.read_parquet(ROOT/'explore_v3/data/epridict_eligibility.parquet')
    sequences=pd.concat([pd.read_csv(oldraw/f'epridict_{k}_batch.txt',sep='\t') for k in ('highlow','additional')])
    sequences['short']=sequences.name.map(shortname)
    seqmap={}
    for r in sequences.itertuples():
        w,e=str(r.amplicon_seq).upper(),str(r.expected_hdr_amplicon_seq).upper()
        if set(w+e)<=set('ACGT') and w!=e:
            seqmap.setdefault(r.short,{})[canonical_keys(w,e)['edit_key']]=(w,e)
    records=[]
    for r in eligibility.itertuples():
        if pd.isna(r.edit_key): continue
        pieces=r.guide.split(SCAFFOLD)
        wt,ed=seqmap[r.sequence_key][r.edit_key]
        answers=reconstruct(pieces[0],pieces[1],wt,ed) if len(pieces)==2 else []
        row={'shortname':r.shortname,'edit_key':r.edit_key,'spacer':r.spacer,'guide':r.guide,
             'has_K562_outcome':r.has_K562_outcome,'has_HEK293T_outcome':r.has_HEK293T_outcome,
             'reconstructions':len(answers),'unambiguous_input':len(answers)==1}
        if len(answers)==1: row.update(answers[0])
        records.append(row)
    f=pd.DataFrame(records)
    source=pd.read_parquet(ROOT/'explore_v2/cache/corpus_canonical_v2.parquet',
         columns=['edit_key','spacer','source_study','cell_type','pe_type','decision_group','design_key'])
    hd,nsp=min_hamming(f.spacer.str[-19:].tolist(),source.spacer.str[-19:].unique(),19)
    f['source_core19_min_hamming']=hd
    def flanks(key):
        parts=key.split('|');return parts[0]+parts[-1] if len(parts)==3 else ''
    hd,nfl=min_hamming(f.edit_key.map(flanks).tolist(),source.edit_key.map(flanks).unique(),24)
    f['source_flanks24_min_hamming']=hd
    f['local_homology_clear']=(f.source_core19_min_hamming>2)&(f.source_flanks24_min_hamming>2)
    f['token_safe']=f.unambiguous_input & (f.aligned_tokens<=100)&(f.peg_tokens<=90)&(f.edit_preserved==True)
    f.to_parquet(OUT/'data/epridict_input_audit.parquet',index=False)
    populations={}
    for cell in ('K562','HEK293T'):
        for name,mask in [('reconstructed',f.unambiguous_input),('token_safe',f.token_safe),
                          ('homology_and_token_safe',f.token_safe&f.local_homology_clear)]:
            sub=f.loc[mask&f[f'has_{cell}_outcome']]
            depth=sub.groupby('edit_key').guide.nunique()
            populations[cell+'_'+name]={'rows':len(sub),'alleles':len(depth),'groups_ge2':int((depth>=2).sum()),
                                        'groups_ge5':int((depth>=5).sum()),'max_depth':int(depth.max()) if len(depth) else 0}
    inventory=[]
    for (study,cell,editor),g in source.groupby(['source_study','cell_type','pe_type']):
        d=g.groupby('decision_group').design_key.nunique()
        inventory.append({'study':study,'cell':cell,'editor':editor,'rows':len(g),'groups':len(d),
                          'depth_ge2':int((d>=2).sum()),'depth_ge5':int((d>=5).sum()),
                          'checkpoint_pretraining_exposure':'source corpus; must exclude before retraining a domain holdout'})
    out={'oped':{'experiments':experiments,'primer_rows':len(primers),'primer_columns':list(primers.columns),
                'linkage_ready':False,'reason':'Descriptions identify replicates and pooled loci/conditions, not barcode-to-design/condition assignment; primer sheet gives gene/forward/reverse primers only.'},
         'epridict':{'mapped_rows':len(f),'unique_reconstructions':int(f.unambiguous_input.sum()),
                     'token_safe_rows':int(f.token_safe.sum()),'local_homology_flagged':int((~f.local_homology_clear).sum()),
                     'source_unique_cores':nsp,'source_unique_flanks':nfl,'populations':populations,
                     'no_model_scoring':True,'remaining':'Context/scaffold convention and cropped-input deployment parity require verification; local Hamming audit is not exhaustive homology.'},
         'source_context_inventory':inventory,'independent_confirmation_ready':False,
         'inputs_sha256':provenance([Path(__file__),OUT/'DATA_PROTOCOL.md',OUT/'metadata_acquisition.json',
              ROOT/'explore_v3/data_inventory.json',ROOT/'explore_v2/cache/corpus_canonical_v2.parquet'])}
    write_json(OUT/'data_feasibility.json',out)
    print(json.dumps({'epridict':out['epridict'],'oped_primers':len(primers),'source_contexts':len(inventory)},indent=2),flush=True)

if __name__=='__main__': main()
