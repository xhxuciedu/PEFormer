"""Build the portable manuscript; optionally assemble a submission-source ZIP."""
from pathlib import Path
import argparse
import subprocess
import zipfile

HERE=Path(__file__).resolve().parent

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--zip',action='store_true',help='Package sources, aggregate source data, figures and PDF')
    parser.add_argument('--package-only',action='store_true',help='Package an already compiled and verified PDF without changing it')
    args=parser.parse_args()
    commands=[['pdflatex','-interaction=nonstopmode','-halt-on-error','main.tex'],
              ['bibtex','main'],
              ['pdflatex','-interaction=nonstopmode','-halt-on-error','main.tex'],
              ['pdflatex','-interaction=nonstopmode','-halt-on-error','main.tex']]
    for command in ([] if args.package_only else commands):
        done=subprocess.run(command,cwd=HERE,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
        if done.returncode:
            print(done.stdout)
            raise SystemExit(done.returncode)
    log=(HERE/'main.log').read_text()
    warnings=[x for x in log.splitlines() if 'Warning' in x or 'Overfull' in x]
    print('\n'.join(warnings) if warnings else 'Build clean: no LaTeX warnings or overfull boxes.')
    print(HERE/'main.pdf')
    if args.zip:
        files=[p for p in HERE.rglob('*') if p.is_file()
               and '__pycache__' not in p.parts and p.suffix in {'.tex','.bib','.bbl','.pdf','.png','.py','.md','.json','.csv'}]
        with zipfile.ZipFile(HERE/'submission_sources.zip','w',zipfile.ZIP_DEFLATED) as archive:
            for p in sorted(files):
                archive.write(p,arcname='pe_rankformer_submission/'+str(p.relative_to(HERE)))
        print(f'Packaged {len(files)} files in submission_sources.zip (no private candidate tables or checkpoints).')

if __name__=='__main__':
    main()
