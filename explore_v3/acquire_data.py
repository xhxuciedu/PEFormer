"""Acquire bounded public supplementary files; never overwrite an existing file."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import requests

OUT = Path(__file__).resolve().parent / "data" / "raw"
FILES = {
    "oped_clinvar.xlsx": "https://media.springernature.com/original/springer-static/esm/art%3A10.1038%2Fs42256-023-00739-w/MediaObjects/42256_2023_739_MOESM5_ESM.xlsx",
    "epridict_supplements.xlsx": "https://raw.githubusercontent.com/Schwank-Lab/epridict/supplementary_files/SupplFile1_primers_libraries_supplementary_file.xlsx",
    "epridict_highlow_batch.txt": "https://raw.githubusercontent.com/Schwank-Lab/epridict/supplementary_files/arrayed_editing/CRISPResso2_batch_files/20230705_highlow_PE_all_batchfile.txt",
    "epridict_additional_batch.txt": "https://raw.githubusercontent.com/Schwank-Lab/epridict/supplementary_files/arrayed_editing/CRISPResso2_batch_files/20230207_additional_endo_targets_batch_all_with_controls.txt",
    "oped_run_metadata.tsv": "https://www.ebi.ac.uk/ena/portal/api/filereport?accession=PRJNA882795&result=read_run&fields=run_accession,experiment_accession,sample_accession,sample_title,experiment_title,library_name,read_count,fastq_bytes,fastq_ftp&format=tsv",
}

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for name, url in FILES.items():
        path = OUT / name
        side = path.with_suffix(".provenance.json")
        if path.exists():
            meta = json.loads(side.read_text())
            assert meta["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
            print("verified existing", name, flush=True)
            continue
        r = requests.get(url, timeout=90)
        r.raise_for_status()
        if len(r.content) > 100_000_000 or (name.endswith(".xlsx") and not r.content.startswith(b"PK")):
            raise ValueError("Unexpected payload or size")
        with path.open("xb") as f:
            f.write(r.content)
        side.write_text(json.dumps({"requested_url": url, "final_url": r.url,
                                   "sha256": hashlib.sha256(r.content).hexdigest(),
                                   "bytes": len(r.content), "acquired_date": "2026-09-10"}, indent=2))
        print("acquired", name, len(r.content), flush=True)

if __name__ == "__main__":
    main()
