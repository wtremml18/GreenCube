"""Acquire pinned public inputs; no account or private data required."""
import hashlib
import json
from pathlib import Path
import urllib.request

ROOT=Path(__file__).resolve().parent
DATA=ROOT.parent/'auroral_data/data'

def main():
    DATA.mkdir(exist_ok=True,parents=True)
    rows=json.loads((ROOT/'sources.json').read_text())
    for row in rows:
        p=DATA/row['file']
        if not p.exists():
            print('Downloading',row['file'],flush=True)
            tmp=p.with_suffix(p.suffix+'.part')
            with urllib.request.urlopen(row['url'],timeout=120) as r, tmp.open('wb') as f:
                while chunk:=r.read(1024*1024):f.write(chunk)
            if hashlib.sha256(tmp.read_bytes()).hexdigest()!=row['sha256']:
                raise ValueError('Downloaded bytes differ from pinned source: '+row['file'])
            tmp.replace(p)
        with p.open('rb') as f:actual=hashlib.file_digest(f,'sha256').hexdigest()
        if actual!=row['sha256'] or p.stat().st_size!=row['bytes']:
            raise ValueError('Source verification failed: '+row['file'])
    print(f"Verified {len(rows)} files, {sum(r['bytes'] for r in rows):,} bytes.")

if __name__=='__main__':main()
