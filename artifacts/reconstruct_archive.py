#!/usr/bin/env python3
from pathlib import Path
import hashlib,json
root=Path(__file__).parent
m=json.loads((root/'archive_manifest.json').read_text())
out=root/m['filename']
with out.open('wb') as f:
 for c in m['chunks']:
  p=root/c['path']
  if p.stat().st_size != c['bytes']: raise SystemExit(f'size mismatch: {p}')
  with p.open('rb') as g:
   while True:
    b=g.read(1024*1024)
    if not b: break
    f.write(b)
h=hashlib.sha256(out.read_bytes()).hexdigest()
if out.stat().st_size != m['bytes'] or h != m['sha256']: raise SystemExit(f'verification failed: bytes={out.stat().st_size}, sha256={h}')
print(f'restored {out} ({out.stat().st_size} bytes, sha256={h})')
