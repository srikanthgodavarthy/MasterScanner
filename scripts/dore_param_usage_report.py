import re, os, dataclasses, sys
sys.path.insert(0,'.')
from utils.dore_options_engine import DoreOptionsSettings
fields=[f.name for f in dataclasses.fields(DoreOptionsSettings)]
files=[]
for r,d,fs in os.walk('.'):
    if '.git' in r: continue
    for f in fs:
        if f.endswith('.py'): files.append(os.path.join(r,f))
src={p:open(p,encoding='utf8',errors='ignore').read().splitlines() for p in files}
rows=[]
for n in fields:
    pat=re.compile(r'(?<![A-Za-z0-9_])'+re.escape(n)+r'(?![A-Za-z0-9_])')
    prod=[];tests=[];defn=0
    for p,lines in src.items():
        for i,l in enumerate(lines,1):
            if pat.search(l):
                s=l.strip()
                if p.endswith('dore_options_engine.py') and re.match(r'^\s*'+re.escape(n)+r'\s*:',l): defn+=1; continue
                if s.startswith('#'): continue
                (tests if '/tests/' in p or '/backtests/' in p else prod).append(f"{p}:{i}")
    rows.append((n,len(prod),len(tests),prod))
print("TOTAL FIELDS",len(fields))
print("\n== ZERO non-test, non-comment references ==")
for n,a,b,pr in rows:
    if a==0: print(f"{n:48s} prod={a} tests={b}")
print("\n== Only referenced in 1 prod line (check it is a real read) ==")
for n,a,b,pr in rows:
    if a==1: print(f"{n:48s} {pr[0]}")
