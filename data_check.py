import csv, re
from pathlib import Path

D = Path('data/shopee_refund')
REQ = ['doc_id', 'title', 'source_url', 'retrieved_at', 'document_version', 'audience']
mds = sorted(D.glob('*.md'))
rows = list(csv.DictReader(open(D / 'sources.csv', encoding='utf-8')))
ids, auds = [], {}

for p in mds:
    content = p.read_text(encoding='utf-8')
    frontmatter_text = content.split('---')[1]
    raw_matches = re.findall(r'^(\w+):\s*(.+)$', frontmatter_text, re.M)
    fm = {k: v.strip(' "\'') for k, v in raw_matches}
    
    ids.append(fm.get('doc_id'))
    aud = fm.get('audience')
    auds[aud] = auds.get(aud, 0) + 1
    
    # 2. Dùng nháy kép trực tiếp bên trong f-string
    status = "OK" if all(k in fm for k in REQ) and fm.get("doc_id") == p.stem else "THIEU METADATA"
    print(f'{p.name:40} {status}')