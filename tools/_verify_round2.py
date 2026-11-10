import json
from pathlib import Path
root=Path(r'D:\Day2-Lab\K5-L2-DAY02-NgoThiMinhQuang-2A202603146-ObjectDetection')
out=root/'outputs/manual_review_round2'
d=json.loads((out/'data.json').read_text(encoding='utf-8'))
assert len(d['images'])==50
missing=[]
for im in d['images']:
    for rel in ['images/original/'+im['file'], 'images/manual/'+im['file'], 'images/diagnostic/'+im['file']]:
        if not (out/rel).exists(): missing.append(rel)
    for s in im['students']:
        if s.get('crop') and not (out/s['crop']).exists(): missing.append(s['crop'])
    for f in im['fns']:
        if f.get('crop') and not (out/f['crop']).exists(): missing.append(f['crop'])
print('images', len(d['images']), 'missing_files', len(missing))
def read(p):
    b=[]
    for i,l in enumerate(p.read_text(encoding='utf-8').splitlines()):
        if not l.strip(): continue
        c,xc,yc,w,h=map(float,l.split()[:5]); b.append((i,int(c),round(xc,6),round(yc,6),round(w,6),round(h,6)))
    return b
mism=[]
for im in d['images']:
    ref=read(root/f'dataset/labels_manual/train/{im["stem"]}.txt')
    cur=[(s['id'],s['c'],s['xc'],s['yc'],s['w'],s['h']) for s in im['students']]
    if ref!=cur: mism.append(im['stem'])
print('coord_id_mismatches', len(mism))
print('sample', d['images'][0]['stem'], 'frame', d['images'][0]['frame_cvat'], 'boxes', len(d['images'][0]['students']), 'tp/fp/fn', d['images'][0]['tp'],d['images'][0]['fp'],d['images'][0]['fn'])
print('crops_dir', len(list((out/'crops').glob('*.jpg'))))
