import json
from pathlib import Path
d=json.loads(Path(r'D:\Day2-Lab\K5-L2-DAY02-NgoThiMinhQuang-2A202603146-ObjectDetection\outputs/manual_eval.json').read_text(encoding='utf-8'))
rows=[]
for p in d['per_image']:
    im=p['image']; fp=p['student_boxes']-p['matches']; fn=p['gold_boxes']-p['matches']
    frame=int(im.split('_')[1])-1
    rows.append((im,frame,fp,fn,fp+fn,p['student_boxes'],p['gold_boxes'],p['matches']))
print('TOP FP (nhiều FP nhất):')
for r in sorted(rows,key=lambda x:(-x[2],-x[4]))[:12]:
    print(f'  {r[0]} (frame {r[1]}): FP={r[2]} FN={r[3]} (student={r[5]} gold={r[6]} match={r[7]})')
print('TOP FN (nhiều FN nhất):')
for r in sorted(rows,key=lambda x:(-x[3],-x[4]))[:12]:
    print(f'  {r[0]} (frame {r[1]}): FN={r[3]} FP={r[2]} (student={r[5]} gold={r[6]} match={r[7]})')
