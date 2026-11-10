import json
from pathlib import Path
root=Path(r'D:\Day2-Lab\K5-L2-DAY02-NgoThiMinhQuang-2A202603146-ObjectDetection')
r=json.load(open(root/'outputs/gold_review_phase1/phase1_report.json',encoding='utf-8'))
for d in r['top5']:
    print('===',d['image'], f"TP={d['tp']} FP={d['fp']} FN={d['fn']} SV={d['student']} G={d['gold']} lowIoU={d['low_iou_matches']} mean={d['mean_iou_matches']}")
    print('  cross_class_hints:', json.dumps(d['cross_class_hints'][:10],ensure_ascii=False))
    print('  FP:', json.dumps(d['fp_boxes'],ensure_ascii=False))
    print('  FN:', json.dumps(d['fn_boxes'],ensure_ascii=False))
    print('  low_matches:', json.dumps(d['low_matches'],ensure_ascii=False))
# student-student duplicates
def parse(p):
    b=[]
    for line in p.read_text(encoding='utf-8').splitlines():
        if not line.strip(): continue
        v=line.split(); c=int(float(v[0])); xc,yc,w,h=map(float,v[1:]); b.append((c,xc,yc,w,h))
    return b
def iou(a,b):
    ax1,ay1,ax2,ay2=a[1]-a[3]/2,a[2]-a[4]/2,a[1]+a[3]/2,a[2]+a[4]/2
    bx1,by1,bx2,by2=b[1]-b[3]/2,b[2]-b[4]/2,b[1]+b[3]/2,b[2]+b[4]/2
    iw=max(0,min(ax2,bx2)-max(ax1,bx1)); ih=max(0,min(ay2,by2)-max(ay1,by1))
    inter=iw*ih; uni=a[3]*a[4]+b[3]*b[4]-inter
    return inter/uni if uni else 0
names={0:'car',1:'truck',2:'bus',3:'van'}
print('\nDUPS')
for stem in [x['image'] for x in r['top5']]:
    b=parse(root/f'dataset/labels/train/{stem}.txt')
    dups=[]
    for i in range(len(b)):
        for j in range(i+1,len(b)):
            ov=iou(b[i],b[j])
            if ov>=0.5: dups.append((i,j,round(ov,3),names.get(b[i][0]),names.get(b[j][0])))
    dups.sort(key=lambda x:-x[2])
    print('==',stem,'stu',len(b),'dup_pairs>=0.5',len(dups))
    for d in dups: print(' ',d)
