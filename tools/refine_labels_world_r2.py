"""
Round 2: ultra-conservative - only dedup (IoU>=0.82) and add very high-confidence missing vehicles.
No class changes, no removals except duplicates. Evidence-based, no Gold.
"""
import json, copy
from pathlib import Path
from collections import defaultdict

ROOT = Path(r"D:\Day2-Lab\K5-L2-DAY02-NgoThiMinhQuang-2A202603146-ObjectDetection")
REVIEWED = ROOT / "dataset/labels_reviewed/train"
WORLD_JSON = ROOT / "outputs/review_automation/world_detections_r1.json"
OUT_DIR = ROOT / "outputs/review_automation"
CLASS_NAMES = {0:'car',1:'truck',2:'bus',3:'van'}
WORLD_MAP = {'car':0,'suv':0,'truck':1,'pickup truck':1,'bus':2,'van':3,'minivan':3}
IGNORE = {'person','motorcycle','bicycle','traffic sign','traffic light'}

def yolo_to_xyxy(b):
    xc,yc,w,h=b; return [xc-w/2,yc-h/2,xc+w/2,yc+h/2]
def clamp_yolo(b):
    xc,yc,w,h=b
    x1=max(0,xc-w/2); y1=max(0,yc-h/2); x2=min(1,xc+w/2); y2=min(1,yc+h/2)
    w=max(1e-6,x2-x1); h=max(1e-6,y2-y1)
    return [(x1+x2)/2,(y1+y2)/2,w,h]
def iou_xyxy(a,b):
    x1=max(a[0],b[0]); y1=max(a[1],b[1]); x2=min(a[2],b[2]); y2=min(a[3],b[3])
    inter=max(0,x2-x1)*max(0,y2-y1)
    aa=max(0,a[2]-a[0])*max(0,a[3]-a[1]); ab=max(0,b[2]-b[0])*max(0,b[3]-b[1])
    return inter/(aa+ab-inter+1e-9)
def read_labels(p):
    boxes=[]
    for line in p.read_text(encoding='utf-8').splitlines():
        s=line.strip()
        if not s: continue
        parts=s.split()
        if len(parts)!=5: continue
        c=int(float(parts[0])); xc,yc,w,h=map(float,parts[1:])
        boxes.append({'cls':c,'yolo':[xc,yc,w,h],'xyxy':yolo_to_xyxy([xc,yc,w,h])})
    return boxes
def write_labels(p, boxes):
    lines=[]
    for b in boxes:
        y=clamp_yolo(b['yolo'])
        lines.append(f"{b['cls']} {y[0]:.6f} {y[1]:.6f} {y[2]:.6f} {y[3]:.6f}")
    p.write_text('\n'.join(lines)+('\n' if lines else ''), encoding='utf-8')

world = json.loads(WORLD_JSON.read_text(encoding='utf-8'))
log = {'round':'r2_ultra_conservative','stats':defaultdict(int),'images':{}}
for lp in sorted(REVIEWED.glob('drive_*.txt')):
    stem=lp.stem
    cur=read_labels(lp); before_n=len(cur)
    dets=world.get(stem,[])
    veh=[{'cls':WORLD_MAP[d['cls']],'raw_cls':d['cls'],'conf':d['conf'],'yolo':d['yolo'],'xyxy':yolo_to_xyxy(d['yolo'])} for d in dets if d['cls'] in WORLD_MAP]
    changes=[]; boxes=[copy.deepcopy(b) for b in cur]; removed=set()
    # dedup
    for i in range(len(boxes)):
        if i in removed: continue
        for j in range(i+1,len(boxes)):
            if j in removed: continue
            if iou_xyxy(boxes[i]['xyxy'],boxes[j]['xyxy'])>=0.82:
                removed.add(j); changes.append({'type':'remove_duplicate','kept':i,'dropped':j,'evidence':'IoU>=0.82 duplicate'})
    after=[b for i,b in enumerate(boxes) if i not in removed]
    # add only very high conf missing
    for d in veh:
        if d['conf']<0.65: continue
        _,_,bw,bh=d['yolo']
        if bw*bh<0.0015 and d['conf']<0.8: continue
        if max([iou_xyxy(d['xyxy'],b['xyxy']) for b in after], default=0)>=0.25: continue
        after.append({'cls':d['cls'],'yolo':d['yolo'],'xyxy':d['xyxy']})
        changes.append({'type':'add','cls':CLASS_NAMES[d['cls']],'world':d['raw_cls'],'conf':round(d['conf'],3),'evidence':'very high-conf world vehicle, no existing match'})
    write_labels(lp, after)
    a=sum(1 for c in changes if c['type']=='add'); r=sum(1 for c in changes if c['type']=='remove_duplicate')
    log['stats']['added']+=a; log['stats']['removed']+=r; log['stats']['images_touched']+= (1 if changes else 0)
    log['images'][stem]={'before':before_n,'after':len(after),'changes':changes}
log['stats']=dict(log['stats'])
(OUT_DIR/'world_refine_r2_log.json').write_text(json.dumps(log,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(log['stats'],indent=2))
