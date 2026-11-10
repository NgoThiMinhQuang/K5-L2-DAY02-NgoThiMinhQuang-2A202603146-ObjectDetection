"""
Evidence-based label refinement using YOLO-World detections.
- Works on dataset/labels_reviewed/train (never touches labels/train or gold).
- Decisions based on model evidence + geometry rules, NOT gold matching.
- Logs all changes with evidence/confidence.
"""
import json, copy, math, os, shutil, subprocess, sys
from pathlib import Path
from collections import defaultdict

ROOT = Path(r"D:\Day2-Lab\K5-L2-DAY02-NgoThiMinhQuang-2A202603146-ObjectDetection")
REVIEWED = ROOT / "dataset/labels_reviewed/train"
WORLD_JSON = ROOT / "outputs/review_automation/world_detections_r1.json"
OUT_DIR = ROOT / "outputs/review_automation"
CLASS_NAMES = {0:'car',1:'truck',2:'bus',3:'van'}
NAME_TO_ID = {v:k for k,v in CLASS_NAMES.items()}

# Map world labels to ADAS class id or None (ignore)
WORLD_MAP = {
    'car': 0, 'suv': 0,
    'truck': 1, 'pickup truck': 1,
    'bus': 2,
    'van': 3, 'minivan': 3,
}
IGNORE = {'person','motorcycle','bicycle','traffic sign','traffic light'}

def yolo_to_xyxy(b):
    xc,yc,w,h=b
    return [xc-w/2, yc-h/2, xc+w/2, yc+h/2]

def xyxy_to_yolo(b):
    x1,y1,x2,y2=b
    w=max(0,x2-x1); h=max(0,y2-y1)
    return [(x1+x2)/2, (y1+y2)/2, w, h]

def clamp_yolo(b):
    xc,yc,w,h=b
    x1=max(0,xc-w/2); y1=max(0,yc-h/2); x2=min(1,xc+w/2); y2=min(1,yc+h/2)
    w=max(0.001,x2-x1); h=max(0.001,y2-y1)
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

def nms_merge(boxes, iou_thr=0.75):
    # boxes: list of dict with cls,yolo,xyxy,conf,evidence
    boxes = sorted(boxes, key=lambda b: b.get('conf',0.5), reverse=True)
    keep=[]
    for b in boxes:
        merged=False
        for k in keep:
            if iou_xyxy(b['xyxy'],k['xyxy']) >= iou_thr:
                # if same effective area, keep higher conf; don't auto-change class
                merged=True
                break
        if not merged: keep.append(b)
    return keep

def best_overlap(box, dets, iou_thr=0.3):
    best=None; biou=0
    for d in dets:
        ov=iou_xyxy(box['xyxy'],d['xyxy'])
        if ov>biou: biou=ov; best=d
    if biou>=iou_thr: return best, biou
    return None, biou

def run(round_name='r1'):
    world = json.loads(WORLD_JSON.read_text(encoding='utf-8'))
    log = {'round': round_name, 'stats': defaultdict(int), 'images': {}}
    all_imgs = sorted(REVIEWED.glob('drive_*.txt'))
    for lp in all_imgs:
        stem = lp.stem
        cur = read_labels(lp)
        before_n = len(cur)
        dets = world.get(stem, [])
        veh_dets = []
        nonveh = []
        for d in dets:
            if d['cls'] in IGNORE:
                nonveh.append(d); continue
            if d['cls'] in WORLD_MAP:
                xyxy = yolo_to_xyxy(d['yolo'])
                veh_dets.append({'cls': WORLD_MAP[d['cls']], 'raw_cls': d['cls'], 'conf': d['conf'], 'yolo': d['yolo'], 'xyxy': xyxy})
        # sort veh dets by conf
        veh_dets.sort(key=lambda x:x['conf'], reverse=True)

        changes = []
        new_boxes = [copy.deepcopy(b) for b in cur]
        removed = set()

        # 1) remove intra-label duplicates (IoU>=0.80)
        for i in range(len(new_boxes)):
            if i in removed: continue
            for j in range(i+1,len(new_boxes)):
                if j in removed: continue
                if iou_xyxy(new_boxes[i]['xyxy'],new_boxes[j]['xyxy'])>=0.80:
                    removed.add(j)
                    changes.append({'type':'remove_duplicate','dropped':j,'kept':i,'iou':round(iou_xyxy(new_boxes[i]['xyxy'],new_boxes[j]['xyxy']),3),'evidence':'high IoU duplicate, 1 vehicle/1 box'})
        # 2) class correction: only when strong, unambiguous world evidence matches geometry
        for i,b in enumerate(new_boxes):
            if i in removed: continue
            det, ov = best_overlap(b, veh_dets, iou_thr=0.45)
            if not det: continue
            if det['cls'] == b['cls']: continue
            # conservative: require high conf and clear separation
            # find competing dets for same box
            competing = [d for d in veh_dets if iou_xyxy(b['xyxy'],d['xyxy'])>=0.45 and d['cls']==b['cls']]
            same_conf = max([d['conf'] for d in competing], default=0)
            # change only if new class conf is strong and notably better, and involves van/bus/truck disambiguation where world has specific signal
            if det['conf'] >= 0.55 and det['conf'] >= same_conf + 0.15 and ov >= 0.5:
                old = b['cls']
                b['cls'] = det['cls']
                b['yolo'] = b['yolo']  # keep coords
                changes.append({'type':'change_class','idx':i,'from':CLASS_NAMES[old],'to':CLASS_NAMES[det['cls']],'world':det['raw_cls'],'conf':round(det['conf'],3),'iou':round(ov,3),'evidence':'strong world-model class signal with geometric match; coords preserved; no Gold used'})
            elif det['raw_cls'] in ('minivan','van') and b['cls'] in (0,1) and det['conf']>=0.60 and ov>=0.55 and same_conf < 0.35:
                old=b['cls']; b['cls']=3
                changes.append({'type':'change_class','idx':i,'from':CLASS_NAMES[old],'to':'van','world':det['raw_cls'],'conf':round(det['conf'],3),'iou':round(ov,3),'evidence':'strong minivan/van signal overriding weak car/truck; coords preserved'})
            elif det['raw_cls']=='bus' and b['cls']==3 and det['conf']>=0.60 and ov>=0.55 and same_conf < 0.35:
                old=b['cls']; b['cls']=2
                changes.append({'type':'change_class','idx':i,'from':'van','to':'bus','world':det['raw_cls'],'conf':round(det['conf'],3),'iou':round(ov,3),'evidence':'strong bus signal overriding weak van; coords preserved'})

        # 3) add missing high-conf vehicle detections
        after_boxes = [b for i,b in enumerate(new_boxes) if i not in removed]
        for d in veh_dets:
            if d['conf'] < 0.50: continue
            # skip tiny boxes (likely noise/distant beyond labeling reliability) unless very confident
            _,_,bw,bh = d['yolo']
            if bw*bh < 0.0012 and d['conf'] < 0.70: continue
            ov_best = max([iou_xyxy(d['xyxy'],b['xyxy']) for b in after_boxes], default=0)
            if ov_best >= 0.30: continue
            # also avoid adding on top of strong non-vehicle
            nv_ov = max([iou_xyxy(d['xyxy'],n['xyxy']) for n in nonveh if n['conf']>=0.4], default=0)
            if nv_ov >= 0.4 and d['conf'] < 0.7: continue
            nb = {'cls':d['cls'],'yolo':d['yolo'],'xyxy':d['xyxy']}
            after_boxes.append(nb)
            changes.append({'type':'add','cls':CLASS_NAMES[d['cls']],'world':d['raw_cls'],'conf':round(d['conf'],3),'yolo':[round(x,6) for x in d['yolo']],'evidence':'high-conf world detection, no matching existing box; ADAS vehicle class'})

        # 4) final dedup pass on after_boxes
        before_dedup=len(after_boxes)
        after_boxes = nms_merge(after_boxes, iou_thr=0.78)
        if len(after_boxes)<before_dedup:
            changes.append({'type':'final_dedup','removed':before_dedup-len(after_boxes)})

        write_labels(lp, after_boxes)
        added=sum(1 for c in changes if c['type']=='add')
        removed_n=sum(1 for c in changes if c['type']=='remove_duplicate') + sum(c.get('removed',0) for c in changes if c['type']=='final_dedup')
        cls_changed=sum(1 for c in changes if c['type']=='change_class')
        log['stats']['added']+=added; log['stats']['removed']+=removed_n; log['stats']['class_changed']+=cls_changed
        log['stats']['images_touched'] += (1 if changes else 0)
        log['images'][stem] = {'before': before_n, 'after': len(after_boxes), 'changes': changes}

    log['stats'] = dict(log['stats'])
    outp = OUT_DIR / f'world_refine_{round_name}_log.json'
    outp.write_text(json.dumps(log, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(log['stats'], indent=2))
    print('log', outp)
    return log

if __name__=='__main__':
    run(sys.argv[1] if len(sys.argv)>1 else 'r1')
