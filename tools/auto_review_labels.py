"""
Automated label review (Giai đoạn 2) - NO GOLD USED FOR EDITS.
- Reads original image + student labels (from dataset/labels_reviewed/train, a working copy)
- Uses YOLO26n (COCO pretrained) as auxiliary evidence ONLY (not ground truth)
- Applies ONLY high-confidence automated edits; keeps uncertain cases and logs them
- Writes changes in-place to dataset/labels_reviewed/train; never touches dataset/labels/train or gold
- Outputs: change log JSON/CSV-like entries, review queue, before/after visuals
"""
from __future__ import annotations
import json, copy, math
from pathlib import Path
from dataclasses import dataclass, asdict
from collections import Counter

ROOT = Path(__file__).resolve().parents[1]
IMG_DIR = ROOT / "dataset" / "images" / "train"
ORIG_LABELS = ROOT / "dataset" / "labels" / "train"   # read-only reference for before
WORK_LABELS = ROOT / "dataset" / "labels_reviewed" / "train"  # edited copy
OUT = ROOT / "outputs" / "review_automation"
NAMES = {0: "car", 1: "truck", 2: "bus", 3: "van"}
COCO_TO_ADAS = {2: 0, 7: 1, 5: 2}  # car, truck, bus; no van
# thresholds (conservative)
DUP_IOU = 0.82
DUP_SAME_CLS_IOU = 0.72
ADD_CONF = 0.78
ADD_OVERLAP_MAX = 0.20  # don't add if overlaps existing student box this much
YOLO_CONF_REF = 0.25

@dataclass
class Box:
    cls: int
    xc: float; yc: float; w: float; h: float
    origin: str = "student"  # student|added
    note: str = ""
    @property
    def corners(self):
        return (self.xc-self.w/2, self.yc-self.h/2, self.xc+self.w/2, self.yc+self.h/2)
    def line(self):
        return f"{self.cls} {self.xc:.6f} {self.yc:.6f} {self.w:.6f} {self.h:.6f}"

def iou(a, b):
    ax1,ay1,ax2,ay2=a.corners; bx1,by1,bx2,by2=b.corners
    iw=max(0.0,min(ax2,bx2)-max(ax1,bx1)); ih=max(0.0,min(ay2,by2)-max(ay1,by1))
    inter=iw*ih; uni=a.w*a.h+b.w*b.h-inter
    return inter/uni if uni else 0.0

def clamp(b: Box):
    b.w=max(1e-6,min(1.0,b.w)); b.h=max(1e-6,min(1.0,b.h))
    b.xc=max(b.w/2,min(1-b.w/2,b.xc)); b.yc=max(b.h/2,min(1-b.h/2,b.yc))

def parse(p: Path):
    if not p.exists(): return []
    out=[]
    for line in p.read_text(encoding='utf-8',errors='ignore').splitlines():
        if not line.strip(): continue
        v=line.split();
        if len(v)!=5: continue
        c=int(float(v[0])); xc,yc,w,h=map(float,v[1:])
        b=Box(c,xc,yc,w,h); clamp(b); out.append(b)
    return out

def save(p: Path, boxes):
    p.parent.mkdir(parents=True, exist_ok=True)
    txt = "\n".join(b.line() for b in boxes)
    p.write_text((txt+"\n") if txt else "", encoding='utf-8')

def yolo_detect(model, img_path):
    res = model.predict(source=str(img_path), verbose=False, conf=YOLO_CONF_REF, iou=0.45)[0]
    dets=[]
    import cv2
    im=cv2.imread(str(img_path)); h,w=im.shape[:2]
    for box in res.boxes:
        coco=int(box.cls[0].item())
        if coco not in COCO_TO_ADAS: continue
        adas=COCO_TO_ADAS[coco]
        conf=float(box.conf[0].item())
        x1,y1,x2,y2=box.xyxy[0].tolist()
        x1=max(0,min(w,x1)); x2=max(0,min(w,x2)); y1=max(0,min(h,y1)); y2=max(0,min(h,y2))
        bw=max(1.0,x2-x1); bh=max(1.0,y2-y1)
        b=Box(adas,(x1+bw/2)/w,(y1+bh/2)/h,bw/w,bh/h,origin="yolo",note=f"yolo_conf={conf:.3f}")
        clamp(b); dets.append((b,conf,coco))
    return dets

def best_yolo_support(stu: Box, ydets):
    best=(None,0.0,None)
    for b,conf,coco in ydets:
        ov=iou(stu,b)
        if ov>best[1]: best=(b,ov,conf)
    return best  # (ybox, iou, conf)

def draw_set(img_path, boxes, title, out_path, color=(0,200,0)):
    import cv2
    im=cv2.imread(str(img_path));
    if im is None: return
    h,w=im.shape[:2]
    for b in boxes:
        x1,y1,x2,y2=b.corners
        p1=(max(0,int(x1*w)),max(0,int(y1*h))); p2=(min(w-1,int(x2*w)),min(h-1,int(y2*h)))
        c=color if b.origin!='added' else (255,180,0)
        cv2.rectangle(im,p1,p2,c,2)
        lab=f"{NAMES.get(b.cls,b.cls)}{'*' if b.origin=='added' else ''}"
        cv2.putText(im,lab,(p1[0],max(12,p1[1]-3)),cv2.FONT_HERSHEY_SIMPLEX,0.4,c,1,cv2.LINE_AA)
    cv2.putText(im,f"{title} n={len(boxes)}",(8,18),cv2.FONT_HERSHEY_SIMPLEX,0.55,(255,255,255),2,cv2.LINE_AA)
    cv2.putText(im,f"{title} n={len(boxes)}",(8,18),cv2.FONT_HERSHEY_SIMPLEX,0.55,color,1,cv2.LINE_AA)
    cv2.imwrite(str(out_path), im)

def process_image(model, stem, apply_auto=True):
    img = IMG_DIR/f"{stem}.jpg"
    if not img.exists(): img = IMG_DIR/f"{stem}.png"
    before = parse(ORIG_LABELS/f"{stem}.txt")
    work = parse(WORK_LABELS/f"{stem}.txt")
    ydets = yolo_detect(model, img)
    changes=[]; review=[]
    removed=set()

    # 1) duplicates (high confidence)
    n=len(work)
    for i in range(n):
        if i in removed: continue
        for j in range(i+1,n):
            if j in removed: continue
            ov=iou(work[i],work[j])
            same = work[i].cls==work[j].cls
            if ov>=DUP_IOU or (same and ov>=DUP_SAME_CLS_IOU):
                # keep the one with better YOLO support; else keep larger
                _,oi,ci = best_yolo_support(work[i], ydets)
                _,oj,cj = best_yolo_support(work[j], ydets)
                si = oi*(ci or 0); sj = oj*(cj or 0)
                drop = j if (si>sj or (si==sj and work[i].w*work[i].h >= work[j].w*work[j].h)) else i
                keep = i if drop==j else j
                conf = 'high' if ov>=DUP_IOU else 'medium'
                if apply_auto and conf=='high':
                    removed.add(drop)
                    changes.append({"type":"remove_duplicate","indices":[i,j],"dropped":drop,"kept":keep,"iou":round(ov,4),"same_class":same,"confidence":conf,"evidence":f"student-student IoU={ov:.3f}; yolo_support i={oi:.2f}/{ci} j={oj:.2f}/{cj}"})
                else:
                    review.append({"type":"possible_duplicate","indices":[i,j],"iou":round(ov,4),"same_class":same,"confidence":conf,"evidence":"nghi trung box; can xem anh goc"})

    cur = [b for k,b in enumerate(work) if k not in removed]

    # 2) flag invalid/wrong-class/misaligned/add candidates (evidence-based, mostly review unless very clear add)
    for k,b in enumerate(cur):
        yb,ov,yc = best_yolo_support(b, ydets)
        if ov < 0.1 and (b.w*b.h < 0.0008):
            review.append({"type":"possible_invalid_or_tiny","index":k,"class":NAMES.get(b.cls,b.cls),"box":b.line(),"confidence":"low","evidence":f"no YOLO vehicle support (best IoU={ov:.2f}) and very small area={b.w*b.h:.6f}; doi chieu anh goc - khong tu xoa"})
        elif ov < 0.05:
            review.append({"type":"possible_invalid_no_support","index":k,"class":NAMES.get(b.cls,b.cls),"box":b.line(),"confidence":"low","evidence":f"no YOLO vehicle support (best IoU={ov:.2f}); co the ngoai schema hoac model bo sot - doi chieu anh goc, KHONG tu xoa"})
        if yb is not None and ov>=0.55 and yc and yc>=0.5 and yb.cls!=b.cls:
            # class disagreement
            if b.cls==3 or yb.cls==3:
                review.append({"type":"class_ambiguous_van","index":k,"student":NAMES.get(b.cls,b.cls),"yolo":NAMES.get(yb.cls,yb.cls),"iou":round(ov,3),"yconf":round(yc,3),"confidence":"low","evidence":"COCO khong co van; lien quan van/truck/bus/car - can xem anh goc theo guideline"})
            else:
                review.append({"type":"possible_wrong_class","index":k,"student":NAMES.get(b.cls,b.cls),"yolo_suggest":NAMES.get(yb.cls,yb.cls),"iou":round(ov,3),"yconf":round(yc,3),"confidence":"medium","evidence":"YOLO(car/truck/bus) khac class SV & overlap cao; co the dung hoac sai (pickup/truck/bus) - can xem anh goc, KHONG tu doi"})

    # 3) high-confidence additions (car/truck/bus only, very conservative)
    adds=[]
    for yb,yc,coco in ydets:
        if yc < ADD_CONF: continue
        maxov=max((iou(yb,b) for b in cur), default=0.0)
        if maxov <= ADD_OVERLAP_MAX:
            nb=Box(yb.cls,yb.xc,yb.yc,yb.w,yb.h,origin='added',note=f'auto_add_yolo_conf={yc:.3f}')
            clamp(nb); adds.append(nb)
            changes.append({"type":"add_high_conf","class":NAMES.get(nb.cls,nb.cls),"box":nb.line(),"yolo_conf":round(yc,3),"max_overlap_existing":round(maxov,3),"confidence":"high_but_model_based","evidence":"YOLO26n high-conf car/truck/bus, no existing overlap; model NOT ground truth - can xoa/sua khi review"})
    if apply_auto:
        cur = cur + adds
    else:
        for a in adds: review.append({"type":"suggest_add","class":NAMES.get(a.cls,a.cls),"box":a.line(),"confidence":"medium"})

    # save reviewed
    save(WORK_LABELS/f"{stem}.txt", cur)

    # visuals
    (OUT/"before_after"/stem).mkdir(parents=True, exist_ok=True)
    draw_set(img,before,f"BEFORE {stem}",OUT/"before_after"/stem/"before.jpg",(180,180,180))
    draw_set(img,cur,f"AFTER {stem}",OUT/"before_after"/stem/"after.jpg",(0,170,0))
    # side by side
    import cv2
    a=cv2.imread(str(OUT/"before_after"/stem/"before.jpg")); b=cv2.imread(str(OUT/"before_after"/stem/"after.jpg"))
    if a is not None and b is not None:
        side=cv2.hconcat([a,b]); cv2.imwrite(str(OUT/"before_after"/f"{stem}_compare.jpg"), side)

    return {"image":stem,"before_n":len(before),"after_n":len(cur),"added":len(adds),"removed":len(removed),"changes":changes,"review_queue":review,"yolo_ref_count":len(ydets)}

def main():
    import sys
    from ultralytics import YOLO
    OUT.mkdir(parents=True, exist_ok=True)
    model=YOLO('yolo26n.pt')
    targets = sys.argv[1:] if len(sys.argv)>1 else ['drive_009','drive_041','drive_030','drive_002','drive_012']
    results=[]
    for s in targets:
        print('processing',s)
        r=process_image(model,s,apply_auto=True)
        results.append(r)
        print(' ', {k:r[k] for k in ['image','before_n','after_n','added','removed','review_queue']})
        print('  review_n', len(r['review_queue']), 'changes_n', len(r['changes']))
    (OUT/'auto_review_log.json').write_text(json.dumps({"targets":targets,"results":results,"policy":{"no_gold_used_for_edits":True,"model_not_ground_truth":True,"conservative_auto":True}},ensure_ascii=False,indent=2),encoding='utf-8')
    print('LOG', OUT/'auto_review_log.json')

if __name__=='__main__':
    main()
