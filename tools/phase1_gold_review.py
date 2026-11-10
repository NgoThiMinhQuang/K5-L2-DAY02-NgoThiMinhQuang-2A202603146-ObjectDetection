"""
Phase 1 (READ-ONLY analysis): compare student labels vs gold_train_labels.
Replicates tools/evaluate_annotations.py matching (same class, IoU>=0.5, one-to-one greedy by IoU desc).
Produces per-image stats, top-5 severe visuals, and a review report. Does NOT modify labels or gold.
"""
from __future__ import annotations
import json, math
from dataclasses import dataclass, asdict
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parents[1]
STUDENT = ROOT / "dataset" / "labels" / "train"
GOLD = ROOT / "gold_train_labels"
IMAGES = ROOT / "dataset" / "images" / "train"
OUT = ROOT / "outputs" / "gold_review_phase1"
VIS = OUT / "visuals"
IOU_T = 0.50
NAMES = {0: "car", 1: "truck", 2: "bus", 3: "van"}
COL = {
    "tp_s": (40, 180, 40), "tp_g": (0, 120, 0),
    "fp": (40, 40, 230), "fn": (230, 180, 40),
    "low": (0, 165, 255),
}

@dataclass(frozen=True)
class Box:
    idx: int
    cls: int
    xc: float; yc: float; w: float; h: float
    @property
    def corners(self):
        return (self.xc-self.w/2, self.yc-self.h/2, self.xc+self.w/2, self.yc+self.h/2)

def parse(p: Path):
    if not p.exists(): return []
    boxes=[]
    for i,line in enumerate(p.read_text(encoding='utf-8').splitlines(),1):
        if not line.strip(): continue
        v=line.split();
        if len(v)!=5: continue
        c=int(float(v[0])); xc,yc,w,h=map(float,v[1:])
        boxes.append(Box(len(boxes),c,xc,yc,w,h))
    return boxes

def iou(a:Box,b:Box):
    ax1,ay1,ax2,ay2=a.corners; bx1,by1,bx2,by2=b.corners
    iw=max(0.0,min(ax2,bx2)-max(ax1,bx1)); ih=max(0.0,min(ay2,by2)-max(ay1,by1))
    inter=iw*ih; uni=a.w*a.h+b.w*b.h-inter
    return inter/uni if uni else 0.0

def match(student,gold,thr=IOU_T):
    cands=sorted(((iou(s,g),si,gi) for si,s in enumerate(student) for gi,g in enumerate(gold) if s.cls==g.cls), reverse=True)
    us,ug=set(),set(); m=[]
    for ov,si,gi in cands:
        if ov<thr: break
        if si not in us and gi not in ug:
            us.add(si); ug.add(gi); m.append((si,gi,ov))
    return m, us, ug

def cross_class(student,gold,thr=0.3):
    # student-gold overlaps >=thr but different class (diagnostic)
    out=[]
    for si,s in enumerate(student):
        for gi,g in enumerate(gold):
            if s.cls==g.cls: continue
            ov=iou(s,g)
            if ov>=thr: out.append({"s_i":si,"s_cls":NAMES.get(s.cls,s.cls),"g_i":gi,"g_cls":NAMES.get(g.cls,g.cls),"iou":round(ov,4)})
    out.sort(key=lambda x:-x['iou'])
    return out[:12]

def draw_box(im, b, color, label, thick=2):
    import cv2
    h,w=im.shape[:2]
    x1,y1,x2,y2=b.corners
    p1=(max(0,int(x1*w)),max(0,int(y1*h))); p2=(min(w-1,int(x2*w)),min(h-1,int(y2*h)))
    cv2.rectangle(im,p1,p2,color,thick)
    (tw,th),_=cv2.getTextSize(label,cv2.FONT_HERSHEY_SIMPLEX,0.4,1)
    y=p1[1]-4 if p1[1]-4>th else p1[1]+th+4
    cv2.rectangle(im,(p1[0],y-th-3),(p1[0]+tw+2,y+2),color,-1)
    cv2.putText(im,label,(p1[0]+1,y),cv2.FONT_HERSHEY_SIMPLEX,0.4,(255,255,255),1,cv2.LINE_AA)

def main():
    import cv2
    OUT.mkdir(parents=True, exist_ok=True); VIS.mkdir(parents=True, exist_ok=True)
    gold_files=sorted(GOLD.glob("*.txt"))
    per=[]; all_overlaps=[]; TP=FP=FN=0
    for gf in gold_files:
        stem=gf.stem
        gold=parse(gf); stu=parse(STUDENT/f"{stem}.txt")
        m,us,ug=match(stu,gold)
        tp=len(m); fp=len(stu)-tp; fn=len(gold)-tp
        TP+=tp; FP+=fp; FN+=fn
        for _,_,ov in m: all_overlaps.append(ov)
        low=[x for x in m if x[2]<0.7]
        fp_boxes=[{"i":s.idx,"cls":NAMES.get(s.cls,s.cls),"xywh":[s.xc,s.yc,s.w,s.h]} for i,s in enumerate(stu) if i not in us]
        fn_boxes=[{"i":g.idx,"cls":NAMES.get(g.cls,g.cls),"xywh":[g.xc,g.yc,g.w,g.h]} for i,g in enumerate(gold) if i not in ug]
        tp_boxes=[{"s":a,"g":b,"iou":round(o,4),"s_cls":NAMES.get(stu[a].cls,stu[a].cls),"g_cls":NAMES.get(gold[b].cls,gold[b].cls)} for a,b,o in m]
        cross=cross_class(stu,gold)
        # severity: prioritize images where fixes help F1 most; also absolute errors
        severity = fp + fn*2  # missing weighted a bit higher for recall? keep simple + include components
        per.append({"image":stem,"gold":len(gold),"student":len(stu),"tp":tp,"fp":fp,"fn":fn,
                    "severity":severity,"low_iou_matches":len(low),
                    "mean_iou_matches": round(sum(o for _,_,o in m)/len(m),4) if m else None,
                    "fp_boxes":fp_boxes,"fn_boxes":fn_boxes,"tp_boxes":tp_boxes,
                    "low_matches":[{"s":a,"g":b,"iou":round(o,4)} for a,b,o in low],
                    "cross_class_hints":cross})

    per.sort(key=lambda x:(-x['severity'],-x['fp'],-x['fn'],x['image']))
    prec=TP/(TP+FP) if TP+FP else 0; rec=TP/(TP+FN) if TP+FN else 0
    f1=2*prec*rec/(prec+rec) if prec+rec else 0; miou=sum(all_overlaps)/len(all_overlaps) if all_overlaps else 0
    summary={"precision":round(prec,4),"recall":round(rec,4),"f1":round(f1,4),"mean_iou":round(miou,4),"TP":TP,"FP":FP,"FN":FN,"student_boxes":TP+FP,"gold_boxes":TP+FN}
    top5=per[:5]

    # visuals for top 5
    for item in top5:
        stem=item['image']
        imgp=IMAGES/f"{stem}.jpg"
        if not imgp.exists():
            # try png
            for ext in ['.png','.jpeg']: 
                q=IMAGES/f"{stem}{ext}";
                if q.exists(): imgp=q; break
        im=cv2.imread(str(imgp))
        if im is None: continue
        gold=parse(GOLD/f"{stem}.txt"); stu=parse(STUDENT/f"{stem}.txt")
        m,us,ug=match(stu,gold)
        lowset={(a,b) for a,b,o in m if o<0.7}
        # draw FN (gold unmatched) dashed-ish via thick cyan
        for i,g in enumerate(gold):
            if i not in ug: draw_box(im,g,COL['fn'],f"FN g{i}:{NAMES.get(g.cls,g.cls)}",2)
        # TP
        for a,b,o in m:
            c=COL['low'] if (a,b) in lowset else COL['tp_s']
            draw_box(im,stu[a],c,f"TP s{a}:{NAMES.get(stu[a].cls,stu[a].cls)} iou={o:.2f}",2)
        # FP
        for i,s in enumerate(stu):
            if i not in us: draw_box(im,s,COL['fp'],f"FP s{i}:{NAMES.get(s.cls,s.cls)}",2)
        # legend
        y=18
        for txt,c in [("TP/khop (xanh la)",COL['tp_s']),("TP IoU<0.7 (cam)",COL['low']),("FP: box SV khong khop gold (do)",COL['fp']),("FN: box gold SV thieu (xanh duong)",COL['fn'])]:
            cv2.putText(im,txt,(8,y),cv2.FONT_HERSHEY_SIMPLEX,0.45,c,1,cv2.LINE_AA); y+=16
        cv2.putText(im,f"{stem} TP={item['tp']} FP={item['fp']} FN={item['fn']} (SV={item['student']} Gold={item['gold']})",(8,y+4),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1,cv2.LINE_AA)
        cv2.imwrite(str(VIS/f"{stem}_review.jpg"), im)

    report={"summary":summary,"top5":[{k:v for k,v in x.items()} for x in top5],
            "priority_all":[{"image":x['image'],"tp":x['tp'],"fp":x['fp'],"fn":x['fn'],"severity":x['severity'],"student":x['student'],"gold":x['gold'],"cross_class_hints_n":len(x['cross_class_hints'])} for x in per],
            "notes":["Matching: same class, IoU>=0.5, one-to-one greedy IoU desc (as evaluate_annotations.py).","FP/FN theo gold-match; box FP chua chac sai thuc te - can doi chieu anh goc + guideline.","No labels/gold/CVAT modified."]}
    (OUT/"phase1_report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({"summary":summary,"top5":[x['image'] for x in top5],"out":str(OUT)},ensure_ascii=False,indent=2))

if __name__=='__main__':
    main()
