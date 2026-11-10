"""Queue-only evidence pass for all 50 images. NO label edits, NO Gold used."""
import json
from pathlib import Path
from dataclasses import dataclass
ROOT = Path(__file__).resolve().parents[1]
IMG_DIR = ROOT/"dataset/images/train"
LBL_DIR = ROOT/"dataset/labels_reviewed/train"
OUT = ROOT/"outputs/review_automation/review_queue_all.json"
NAMES={0:"car",1:"truck",2:"bus",3:"van"}
COCO_TO_ADAS={2:0,7:1,5:2}
@dataclass
class Box:
    cls:int; xc:float; yc:float; w:float; h:float
    @property
    def corners(self): return (self.xc-self.w/2,self.yc-self.h/2,self.xc+self.w/2,self.yc+self.h/2)
    def line(self): return f"{self.cls} {self.xc:.6f} {self.yc:.6f} {self.w:.6f} {self.h:.6f}"
def iou(a,b):
    ax1,ay1,ax2,ay2=a.corners; bx1,by1,bx2,by2=b.corners
    iw=max(0,min(ax2,bx2)-max(ax1,bx1)); ih=max(0,min(ay2,by2)-max(ay1,by1))
    inter=iw*ih; uni=a.w*a.h+b.w*b.h-inter
    return inter/uni if uni else 0
def parse(p):
    out=[]
    if not p.exists(): return out
    for line in p.read_text(encoding='utf-8',errors='ignore').splitlines():
        if not line.strip(): continue
        v=line.split();
        if len(v)!=5: continue
        c=int(float(v[0])); xc,yc,w,h=map(float,v[1:]); out.append(Box(c,xc,yc,w,h))
    return out
def main():
    from ultralytics import YOLO
    model=YOLO('yolo26n.pt')
    results=[]
    for i in range(1,51):
        stem=f"drive_{i:03d}"
        img=IMG_DIR/f"{stem}.jpg"
        boxes=parse(LBL_DIR/f"{stem}.txt")
        res=model.predict(source=str(img),verbose=False,conf=0.25,iou=0.45)[0]
        ydets=[]
        im_h,im_w=res.orig_shape
        for box in res.boxes:
            coco=int(box.cls[0].item());
            if coco not in COCO_TO_ADAS: continue
            adas=COCO_TO_ADAS[coco]; conf=float(box.conf[0].item())
            x1,y1,x2,y2=box.xyxy[0].tolist()
            x1=max(0,min(im_w,x1));x2=max(0,min(im_w,x2));y1=max(0,min(im_h,y1));y2=max(0,min(im_h,y2))
            bw=max(1,x2-x1); bh=max(1,y2-y1)
            ydets.append((Box(adas,(x1+bw/2)/im_w,(y1+bh/2)/im_h,bw/im_w,bh/im_h),conf))
        review=[]; dups=[]
        for a in range(len(boxes)):
            for b in range(a+1,len(boxes)):
                ov=iou(boxes[a],boxes[b])
                if ov>=0.72 and boxes[a].cls==boxes[b].cls: dups.append({"i":a,"j":b,"iou":round(ov,3),"cls":NAMES[boxes[a].cls]})
                elif ov>=0.82: dups.append({"i":a,"j":b,"iou":round(ov,3),"cls_a":NAMES[boxes[a].cls],"cls_b":NAMES[boxes[b].cls]})
        for k,b in enumerate(boxes):
            best=max(((iou(b,yb),yc,yb) for yb,yc in ydets), default=(0,0,None), key=lambda x:x[0])
            ov,yc,yb=best
            if ov<0.05: review.append({"type":"no_yolo_support","index":k,"class":NAMES[b.cls],"box":b.line(),"confidence":"low","evidence":"no YOLO vehicle support; doi chieu anh goc, KHONG tu xoa"})
            elif ov>=0.55 and yc>=0.5 and yb.cls!=b.cls:
                t="class_ambiguous_van" if (b.cls==3 or yb.cls==3) else "possible_wrong_class"
                review.append({"type":t,"index":k,"student":NAMES[b.cls],"yolo_suggest":NAMES[yb.cls],"iou":round(ov,3),"yconf":round(yc,3),"confidence":"low" if t=="class_ambiguous_van" else "medium"})
        results.append({"image":stem,"n":len(boxes),"yolo_ref":len(ydets),"dup_candidates":dups,"review_queue":review})
        if i%10==0: print('done',i)
    OUT.parent.mkdir(parents=True,exist_ok=True)
    OUT.write_text(json.dumps({"no_gold_used":True,"results":results},ensure_ascii=False,indent=2),encoding='utf-8')
    print('saved',OUT, 'total_flags', sum(len(r['review_queue']) for r in results), 'dup_cands', sum(len(r['dup_candidates']) for r in results))
if __name__=='__main__': main()
