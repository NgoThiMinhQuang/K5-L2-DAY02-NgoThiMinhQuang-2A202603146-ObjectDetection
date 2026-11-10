"""
Generate a fresh label set from YOLO-World (open-vocab) for ADAS 4 classes.
Output: dataset/labels_yolo_new/train/*.txt + outputs/yolo_new_previews/*.jpg
No Gold used. No overwrite of dataset/labels/train.
"""
import json, os, sys
from pathlib import Path
import cv2
from ultralytics import YOLO

ROOT = Path(r"D:\Day2-Lab\K5-L2-DAY02-NgoThiMinhQuang-2A202603146-ObjectDetection")
IMG_DIR = ROOT / "dataset/images/train"
OUT_LBL = ROOT / "dataset/labels_yolo_new/train"
OUT_PREV = ROOT / "outputs/yolo_new_previews"
OUT_LBL.mkdir(parents=True, exist_ok=True)
OUT_PREV.mkdir(parents=True, exist_ok=True)

ADAS = {0:'car',1:'truck',2:'bus',3:'van'}
COLORS = {0:(60,220,60),1:(255,160,0),2:(255,80,80),3:(220,80,255)} # BGR

# World classes to prompt - include variants to improve recall, then map carefully
PROMPTS = ['car','automobile','truck','pickup truck','lorry','bus','coach','van','minivan','minibus','suv','person','motorcycle','bicycle']
MAP = {
    'car':0,'automobile':0,'suv':0,
    'truck':1,'lorry':1,'pickup truck':1,  # pickup ambiguous but closer to truck/cargo in many cases; logged
    'bus':2,'coach':2,'minibus':2,  # minibus can be van-like; mapping to bus per common large-passenger; imperfect
    'van':3,'minivan':3,
}
IGNORE = {'person','motorcycle','bicycle'}

def iou_xyxy(a,b):
    x1=max(a[0],b[0]); y1=max(a[1],b[1]); x2=min(a[2],b[2]); y2=min(a[3],b[3])
    inter=max(0,x2-x1)*max(0,y2-y1)
    aa=max(0,a[2]-a[0])*max(0,a[3]-a[1]); ab=max(0,b[2]-b[0])*max(0,b[3]-b[1])
    return inter/(aa+ab-inter+1e-9)

def xyxy_to_norm(xyxy,w,h):
    x1,y1,x2,y2=xyxy
    x1=max(0,min(w-1,x1)); y1=max(0,min(h-1,y1)); x2=max(0,min(w,x2)); y2=max(0,min(h,y2))
    bw=max(1.0,x2-x1); bh=max(1.0,y2-y1)
    xc=(x1+x2)/2/w; yc=(y1+y2)/2/h; nw=bw/w; nh=bh/h
    return [xc,yc,nw,nh]

def nms_adas(dets, iou_thr=0.55):
    # dets: dict cls,conf,xyxy,norm,raw
    dets=sorted(dets,key=lambda d:d['conf'],reverse=True)
    keep=[]
    for d in dets:
        drop=False
        for k in keep:
            ov=iou_xyxy(d['xyxy'],k['xyxy'])
            if ov>=iou_thr:
                # same location -> 1 vehicle/1 box. keep higher conf; if classes differ keep first (higher conf)
                drop=True; break
        if not drop: keep.append(d)
    return keep

def main(model_name='yolov8s-worldv2.pt', conf=0.18, imgsz=1280, tag='world'):
    print('loading', model_name)
    model = YOLO(model_name)
    model.set_classes(PROMPTS)
    report={'model':model_name,'conf':conf,'imgsz':imgsz,'tag':tag,'images':{},'totals':{'boxes':0,'by_class':{v:0 for v in ADAS.values()}}}

    for i in range(1,51):
        stem=f'drive_{i:03d}'
        imgp=IMG_DIR/f'{stem}.jpg'
        img=cv2.imread(str(imgp))
        if img is None:
            print('missing', imgp); continue
        h,w=img.shape[:2]
        res=model.predict(str(imgp), conf=conf, imgsz=imgsz, verbose=False, agnostic_nms=False)[0]
        names=res.names
        cands=[]
        for b in res.boxes:
            ci=int(b.cls[0]); raw=names[ci]; c=float(b.conf[0])
            xyxy=[float(x) for x in b.xyxy[0].tolist()]
            if raw in IGNORE: continue
            if raw not in MAP: continue
            ac=MAP[raw]
            # basic size sanity: filter absurdly tiny unless confident
            x1,y1,x2,y2=xyxy
            bw=max(0,x2-x1); bh=max(0,y2-y1)
            area_norm=(bw*bh)/(w*h+1e-9)
            if area_norm < 0.0006 and c < 0.45: continue
            if bw < 4 or bh < 4: continue
            cands.append({'cls':ac,'conf':c,'xyxy':xyxy,'norm':xyxy_to_norm(xyxy,w,h),'raw':raw})
        kept=nms_adas(cands, iou_thr=0.50)
        # write label
        lp=OUT_LBL/f'{stem}.txt'
        lines=[]
        for d in sorted(kept, key=lambda z:(z['norm'][1],z['norm'][0])):
            n=d['norm']
            n=[max(0.0,min(1.0,x)) for x in n]
            lines.append(f"{d['cls']} {n[0]:.6f} {n[1]:.6f} {n[2]:.6f} {n[3]:.6f}")
        lp.write_text('\n'.join(lines)+('\n' if lines else ''), encoding='utf-8')
        # preview
        vis=img.copy()
        for idx,d in enumerate(kept):
            x1,y1,x2,y2=map(int,d['xyxy'])
            col=COLORS[d['cls']]
            cv2.rectangle(vis,(x1,y1),(x2,y2),col,2)
            label=f"#{idx} {ADAS[d['cls']]} {d['conf']:.2f}"
            (tw,th),_=cv2.getTextSize(label,cv2.FONT_HERSHEY_SIMPLEX,0.45,1)
            ytxt=max(12,y1-4)
            cv2.rectangle(vis,(x1,ytxt-th-4),(x1+tw+2,ytxt+2),col,-1)
            cv2.putText(vis,label,(x1+1,ytxt),cv2.FONT_HERSHEY_SIMPLEX,0.45,(0,0,0),1,cv2.LINE_AA)
        cv2.imwrite(str(OUT_PREV/f'{stem}.jpg'), vis)
        by={}
        for d in kept: by[ADAS[d['cls']]]=by.get(ADAS[d['cls']],0)+1
        report['images'][stem]={'count':len(kept),'by_class':by}
        report['totals']['boxes']+=len(kept)
        for d in kept: report['totals']['by_class'][ADAS[d['cls']]]+=1
        if i%10==0: print('done',i,'boxes so far',report['totals']['boxes'])
    rp=ROOT/f'outputs/yolo_new_{tag}_report.json'
    rp.parent.mkdir(parents=True, exist_ok=True)
    rp.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print('TOTAL', report['totals'])
    print('report', rp)
    return report

if __name__=='__main__':
    tag=sys.argv[1] if len(sys.argv)>1 else 'world1280'
    conf=float(sys.argv[2]) if len(sys.argv)>2 else 0.18
    imgsz=int(sys.argv[3]) if len(sys.argv)>3 else 1280
    main(conf=conf,imgsz=imgsz,tag=tag)
