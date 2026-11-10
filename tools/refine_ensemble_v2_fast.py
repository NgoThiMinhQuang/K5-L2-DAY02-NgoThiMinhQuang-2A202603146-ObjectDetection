"""Targeted refinement (image-evidence, NO Gold): re-verify ONLY minority-class boxes
(truck/bus/van) + car boxes with van/bus/truck world signal via crop inference. Fast enough on CPU."""
import json
from pathlib import Path
import cv2
from ultralytics import YOLO
ROOT=Path(r"D:\Day2-Lab\K5-L2-DAY02-NgoThiMinhQuang-2A202603146-ObjectDetection")
SRC=ROOT/'dataset/labels_ensemble/train'; DST=ROOT/'dataset/labels_ensemble_v2/train'; IMG=ROOT/'dataset/images/train'
ADAS={0:'car',1:'truck',2:'bus',3:'van'}
log={'changes':[],'stats':{'checked':0,'changed':0,'total':0}}

def read(p):
    b=[]
    for l in p.read_text().splitlines():
        if not l.strip(): continue
        c,xc,yc,w,h=map(float,l.split()[:5]); b.append({'c':int(c),'xc':xc,'yc':yc,'w':w,'h':h})
    return b
def write(p,boxes):
    p.write_text('\n'.join(f"{b['c']} {b['xc']:.6f} {b['yc']:.6f} {b['w']:.6f} {b['h']:.6f}" for b in boxes)+('\n' if boxes else ''),encoding='utf-8')

mw=YOLO('yolov8s-worldv2.pt')
PROMPTS=['car','suv','truck','pickup truck','bus','coach','van','minivan']
VMAP={'car':0,'suv':0,'truck':1,'pickup truck':1,'bus':2,'coach':2,'van':3,'minivan':3}
mw.set_classes(PROMPTS)

for i in range(1,51):
    stem=f'drive_{i:03d}'; im=cv2.imread(str(IMG/f'{stem}.jpg')); h,w=im.shape[:2]
    boxes=read(SRC/f'{stem}.txt'); nb=[]
    for bi,b in enumerate(boxes):
        log['stats']['total']+=1; newc=b['c']
        # target minority classes and a sample of cars? check all non-car + some heuristic
        target = (b['c'] in (1,2,3)) or (b['w']*b['h']>0.015) # non-car or large vehicle
        if target:
            log['stats']['checked']+=1
            x1=int((b['xc']-b['w']/2)*w); y1=int((b['yc']-b['h']/2)*h); x2=int((b['xc']+b['w']/2)*w); y2=int((b['yc']+b['h']/2)*h)
            pad=int(0.22*max(x2-x1,y2-y1,16))
            crop=im[max(0,y1-pad):min(h,y2+pad), max(0,x1-pad):min(w,x2+pad)]
            if crop.size and crop.shape[0]>=14 and crop.shape[1]>=14:
                r=mw.predict(crop,conf=0.08,imgsz=256,verbose=False)[0]
                scores={}
                for det in r.boxes:
                    raw=r.names[int(det.cls[0])]; cf=float(det.conf[0])
                    if raw in VMAP: scores[VMAP[raw]]=max(scores.get(VMAP[raw],0),cf)
                if scores:
                    best=max(scores,key=scores.get); bconf=scores[best]; cur=scores.get(b['c'],0)
                    if best!=b['c'] and bconf>=0.40 and bconf>=cur+0.15:
                        if best==3 and bconf<0.48: pass
                        else:
                            newc=best
                            log['changes'].append({'img':stem,'idx':bi,'from':ADAS[b['c']],'to':ADAS[newc],'conf':round(bconf,3),'cur_conf':round(cur,3)})
                            log['stats']['changed']+=1
        nb.append({'c':newc,'xc':b['xc'],'yc':b['yc'],'w':b['w'],'h':b['h']})
    write(DST/f'{stem}.txt',nb)
    if i%10==0: print('done',i,log['stats'],flush=True)
(ROOT/'outputs/ensemble_v2_fast_log.json').write_text(json.dumps(log,ensure_ascii=False,indent=2),encoding='utf-8')
print('FINAL',log['stats'],flush=True)
