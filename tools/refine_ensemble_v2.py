"""
Feasible improvement attempt (image-evidence only, NO Gold):
Reclassify ensemble boxes using YOLO-World crop verification on original images
+ simple geometric heuristics (aspect/relative size). Conservative: change class only
with sufficient image-model evidence. Output: dataset/labels_ensemble_v2/train.
"""
import json
from pathlib import Path
import cv2
from ultralytics import YOLO

ROOT=Path(r"D:\Day2-Lab\K5-L2-DAY02-NgoThiMinhQuang-2A202603146-ObjectDetection")
SRC=ROOT/'dataset/labels_ensemble/train'; DST=ROOT/'dataset/labels_ensemble_v2/train'; IMG=ROOT/'dataset/images/train'
DST.mkdir(parents=True,exist_ok=True)
ADAS={0:'car',1:'truck',2:'bus',3:'van'}
log={'changes':[],'stats':{'same':0,'changed':0,'total':0}}

def read(p):
    b=[]
    for l in p.read_text().splitlines():
        if not l.strip(): continue
        c,xc,yc,w,h=map(float,l.split()[:5]); b.append({'c':int(c),'xc':xc,'yc':yc,'w':w,'h':h})
    return b
def write(p,boxes):
    (p).write_text('\n'.join(f"{b['c']} {b['xc']:.6f} {b['yc']:.6f} {b['w']:.6f} {b['h']:.6f}" for b in boxes)+('\n' if boxes else ''),encoding='utf-8')

mw=YOLO('yolov8s-worldv2.pt')
PROMPTS=['car','sedan','suv','truck','pickup truck','lorry','delivery truck','bus','coach','minibus','van','minivan','cargo van']
VMAP={'car':0,'sedan':0,'suv':0,'truck':1,'pickup truck':1,'lorry':1,'delivery truck':1,'bus':2,'coach':2,'minibus':2,'van':3,'minivan':3,'cargo van':3}

for i in range(1,51):
    stem=f'drive_{i:03d}'
    im=cv2.imread(str(IMG/f'{stem}.jpg')); h,w=im.shape[:2]
    boxes=read(SRC/f'{stem}.txt')
    nb=[]
    for bi,b in enumerate(boxes):
        log['stats']['total']+=1
        x1=int((b['xc']-b['w']/2)*w); y1=int((b['yc']-b['h']/2)*h); x2=int((b['xc']+b['w']/2)*w); y2=int((b['yc']+b['h']/2)*h)
        pad=int(0.22*max(x2-x1,y2-y1,16))
        xa=max(0,x1-pad); ya=max(0,y1-pad); xb=min(w,x2+pad); yb=min(h,y2+pad)
        crop=im[ya:yb,xa:xb]
        newc=b['c']; evidence='kept'
        if crop.size>0 and crop.shape[0]>=14 and crop.shape[1]>=14:
            mw.set_classes(PROMPTS)
            r=mw.predict(crop,conf=0.08,imgsz=320,verbose=False)[0]
            scores={}
            for det in r.boxes:
                raw=r.names[int(det.cls[0])]; cf=float(det.conf[0])
                if raw in VMAP:
                    ac=VMAP[raw]; scores[ac]=max(scores.get(ac,0),cf)
            if scores:
                best=max(scores,key=scores.get); bconf=scores[best]
                cur=scores.get(b['c'],0)
                # geometric hints from crop/box (image evidence)
                ar=(x2-x1)/max(1,(y2-y1)); area=(b['w']*b['h'])
                geo=None
                if area>0.025 and ar<1.6 and ar>0.6 and bconf<0.55:
                    geo=2 # large/tall-ish -> bus candidate hint (weak)
                # conservative change rules
                if best!=b['c'] and bconf>=0.38 and bconf>=cur+0.12:
                    # extra caution for van: require stronger evidence
                    if best==3 and bconf<0.45: pass
                    elif best==2 and geo!=2 and bconf<0.48 and cur>0.15: pass
                    else:
                        newc=best; evidence=f'world_crop best={ADAS[best]}:{bconf:.2f} cur={ADAS[b["c"]]}:{cur:.2f}'
                # if current is car but strong van signal
                if b['c']==0 and scores.get(3,0)>=0.46 and scores.get(3,0)>=max(scores.get(0,0),0)+0.10:
                    newc=3; evidence=f'crop_van_signal {scores.get(3,0):.2f}'
        if newc!=b['c']:
            log['changes'].append({'img':stem,'idx':bi,'from':ADAS[b['c']],'to':ADAS[newc],'evidence':evidence})
            log['stats']['changed']+=1
        else: log['stats']['same']+=1
        nb.append({'c':newc,'xc':b['xc'],'yc':b['yc'],'w':b['w'],'h':b['h']})
    write(DST/f'{stem}.txt', nb)
    if i%10==0: print('done',i, log['stats'], flush=True)
(ROOT/'outputs/ensemble_v2_log.json').write_text(json.dumps(log,ensure_ascii=False,indent=2),encoding='utf-8')
print('FINAL',log['stats'],'log outputs/ensemble_v2_log.json', flush=True)
