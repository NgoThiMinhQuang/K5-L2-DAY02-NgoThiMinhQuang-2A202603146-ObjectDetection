"""
High-quality ensemble annotation pipeline (CPU-efficient):
- Sources: YOLO26n (COCO car/bus/truck) + YOLO-World v2 (open-vocab incl. van/minivan).
- IoU clustering/matching + NMS (not simple union), 1 vehicle/1 box.
- Class decision via source agreement/confidence; world used for van signal (26n lacks van).
- Filters out-of-scope (person/motorcycle/bicycle) using world detections.
- Writes dataset/labels_ensemble/train + outputs/ensemble_previews + detailed log.
- NO Gold used for any label decision.
"""
import json, sys
from pathlib import Path
from collections import defaultdict, Counter
import cv2
from ultralytics import YOLO

ROOT = Path(r"D:\Day2-Lab\K5-L2-DAY02-NgoThiMinhQuang-2A202603146-ObjectDetection")
IMG_DIR = ROOT / "dataset/images/train"
OUT_LBL = ROOT / "dataset/labels_ensemble/train"
OUT_PREV = ROOT / "outputs/ensemble_previews"
OUT_LOG = ROOT / "outputs/ensemble_pipeline_log.json"
ADAS = {0:'car',1:'truck',2:'bus',3:'van'}
COLORS = {0:(60,220,60),1:(255,160,0),2:(255,80,80),3:(220,80,255)}
OUT_LBL.mkdir(parents=True, exist_ok=True); OUT_PREV.mkdir(parents=True, exist_ok=True)

def xyxy_to_norm(xyxy,w,h):
    x1,y1,x2,y2=xyxy
    x1=max(0,min(w,x1)); y1=max(0,min(h,y1)); x2=max(0,min(w,x2)); y2=max(0,min(h,y2))
    bw=max(1.0,x2-x1); bh=max(1.0,y2-y1)
    return [(x1+x2)/2/w,(y1+y2)/2/h,bw/w,bh/h]
def clamp_norm(n):
    xc,yc,bw,bh=n
    x1=max(0,xc-bw/2); y1=max(0,yc-bh/2); x2=min(1,xc+bw/2); y2=min(1,yc+bh/2)
    bw=max(1e-6,x2-x1); bh=max(1e-6,y2-y1)
    return [(x1+x2)/2,(y1+y2)/2,bw,bh]
def iou_xyxy(a,b):
    x1=max(a[0],b[0]); y1=max(a[1],b[1]); x2=min(a[2],b[2]); y2=min(a[3],b[3])
    inter=max(0,x2-x1)*max(0,y2-y1)
    aa=max(0,a[2]-a[0])*max(0,a[3]-a[1]); ab=max(0,b[2]-b[0])*max(0,b[3]-b[1])
    return inter/(aa+ab-inter+1e-9)
def center(a): return ((a[0]+a[2])/2,(a[1]+a[3])/2)

def detect_26n(model, imgp, conf=0.12, imgsz=1280):
    cmap={2:0,7:1,5:2}
    r=model.predict(str(imgp), conf=conf, imgsz=imgsz, verbose=False)[0]
    dets=[]
    for b in r.boxes:
        c=int(b.cls[0])
        if c not in cmap: continue
        xyxy=[float(x) for x in b.xyxy[0].tolist()]
        dets.append({'cls':cmap[c],'conf':float(b.conf[0]),'xyxy':xyxy,'src':'yolo26n','raw':r.names[c]})
    return dets

def detect_world(model, imgp, conf=0.10, imgsz=1280):
    prompts=['car','truck','pickup truck','bus','coach','van','minivan','suv','person','motorcycle','bicycle']
    model.set_classes(prompts)
    r=model.predict(str(imgp), conf=conf, imgsz=imgsz, verbose=False)[0]
    wmap={'car':0,'suv':0,'truck':1,'pickup truck':1,'bus':2,'coach':2,'van':3,'minivan':3}
    ignore={'person','motorcycle','bicycle'}
    dets=[]; nonveh=[]
    for b in r.boxes:
        raw=r.names[int(b.cls[0])]; c=float(b.conf[0]); xyxy=[float(x) for x in b.xyxy[0].tolist()]
        if raw in ignore: nonveh.append({'raw':raw,'conf':c,'xyxy':xyxy}); continue
        if raw in wmap: dets.append({'cls':wmap[raw],'conf':c,'xyxy':xyxy,'src':'world','raw':raw})
    return dets, nonveh

def cluster_detections(all_dets, iou_thr=0.42):
    dets=sorted(all_dets,key=lambda d:d['conf'],reverse=True)
    used=[False]*len(dets); clusters=[]
    for i,d in enumerate(dets):
        if used[i]: continue
        cl=[d]; used[i]=True
        changed=True
        while changed:
            changed=False
            for j,e in enumerate(dets):
                if used[j]: continue
                if any(iou_xyxy(e['xyxy'],m['xyxy'])>=iou_thr for m in cl):
                    cl.append(e); used[j]=True; changed=True
        clusters.append(cl)
    return clusters

def weighted_box(cluster):
    sw=sum(d['conf'] for d in cluster)+1e-9
    x1=sum(d['xyxy'][0]*d['conf'] for d in cluster)/sw
    y1=sum(d['xyxy'][1]*d['conf'] for d in cluster)/sw
    x2=sum(d['xyxy'][2]*d['conf'] for d in cluster)/sw
    y2=sum(d['xyxy'][3]*d['conf'] for d in cluster)/sw
    best=max(cluster,key=lambda d:d['conf'])
    return [0.65*best['xyxy'][k]+0.35*[x1,y1,x2,y2][k] for k in range(4)]

def choose_class(cluster):
    by_src=defaultdict(list)
    for d in cluster: by_src[d['src']].append(d)
    confsum=defaultdict(float)
    for d in cluster: confsum[d['cls']]+=d['conf']
    # both sources agree
    if 'yolo26n' in by_src and 'world' in by_src:
        s26=set(d['cls'] for d in by_src['yolo26n'])
        sw=set(d['cls'] for d in by_src['world'])
        inter=s26 & sw
        if inter:
            c=max(inter,key=lambda x:confsum[x]); return c,'both_sources_agree'
        # disagree: world can signal van (26n cannot)
        bw=max(by_src['world'],key=lambda d:d['conf'])
        if bw['cls']==3 and bw['conf']>=0.30:
            return 3,'world_van_signal(no_van_in_coco)'
        # else prefer 26n class if confident (good at car/bus/truck), unless world much stronger
        b26=max(by_src['yolo26n'],key=lambda d:d['conf'])
        if b26['conf']>=bw['conf']*0.85:
            return b26['cls'],'prefer_yolo26n_on_conflict'
        return bw['cls'],'world_stronger_on_conflict'
    # single source
    c=max(confsum.keys(), key=lambda x:confsum[x]); return c,f"single_source_{list(by_src.keys())[0]}"

def overlaps_nonvehicle(xyxy, nonveh, thr=0.42, conf_thr=0.45):
    for n in nonveh:
        if n['conf']>=conf_thr and iou_xyxy(xyxy,n['xyxy'])>=thr: return n
    return None

def main():
    print('loading models...'); sys.stdout.flush()
    m26 = YOLO(str(ROOT/'yolo26n.pt'))
    mw  = YOLO('yolov8s-worldv2.pt')
    log={'images':{},'totals':{'kept':0,'removed':0,'reclassified':0,'total':0},'note':'no Gold used'}
    for i in range(1,51):
        stem=f'drive_{i:03d}'; imgp=IMG_DIR/f'{stem}.jpg'
        img=cv2.imread(str(imgp)); h,w=img.shape[:2]
        d26=detect_26n(m26,imgp,conf=0.12,imgsz=1280)
        dw,nonveh=detect_world(mw,imgp,conf=0.10,imgsz=1280)
        clusters=cluster_detections(d26+dw,iou_thr=0.42)
        img_changes=[]; finals=[]
        for cl in clusters:
            bx=weighted_box(cl)
            nv=overlaps_nonvehicle(bx,nonveh)
            maxveh=max(d['conf'] for d in cl)
            if nv and maxveh < 0.55:
                img_changes.append({'action':'removed_out_of_scope','match':nv['raw'],'nv_conf':round(nv['conf'],3),'veh_conf':round(maxveh,3)}); log['totals']['removed']+=1; continue
            cls,reason=choose_class(cl)
            conf=max(d['conf'] for d in cl)
            if conf < 0.16 and len(cl)==1:
                img_changes.append({'action':'removed_low_conf_isolated','conf':round(conf,3)}); log['totals']['removed']+=1; continue
            an=(bx[2]-bx[0])*(bx[3]-bx[1])/(w*h+1e-9)
            if an < 0.00045 and conf < 0.45:
                img_changes.append({'action':'removed_tiny_low_conf','area_norm':round(an,6)}); log['totals']['removed']+=1; continue
            finals.append({'cls':cls,'xyxy':bx,'conf':conf,'reason':reason})
            img_changes.append({'action':'kept','cls':ADAS[cls],'conf':round(conf,3),'reason':reason,'n':len(cl)})
        finals.sort(key=lambda d:d['conf'],reverse=True)
        keep=[]
        for d in finals:
            if any(iou_xyxy(d['xyxy'],k['xyxy'])>=0.55 for k in keep):
                img_changes.append({'action':'removed_nms_dup','cls':ADAS[d['cls']]}); log['totals']['removed']+=1; continue
            keep.append(d)
        lines=[]; vis=img.copy()
        for idx,d in enumerate(sorted(keep,key=lambda z:center(z['xyxy'])[1])):
            n=clamp_norm(xyxy_to_norm(d['xyxy'],w,h))
            lines.append(f"{d['cls']} {n[0]:.6f} {n[1]:.6f} {n[2]:.6f} {n[3]:.6f}")
            x1,y1,x2,y2=map(int,d['xyxy']); col=COLORS[d['cls']]
            cv2.rectangle(vis,(x1,y1),(x2,y2),col,2)
            label=f"#{idx} {ADAS[d['cls']]} {d['conf']:.2f}"
            (tw,th),_=cv2.getTextSize(label,cv2.FONT_HERSHEY_SIMPLEX,0.42,1); ytxt=max(12,y1-3)
            cv2.rectangle(vis,(x1,ytxt-th-4),(x1+tw+2,ytxt+2),col,-1)
            cv2.putText(vis,label,(x1+1,ytxt),cv2.FONT_HERSHEY_SIMPLEX,0.42,(0,0,0),1,cv2.LINE_AA)
        (OUT_LBL/f'{stem}.txt').write_text('\n'.join(lines)+('\n' if lines else ''),encoding='utf-8')
        cv2.imwrite(str(OUT_PREV/f'{stem}.jpg'),vis)
        log['images'][stem]={'n26':len(d26),'nworld':len(dw),'n_nonveh':len(nonveh),'final':len(keep),'changes':img_changes}
        log['totals']['kept']+=len(keep); log['totals']['total']+=len(keep)
        print(f'done {i:02d} {stem} 26n={len(d26)} world={len(dw)} final={len(keep)}', flush=True)
    OUT_LOG.write_text(json.dumps(log,ensure_ascii=False,indent=2),encoding='utf-8')
    print('TOTALS',log['totals'], flush=True)

if __name__=='__main__':
    main()
