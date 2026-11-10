"""
Build offline manual review guide for CVAT editing.
- Source: dataset/images/train + dataset/labels_ensemble/train (Ensemble v1)
- Gold (gold_train_labels) used ONLY for diagnostic matching/visualization data in this guide.
  It is NOT copied into labels_manual/train or any submission label set.
Outputs -> outputs/manual_review_guide/
"""
import json, csv, os, shutil
from pathlib import Path
from collections import defaultdict, Counter
import cv2

ROOT = Path(r"D:\Day2-Lab\K5-L2-DAY02-NgoThiMinhQuang-2A202603146-ObjectDetection")
IMG_DIR = ROOT / "dataset/images/train"
STUD_DIR = ROOT / "dataset/labels_ensemble/train"
GOLD_DIR = ROOT / "gold_train_labels"  # diagnostic only
OUT = ROOT / "outputs/manual_review_guide"
NAMES = {0:'car', 1:'truck', 2:'bus', 3:'van'}
for sub in ["images/original","images/ensemble","images/diagnostic"]:
    (OUT/sub).mkdir(parents=True, exist_ok=True)

def read_boxes(p):
    boxes=[]
    if not p.exists(): return boxes
    for i, line in enumerate(p.read_text(encoding='utf-8').splitlines()):
        s=line.strip()
        if not s: continue
        parts=s.split()
        if len(parts)<5: continue
        c=int(float(parts[0])); xc,yc,w,h=map(float,parts[1:5])
        boxes.append({'i':i,'c':c,'xc':xc,'yc':yc,'w':w,'h':h,'area':w*h,
                      'xyxy':[xc-w/2,yc-h/2,xc+w/2,yc+h/2]})
    return boxes

def iou(a,b):
    x1=max(a[0],b[0]); y1=max(a[1],b[1]); x2=min(a[2],b[2]); y2=min(a[3],b[3])
    inter=max(0,x2-x1)*max(0,y2-y1)
    aa=max(0,a[2]-a[0])*max(0,a[3]-a[1]); ab=max(0,b[2]-b[0])*max(0,b[3]-b[1])
    den=aa+ab-inter
    return inter/den if den>1e-12 else 0.0

def denorm(b,w,h):
    x1=int((b['xc']-b['w']/2)*w); y1=int((b['yc']-b['h']/2)*h)
    x2=int((b['xc']+b['w']/2)*w); y2=int((b['yc']+b['h']/2)*h)
    x1=max(0,min(w-1,x1)); y1=max(0,min(h-1,y1)); x2=max(0,min(w,x2)); y2=max(0,min(h,y2))
    return x1,y1,x2,y2

def match_greedy_same_class(sb, gb, thr=0.5):
    cands=[]
    for si,s in enumerate(sb):
        for gi,g in enumerate(gb):
            if s['c']==g['c']:
                v=iou(s['xyxy'],g['xyxy'])
                if v>=thr: cands.append((v,si,gi))
    cands.sort(reverse=True)
    ms,mg=set(),set(); pairs=[]
    for v,si,gi in cands:
        if si in ms or gi in mg: continue
        ms.add(si); mg.add(gi); pairs.append((si,gi,v))
    return pairs, ms, mg

all_data={'meta':{'note':'Gold used ONLY for diagnostic matching/visualization in this guide; not exported as labels, not copied to submission.','classes':NAMES,'cvat_base_frame':{'drive_001':0}},'images':[]}
checklist=[]

for idx in range(1,51):
    stem=f'drive_{idx:03d}'
    frame=idx-1  # drive_001 = frame 0
    imgp=IMG_DIR/f'{stem}.jpg'
    im=cv2.imread(str(imgp))
    if im is None: raise RuntimeError(f'missing {imgp}')
    H,W=im.shape[:2]
    sb=read_boxes(STUD_DIR/f'{stem}.txt')
    gb=read_boxes(GOLD_DIR/f'{stem}.txt') if GOLD_DIR.exists() else []
    pairs, ms, mg = match_greedy_same_class(sb, gb, 0.5)
    match_iou={si:v for si,gi,v in pairs}

    # student box diagnostics
    student_out=[]
    # find overlaps among students
    overlaps=[]
    for a in range(len(sb)):
        for b in range(a+1,len(sb)):
            ov=iou(sb[a]['xyxy'],sb[b]['xyxy'])
            if ov>=0.30: overlaps.append((a,b,ov))
    overlap_set=set()
    for a,b,ov in overlaps: overlap_set.add(a); overlap_set.add(b)

    for si,s in enumerate(sb):
        status='TP' if si in ms else 'FP'
        mi=match_iou.get(si)
        flags=[]
        if s['area'] < 0.0010: flags.append('tiny')
        if si in overlap_set: flags.append('overlap')
        if status=='TP' and mi is not None and mi < 0.60: flags.append('misaligned')
        # suspected class error: FP but overlaps a gold box of different class well
        suspect=None
        if status=='FP':
            best=(0,None,None)
            for gi,g in enumerate(gb):
                if gi in mg: continue
                v=iou(s['xyxy'],g['xyxy'])
                if v>best[0]: best=(v,gi,g['c'])
            if best[0]>=0.40 and best[1] is not None and best[2]!=s['c']:
                flags.append('class_suspect')
                suspect={'gold_class':NAMES[best[2]],'gold_class_id':best[2],'iou':round(best[0],3)}
            elif best[0]>=0.40 and best[1] is not None and best[2]==s['c']:
                flags.append('misaligned') # same class, below match thr context
        x1,y1,x2,y2=denorm(s,W,H)
        student_out.append({
            'id': si, 'class_id': s['c'], 'class': NAMES[s['c']],
            'xc':round(s['xc'],6),'yc':round(s['yc'],6),'w':round(s['w'],6),'h':round(s['h'],6),
            'x1':x1,'y1':y1,'x2':x2,'y2':y2,
            'status': status, 'match_iou': round(mi,3) if mi is not None else None,
            'flags': flags, 'class_suspect': suspect
        })

    # FN markers (diagnostic positions - from gold, not for label copy)
    fn_out=[]
    for gi,g in enumerate(gb):
        if gi not in mg:
            x1,y1,x2,y2=denorm(g,W,H)
            # is there a class-mismatch overlap?
            cm=None
            best=(0,None)
            for si,s in enumerate(sb):
                if si in ms: continue
                v=iou(s['xyxy'],g['xyxy'])
                if v>best[0]: best=(v,s['c'])
            if best[0]>=0.40 and best[1]!=g['c']: cm={'pred_class':NAMES[best[1]],'iou':round(best[0],3)}
            fn_out.append({'class_id':g['c'],'class':NAMES[g['c']],'x1':x1,'y1':y1,'x2':x2,'y2':y2,
                           'xc':round(g['xc'],6),'yc':round(g['yc'],6),'w':round(g['w'],6),'h':round(g['h'],6),
                           'class_mismatch_hint':cm})

    tp=sum(1 for s in student_out if s['status']=='TP')
    fp=sum(1 for s in student_out if s['status']=='FP')
    fn=len(fn_out)
    n_class_sus=sum(1 for s in student_out if 'class_suspect' in s['flags']) + sum(1 for f in fn_out if f['class_mismatch_hint'])
    # priority
    score = fp*2 + fn*2 + n_class_sus*3
    if score>=30: pr='CAO'
    elif score>=12: pr='TRUNG BÌNH'
    else: pr='THẤP'

    # save images
    orig = im.copy()
    cv2.imwrite(str(OUT/'images/original'/f'{stem}.jpg'), orig)
    ens = im.copy()
    for s in student_out:
        col=(255,160,60) # orange-ish for ensemble neutral
        cv2.rectangle(ens,(s['x1'],s['y1']),(s['x2'],s['y2']),col,2)
        lbl=f"{s['id']}:{s['class'][0].upper()}"
        cv2.putText(ens,lbl,(s['x1'],max(12,s['y1']-3)),cv2.FONT_HERSHEY_SIMPLEX,0.38,col,1,cv2.LINE_AA)
    cv2.imwrite(str(OUT/'images/ensemble'/f'{stem}.jpg'), ens)

    diag = im.copy()
    # FN (blue) behind
    for f in fn_out:
        cv2.rectangle(diag,(f['x1'],f['y1']),(f['x2'],f['y2']),(255,120,0),2)
        cv2.putText(diag,f"FN:{f['class'][0].upper()}",(f['x1'],min(H-4,f['y2']+12)),cv2.FONT_HERSHEY_SIMPLEX,0.35,(255,120,0),1,cv2.LINE_AA)
    # FP (red), TP (green)
    for s in student_out:
        if s['status']=='TP':
            col=(0,200,0); tag='TP'
        else:
            col=(0,0,230); tag='FP'
        thick=2 if s['status']=='FP' or 'class_suspect' in s['flags'] else 1
        cv2.rectangle(diag,(s['x1'],s['y1']),(s['x2'],s['y2']),col,thick)
        short=f"{s['id']}"
        cv2.putText(diag,short,(s['x1'],max(12,s['y1']-2)),cv2.FONT_HERSHEY_SIMPLEX,0.35,col,1,cv2.LINE_AA)
        if 'class_suspect' in s['flags']:
            cv2.putText(diag,"CLS?",(s['x1'],min(H-4,s['y2']+11)),cv2.FONT_HERSHEY_SIMPLEX,0.32,(0,165,255),1,cv2.LINE_AA)
        if 'tiny' in s['flags']:
            cv2.drawMarker(diag,((s['x1']+s['x2'])//2,(s['y1']+s['y2'])//2),(200,0,200),cv2.MARKER_CROSS,8,1)
    # legend
    cv2.rectangle(diag,(0,0),(W,22),(0,0,0),-1)
    cv2.putText(diag,'TP xanh la | FP do | FN xanh duong | CLS? nghi sai class | + box nho (chan doan, gold chi de tham chieu)',(4,15),cv2.FONT_HERSHEY_SIMPLEX,0.38,(255,255,255),1,cv2.LINE_AA)
    cv2.imwrite(str(OUT/'images/diagnostic'/f'{stem}.jpg'), diag)

    all_data['images'].append({
        'stem':stem,'file':f'{stem}.jpg','frame_cvat':frame,'width':W,'height':H,
        'tp':tp,'fp':fp,'fn':fn,'n_class_suspect':n_class_sus,'priority':pr,
        'students':student_out,'fns':fn_out
    })
    checklist.append({'image':stem,'frame_cvat':frame,'tp':tp,'fp':fp,'fn':fn,
                      'n_class_suspect':n_class_sus,'priority':pr,'student_boxes':len(sb),'gold_boxes_diag':len(gb)})
    if idx%10==0: print('done',idx)

# save data
(OUT/'data.json').write_text(json.dumps(all_data,ensure_ascii=False,indent=2),encoding='utf-8')
(OUT/'data.js').write_text('window.REVIEW_DATA = '+json.dumps(all_data,ensure_ascii=False)+';',encoding='utf-8')
with (OUT/'checklist_50.csv').open('w',newline='',encoding='utf-8-sig') as f:
    w=csv.DictWriter(f,fieldnames=['image','frame_cvat','tp','fp','fn','n_class_suspect','priority','student_boxes','gold_boxes_diag'])
    w.writeheader(); w.writerows(checklist)
(OUT/'checklist_50.json').write_text(json.dumps(checklist,ensure_ascii=False,indent=2),encoding='utf-8')
print('data+checklist done', len(checklist))
