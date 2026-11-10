"""
Build round2 QC guide for manual labels (no auto-fix). Gold used ONLY for diagnostic.
"""
import json, csv
from pathlib import Path
from collections import defaultdict
import cv2

ROOT = Path(r"D:\Day2-Lab\K5-L2-DAY02-NgoThiMinhQuang-2A202603146-ObjectDetection")
IMG_DIR = ROOT / "dataset/images/train"
MAN_DIR = ROOT / "dataset/labels_manual/train"
GOLD_DIR = ROOT / "gold_train_labels"
EVAL = json.loads((ROOT/"outputs/manual_eval.json").read_text(encoding='utf-8'))
OUT = ROOT / "outputs/manual_review_round2"
NAMES={0:'car',1:'truck',2:'bus',3:'van'}
PRIO_LIST = ['drive_002','drive_003','drive_009','drive_030','drive_012','drive_041','drive_024','drive_027']
for sub in ['images/original','images/manual','images/diagnostic','crops']:
    (OUT/sub).mkdir(parents=True,exist_ok=True)

def read(p):
    b=[]
    if not p.exists(): return b
    for i,l in enumerate(p.read_text(encoding='utf-8').splitlines()):
        s=l.strip()
        if not s: continue
        parts=s.split()
        if len(parts)<5: continue
        c=int(float(parts[0])); xc,yc,w,h=map(float,parts[1:5])
        b.append({'id':i,'c':c,'xc':xc,'yc':yc,'w':w,'h':h,'area':w*h,'xyxy':[xc-w/2,yc-h/2,xc+w/2,yc+h/2]})
    return b

def iou(a,b):
    x1=max(a[0],b[0]); y1=max(a[1],b[1]); x2=min(a[2],b[2]); y2=min(a[3],b[3])
    inter=max(0,x2-x1)*max(0,y2-y1); aa=max(0,a[2]-a[0])*max(0,a[3]-a[1]); ab=max(0,b[2]-b[0])*max(0,b[3]-b[1]); d=aa+ab-inter
    return inter/d if d>1e-12 else 0.0

def denorm(b,w,h):
    x1=int((b['xc']-b['w']/2)*w); y1=int((b['yc']-b['h']/2)*h); x2=int((b['xc']+b['w']/2)*w); y2=int((b['yc']+b['h']/2)*h)
    return max(0,x1),max(0,y1),min(w,x2),min(h,y2)

def crop_img(im,x1,y1,x2,y2,pad=40):
    h,w=im.shape[:2]
    x1=max(0,x1-pad); y1=max(0,y1-pad); x2=min(w,x2+pad); y2=min(h,y2+pad)
    if x2<=x1 or y2<=y1: return None
    return im[y1:y2,x1:x2].copy()

data={'meta':{'gold_diagnostic_only':True,'classes':NAMES,'cvat_base_frame':0,'priority':PRIO_LIST},'images':[]}
checklist=[]
for idx in range(1,51):
    stem=f'drive_{idx:03d}'; frame=idx-1
    imgp=IMG_DIR/f'{stem}.jpg'; im=cv2.imread(str(imgp))
    if im is None: raise SystemExit(imgp)
    H,W=im.shape[:2]
    sb=read(MAN_DIR/f'{stem}.txt'); gb=read(GOLD_DIR/f'{stem}.txt')
    # match
    cands=[]; 
    for si,s in enumerate(sb):
        for gi,g in enumerate(gb):
            if s['c']==g['c']:
                v=iou(s['xyxy'],g['xyxy'])
                if v>=0.5: cands.append((v,si,gi))
    cands.sort(reverse=True); ms,mg=set(),set(); pairs=[]
    for v,si,gi in cands:
        if si in ms or gi in mg: continue
        ms.add(si); mg.add(gi); pairs.append((si,gi,v))
    miou={si:v for si,gi,v in pairs}
    # overlaps
    ovset=set()
    for a in range(len(sb)):
        for b in range(a+1,len(sb)):
            if iou(sb[a]['xyxy'],sb[b]['xyxy'])>=0.30: ovset.add(a); ovset.add(b)
    students=[]
    for si,s in enumerate(sb):
        status='TP' if si in ms else 'FP'
        flags=[]; reason=''; action='KIỂM TRA'
        if s['area']<0.0010: flags.append('tiny')
        if si in ovset: flags.append('overlap')
        if status=='TP' and miou.get(si,1.0)<0.60: flags.append('misaligned')
        cs=None
        if status=='FP':
            best=(0,None,None)
            for gi,g in enumerate(gb):
                if gi in mg: continue
                v=iou(s['xyxy'],g['xyxy'])
                if v>best[0]: best=(v,gi,g)
            if best[0]>=0.40 and best[1] is not None and best[2]['c']!=s['c']:
                flags.append('class_suspect'); cs={'gold_c':NAMES[best[2]['c']],'gold_cid':best[2]['c'],'iou':round(best[0],3)}
                reason=f"Vị trí trùng khá với gold {NAMES[best[2]['c']]} nhưng class khác (IoU {cs['iou']:.2f}). So sánh đặc điểm trên ảnh gốc theo GUIDELINE_MINI."
                action='KIỂM TRA ĐỔI CLASS'
            else:
                reason='Không khớp gold cùng class (IoU<0.5). Kiểm tra có phải thừa/out-of-scope/box gộp hay không.'
                action='KIỂM TRA XÓA'
        elif status=='TP':
            if miou.get(si,1.0)<0.60: reason='Khớp đúng class nhưng IoU thấp → có thể lệch/box lỏng, cân nhắc chỉnh sát hơn.'; action='KIỂM TRA CHỈNH BOX'
            else: reason='Khớp đúng class + IoU đủ (tham chiếu). Vẫn rà nhanh xem có gộp/trùng hay không.'; action='GIỮ'
        if 'overlap' in flags and action!='KIỂM TRA XÓA': reason+=' Có khả năng trùng 1 xe với box lân cận → chỉ giữ 1 box.'; action='KIỂM TRA XÓA' if status=='FP' else 'KIỂM TRA'
        if 'tiny' in flags: reason+=' Box rất nhỏ (area<0.001) → dễ sai class/ngoài phạm vi, kiểm tra cẩn thận.' 
        x1,y1,x2,y2=denorm(s,W,H)
        cimg=crop_img(im,x1,y1,x2,y2,48)
        cpath=None
        if cimg is not None and cimg.size: cpath=f'crops/{stem}_b{s["id"]}.jpg'; cv2.imwrite(str(OUT/cpath),cimg)
        students.append({'id':s['id'],'c':s['c'],'class':NAMES[s['c']],'x1':x1,'y1':y1,'x2':x2,'y2':y2,
                        'xc':round(s['xc'],6),'yc':round(s['yc'],6),'w':round(s['w'],6),'h':round(s['h'],6),
                        'status':status,'match_iou':round(miou.get(s['id']),3) if s['id'] in miou else None,
                        'flags':flags,'class_suspect':cs,'reason':reason,'action':action,'crop':cpath})
    # FN
    fns=[]
    for gi,g in enumerate(gb):
        if gi not in mg:
            x1,y1,x2,y2=denorm(g,W,H)
            cm=None; best=(0,None)
            for si,s in enumerate(sb):
                if si in ms: continue
                v=iou(s['xyxy'],g['xyxy'])
                if v>best[0]: best=(v,s['c'])
            if best[0]>=0.40 and best[1]!=g['c']: cm={'pred_c':NAMES[best[1]],'iou':round(best[0],3)}
            cimg=crop_img(im,x1,y1,x2,y2,48)
            cpath=None
            if cimg is not None and cimg.size: cpath=f'crops/{stem}_fn{gi}.jpg'; cv2.imwrite(str(OUT/cpath),cimg)
            fns.append({'gidx':gi,'c':g['c'],'class':NAMES[g['c']],'x1':x1,'y1':y1,'x2':x2,'y2':y2,
                       'xc':round(g['xc'],6),'yc':round(g['yc'],6),'w':round(g['w'],6),'h':round(g['h'],6),
                       'class_mismatch_hint':cm,'crop':cpath,'reason':('Có box dự đoán khác class ở gần (IoU '+str(cm['iou'])+') — kiểm tra class/trùng.' if cm else 'Không tìm thấy box ghép đúng class IoU>=0.5. Có thể bỏ sót hoặc box quá lệch — kiểm tra ảnh gốc (xe có thể bị che/cắt).'),'action':'KIỂM TRA THÊM BOX' if not cm else 'KIỂM TRA ĐỔI CLASS HOẶC THÊM BOX'})
    # images
    cv2.imwrite(str(OUT/'images/original'/f'{stem}.jpg'), im)
    manim=im.copy()
    for s in students: x1,y1,x2,y2=s['x1'],s['y1'],s['x2'],s['y2']; cv2.rectangle(manim,(x1,y1),(x2,y2),(80,200,255),2); cv2.putText(manim,f"{s['id']}:{s['class'][0]}",(x1,max(12,y1-2)),cv2.FONT_HERSHEY_SIMPLEX,0.35,(80,200,255),1)
    cv2.imwrite(str(OUT/'images/manual'/f'{stem}.jpg'), manim)
    di=im.copy()
    for f in fns: cv2.rectangle(di,(f['x1'],f['y1']),(f['x2'],f['y2']),(255,140,0),2); cv2.putText(di,f"FN:{f['class'][0]}",(f['x1'],min(H-4,f['y2']+12)),cv2.FONT_HERSHEY_SIMPLEX,0.33,(255,140,0),1)
    for s in students:
        if s['status']=='TP': col=(0,210,0)
        else: col=(0,0,235)
        th=2 if s['status']=='FP' or 'class_suspect' in s['flags'] else 1
        cv2.rectangle(di,(s['x1'],s['y1']),(s['x2'],s['y2']),col,th)
        cv2.putText(di,str(s['id']),(s['x1'],max(12,s['y1']-2)),cv2.FONT_HERSHEY_SIMPLEX,0.33,col,1)
        if 'class_suspect' in s['flags']: cv2.putText(di,'CLS?',(s['x1'],min(H-4,s['y2']+11)),cv2.FONT_HERSHEY_SIMPLEX,0.3,(0,200,255),1)
        if 'tiny' in s['flags']: cv2.drawMarker(di,((s['x1']+s['x2'])//2,(s['y1']+s['y2'])//2),(200,0,200),cv2.MARKER_CROSS,7,1)
    cv2.rectangle(di,(0,0),(W,20),(0,0,0),-1); cv2.putText(di,'TP xanh la | FP do | FN xanh duong | CLS? vang | + nho',(3,14),cv2.FONT_HERSHEY_SIMPLEX,0.36,(255,255,255),1)
    cv2.imwrite(str(OUT/'images/diagnostic'/f'{stem}.jpg'), di)
    tp_c=sum(1 for s in students if s['status']=='TP'); fp_c=sum(1 for s in students if s['status']=='FP'); fn_c=len(fns); cls_c=sum(1 for s in students if 'class_suspect' in s['flags'])
    err = fp_c*2 + fn_c*2 + cls_c*3
    pr='CAO' if err>=28 else ('TRUNG BINH' if err>=10 else 'THAP')
    # ensure priority order push up
    if stem in PRIO_LIST and pr!='CAO': pr='CAO'
    obj={'stem':stem,'file':f'{stem}.jpg','frame_cvat':frame,'W':W,'H':H,'tp':tp_c,'fp':fp_c,'fn':fn_c,'n_class_suspect':cls_c,'err_score':err,'priority':pr,'students':students,'fns':fns}
    data['images'].append(obj)
    checklist.append({'image':stem,'frame_cvat':frame,'tp':tp_c,'fp':fp_c,'fn':fn_c,'n_class_suspect':cls_c,'priority':pr,'student_boxes':len(students)})
    if idx%10==0: print('done',idx)

(OUT/'data.json').write_text(json.dumps(data,ensure_ascii=False),encoding='utf-8')
(OUT/'data.js').write_text('window.REVIEW2 = '+json.dumps(data,ensure_ascii=False)+';',encoding='utf-8')
with (OUT/'checklist_50.csv').open('w',newline='',encoding='utf-8-sig') as f:
    w=csv.DictWriter(f,fieldnames=['image','frame_cvat','tp','fp','fn','n_class_suspect','priority','student_boxes']); w.writeheader(); w.writerows(checklist)
(OUT/'checklist_50.json').write_text(json.dumps(checklist,ensure_ascii=False,indent=2),encoding='utf-8')
print('done all', len(data['images']))
