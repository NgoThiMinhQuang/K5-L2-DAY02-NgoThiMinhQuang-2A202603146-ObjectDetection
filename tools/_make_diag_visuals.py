import json, cv2
from pathlib import Path
root=Path(r'D:\Day2-Lab\K5-L2-DAY02-NgoThiMinhQuang-2A202603146-ObjectDetection')
out=root/'outputs/deep_error_visuals'; out.mkdir(parents=True,exist_ok=True)
names={0:'car',1:'truck',2:'bus',3:'van'}
def read(p):
    b=[]
    if not p.exists(): return b
    for l in p.read_text().splitlines():
        if not l.strip(): continue
        c,xc,yc,w,h=map(float,l.split()[:5]); c=int(c)
        b.append({'c':c,'xyxy':[xc-w/2,yc-h/2,xc+w/2,yc+h/2],'a':w*h})
    return b
def iou(a,b):
    x1=max(a[0],b[0]); y1=max(a[1],b[1]); x2=min(a[2],b[2]); y2=min(a[3],b[3])
    inter=max(0,x2-x1)*max(0,y2-y1); aa=max(0,a[2]-a[0])*max(0,a[3]-a[1]); ab=max(0,b[2]-b[0])*max(0,b[3]-b[1])
    return inter/(aa+ab-inter+1e-9)
def denorm(xyxy,w,h): return [int(xyxy[0]*w),int(xyxy[1]*h),int(xyxy[2]*w),int(xyxy[3]*h)]

stud_dir=root/'dataset/labels_ensemble/train'; gold_dir=root/'gold_train_labels'; img_dir=root/'dataset/images/train'
conf_examples=[]; tiny_examples=[]; overlap_examples=[]
for im in [f'drive_{i:03d}' for i in range(1,51)]:
    sb=read(stud_dir/f'{im}.txt'); gb=read(gold_dir/f'{im}.txt')
    for si,s in enumerate(sb):
        for gi,g in enumerate(gb):
            ov=iou(s['xyxy'],g['xyxy'])
            if ov>=0.45 and s['c']!=g['c'] and len(conf_examples)<8:
                conf_examples.append((im,si,gi,ov,s,g))
        if s['a']<0.0008 and len(tiny_examples)<5: tiny_examples.append((im,si,s))
    for i in range(len(sb)):
        for j in range(i+1,len(sb)):
            ov=iou(sb[i]['xyxy'],sb[j]['xyxy'])
            if ov>=0.35 and len(overlap_examples)<4: overlap_examples.append((im,i,j,ov,sb[i],sb[j]))

def draw_base(im):
    img=cv2.imread(str(img_dir/f'{im}.jpg')); h,w=img.shape[:2]
    cv2.rectangle(img,(0,0),(w,28),(0,0,0),-1)
    cv2.putText(img,'DIAGNOSTIC: gold overlay for error analysis only; NOT used for labeling',(5,18),cv2.FONT_HERSHEY_SIMPLEX,0.45,(255,255,255),1,cv2.LINE_AA)
    return img,h,w

for idx,(im,si,gi,ov,s,g) in enumerate(conf_examples):
    img,h,w=draw_base(im)
    gx1,gy1,gx2,gy2=denorm(g['xyxy'],w,h); x1,y1,x2,y2=denorm(s['xyxy'],w,h)
    cv2.rectangle(img,(gx1,gy1),(gx2,gy2),(0,255,0),2)
    cv2.putText(img,'GOLD '+names[g['c']],(gx1,max(40,gy1-5)),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),2)
    cv2.rectangle(img,(x1,y1),(x2,y2),(0,0,255),2)
    cv2.putText(img,'PRED '+names[s['c']]+' IoU='+f'{ov:.2f}',(x1,min(h-5,y2+18)),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),2)
    cv2.imwrite(str(out/f'class_confusion_{idx+1}_{im}_pred-{names[s["c"]]}_true-{names[g["c"]]}.jpg'),img)

for idx,(im,si,s) in enumerate(tiny_examples):
    img,h,w=draw_base(im)
    x1,y1,x2,y2=denorm(s['xyxy'],w,h)
    cv2.rectangle(img,(x1,y1),(x2,y2),(0,0,255),2)
    cv2.putText(img,'TINY PRED '+names[s['c']]+' area='+f'{s["a"]:.6f}',(5,48),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),2)
    cv2.imwrite(str(out/f'tiny_box_{idx+1}_{im}.jpg'),img)

for idx,(im,i,j,ov,a,b) in enumerate(overlap_examples):
    img,h,w=draw_base(im)
    for box,col,lab in [(a,(0,0,255),'A'),(b,(255,0,0),'B')]:
        x1,y1,x2,y2=denorm(box['xyxy'],w,h); cv2.rectangle(img,(x1,y1),(x2,y2),col,2)
    cv2.putText(img,f'OVERLAP IoU={ov:.2f} {names[a["c"]]}/{names[b["c"]]}',(5,48),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,255),2)
    cv2.imwrite(str(out/f'overlap_{idx+1}_{im}.jpg'),img)

# also missed/occluded illustration: show gold boxes that had no match (FN context) - diagnostic
miss_ex=[]
for im in [f'drive_{i:03d}' for i in range(1,51)]:
    sb=read(stud_dir/f'{im}.txt'); gb=read(gold_dir/f'{im}.txt')
    for gi,g in enumerate(gb):
        best=max([iou(s['xyxy'],g['xyxy']) for s in sb], default=0)
        if best<0.3 and len(miss_ex)<4: miss_ex.append((im,gi,g,best))
for idx,(im,gi,g,best) in enumerate(miss_ex):
    img,h,w=draw_base(im)
    x1,y1,x2,y2=denorm(g['xyxy'],w,h)
    cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
    cv2.putText(img,'GOLD UNMATCHED '+names[g['c']]+' bestIoU='+f'{best:.2f}',(5,48),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),2)
    cv2.imwrite(str(out/f'missed_fn_{idx+1}_{im}_{names[g["c"]]}.jpg'),img)

print('conf',len(list(out.glob('class_confusion_*'))),'tiny',len(list(out.glob('tiny_box_*'))),'overlap',len(list(out.glob('overlap_*'))),'missed',len(list(out.glob('missed_fn_*'))))
print(out)
