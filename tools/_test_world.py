from ultralytics import YOLO
from pathlib import Path
root = Path(r"D:\Day2-Lab\K5-L2-DAY02-NgoThiMinhQuang-2A202603146-ObjectDetection")
m = YOLO('yolov8s-worldv2.pt')
m.set_classes(['car','truck','bus','van','minivan','pickup truck','person','motorcycle','bicycle','traffic sign'])
for img in ['drive_009.jpg','drive_031.jpg','drive_003.jpg']:
    r = m.predict(str(root/f'dataset/images/train/{img}'), conf=0.12, verbose=False)[0]
    print(f'=={img} {r.names}==')
    for b in r.boxes:
        c = int(b.cls[0]); xyxy = [round(x,1) for x in b.xyxy[0].tolist()]
        print(f"  {c} {r.names[c]} {float(b.conf[0]):.3f} {xyxy}")
    print()
