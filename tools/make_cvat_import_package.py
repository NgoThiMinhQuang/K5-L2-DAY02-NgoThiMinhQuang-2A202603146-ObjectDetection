import shutil, zipfile
from pathlib import Path
root = Path(r"D:\Day2-Lab\K5-L2-DAY02-NgoThiMinhQuang-2A202603146-ObjectDetection")
img_src = root/"dataset/images/train"
lbl_src = root/"outputs/auto_prelabels"
out = root/"outputs/cvat_import_review"
obj = out/"obj_train_data"
if obj.exists():
    shutil.rmtree(obj)
obj.mkdir(parents=True, exist_ok=True)
imgs = [img_src/f"drive_{i:03d}.jpg" for i in range(1,51)]
missing = [p for p in imgs if not p.exists()]
if missing: raise SystemExit(f"missing {missing[:3]}")
for p in imgs:
    shutil.copy2(p, obj/p.name)
    lt = lbl_src/f"{p.stem}.txt"
    if not lt.exists(): raise SystemExit(f"missing label {lt}")
    shutil.copy2(lt, obj/lt.name)
(out/"obj.names").write_text("car\ntruck\nbus\nvan\n", encoding="utf-8")
(out/"obj.data").write_text("classes = 4\ntrain = train.txt\nnames = obj.names\nbackup = backup/\n", encoding="utf-8")
# also copy into obj_train_data for some importers
shutil.copy2(out/"obj.names", obj/"obj.names")
train_list = "\n".join([f"data/obj_train_data/{p.name}" for p in imgs]) + "\n"
(out/"train.txt").write_text(train_list, encoding="utf-8")
zip_path = out/"ADAS_50_Images_PRELABEL_REVIEW_YOLO.zip"
if zip_path.exists(): zip_path.unlink()
with zipfile.ZipFile(zip_path, 'w', compression=zipfile.ZIP_DEFLATED) as z:
    for f in [out/"obj.names", out/"obj.data", out/"train.txt"]:
        z.write(f, arcname=f.name)
    for p in sorted(obj.iterdir()):
        z.write(p, arcname=f"obj_train_data/{p.name}")
jpgs = list(obj.glob('*.jpg'))
txts = [p for p in obj.glob('*.txt') if p.name not in ('train.txt','obj.names')]
print('obj_train_data jpg', len(jpgs), 'txt', len(txts))
print('zip', zip_path.name, zip_path.stat().st_size)
print('order_ok', [p.name for p in sorted(jpgs)[:2]], '...', [p.name for p in sorted(jpgs)[-2:]])
