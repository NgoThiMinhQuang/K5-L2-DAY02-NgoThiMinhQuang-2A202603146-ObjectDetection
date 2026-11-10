"""
Local ADAS Annotation Review (Streamlit)
- Reads dataset/images/train + dataset/labels_reviewed/train (NEVER modifies dataset/labels/train or Gold)
- Displays boxes with ID+class; edit/delete/change class/adjust coords/add; filter by review_queue
- Shows YOLO26n suggestions (auxiliary only, NOT ground truth); no Gold copied
Run (Windows CMD, from repo root):
  python -m streamlit run tools/annotation_review_app.py
"""
from __future__ import annotations
import json, subprocess, sys, copy, datetime
from pathlib import Path
import streamlit as st
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
IMG_DIR = ROOT / "dataset" / "images" / "train"
LBL_DIR = ROOT / "dataset" / "labels_reviewed" / "train"
ORIG_LBL = ROOT / "dataset" / "labels" / "train"
GOLD_DIR = ROOT / "gold_train_labels"
OUT = ROOT / "outputs" / "review_automation"
NAMES = {0: "car", 1: "truck", 2: "bus", 3: "van"}
NAME_TO_ID = {v: k for k, v in NAMES.items()}
COLORS = {0: (46, 204, 113), 1: (230, 126, 34), 2: (231, 76, 60), 3: (155, 89, 182)}
YOLO_COL = (52, 152, 219)

st.set_page_config(page_title="ADAS Label Review", layout="wide")

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def parse_labels(stem: str):
    p = LBL_DIR / f"{stem}.txt"
    boxes = []
    if p.exists():
        for ln in p.read_text(encoding="utf-8", errors="ignore").splitlines():
            if not ln.strip(): continue
            v = ln.split()
            if len(v) != 5: continue
            c = int(float(v[0])); xc,yc,w,h = map(float, v[1:])
            boxes.append({"cls": c, "xc": xc, "yc": yc, "w": w, "h": h})
    return boxes

def save_labels(stem: str, boxes):
    LBL_DIR.mkdir(parents=True, exist_ok=True)
    p = LBL_DIR / f"{stem}.txt"
    lines = []
    for b in boxes:
        c = int(b["cls"]); xc=clamp(float(b["xc"]),0,1); yc=clamp(float(b["yc"]),0,1)
        w=clamp(float(b["w"]),1e-6,1); h=clamp(float(b["h"]),1e-6,1)
        # keep center valid for size
        xc=clamp(xc,w/2,1-w/2); yc=clamp(yc,h/2,1-h/2)
        b.update({"cls":c,"xc":xc,"yc":yc,"w":w,"h":h})
        lines.append(f"{c} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")
    p.write_text(("\n".join(lines)+"\n") if lines else "", encoding="utf-8")

def log_change(entry):
    OUT.mkdir(parents=True, exist_ok=True)
    entry = {"ts": datetime.datetime.now().isoformat(timespec="seconds"), **entry}
    with open(OUT / "ui_change_log.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

@st.cache_data
def load_priority():
    order = [f"drive_{i:03d}" for i in range(1, 51)]
    sev = {s: 0 for s in order}
    p = ROOT / "outputs/gold_review_phase1/phase1_report.json"
    if p.exists():
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
            rank = {x["image"]: len(d.get("priority_all",[])) - i for i,x in enumerate(d.get("priority_all",[]))}
            for k,v in rank.items(): sev[k]=sev.get(k,0)+v*10
        except Exception: pass
    q = OUT / "review_queue_all.json"
    if not q.exists(): q = OUT / "auto_review_log.json"
    qmap = {}
    if q.exists():
        try:
            d = json.loads(q.read_text(encoding="utf-8"))
            for r in d.get("results", []):
                qmap[r["image"]] = r.get("review_queue", [])
                sev[r["image"]] = sev.get(r["image"],0) + len(qmap[r["image"]])*2 + len(r.get("dup_candidates",[]))*20
        except Exception: pass
    order.sort(key=lambda s: (-sev.get(s,0), s))
    return order, qmap

def xywh_to_xyxy(b, W, H):
    x1=(b["xc"]-b["w"]/2)*W; y1=(b["yc"]-b["h"]/2)*H
    x2=(b["xc"]+b["w"]/2)*W; y2=(b["yc"]+b["h"]/2)*H
    return [x1,y1,x2,y2]

def draw(img: Image.Image, boxes, selected=None, yolo=None, show_yolo=False):
    im = img.convert("RGB").copy(); dr = ImageDraw.Draw(im)
    W,H = im.size
    try: font = ImageFont.load_default()
    except Exception: font = None
    if show_yolo and yolo:
        for yb in yolo:
            x1,y1,x2,y2 = xywh_to_xyxy(yb,W,H)
            dr.rectangle([x1,y1,x2,y2], outline=YOLO_COL, width=2)
            lab = f"YOLO:{NAMES.get(yb['cls'],yb['cls'])} {yb.get('conf',0):.2f}"
            dr.text((x1+2, max(0,y1-11)), lab, fill=YOLO_COL, font=font)
    for i,b in enumerate(boxes):
        x1,y1,x2,y2 = xywh_to_xyxy(b,W,H)
        col = COLORS.get(b["cls"], (200,200,200))
        w = 4 if i==selected else 2
        dr.rectangle([x1,y1,x2,y2], outline=col, width=w)
        lab = f"#{i} {NAMES.get(b['cls'],b['cls'])}"
        dr.text((x1+2, max(0,y1-11)), lab, fill=col, font=font)
    return im

@st.cache_resource
def get_yolo():
    try:
        from ultralytics import YOLO
        return YOLO("yolo26n.pt")
    except Exception as e:
        return e

def yolo_suggest(stem):
    model = get_yolo()
    if isinstance(model, Exception): return [], str(model)
    img = IMG_DIR / f"{stem}.jpg"
    COCO_TO_ADAS={2:0,7:1,5:2}
    res = model.predict(source=str(img), verbose=False, conf=0.25)[0]
    H,W = res.orig_shape
    out=[]
    for box in res.boxes:
        coco=int(box.cls[0].item())
        if coco not in COCO_TO_ADAS: continue
        conf=float(box.conf[0].item())
        x1,y1,x2,y2=box.xyxy[0].tolist()
        bw=max(1,x2-x1); bh=max(1,y2-y1)
        out.append({"cls":COCO_TO_ADAS[coco],"xc":(x1+bw/2)/W,"yc":(y1+bh/2)/H,"w":bw/W,"h":bh/H,"conf":conf})
    return out, None

def push_undo(stem, boxes):
    st.session_state.setdefault("history", [])
    st.session_state.history.append((stem, copy.deepcopy(boxes)))
    if len(st.session_state.history)>100: st.session_state.history = st.session_state.history[-100:]

# --- UI ---
st.title("ADAS Local Annotation Review (YOLO labels)")
st.caption("Chỉnh sửa lưu vào `dataset/labels_reviewed/train` • Không sửa `dataset/labels/train` • Không dùng/sao chép Gold • YOLO26n chỉ là gợi ý.")

order, qmap = load_priority()
if "stem" not in st.session_state: st.session_state.stem = order[0] if order else "drive_001"
if "sel" not in st.session_state: st.session_state.sel = None

with st.sidebar:
    st.header("Điều hướng & lọc")
    filt = st.selectbox("Hiển thị", ["Ưu tiên: nhiều vấn đề trước", "Theo tên ảnh", "Chỉ ảnh có review_queue"])
    items = order
    if filt == "Theo tên ảnh": items = sorted(order)
    elif filt == "Chỉ ảnh có review_queue": items = [s for s in order if qmap.get(s)] or order
    stem = st.selectbox("Ảnh", items, index=items.index(st.session_state.stem) if st.session_state.stem in items else 0)
    st.session_state.stem = stem
    q = qmap.get(stem, [])
    st.metric("Review flags (gợi ý)", len(q))
    if q:
        for it in q[:12]:
            idx = it.get('index','?'); t=it.get('type','')
            st.write(f"- `{t}` box #{idx} ({it.get('confidence','')}) {it.get('student',it.get('class',''))}")
        if len(q)>12: st.write(f"... +{len(q)-12}")
    st.divider()
    show_yolo = st.checkbox("Hiện gợi ý YOLO26n (không phải nhãn đúng)", value=False)
    if st.button("Nạp gợi ý YOLO cho ảnh này"):
        ys,err = yolo_suggest(stem)
        st.session_state.yolo_cache = st.session_state.get("yolo_cache",{})
        st.session_state.yolo_cache[stem] = ys
        if err: st.error(err)
        else: st.success(f"{len(ys)} gợi ý (car/truck/bus, không có van)")
    st.divider()
    if st.button("Undo (hoàn tác bước gần nhất)"):
        hist = st.session_state.get("history", [])
        if hist:
            hstem, hboxes = hist.pop(); save_labels(hstem, hboxes); log_change({"action":"undo","image":hstem})
            st.session_state.stem = hstem; st.rerun()
        else: st.warning("Không có lịch sử")
    if st.button("Lưu nhãn ảnh hiện tại"):
        save_labels(stem, st.session_state.boxes); log_change({"action":"save","image":stem,"n":len(st.session_state.boxes)}); st.success(f"Đã lưu {LBL_DIR/stem}.txt")

boxes = parse_labels(stem)
st.session_state.boxes = boxes
img_path = IMG_DIR / f"{stem}.jpg"
img = Image.open(img_path).convert("RGB")
yolo = st.session_state.get("yolo_cache",{}).get(stem, [])

left, right = st.columns([1.35, 1])
with left:
    st.subheader(f"{stem} — {len(boxes)} box")
    sel = st.selectbox("Chọn box (ID)", options=["(không)"]+[str(i) for i in range(len(boxes))], index=0)
    st.session_state.sel = None if sel=="(không)" else int(sel)
    st.image(draw(img, boxes, st.session_state.sel, yolo, show_yolo), use_container_width=True)
with right:
    st.subheader("Chỉnh sửa box")
    if st.session_state.sel is not None and st.session_state.sel < len(boxes):
        b = boxes[st.session_state.sel]
        nc = st.selectbox("Class", ["car","truck","bus","van"], index=[0,1,2,3].index(b["cls"]) if b["cls"] in [0,1,2,3] else 0, key=f"c_{stem}_{st.session_state.sel}")
        nxc = st.number_input("x_center (0-1)", value=float(b["xc"]), format="%.6f", key=f"xc_{stem}_{st.session_state.sel}")
        nyc = st.number_input("y_center (0-1)", value=float(b["yc"]), format="%.6f", key=f"yc_{stem}_{st.session_state.sel}")
        nw  = st.number_input("width (0-1)", value=float(b["w"]), format="%.6f", key=f"w_{stem}_{st.session_state.sel}")
        nh  = st.number_input("height (0-1)", value=float(b["h"]), format="%.6f", key=f"h_{stem}_{st.session_state.sel}")
        c1,c2,c3 = st.columns(3)
        if c1.button("Áp dụng sửa"):
            push_undo(stem, boxes)
            old=copy.deepcopy(b); b.update({"cls":NAME_TO_ID[nc],"xc":nxc,"yc":nyc,"w":nw,"h":nh})
            save_labels(stem, boxes); log_change({"action":"edit","image":stem,"index":st.session_state.sel,"before":old,"after":copy.deepcopy(b)}); st.rerun()
        if c2.button("Xóa box"):
            push_undo(stem, boxes)
            old=copy.deepcopy(b); boxes.pop(st.session_state.sel)
            save_labels(stem, boxes); log_change({"action":"delete","image":stem,"index":st.session_state.sel,"before":old}); st.session_state.sel=None; st.rerun()
        if c3.button("Giữ nguyên (ghi chú)"):
            log_change({"action":"keep","image":stem,"index":st.session_state.sel,"box":copy.deepcopy(b)}); st.info("Đã ghi log giữ nguyên.")
    else:
        st.info("Chọn một box ID để sửa/xóa/đổi class/chỉnh tọa độ.")
    st.subheader("Thêm box mới")
    ac = st.selectbox("Class mới", ["car","truck","bus","van"], key=f"ac_{stem}")
    axc = st.number_input("x_center mới", value=0.5, format="%.6f", key=f"axc_{stem}")
    ayc = st.number_input("y_center mới", value=0.5, format="%.6f", key=f"ayc_{stem}")
    aw  = st.number_input("width mới", value=0.1, format="%.6f", key=f"aw_{stem}")
    ah  = st.number_input("height mới", value=0.1, format="%.6f", key=f"ah_{stem}")
    if st.button("Thêm box"):
        push_undo(stem, boxes)
        nb={"cls":NAME_TO_ID[ac],"xc":axc,"yc":ayc,"w":aw,"h":ah}; boxes.append(nb)
        save_labels(stem, boxes); log_change({"action":"add","image":stem,"box":copy.deepcopy(nb)}); st.rerun()
    if yolo:
        st.subheader("Import từ gợi ý YOLO (tùy chọn, tự kiểm tra)")
        ysel = st.multiselect("Chọn gợi ý", [f"{i}:{NAMES.get(y['cls'])} conf={y.get('conf',0):.2f}" for i,y in enumerate(yolo)])
        if st.button("Thêm gợi ý đã chọn") and ysel:
            push_undo(stem, boxes)
            for s in ysel:
                yi=int(s.split(':')[0]); y=yolo[yi]; nb={k:y[k] for k in ['cls','xc','yc','w','h']}; boxes.append(nb)
                log_change({"action":"add_from_yolo_suggestion","image":stem,"box":nb,"note":"YOLO suggestion, user confirmed add; NOT ground truth"})
            save_labels(stem, boxes); st.rerun()

st.divider()
st.header("Kiểm tra cấu trúc & Đánh giá (Gold chỉ để chấm)")
colA,colB = st.columns(2)
with colA:
    if st.button("Chạy check_yolo_labels.py (dataset/labels_reviewed/train)"):
        env = {**__import__('os').environ, 'PYTHONIOENCODING':'utf-8'}
        r = subprocess.run([sys.executable, str(ROOT/'tools/check_yolo_labels.py'), '--labels', 'dataset/labels_reviewed/train'], capture_output=True, text=True, cwd=str(ROOT), env=env)
        st.code((r.stdout or '') + ('\n'+r.stderr if r.stderr else ''), language='text')
with colB:
    if st.button("Chạy evaluate_annotations.py (reviewed vs Gold)"):
        env = {**__import__('os').environ, 'PYTHONIOENCODING':'utf-8'}
        outp = OUT/'ui_eval.json'
        r = subprocess.run([sys.executable, str(ROOT/'tools/evaluate_annotations.py'), '--student-labels','dataset/labels_reviewed/train','--gold-labels','gold_train_labels','--output',str(outp)], capture_output=True, text=True, cwd=str(ROOT), env=env)
        if outp.exists():
            d=json.loads(outp.read_text(encoding='utf-8'))
            m={k:d[k] for k in ['precision','recall','f1','mean_iou_of_matches','true_positive','false_positive','false_negative','passed','required_f1','required_mean_iou']}
            st.json(m)
            st.caption("Gold chỉ dùng để đánh giá. passed=false => chưa đạt (F1>=0.80 & mean IoU>=0.70).")
        else:
            st.code((r.stdout or '')+'\n'+(r.stderr or ''))
st.caption("Log thay đổi: outputs/review_automation/ui_change_log.jsonl. Ảnh gốc/nhãn gốc/Gold không bị sửa.")
