"""
Generate preliminary ADAS labels from YOLO26n (COCO pretrained).
- Input: dataset/images/train/drive_001.jpg..drive_050.jpg only
- Output: outputs/auto_prelabels/*.txt (YOLO 5-col normalized), previews/*.jpg, report.json
- Mapping: COCO car(2)->0, truck(7)->1, bus(5)->2. No auto van(3).
- Does NOT touch dataset/labels/train or CVAT.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parents[1]
IMG_DIR = ROOT / "dataset" / "images" / "train"
OUT_DIR = ROOT / "outputs" / "auto_prelabels"
PREVIEW_DIR = OUT_DIR / "previews"
MODEL_NAME = "yolo26n.pt"
# COCO id -> ADAS id
MAP = {2: 0, 7: 1, 5: 2}
ADAS_NAMES = {0: "car", 1: "truck", 2: "bus", 3: "van"}
COLORS = {0: (0, 200, 0), 1: (255, 140, 0), 2: (255, 0, 0)}  # BGR-ish for cv2

def main():
    try:
        from ultralytics import YOLO
        import cv2
    except Exception as e:
        print("IMPORT_ERROR", e)
        sys.exit(2)

    images = sorted(IMG_DIR.glob("drive_*.jpg"))
    # keep only 001-050
    selected = []
    for p in images:
        try:
            n = int(p.stem.split("_")[1])
        except Exception:
            continue
        if 1 <= n <= 50:
            selected.append(p)
    selected = sorted(selected, key=lambda p: p.name)
    if len(selected) != 50:
        print(f"WARNING: expected 50 images, found {len(selected)}: {[p.name for p in selected[:5]]} ...")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PREVIEW_DIR.mkdir(parents=True, exist_ok=True)

    model = YOLO(MODEL_NAME)
    names = model.names

    stats = {"model": MODEL_NAME, "total_images": len(selected), "images": [], "totals": Counter(), "no_detection": []}
    review = []

    for img_path in selected:
        results = model.predict(source=str(img_path), verbose=False, conf=0.25, iou=0.45)[0]
        im = cv2.imread(str(img_path))
        h, w = im.shape[:2]
        lines = []
        dets = []
        for box in results.boxes:
            coco = int(box.cls[0].item())
            if coco not in MAP:
                continue
            adas = MAP[coco]
            conf = float(box.conf[0].item())
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            # clamp
            x1 = max(0.0, min(w - 1.0, x1)); x2 = max(0.0, min(w, x2))
            y1 = max(0.0, min(h - 1.0, y1)); y2 = max(0.0, min(h, y2))
            bw = max(1.0, x2 - x1); bh = max(1.0, y2 - y1)
            xc = (x1 + bw / 2) / w
            yc = (y1 + bh / 2) / h
            nw = bw / w; nh = bh / h
            # clamp normalized
            xc = min(max(xc, 0.0), 1.0); yc = min(max(yc, 0.0), 1.0)
            nw = min(max(nw, 1e-6), 1.0); nh = min(max(nh, 1e-6), 1.0)
            lines.append(f"{adas} {xc:.6f} {yc:.6f} {nw:.6f} {nh:.6f}")
            dets.append({"adas": adas, "coco": coco, "conf": round(conf, 4), "xyxy": [round(x1,1),round(y1,1),round(x2,1),round(y2,1)], "whn": [round(nw,6),round(nh,6)]})
            stats["totals"][ADAS_NAMES[adas]] += 1
            color = COLORS.get(adas, (200,200,200))
            cv2.rectangle(im, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            label = f"{ADAS_NAMES[adas]} {conf:.2f}"
            cv2.putText(im, label, (int(x1), max(12, int(y1)-4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

        txt_path = OUT_DIR / f"{img_path.stem}.txt"
        txt_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        cv2.imwrite(str(PREVIEW_DIR / img_path.name), im)

        if not dets:
            stats["no_detection"].append(img_path.name)

        # van/class suspicion heuristics (conservative flags)
        flags = []
        for d in dets:
            adas = d["adas"]; conf = d["conf"]; wn, hn = d["whn"]
            area = wn * hn
            ar = (wn / hn) if hn else 0
            if adas == 2:  # bus: could be minibus (labelled bus per schema) or sometimes van-like? schema: bus includes minibus; van is separate.
                if area < 0.015 and conf < 0.6:
                    flags.append(f"bus conf={conf} small(area={area:.4f}) - kiểm tra minibus/van?")
            if adas == 1:  # truck
                if area < 0.02 and ar > 1.3 and conf < 0.65:
                    flags.append(f"truck conf={conf} small/wide - kiểm tra pickup/van/truck?")
            if adas == 0 and conf < 0.45:
                flags.append(f"car conf thấp {conf} - kiểm tra class/box")
        # if many vehicles expected? we can't know; flag high density? not reliable.
        if flags:
            review.append({"image": img_path.name, "frame_hint": int(img_path.stem.split('_')[1]) - 1, "detections": len(dets), "flags": flags})

        stats["images"].append({"image": img_path.name, "detections": len(dets), "classes": Counter(ADAS_NAMES[d["adas"]] for d in dets)})

    # convert counters
    stats["totals"] = dict(stats["totals"])
    for im in stats["images"]:
        im["classes"] = dict(im["classes"])
    report = {"stats": stats, "review": review, "notes": [
        "COCO car(2)->ADAS car(0), truck(7)->ADAS truck(1), bus(5)->ADAS bus(2).",
        "ADAS van(3): not auto-assigned (COCO has no van). Review flagged truck/bus/low-conf cases.",
        "Prelabels only; manual QC required. Do not overwrite dataset/labels/train.",
    ]}
    (OUT_DIR / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    # image 3 focus
    img3 = next((x for x in stats["images"] if x["image"] == "drive_003.jpg"), None)
    print("DONE")
    print(json.dumps({"totals": stats["totals"], "no_detection": stats["no_detection"], "review_count": len(review), "image3": img3}, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
