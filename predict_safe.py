# predict_safe.py
import yaml, argparse
from ultralytics import RTDETR

def load_names_from_yaml(yaml_path):
    with open(yaml_path, "r", encoding="utf-8") as f:
        y = yaml.safe_load(f)
    names = y.get("names")
    if isinstance(names, dict):
        # {0:'A',1:'B',...} -> list
        names = [names[k] for k in sorted(names.keys())]
    return names

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="runs/detect/train/weights/best.pt")
    ap.add_argument("--data", default="data.yaml")
    ap.add_argument("--source", default="test/images")
    ap.add_argument("--device", default="0")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--save", action="store_true")
    args = ap.parse_args()

    names = load_names_from_yaml(args.data)
    m = RTDETR(args.model)

    # ★ 关键：把模型内部的 names 强制设为 data.yaml 的顺序
    if names:
        m.model.names = {i: n for i, n in enumerate(names)}

    m.predict(
        source=args.source,
        device=args.device,
        workers=args.workers,
        conf=args.conf,
        save=args.save,
        project="runs/detect",
        name="predict_safe",
    )

if __name__ == "__main__":
    main()
