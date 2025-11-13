# rtdetr_repro.py
# Ultralytics RT-DETR minimal reproducible runner (train/val/predict/export/sanity).
# Place this file next to your data.yaml (e.g., D:/COMP9517/Group_project/).

import argparse, os, sys, json, datetime, platform
from ultralytics import RTDETR, __version__ as ULTRA_VER

def env_snapshot():
    try:
        import torch, torchvision
        snap = {
            "ultralytics": ULTRA_VER,
            "python": sys.version.replace("\n", " "),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "torchvision": getattr(torchvision, "__version__", "n/a"),
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_count": torch.cuda.device_count(),
            "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        }
        return snap
    except Exception as e:
        return {"error": f"env snapshot failed: {e}"}

def write_meta(save_dir:str, args:dict):
    os.makedirs(save_dir, exist_ok=True)
    meta = {
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "args": args,
        "env": env_snapshot(),
    }
    with open(os.path.join(save_dir, "repro_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    # 尝试把 data.yaml 也备份一份（如果是相对路径）
    data_path = args.get("data", "")
    if data_path and os.path.isfile(data_path):
        import shutil
        try:
            shutil.copy2(data_path, os.path.join(save_dir, "data.yaml"))
        except Exception:
            pass

def add_common_args(p):
    p.add_argument("--model", default="rtdetr-l.pt", type=str, help="rtdetr-l.pt / rtdetr-x.pt / custom .pt")
    p.add_argument("--data", default="data.yaml", type=str, help="Ultralytics data yaml")
    p.add_argument("--device", default="0", type=str, help="'0' or 'cpu'")
    p.add_argument("--workers", default=0, type=int, help="Windows=0; Linux可2-4")
    p.add_argument("--save_dir", default=None, type=str, help="runs/detect 下自定义目录名")
    p.add_argument("--seed", default=42, type=int)

def main():
    ap = argparse.ArgumentParser("RT-DETR reproducible runner")
    sub = ap.add_subparsers(dest="mode", required=True)

    # train
    tp = sub.add_parser("train", help="Train RT-DETR")
    add_common_args(tp)
    tp.add_argument("--epochs", default=1, type=int)
    tp.add_argument("--imgsz", default=384, type=int)
    tp.add_argument("--batch", default=2, type=int)
    tp.add_argument("--freeze", default=10, type=int)
    tp.add_argument("--lr0", default=0.01, type=float)
    tp.add_argument("--optimizer", default="SGD", choices=["SGD","Adam","AdamW"])
    tp.add_argument("--cos_lr", action="store_true")
    tp.add_argument("--patience", default=3, type=int)
    tp.add_argument("--cache", action="store_true")

    # val
    vp = sub.add_parser("val", help="Validate a trained model")
    add_common_args(vp)

    # predict
    pp = sub.add_parser("predict", help="Predict on images/dir/video")
    add_common_args(pp)
    pp.add_argument("--source", default="valid/images", type=str)
    pp.add_argument("--conf", default=0.25, type=float)
    pp.add_argument("--save", action="store_true")

    # export
    ep = sub.add_parser("export", help="Export model")
    add_common_args(ep)
    ep.add_argument("--format", default="onnx", type=str)

    # sanity (快速CPU检查，不保存)
    sp = sub.add_parser("sanity", help="CPU quick check on a few images")
    add_common_args(sp)
    sp.add_argument("--source", default="valid/images", type=str)

    args = ap.parse_args()
    # 统一初始化模型
    model = RTDETR(args.model)

    # 保存元信息到将生成的目录名（先猜一个）
    # Ultralytics 会在 runs/detect 下创建实际目录；我们用自定义名可控
    name = args.save_dir if args.save_dir else None
    project = "runs/detect"

    if args.mode == "train":
        results = model.train(
            data=args.data,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            workers=args.workers,
            seed=args.seed,
            lr0=args.lr0,
            optimizer=args.optimizer,
            cos_lr=args.cos_lr,
            patience=args.patience,
            cache=args.cache,
            freeze=args.freeze,
            project=project,
            name=name,
        )
        # 推断最终保存目录（Ultralytics返回值里通常有save_dir；兜底按习惯路径）
        save_dir = getattr(results, "save_dir", None) or os.path.join(project, name or "train")
        write_meta(save_dir, vars(args))
        print(f"[TRAIN] done. Saved to: {save_dir}")

    elif args.mode == "val":
        metrics = model.val(
            data=args.data,
            device=args.device,
            workers=args.workers,
            project=project,
            name=name,
        )
        save_dir = getattr(metrics, "save_dir", None) or os.path.join(project, name or "val")
        write_meta(save_dir, vars(args))
        print(f"[VAL] metrics saved to: {save_dir}")

    elif args.mode == "predict":
        preds = model.predict(
            source=args.source,
            device=args.device,
            workers=args.workers,
            conf=args.conf,
            save=args.save,
            project=project,
            name=name,
        )
        # preds 是生成器/列表，保存目录按Ultralytics约定：
        save_dir = os.path.join(project, name or "predict")
        write_meta(save_dir, vars(args))
        print(f"[PREDICT] outputs at: {save_dir}")

    elif args.mode == "export":
        out = model.export(format=args.format, device=args.device, imgsz=384)
        # export 到当前目录，额外写个meta
        save_dir = os.path.join(project, name or "export")
        write_meta(save_dir, vars(args))
        print(f"[EXPORT] exported: {out}")

    elif args.mode == "sanity":
        # 仅用CPU快速过一遍读取流程
        model.predict(source=args.source, device="cpu", workers=0, save=False, conf=0.25)
        save_dir = os.path.join(project, name or "sanity")
        write_meta(save_dir, vars(args))
        print("[SANITY] CPU quick check OK.")

if __name__ == "__main__":
    main()
