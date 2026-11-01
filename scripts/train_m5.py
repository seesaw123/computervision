"""Fine-tune a YOLO model using a custom labeled dataset."""
import argparse
from pathlib import Path
def run():
    p=argparse.ArgumentParser()
    p.add_argument('--data',default='data/custom_objects/data.yaml')
    p.add_argument('--epochs',type=int,default=30);p.add_argument('--imgsz',type=int,default=512)
    p.add_argument('--batch',type=int,default=8);p.add_argument('--device',default='cpu')
    p.add_argument('--model',default='yolo26n.pt');args=p.parse_args()
    path=Path(args.data)
    if not path.is_file():raise SystemExit(f'Missing {path}; see data/custom_objects/README.md')
    from ultralytics import YOLO
    import yaml
    cfg=yaml.safe_load(path.read_text());cfg['path']=str(path.resolve().parent)
    resolved=path.parent/'data.resolved.yaml';resolved.write_text(yaml.safe_dump(cfg,sort_keys=False))
    YOLO(args.model).train(data=str(resolved.resolve()),epochs=args.epochs,imgsz=args.imgsz,batch=args.batch,device=args.device,project='runs/m5',name='custom_objects',exist_ok=True)
if __name__=='__main__':run()
