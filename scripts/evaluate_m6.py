"""Compare trained object detectors on a single labeled dataset split."""
import argparse,csv,json
from pathlib import Path
def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data',required=True);p.add_argument('--models',nargs='+',required=True)
    p.add_argument('--split',default='test',choices=['val','test'])
    p.add_argument('--output',default='runs/m6/model_comparison.csv')
    p.add_argument('--device',default='cpu');args=p.parse_args()
    from ultralytics import YOLO
    import yaml
    source=Path(args.data)
    if not source.is_file():raise SystemExit(f'Missing dataset YAML: {source}')
    cfg=yaml.safe_load(source.read_text());cfg['path']=str(source.resolve().parent)
    resolved=source.parent/'data.resolved.yaml';resolved.write_text(yaml.safe_dump(cfg,sort_keys=False))
    rows=[]
    for path in args.models:
        m=YOLO(path).val(data=str(resolved.resolve()),split=args.split,device=args.device,project='runs/m6',name=Path(path).stem,exist_ok=True)
        rows.append({'checkpoint':path,'precision':float(m.box.mp),'recall':float(m.box.mr),'mAP50':float(m.box.map50),'mAP50_95':float(m.box.map)})
    dest=Path(args.output);dest.parent.mkdir(parents=True,exist_ok=True)
    with dest.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    print(json.dumps(rows,indent=2))
if __name__=='__main__':main()
