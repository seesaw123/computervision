"""Synthetic shape dataset generator."""
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
CLASSES=['circle','square','triangle']
def make_image(class_id,rng,size=64):
    bg=tuple(int(v) for v in rng.integers(200,256,3))
    fg=tuple(int(v) for v in rng.integers(10,130,3))
    im=Image.new('RGB',(size,size),bg);d=ImageDraw.Draw(im)
    x=int(rng.integers(10,24));y=int(rng.integers(10,24));w=int(rng.integers(23,39))
    if class_id==0:d.ellipse((x,y,x+w,y+w),fill=fg)
    elif class_id==1:d.rectangle((x,y,x+w,y+w),fill=fg)
    else:d.polygon([(x+w//2,y),(x+w,y+w),(x,y+w)],fill=fg)
    if rng.random()<.25:im=im.filter(ImageFilter.GaussianBlur(radius=float(rng.uniform(0,.6))))
    return im
def create_dataset(root,per_class=150,seed=42):
    root=Path(root);rng=np.random.default_rng(seed)
    for cls_id,cls in enumerate(CLASSES):
        for i in range(per_class):
            split='train' if i<int(.7*per_class) else ('val' if i<int(.85*per_class) else 'test')
            dest=root/split/cls;dest.mkdir(parents=True,exist_ok=True)
            make_image(cls_id,rng).save(dest/f'{cls}_{i:04}.png')
    return root
