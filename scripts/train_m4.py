"""Train an original CNN from scratch on synthetic images."""
import argparse,csv,json,random,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset,DataLoader
from PIL import Image
from visionlab.shapes import create_dataset,CLASSES
class ShapeDataset(Dataset):
    def __init__(self,root,split):
        self.files=[(p,i) for i,c in enumerate(CLASSES) for p in sorted((Path(root)/split/c).glob('*.png'))]
    def __len__(self):return len(self.files)
    def __getitem__(self,i):
        path,label=self.files[i]
        a=np.asarray(Image.open(path).convert('RGB'),dtype=np.float32)/255
        return torch.from_numpy(a.transpose(2,0,1).copy()),torch.tensor(label,dtype=torch.long)
class TinyCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features=nn.Sequential(nn.Conv2d(3,16,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),nn.Conv2d(16,32,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),nn.Conv2d(32,48,3,padding=1),nn.ReLU(),nn.AdaptiveAvgPool2d((4,4)))
        self.head=nn.Sequential(nn.Flatten(),nn.Linear(48*4*4,64),nn.ReLU(),nn.Linear(64,3))
    def forward(self,x):return self.head(self.features(x))
def run(args):
    random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
    root=Path(args.dataset);out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    if not (root/'train').exists():create_dataset(root,args.samples,args.seed)
    datasets={s:ShapeDataset(root,s) for s in ('train','val','test')}
    if not all(len(d) for d in datasets.values()):raise ValueError('Train, val and test must all contain images')
    loaders={s:DataLoader(d,batch_size=args.batch,shuffle=s=='train') for s,d in datasets.items()}
    model=TinyCNN();opt=torch.optim.Adam(model.parameters(),lr=args.lr);loss_fn=nn.CrossEntropyLoss()
    history=[];best=-1.
    for epoch in range(1,args.epochs+1):
        stats={}
        for split in ('train','val'):
            model.train(split=='train');total=correct=0;losses=0.
            for x,y in loaders[split]:
                with torch.set_grad_enabled(split=='train'):
                    logits=model(x);loss=loss_fn(logits,y)
                    if split=='train':opt.zero_grad();loss.backward();opt.step()
                n=len(y);total+=n;losses+=float(loss.item())*n;correct+=int((logits.argmax(1)==y).sum())
            stats[split+'_loss']=losses/total;stats[split+'_accuracy']=correct/total
        history.append({'epoch':epoch,**stats})
        if stats['val_accuracy']>best:
            best=stats['val_accuracy'];torch.save({'model_state':model.state_dict(),'classes':CLASSES},out/'m4_shapes_cnn.pt')
        print(f"Epoch {epoch}/{args.epochs}: train={stats['train_accuracy']:.3f} val={stats['val_accuracy']:.3f}")
    with (out/'m4_training_history.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(history[0]));w.writeheader();w.writerows(history)
    checkpoint=torch.load(out/'m4_shapes_cnn.pt',map_location='cpu',weights_only=True)
    model.load_state_dict(checkpoint['model_state']);model.eval()
    total=correct=0;confusion=np.zeros((3,3),dtype=int)
    with torch.no_grad():
        for x,y in loaders['test']:
            pred=model(x).argmax(1)
            total+=len(y);correct+=int((pred==y).sum())
            for a,b in zip(y.tolist(),pred.tolist()):confusion[a,b]+=1
    metrics={'test_accuracy':correct/total,'test_count':total,'classes':CLASSES,'confusion_matrix_rows_true_cols_pred':confusion.tolist(),'best_validation_accuracy':best,'dataset_type':'synthetic educational images'}
    (out/'m4_test_metrics.json').write_text(json.dumps(metrics,indent=2));print('TEST:',metrics)
if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--dataset',default='data/shapes');p.add_argument('--output',default='models')
    p.add_argument('--samples',type=int,default=150);p.add_argument('--epochs',type=int,default=8)
    p.add_argument('--batch',type=int,default=32);p.add_argument('--lr',type=float,default=.001)
    p.add_argument('--seed',type=int,default=42);run(p.parse_args())
