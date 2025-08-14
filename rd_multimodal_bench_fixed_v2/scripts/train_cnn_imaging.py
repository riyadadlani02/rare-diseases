
import argparse, os, json, numpy as np, torch
from torch import nn
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from utils.bootstrap_utils import bootstrap_ci

def get_model(num_classes=2):
    m = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
    in_f = m.classifier[1].in_features
    m.classifier[1] = nn.Linear(in_f, num_classes)
    return m

def evaluate(model, dl, device):
    model.eval(); y_true=[]; y_prob=[]
    with torch.no_grad():
        for x,y in dl:
            x=x.to(device); logits=model(x)
            p=torch.softmax(logits,dim=-1)[:,1].cpu().numpy().tolist()
            y_prob.extend(p); y_true.extend(y.numpy().tolist())
    y_true=np.array(y_true); y_prob=np.array(y_prob)
    y_pred=(y_prob>=0.5).astype(int)
    acc=accuracy_score(y_true,y_pred)
    f1=f1_score(y_true,y_pred,average='macro')
    try: auc=roc_auc_score(y_true,y_prob)
    except: auc=float('nan')
    return acc,f1,auc,y_true,y_prob,y_pred

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--train_dir", required=True)
    ap.add_argument("--val_dir", required=True)
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--seed", type=int, default=42)
    args=ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    tfm_train = transforms.Compose([
        transforms.Resize((args.img_size,args.img_size)),
        transforms.Grayscale(num_output_channels=3),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])
    tfm_val = transforms.Compose([
        transforms.Resize((args.img_size,args.img_size)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor()
    ])

    ds_tr=datasets.ImageFolder(args.train_dir, transform=tfm_train)
    ds_va=datasets.ImageFolder(args.val_dir, transform=tfm_val)
    dl_tr=DataLoader(ds_tr, batch_size=args.batch_size, shuffle=True, num_workers=2)
    dl_va=DataLoader(ds_va, batch_size=args.batch_size, num_workers=2)

    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model=get_model(num_classes=2).to(device)
    opt=torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    ce=nn.CrossEntropyLoss()

    best_f1=-1; best_path=os.path.join(args.out_dir,"best.pt")
    for ep in range(1,args.epochs+1):
        model.train()
        for x,y in dl_tr:
            x=x.to(device); y=y.to(device)
            opt.zero_grad(); loss=ce(model(x),y); loss.backward(); opt.step()
        acc,f1,auc,y_true,y_prob,y_pred = evaluate(model, dl_va, device)
        if f1>best_f1:
            best_f1=f1; torch.save(model.state_dict(), best_path)
        ci_acc, ci_f1, ci_auc = bootstrap_ci(y_true, y_prob, y_pred, average="macro", n_iter=1000, ci=95)
        metrics={"accuracy":float(acc),"macro_f1":float(f1),"auroc":float(auc) if auc==auc else None,
                 "ci_acc":[float(ci_acc[0]),float(ci_acc[1])],
                 "ci_f1":[float(ci_f1[0]),float(ci_f1[1])],
                 "ci_auc":[float(ci_auc[0]),float(ci_auc[1])]}
        json.dump(metrics, open(os.path.join(args.out_dir,"val_metrics.json"),"w"), indent=2)
        print(f"[val] acc={acc:.4f} f1={f1:.4f} auroc={auc:.4f}")
    print(f"Saved best checkpoint → {best_path}")

if __name__=="__main__":
    main()
