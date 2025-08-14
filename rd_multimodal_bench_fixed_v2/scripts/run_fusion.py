
import argparse, os, json, numpy as np, torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from utils.bootstrap_utils import bootstrap_ci

class FusionMLP(nn.Module):
    def __init__(self, in_dim, out_dim=2, p=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128), nn.ReLU(), nn.Dropout(p),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(p),
            nn.Linear(64, out_dim)
        )
    def forward(self, x): return self.net(x)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text_train", required=True); ap.add_argument("--text_val", required=True)
    ap.add_argument("--img_train",  required=True); ap.add_argument("--img_val",  required=True)
    ap.add_argument("--ts_train",   required=True); ap.add_argument("--ts_val",   required=True)
    ap.add_argument("--y_train",    required=True); ap.add_argument("--y_val",    required=True)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    np.random.seed(args.seed); torch.manual_seed(args.seed)

    def load(path): return torch.tensor(np.load(path), dtype=torch.float32)
    Xtr = torch.cat([load(args.text_train), load(args.img_train), load(args.ts_train)], dim=1)
    Xva = torch.cat([load(args.text_val),  load(args.img_val),  load(args.ts_val)],  dim=1)
    ytr = torch.tensor(np.load(args.y_train), dtype=torch.long)
    yva = torch.tensor(np.load(args.y_val), dtype=torch.long)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FusionMLP(in_dim=Xtr.shape[1], out_dim=int(ytr.max().item()+1)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    ce = torch.nn.CrossEntropyLoss()

    tr_dl = DataLoader(TensorDataset(Xtr,ytr), batch_size=args.batch_size, shuffle=True)
    va_dl = DataLoader(TensorDataset(Xva,yva), batch_size=args.batch_size)

    best_f1=-1; best_path=os.path.join(args.out_dir,"best.pt")
    for ep in range(1, args.epochs+1):
        model.train()
        for xb,yb in tr_dl:
            xb=xb.to(device); yb=yb.to(device)
            opt.zero_grad(); loss=ce(model(xb), yb); loss.backward(); opt.step()

        # Eval
        model.eval(); y_true=[]; y_prob=[]; y_pred=[]
        with torch.no_grad():
            for xb,yb in va_dl:
                xb=xb.to(device)
                prob=torch.softmax(model(xb), dim=-1).cpu().numpy()
                y_prob.append(prob); y_true.append(yb.numpy()); y_pred.append(np.argmax(prob,axis=1))
        y_prob=np.concatenate(y_prob,axis=0); y_true=np.concatenate(y_true,axis=0); y_pred=np.concatenate(y_pred,axis=0)
        acc=accuracy_score(y_true,y_pred); f1=f1_score(y_true,y_pred,average='macro')
        try: auroc=roc_auc_score(torch.nn.functional.one_hot(torch.tensor(y_true)).numpy(), y_prob, multi_class="ovr")
        except Exception: auroc=float('nan')

        if f1>best_f1:
            best_f1=f1; torch.save(model.state_dict(), best_path)

        ci_acc, ci_f1, ci_auc = bootstrap_ci(y_true, y_prob, y_pred, average="macro", n_iter=1000, ci=95)
        metrics={"accuracy":float(acc),"macro_f1":float(f1),"auroc":float(auroc) if auroc==auroc else None,
                 "ci_acc":[float(ci_acc[0]),float(ci_acc[1])],
                 "ci_f1":[float(ci_f1[0]),float(ci_f1[1])],
                 "ci_auc":[float(ci_auc[0]),float(ci_auc[1])]}
        json.dump(metrics, open(os.path.join(args.out_dir,"val_metrics.json"),"w"), indent=2)
        print(f"[val] acc={acc:.4f} f1={f1:.4f} auroc={auroc:.4f}")
    print(f"Saved best checkpoint → {best_path}")

if __name__=="__main__":
    main()
