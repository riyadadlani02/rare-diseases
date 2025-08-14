
import argparse, yaml, numpy as np, torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

class FusionMLP(nn.Module):
    def __init__(self, in_dim=384, out_dim=2, p=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128), nn.ReLU(), nn.Dropout(p),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(p),
            nn.Linear(64, out_dim)
        )
    def forward(self, x): return self.net(x)

def evaluate(model, dl, device):
    model.eval(); y_true=[]; y_prob=[]
    with torch.no_grad():
        for x,y in dl:
            x=x.to(device)
            prob=torch.softmax(model(x),dim=-1)[:,1].cpu().numpy()
            y_prob.extend(prob); y_true.extend(y.numpy())
    y_pred=(np.array(y_prob)>=0.5).astype(int)
    acc=accuracy_score(y_true,y_pred); f1=f1_score(y_true,y_pred,average='macro')
    try: auc=roc_auc_score(y_true,y_prob)
    except: auc=float('nan')
    return acc,f1,auc

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args=ap.parse_args()
    cfg=yaml.safe_load(open(args.config))

    def load_feats(path): return torch.tensor(np.load(path), dtype=torch.float32)
    X_tr=torch.cat([load_feats(cfg['features']['text_train']),
                    load_feats(cfg['features']['img_train']),
                    load_feats(cfg['features']['ts_train'])], dim=1)
    X_va=torch.cat([load_feats(cfg['features']['text_val']),
                    load_feats(cfg['features']['img_val']),
                    load_feats(cfg['features']['ts_val'])], dim=1)
    y_tr=torch.tensor(np.load(cfg['features']['text_train'].replace('text','labels').replace('.npy','_y.npy')), dtype=torch.long)
    y_va=torch.tensor(np.load(cfg['features']['text_val'].replace('text','labels').replace('.npy','_y.npy')), dtype=torch.long)

    tr_dl=DataLoader(TensorDataset(X_tr,y_tr), batch_size=cfg['training']['batch_size'], shuffle=True)
    va_dl=DataLoader(TensorDataset(X_va,y_va), batch_size=cfg['training']['batch_size'])

    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model=FusionMLP(in_dim=X_tr.shape[1], out_dim=cfg['model']['out_dim'], p=cfg['model']['p_dropout']).to(device)
    opt=torch.optim.AdamW(model.parameters(), lr=float(cfg['training']['lr']), weight_decay=float(cfg['training']['weight_decay']))
    ce=nn.CrossEntropyLoss()

    for ep in range(cfg['training']['epochs']):
        model.train()
        for x,y in tr_dl:
            x=x.to(device); y=y.to(device)
            opt.zero_grad(); loss=ce(model(x),y); loss.backward(); opt.step()
        acc,f1,auc=evaluate(model, va_dl, device)
        print(f"[val] acc={acc:.4f} f1={f1:.4f} auroc={auc:.4f}")

if __name__=="__main__":
    main()
