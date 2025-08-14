
import argparse, yaml, numpy as np, pandas as pd, torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

class TSDataset(Dataset):
    def __init__(self, parquet, max_len=256):
        df = pd.read_parquet(parquet)
        self.pid = df['pid'].values
        self.y = df['label'].astype(int).values
        feats = df.drop(columns=['pid','time','label']).values.reshape((-1, max_len, -1))
        self.x = torch.tensor(feats, dtype=torch.float32)
    def __len__(self): return len(self.y)
    def __getitem__(self, i): return self.x[i], torch.tensor(self.y[i], dtype=torch.long)

class LSTMHead(nn.Module):
    def __init__(self, in_dim, hidden=256, out=2, p=0.3):
        super().__init__()
        self.lstm = nn.LSTM(in_dim, hidden, num_layers=2, batch_first=True, dropout=p)
        self.fc = nn.Sequential(nn.Linear(hidden, 128), nn.ReLU(), nn.Dropout(p), nn.Linear(128, out))
    def forward(self, x):
        _, (h, _) = self.lstm(x)
        return self.fc(h[-1])

def evaluate(model, dl, device):
    model.eval(); y_true=[]; y_prob=[]
    with torch.no_grad():
        for x,y in dl:
            x=x.to(device); logits=model(x)
            prob=torch.softmax(logits,dim=-1)[:,1].cpu().numpy()
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

    tr=TSDataset(cfg['data']['train_parquet'], cfg['training']['max_len'])
    va=TSDataset(cfg['data']['val_parquet'], cfg['training']['max_len'])
    train_dl=DataLoader(tr, batch_size=cfg['training']['batch_size'], shuffle=True)
    val_dl=DataLoader(va, batch_size=cfg['training']['batch_size'])

    in_dim=tr.x.shape[-1]
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model=LSTMHead(in_dim, hidden=cfg['training']['hidden']).to(device)
    opt=torch.optim.AdamW(model.parameters(), lr=float(cfg['training']['lr']), weight_decay=float(cfg['training']['weight_decay']))
    ce=nn.CrossEntropyLoss()

    for ep in range(cfg['training']['epochs']):
        model.train()
        for x,y in train_dl:
            x=x.to(device); y=y.to(device)
            opt.zero_grad(); loss=ce(model(x),y); loss.backward(); opt.step()
        acc,f1,auc=evaluate(model, val_dl, device)
        print(f"[val] acc={acc:.4f} f1={f1:.4f} auroc={auc:.4f}")

if __name__=="__main__":
    main()
