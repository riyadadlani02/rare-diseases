
import argparse, os, json, gzip, numpy as np, pandas as pd, torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from utils.bootstrap_utils import bootstrap_ci
from tqdm import tqdm

def read_gz_csv(path):
    with gzip.open(path, 'rt') as f:
        return pd.read_csv(f)

def build_labels(diagnoses_csv, topk=5):
    d = read_gz_csv(diagnoses_csv)
    d = d.dropna(subset=["icd_code"]).copy()
    d["icd_cat"] = d["icd_code"].astype(str).str[:3]
    # first diagnosis per subject
    d = d.sort_values(["subject_id"]).drop_duplicates("subject_id", keep="first")
    top = d["icd_cat"].value_counts().head(topk).index
    d = d[d["icd_cat"].isin(top)]
    return d[["subject_id","icd_cat"]]

def build_sequences(chartevents_csv, keep_subjects, max_len=256):
    df = read_gz_csv(chartevents_csv)
    df = df[df["subject_id"].isin(keep_subjects)].copy()
    cols = [c for c in ["subject_id","itemid","charttime"] if c in df.columns]
    df = df[cols]
    if "charttime" in df.columns:
        df["charttime"] = pd.to_datetime(df["charttime"], errors="coerce")
        df = df.sort_values(["subject_id","charttime"])
    seqs = df.groupby("subject_id")["itemid"].apply(list)
    all_items = sorted(set(i for seq in seqs for i in seq))
    vocab = {iid:i+1 for i,iid in enumerate(all_items)}  # 0=pad
    seqs_idx = [np.array([vocab[i] for i in seq], dtype=np.int64) for seq in seqs.values]
    X = []
    for s in seqs_idx:
        if len(s)>=max_len: X.append(s[:max_len])
        else: X.append(np.pad(s, (0,max_len-len(s)), constant_values=0))
    X = np.stack(X)
    sids = seqs.index.values
    return X, sids, len(vocab)+1

class SeqDS(Dataset):
    def __init__(self, X, y): self.X=torch.tensor(X, dtype=torch.long); self.y=torch.tensor(y, dtype=torch.long)
    def __len__(self): return len(self.y)
    def __getitem__(self, i): return self.X[i], self.y[i]

class LSTMNet(nn.Module):
    def __init__(self, vocab_size, emb=128, hidden=256, num_classes=2, p=0.3):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb, padding_idx=0)
        self.lstm1 = nn.LSTM(emb, hidden, num_layers=1, batch_first=True)
        self.drop1 = nn.Dropout(p)
        self.lstm2 = nn.LSTM(hidden, 128, num_layers=1, batch_first=True)
        self.drop2 = nn.Dropout(p)
        self.fc = nn.Linear(128, num_classes)
    def forward(self, x):
        x = self.emb(x)
        out,_ = self.lstm1(x); out=self.drop1(out)
        out,_ = self.lstm2(out); out=self.drop2(out)
        h = out[:,-1,:]
        return self.fc(h)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chartevents", required=True)
    ap.add_argument("--diagnoses",   required=True)
    ap.add_argument("--max_len", type=int, default=256)
    ap.add_argument("--topk_labels", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    np.random.seed(args.seed); torch.manual_seed(args.seed)

    labels_df = build_labels(args.diagnoses, topk=args.topk_labels)
    keep_subjects = set(labels_df["subject_id"].unique())
    X, sids, vocab_size = build_sequences(args.chartevents, keep_subjects, max_len=args.max_len)

    sid2label = dict(zip(labels_df["subject_id"], labels_df["icd_cat"]))
    y_labels = [sid2label.get(s, None) for s in sids]
    mask = np.array([lbl is not None for lbl in y_labels])
    X = X[mask]; y_labels = [lbl for lbl in y_labels if lbl is not None]

    le = LabelEncoder(); y = le.fit_transform(y_labels)
    num_classes = len(le.classes_)

    X_tr, X_va, y_tr, y_va = train_test_split(X, y, test_size=0.2, random_state=args.seed, stratify=y)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMNet(vocab_size=vocab_size, num_classes=num_classes).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    ce = nn.CrossEntropyLoss()

    best_f1=-1; best_path=os.path.join(args.out_dir,"best.pt")
    for ep in range(1, args.epochs+1):
        model.train()
        dl_tr = DataLoader(SeqDS(X_tr,y_tr), batch_size=args.batch_size, shuffle=True)
        for xb,yb in tqdm(dl_tr, desc=f"epoch {ep}"):
            xb=xb.to(device); yb=yb.to(device)
            opt.zero_grad(); loss=ce(model(xb), yb); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()

        # Eval
        model.eval()
        dl_va = DataLoader(SeqDS(X_va,y_va), batch_size=args.batch_size)
        y_true=[]; y_prob=[]; y_pred=[]
        with torch.no_grad():
            for xb,yb in dl_va:
                xb=xb.to(device)
                logits = model(xb)
                prob = torch.softmax(logits, dim=-1).cpu().numpy()
                y_prob.append(prob); y_true.append(yb.numpy()); y_pred.append(np.argmax(prob,axis=1))
        y_prob = np.concatenate(y_prob, axis=0)
        y_true = np.concatenate(y_true, axis=0)
        y_pred = np.concatenate(y_pred, axis=0)

        acc = accuracy_score(y_true, y_pred)
        f1  = f1_score(y_true, y_pred, average="macro")
        try:
            auroc = roc_auc_score(pd.get_dummies(y_true), y_prob, multi_class="ovr" if num_classes>2 else "raise")
        except Exception:
            auroc = float('nan')

        if f1>best_f1:
            best_f1=f1; torch.save(model.state_dict(), best_path)

        ci_acc, ci_f1, ci_auc = bootstrap_ci(y_true, y_prob, y_pred, average="macro", n_iter=1000, ci=95)
        metrics = {"accuracy": float(acc), "macro_f1": float(f1), "auroc": float(auroc) if auroc==auroc else None,
                   "ci_acc": [float(ci_acc[0]), float(ci_acc[1])],
                   "ci_f1":  [float(ci_f1[0]),  float(ci_f1[1])],
                   "ci_auc": [float(ci_auc[0]), float(ci_auc[1])],
                   "classes": [str(c) for c in le.classes_]}
        json.dump(metrics, open(os.path.join(args.out_dir,"val_metrics.json"),"w"), indent=2)
        print(f"[val] acc={acc:.4f} f1={f1:.4f} auroc={auroc:.4f}")
    print(f"Saved best checkpoint → {best_path}")

if __name__=="__main__":
    main()
