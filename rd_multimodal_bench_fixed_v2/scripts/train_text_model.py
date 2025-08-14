
import argparse, os, json, numpy as np, pandas as pd, torch
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from utils.bootstrap_utils import bootstrap_ci

class TextDS(Dataset):
    def __init__(self, csv_path, text_col="text", label_col="label", tokenizer=None, max_length=256):
        df = pd.read_csv(csv_path)
        if text_col not in df.columns or label_col not in df.columns:
            raise ValueError(f"CSV must contain columns '{text_col}' and '{label_col}'")
        self.texts = df[text_col].astype(str).tolist()
        self.labels = df[label_col].astype(int).tolist()
        self.tok = tokenizer; self.max_length = max_length
    def __len__(self): return len(self.texts)
    def __getitem__(self, i):
        enc = self.tok(self.texts[i], truncation=True, padding='max_length', max_length=self.max_length, return_tensors='pt')
        return {k:v.squeeze(0) for k,v in enc.items()}, torch.tensor(self.labels[i], dtype=torch.long)

def evaluate(model, dl, device):
    model.eval(); y_true=[]; y_prob=[]
    with torch.no_grad():
        for x,y in dl:
            x={k:v.to(device) for k,v in x.items()}
            logits = model(**x).logits
            prob = torch.softmax(logits, dim=-1)[:,1].detach().cpu().numpy()
            y_prob.extend(prob); y_true.extend(y.numpy())
    y_true = np.array(y_true); y_prob = np.array(y_prob)
    y_pred = (y_prob>=0.5).astype(int)
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='macro')
    try: auroc = roc_auc_score(y_true, y_prob)
    except: auroc = float('nan')
    return acc, f1, auroc, y_true, y_prob, y_pred

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_csv", required=True)
    ap.add_argument("--val_csv",   required=True)
    ap.add_argument("--text_col", default="text")
    ap.add_argument("--label_col", default="label")
    ap.add_argument("--model_name", default="dmis-lab/biobert-v1.1")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--max_length", type=int, default=256)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=2).to(device)

    tr_ds = TextDS(args.train_csv, args.text_col, args.label_col, tok, args.max_length)
    va_ds = TextDS(args.val_csv, args.text_col, args.label_col, tok, args.max_length)
    tr_dl = DataLoader(tr_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)
    va_dl = DataLoader(va_ds, batch_size=args.batch_size, num_workers=2)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = len(tr_dl) * args.epochs
    sch = get_linear_schedule_with_warmup(opt, int(0.1*total_steps), total_steps)

    best_f1 = -1; best_path = os.path.join(args.out_dir, "best.pt")
    for ep in range(1, args.epochs+1):
        model.train()
        pbar = tqdm(tr_dl, desc=f"epoch {ep}")
        for x,y in pbar:
            x={k:v.to(device) for k,v in x.items()}; y=y.to(device)
            out = model(**x, labels=y)
            loss = out.loss
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sch.step()
            pbar.set_postfix(loss=float(loss))

        acc,f1,auc, y_true,y_prob,y_pred = evaluate(model, va_dl, device)
        if f1>best_f1:
            best_f1=f1; torch.save(model.state_dict(), best_path)
        ci_acc, ci_f1, ci_auc = bootstrap_ci(y_true, y_prob, y_pred, average="macro", n_iter=1000, ci=95)
        metrics = {"accuracy": float(acc), "macro_f1": float(f1), "auroc": float(auc) if auc==auc else None,
                   "ci_acc": [float(ci_acc[0]), float(ci_acc[1])],
                   "ci_f1":  [float(ci_f1[0]),  float(ci_f1[1])],
                   "ci_auc": [float(ci_auc[0]), float(ci_auc[1])]}
        json.dump(metrics, open(os.path.join(args.out_dir, "val_metrics.json"), "w"), indent=2)
        print(f"[val] acc={acc:.4f} f1={f1:.4f} auroc={auc:.4f}")
    print(f"Saved best checkpoint → {best_path}")

if __name__ == "__main__":
    main()
