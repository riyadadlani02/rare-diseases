
import argparse, os, random, numpy as np, pandas as pd, torch, yaml
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from tqdm import tqdm

class TextDS(Dataset):
    def __init__(self, csv_path, tokenizer, max_length):
        df = pd.read_csv(csv_path)
        self.texts = df['text'].astype(str).tolist()
        self.labels = df['label'].astype(int).tolist()
        self.tok = tokenizer; self.max_length = max_length
    def __len__(self): return len(self.texts)
    def __getitem__(self, i):
        enc = self.tok(self.texts[i], truncation=True, padding='max_length', max_length=self.max_length, return_tensors='pt')
        return {k:v.squeeze(0) for k,v in enc.items()}, torch.tensor(self.labels[i], dtype=torch.long)

def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

def evaluate(model, dl, device):
    model.eval(); y_true=[]; y_prob=[]
    with torch.no_grad():
        for batch in dl:
            x,y = batch
            x={k:v.to(device) for k,v in x.items()}
            logits = model(**x).logits
            prob = torch.softmax(logits, dim=-1)[:,1].detach().cpu().numpy()
            y_prob.extend(prob); y_true.extend(y.numpy())
    y_pred = (np.array(y_prob)>=0.5).astype(int)
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='macro')
    try:
        auroc = roc_auc_score(y_true, y_prob)
    except:
        auroc = float('nan')
    return acc, f1, auroc

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--model_name", default="dmis-lab/biobert-v1.1")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    set_seed(cfg['training']['seed'])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tok = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=cfg['model']['num_labels']).to(device)

    ds_tr = TextDS(cfg['data']['train_csv'], tok, cfg['data']['max_length'])
    ds_va = TextDS(cfg['data']['val_csv'], tok, cfg['data']['max_length'])
    dl_tr = DataLoader(ds_tr, batch_size=cfg['training']['batch_size'], shuffle=True)
    dl_va = DataLoader(ds_va, batch_size=cfg['training']['batch_size'])

    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg['training']['lr']), weight_decay=float(cfg['training']['weight_decay']))
    total_steps = len(dl_tr)*cfg['training']['epochs']
    sch = get_linear_schedule_with_warmup(opt, int(0.1*total_steps), total_steps)

    for ep in range(cfg['training']['epochs']):
        model.train()
        pbar = tqdm(dl_tr, desc=f"epoch {ep+1}")
        for x,y in pbar:
            x={k:v.to(device) for k,v in x.items()}; y=y.to(device)
            out = model(**x, labels=y)
            loss = out.loss
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sch.step()
            pbar.set_postfix(loss=float(loss))

        acc,f1,auc = evaluate(model, dl_va, device)
        print(f"[val] acc={acc:.4f} f1={f1:.4f} auroc={auc:.4f}")

if __name__=="__main__":
    main()
