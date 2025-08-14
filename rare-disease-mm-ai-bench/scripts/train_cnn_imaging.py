
import argparse, os, yaml, numpy as np, torch
from torch import nn
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

def get_model(num_classes=2):
    m = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
    m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)
    return m

def evaluate(model, dl, device):
    model.eval(); y_true=[]; y_prob=[]
    with torch.no_grad():
        for x,y in dl:
            x=x.to(device); y_true.extend(y.numpy().tolist())
            logits = model(x)
            prob = torch.softmax(logits, dim=-1)[:,1].cpu().numpy().tolist()
            y_prob.extend(prob)
    y_pred = (np.array(y_prob)>=0.5).astype(int)
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='macro')
    try:
        auc = roc_auc_score(y_true, y_prob)
    except:
        auc = float('nan')
    return acc,f1,auc

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))

    tfm = transforms.Compose([
        transforms.Resize((cfg['training']['img_size'], cfg['training']['img_size'])),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor()
    ])
    train_ds = datasets.ImageFolder(cfg['data']['train_dir'], transform=tfm)
    val_ds   = datasets.ImageFolder(cfg['data']['val_dir'], transform=tfm)
    train_dl = DataLoader(train_ds, batch_size=cfg['training']['batch_size'], shuffle=True, num_workers=2)
    val_dl   = DataLoader(val_ds, batch_size=cfg['training']['batch_size'], num_workers=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg['training']['lr']), weight_decay=float(cfg['training']['weight_decay']))
    ce = nn.CrossEntropyLoss()

    for ep in range(cfg['training']['epochs']):
        model.train()
        for x,y in train_dl:
            x=x.to(device); y=y.to(device)
            opt.zero_grad(); loss = ce(model(x), y); loss.backward(); opt.step()
        acc,f1,auc = evaluate(model, val_dl, device)
        print(f"[val] acc={acc:.4f} f1={f1:.4f} auroc={auc:.4f}")

if __name__=="__main__":
    main()
