"""
This script trains and evaluates a Convolutional Neural Network (CNN) for image classification.

The script is designed to perform the following steps:
1.  **Model Initialization**: It uses a pre-trained EfficientNet-B0 model from `torchvision`.
    The final classifier layer is replaced with a new sequential layer containing dropout
    and a linear layer to match the number of classes in the target dataset.

2.  **Argument Parsing**: It uses `argparse` to manage all command-line arguments,
    including paths to data directories, training hyperparameters (epochs, batch size,
    learning rate), and output settings.

3.  **Data Loading and Preprocessing**:
    -   It defines a series of image transformations using `torchvision.transforms`,
      including resizing, converting to grayscale (and duplicating to 3 channels to
      match the pre-trained model's input), and converting to PyTorch tensors.
    -   It uses `datasets.ImageFolder` to load training and validation images from
      specified directories, where each subdirectory corresponds to a class.
    -   `DataLoader` is used to create batches of data for training and validation, with
      support for shuffling and parallel data loading.

4.  **Training and Evaluation Loop**:
    -   The model is trained using the AdamW optimizer and Cross-Entropy Loss.
    -   The script iterates through the training data for a specified number of epochs.
    -   After each epoch, the model's performance is evaluated on the validation set.
    -   The best model checkpoint (based on macro F1-score) is saved.

5.  **Metrics and Confidence Intervals**:
    -   During evaluation, it calculates key classification metrics: accuracy, macro F1-score,
      and multi-class AUROC.
    -   It also computes 95% confidence intervals for these metrics using a bootstrap
      resampling method from `utils.bootstrap_utils`.
    -   The final metrics and CIs are saved to a JSON file for later analysis.

To run the script, you must provide paths to the training and validation data directories.

Example Usage:
    python train_cnn_imaging.py --train_dir path/to/train_data --val_dir path/to/val_data \\
                                --out_dir results/cnn_run1
"""
import argparse, os, json, numpy as np, torch
from torch import nn
import cv2
from PIL import Image
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
    
class MedicalImagePreprocess:
    def __init__(self, img_size=224):
        self.img_size = img_size
        self.to_tensor = transforms.ToTensor()

    def __call__(self, img):
        # Convert to grayscale
        img = img.convert("L")
        # CLAHE for contrast enhancement
        np_img = np.array(img)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        np_img = clahe.apply(np_img)
        # Gaussian smoothing for denoising
        np_img = cv2.GaussianBlur(np_img, (3,3), 0)
        # Intensity normalization
        np_img = cv2.normalize(np_img, None, 0, 255, cv2.NORM_MINMAX)
        # Resize and convert to 3 channels
        img = Image.fromarray(np_img)
        img = img.resize((self.img_size, self.img_size))
        img = img.convert("RGB")
        return self.to_tensor(img)


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

    tfm_train = MedicalImagePreprocess(img_size=args.img_size)
    tfm_val = MedicalImagePreprocess(img_size=args.img_size)


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
