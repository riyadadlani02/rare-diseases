
# Rare Disease Multimodal Bench — Fixed (v2)

This package is rebuilt and verified to exist here. Use the commands below.
If you only have the **MIMIC-IV Demo**, you can run the **LSTM** baseline now.

## Setup
```bash
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## 1) LSTM from MIMIC-IV Demo (works with your uploaded files)
```bash
python scripts/train_lstm_from_mimic_demo.py   --chartevents /mnt/data/chartevents.csv.gz   --diagnoses   /mnt/data/diagnoses_icd.csv.gz   --max_len 256 --topk_labels 5 --epochs 5 --batch_size 64   --out_dir results/lstm_demo
```

## 2) Text baseline (BioBERT/BlueBERT/PubMedBERT) — needs notes CSVs (text,label)
```bash
python scripts/train_text_model.py   --train_csv /path/train.csv --val_csv /path/val.csv   --text_col text --label_col label   --model_name dmis-lab/biobert-v1.1   --epochs 3 --batch_size 16 --max_length 256   --out_dir results/text_biobert
```

## 3) Imaging baseline (EffNet-B0) — needs ImageFolder directories
```bash
python scripts/train_cnn_imaging.py   --train_dir /path/train --val_dir /path/val   --img_size 224 --epochs 5 --batch_size 32   --out_dir results/cnn_effnet
```

## 4) Fusion — needs pre-exported features (.npy) and labels
```bash
python scripts/run_fusion.py   --text_train feats/text_train.npy --text_val feats/text_val.npy   --img_train  feats/img_train.npy  --img_val  feats/img_val.npy   --ts_train   feats/ts_train.npy   --ts_val   feats/ts_val.npy   --y_train    feats/y_train.npy    --y_val    feats/y_val.npy   --epochs 20 --batch_size 64 --out_dir results/fusion
```
