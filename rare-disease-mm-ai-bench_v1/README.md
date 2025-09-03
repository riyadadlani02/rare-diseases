
# Rare Disease Multimodal AI — Reproduction Bench

This repository provides **reproducible training/evaluation scripts** for the table of results you shared
(MIMIC-III/IV baselines and multimodal fusion). It includes text-only (BioBERT/BlueBERT/PubMedBERT),
imaging-only (EfficientNet-B0), time-series-only (LSTM), and multimodal fusion (SLM+CNN+LSTM).

> ⚠️ **Data access**: MIMIC datasets require **credentialed access** via PhysioNet (links below).
This repo contains *no patient data*. You must download and preprocess data locally, then point the scripts
to your paths in `configs/*.yaml`.

## Datasets (Official Links)
- **MIMIC-III Clinical Database** (PhysioNet, credentialed): https://archive.physionet.org/physiobank/database/mimic3cdb/  
- **MIMIC-IV v2.2 Clinical Database** (PhysioNet, credentialed): https://physionet.org/content/mimiciv/2.2/  
- **MIMIC-CXR (Chest X-ray) v1.0.0** (imaging): https://physionet.org/content/mimic-cxr/1.0.0/  
- **MIMIC-IV Demo** (open subset): https://physionet.org/content/mimic-iv-demo/  
- **MIMIC-IV ECG module** (optional): https://physionet.org/content/?topic=mimic-iv  
- **VQA-RAD** (optional QA supervision): https://huggingface.co/datasets/flaviagiammarino/vqa-rad

## Pretrained Models
- **BioBERT** (dmis-lab/biobert-v1.1): https://huggingface.co/dmis-lab/biobert-v1.1  
- **PubMedBERT / BiomedBERT** (from-scratch on PubMed/PMC):  
  - Abstracts-only: https://huggingface.co/microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract  
  - Abstract+Fulltext: https://huggingface.co/microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext  
- **BlueBERT** (PubMed + MIMIC-III): https://github.com/ncbi-nlp/bluebert  
- **EfficientNet-B0** (torchvision): https://docs.pytorch.org/vision/main/models/generated/torchvision.models.efficientnet_b0.html

## Quick Start
```bash
# 1) Install
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 2) Edit a config with your local paths
vim configs/mimic_iv_text.yaml

# 3) Train a baseline
python scripts/train_text_model.py --config configs/mimic_iv_text.yaml --model_name dmis-lab/biobert-v1.1

# 4) Train imaging baseline (MIMIC-CXR)
python scripts/train_cnn_imaging.py --config configs/mimic_cxr.yaml

# 5) Train time-series baseline (extract vital/lab time series to CSV/Parquet first)
python scripts/train_lstm_timeseries.py --config configs/mimic_timeseries.yaml

# 6) Multimodal fusion (loads features exported by the three streams)
python scripts/run_multimodal_fusion.py --config configs/fusion.yaml
```

## Results Table Targets (for verification)
- **MIMIC-III BioBERT (baseline)**: Acc ~78.3, Macro F1 ~0.71, AUROC ~0.85
- **MIMIC-III CNN (baseline)**: Acc ~75.9, Macro F1 ~0.68, AUROC ~0.82
- **MIMIC-III LSTM (baseline)**: Acc ~76.5, Macro F1 ~0.70, AUROC ~0.83
- **MIMIC-IV BioBERT (retrained)**: Acc ~80.1, Macro F1 ~0.75, AUROC ~0.87
- **BlueBERT (retrained)**: Acc ~80.8, Macro F1 ~0.758, AUROC ~0.876
- **PubMedBERT (retrained)**: Acc ~81.7, Macro F1 ~0.77, AUROC ~0.883
- **MIMIC-IV CNN (EffNet)**: Acc ~79.2, Macro F1 ~0.72, AUROC ~0.85
- **MIMIC-IV SLM+CNN**: Acc ~79.2, Macro F1 ~0.75, AUROC ~0.81
- **MIMIC-IV SLM+CNN+LSTM (ours)**: Acc ~84.6, Macro F1 ~0.80, AUROC ~0.89

## License
MIT
