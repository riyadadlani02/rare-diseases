
import os
import gzip
import argparse
import numpy as np
import pandas as pd
from collections import Counter, defaultdict

# ML / Metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.utils import resample

# Deep learning
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding, Dropout
from tensorflow.keras.preprocessing.sequence import pad_sequences
import os
from Crypto.Cipher import DES3, AES
from Crypto.Random import get_random_bytes
from Crypto.Util.Padding import pad, unpad
from PIL import Image
import hashlib

class ImageEncryption:
    def __init__(self):
        # Generate keys
        self.des3_key = DES3.adjust_key_parity(get_random_bytes(24))  # 192-bit key for 3DES
        self.aes_key = get_random_bytes(32)  # 256-bit key for AES
        
    def read_image(self, filepath):
        """Read image file as bytes"""
        try:
            with open(filepath, 'rb') as file:
                return file.read()
        except FileNotFoundError:
            print(f"Error: File '{filepath}' not found.")
            return None
    
    def save_bytes_to_file(self, data, filepath):
        """Save bytes data to file"""
        with open(filepath, 'wb') as file:
            file.write(data)
    
    def triple_des_encrypt(self, data):
        """Encrypt data using Triple DES"""
        cipher = DES3.new(self.des3_key, DES3.MODE_CBC)
        iv = cipher.iv
        
        # Pad data to be multiple of 8 bytes (DES block size)
        padded_data = pad(data, DES3.block_size)
        encrypted_data = cipher.encrypt(padded_data)
        
        return iv + encrypted_data  # Prepend IV to encrypted data
    
    def triple_des_decrypt(self, encrypted_data):
        """Decrypt data using Triple DES"""
        iv = encrypted_data[:DES3.block_size]  # Extract IV
        ciphertext = encrypted_data[DES3.block_size:]
        
        cipher = DES3.new(self.des3_key, DES3.MODE_CBC, iv)
        padded_data = cipher.decrypt(ciphertext)
        
        # Remove padding
        return unpad(padded_data, DES3.block_size)
    
    def aes_encrypt(self, data):
        """Encrypt data using AES"""
        cipher = AES.new(self.aes_key, AES.MODE_CBC)
        iv = cipher.iv
        
        # Pad data to be multiple of 16 bytes (AES block size)
        padded_data = pad(data, AES.block_size)
        encrypted_data = cipher.encrypt(padded_data)
        
        return iv + encrypted_data  # Prepend IV to encrypted data
    
    def aes_decrypt(self, encrypted_data):
        """Decrypt data using AES"""
        iv = encrypted_data[:AES.block_size]  # Extract IV
        ciphertext = encrypted_data[AES.block_size:]
        
        cipher = AES.new(self.aes_key, AES.MODE_CBC, iv)
        padded_data = cipher.decrypt(ciphertext)
        
        # Remove padding
        return unpad(padded_data, AES.block_size)
    
    def process_image(self, input_filepath):
        """Main function to encrypt and decrypt image with both algorithms"""
        print(f"Processing image: {input_filepath}")
        
        # Read original image
        original_data = self.read_image(input_filepath)
        if original_data is None:
            return
        
        print(f"Original image size: {len(original_data)} bytes")
        
        # Triple DES Encryption/Decryption
        print("\n=== Triple DES Processing ===")
        des3_encrypted = self.triple_des_encrypt(original_data)
        print(f"Encrypted size (3DES): {len(des3_encrypted)} bytes")
        
        # Save encrypted data
        self.save_bytes_to_file(des3_encrypted, "passport_3des_encrypted.bin")
        print("3DES encrypted data saved as: passport_3des_encrypted.bin")
        
        # Decrypt and save
        des3_decrypted = self.triple_des_decrypt(des3_encrypted)
        self.save_bytes_to_file(des3_decrypted, "passport_3des_decrypted.jpg")
        print("3DES decrypted image saved as: passport_3des_decrypted.jpg")
        
        # AES Encryption/Decryption
        print("\n=== AES Processing ===")
        aes_encrypted = self.aes_encrypt(original_data)
        print(f"Encrypted size (AES): {len(aes_encrypted)} bytes")
        
        # Save encrypted data
        self.save_bytes_to_file(aes_encrypted, "passport_aes_encrypted.bin")
        print("AES encrypted data saved as: passport_aes_encrypted.bin")
        
        # Decrypt and save
        aes_decrypted = self.aes_decrypt(aes_encrypted)
        self.save_bytes_to_file(aes_decrypted, "passport_aes_decrypted.jpg")
        print("AES decrypted image saved as: passport_aes_decrypted.jpg")
        
        # Verify integrity
        print("\n=== Verification ===")
        print(f"Original data hash: {hashlib.sha256(original_data).hexdigest()[:16]}...")
        print(f"3DES decrypted hash: {hashlib.sha256(des3_decrypted).hexdigest()[:16]}...")
        print(f"AES decrypted hash:  {hashlib.sha256(aes_decrypted).hexdigest()[:16]}...")
        
        if original_data == des3_decrypted == aes_decrypted:
            print("✅ All decryptions successful - data integrity verified!")
        else:
            print("❌ Data integrity check failed!")
        
        # Compare the encrypted data
        comparison_results = self.compare_encrypted_data(des3_encrypted, aes_encrypted, original_data)
        
        # Create visual comparison
        self.visual_comparison(des3_encrypted, aes_encrypted)
        
        return comparison_results
    
    def compare_encrypted_data(self, des3_encrypted, aes_encrypted, original_data):
        """Compare the encrypted outputs from both algorithms"""
        print("\n=== Encrypted Data Comparison ===")
        
        # Size comparison
        print(f"Original data size:    {len(original_data):,} bytes")
        print(f"3DES encrypted size:   {len(des3_encrypted):,} bytes")
        print(f"AES encrypted size:    {len(aes_encrypted):,} bytes")
        
        # Calculate overhead
        des3_overhead = len(des3_encrypted) - len(original_data)
        aes_overhead = len(aes_encrypted) - len(original_data)
        print(f"3DES overhead:         {des3_overhead} bytes")
        print(f"AES overhead:          {aes_overhead} bytes")
        
        # Entropy analysis (randomness measure)
        def calculate_entropy(data):
            """Calculate Shannon entropy of data"""
            import math
            byte_counts = [0] * 256
            for byte in data:
                byte_counts[byte] += 1
            
            entropy = 0
            data_len = len(data)
            for count in byte_counts:
                if count > 0:
                    p = count / data_len
                    entropy -= p * math.log2(p)
            return entropy
        
        original_entropy = calculate_entropy(original_data)
        des3_entropy = calculate_entropy(des3_encrypted)
        aes_entropy = calculate_entropy(aes_encrypted)
        
        print(f"\nEntropy Analysis (higher = more random):")
        print(f"Original image entropy: {original_entropy:.3f}")
        print(f"3DES encrypted entropy: {des3_entropy:.3f}")
        print(f"AES encrypted entropy:  {aes_entropy:.3f}")
        
        # Byte frequency analysis
        print(f"\nByte Distribution:")
        print(f"Original unique bytes:  {len(set(original_data))}/256")
        print(f"3DES unique bytes:      {len(set(des3_encrypted))}/256")
        print(f"AES unique bytes:       {len(set(aes_encrypted))}/256")
        
        # Correlation test (first 100 bytes)
        sample_size = min(100, len(original_data))
        print(f"\nFirst {sample_size} bytes comparison:")
        print(f"Original: {original_data[:sample_size].hex()[:50]}...")
        print(f"3DES:     {des3_encrypted[8:8+sample_size].hex()[:50]}...")  # Skip IV
        print(f"AES:      {aes_encrypted[16:16+sample_size].hex()[:50]}...")  # Skip IV
        
        # Check if encrypted data looks random
        def has_patterns(data, pattern_length=4):
            """Check for repeating patterns"""
            patterns = {}
            for i in range(len(data) - pattern_length + 1):
                pattern = data[i:i+pattern_length]
                patterns[pattern] = patterns.get(pattern, 0) + 1
            
            max_repeats = max(patterns.values()) if patterns else 0
            return max_repeats, len(patterns)
        
        _, original_patterns = has_patterns(original_data)
        _, des3_patterns = has_patterns(des3_encrypted)
        _, aes_patterns = has_patterns(aes_encrypted)
        
        print(f"\n4-byte Pattern Analysis:")
        print(f"Original patterns:      {original_patterns}")
        print(f"3DES patterns:          {des3_patterns}")
        print(f"AES patterns:           {aes_patterns}")
        
        return {
            'sizes': {'original': len(original_data), '3des': len(des3_encrypted), 'aes': len(aes_encrypted)},
            'entropy': {'original': original_entropy, '3des': des3_entropy, 'aes': aes_entropy},
            'patterns': {'original': original_patterns, '3des': des3_patterns, 'aes': aes_patterns}
        }
    
    def visual_comparison(self, des3_encrypted, aes_encrypted):
        """Create visual representation of encrypted data"""
        try:
            from PIL import Image
            import numpy as np
            
            print("\n=== Visual Comparison ===")
            
            # Take first 10000 bytes for visualization (or less if data is smaller)
            sample_size = min(10000, len(des3_encrypted), len(aes_encrypted))
            
            # Create square dimensions
            side_length = int(sample_size ** 0.5)
            actual_size = side_length * side_length
            
            # Prepare data
            des3_sample = np.array(list(des3_encrypted[:actual_size]), dtype=np.uint8)
            aes_sample = np.array(list(aes_encrypted[:actual_size]), dtype=np.uint8)
            
            # Reshape to square
            des3_square = des3_sample.reshape(side_length, side_length)
            aes_square = aes_sample.reshape(side_length, side_length)
            
            # Create images
            des3_img = Image.fromarray(des3_square, mode='L')
            aes_img = Image.fromarray(aes_square, mode='L')
            
            # Save visualization
            des3_img.save("3des_encrypted_visualization.png")
            aes_img.save("aes_encrypted_visualization.png")
            
            print(f"Visual comparison saved:")
            print(f"- 3des_encrypted_visualization.png ({side_length}x{side_length})")
            print(f"- aes_encrypted_visualization.png ({side_length}x{side_length})")
            print("Both should look like random noise if encryption is working properly")
            
        except ImportError:
            print("PIL/numpy not available for visual comparison")
    
    def display_keys(self):
        """Display the encryption keys (for demonstration purposes)"""
        print("\n=== Encryption Keys ===")
        print(f"3DES Key: {self.des3_key.hex()}")
        print(f"AES Key:  {self.aes_key.hex()}")

# Example usage
if __name__ == "__main__":
    # Initialize encryption system
    encryptor = ImageEncryption()
    
    # Display keys
    encryptor.display_keys()
    
    # Process the image
    encryptor.process_image("passport.jpg")
    
    print("\n=== Summary ===")
    print("Files created:")
    print("- passport_3des_encrypted.bin (encrypted with Triple DES)")
    print("- passport_3des_decrypted.jpg (decrypted with Triple DES)")
    print("- passport_aes_encrypted.bin (encrypted with AES)")
    print("- passport_aes_decrypted.jpg (decrypted with AES)")
def read_gz_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file: {path}")
    with gzip.open(path, 'rt') as f:
        return pd.read_csv(f)

def infer_device():
    try:
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            return "GPU"
        else:
            return "CPU"
    except Exception:
        return "CPU"

def make_sequences(chartevents: pd.DataFrame, max_len: int = 256) -> (np.ndarray, np.ndarray):
    # Keep necessary columns and sort by time per subject for deterministic sequences
    cols = [c for c in ["subject_id", "itemid", "charttime"] if c in chartevents.columns]
    df = chartevents[cols].copy()
    if "charttime" in df.columns:
        df["charttime"] = pd.to_datetime(df["charttime"], errors="coerce")
        df = df.sort_values(["subject_id", "charttime"])
    # Build lists of itemids per subject
    seqs = df.groupby("subject_id")["itemid"].apply(list)
    # Map itemids to a compact vocabulary (start at 1 to reserve 0 for padding)
    all_itemids = [iid for seq in seqs for iid in seq]
    vocab = {iid: i+1 for i, iid in enumerate(sorted(set(all_itemids)))}
    seqs_idx = [np.array([vocab[iid] for iid in seq], dtype=np.int32) for seq in seqs]
    X = pad_sequences(seqs_idx, maxlen=max_len, padding="post", truncating="post", value=0)
    sids = seqs.index.values
    return X, sids

def pick_labels(diagnoses: pd.DataFrame, topk: int = 5) -> pd.DataFrame:
    # Choose a single label per subject: first 3 chars of ICD code (category) of the FIRST diagnosis
    d = diagnoses.dropna(subset=["icd_code"]).copy()
    d["icd_cat"] = d["icd_code"].astype(str).str[:3]
    # primary diagnosis per subject (first occurrence)
    d = d.sort_values(["subject_id"]).drop_duplicates(subset=["subject_id"], keep="first")
    # Reduce to top-K frequent categories to avoid extreme class imbalance
    counts = d["icd_cat"].value_counts()
    keep = set(counts.head(topk).index)
    d = d[d["icd_cat"].isin(keep)].copy()
    return d[["subject_id", "icd_cat"]]

def bootstrap_ci(y_true, y_prob, y_pred, average="macro", n_iter=1000, ci=95):
    rng = np.random.default_rng(42)
    n = len(y_true)
    accs, f1s, aucs = [], [], []
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    y_pred = np.array(y_pred)

    # Determine multi-class or binary
    num_classes = y_prob.shape[1] if y_prob.ndim == 2 else 2
    for _ in range(n_iter):
        idx = rng.choice(n, size=n, replace=True)
        yt = y_true[idx]
        yp = y_pred[idx]
        ypr = y_prob[idx]
        try:
            auc = roc_auc_score(pd.get_dummies(yt), ypr, multi_class="ovr" if num_classes > 2 else "raise")
        except Exception:
            auc = np.nan
        accs.append(accuracy_score(yt, yp))
        f1s.append(f1_score(yt, yp, average=average))
        aucs.append(auc)
    lo = (100 - ci) / 2.0
    hi = 100 - lo
    ci_acc = (np.nanpercentile(accs, lo), np.nanpercentile(accs, hi))
    ci_f1  = (np.nanpercentile(f1s,  lo), np.nanpercentile(f1s,  hi))
    ci_auc = (np.nanpercentile(aucs, lo), np.nanpercentile(aucs, hi))
    return ci_acc, ci_f1, ci_auc

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--admissions", default="admissions.csv.gz")
    ap.add_argument("--diagnoses", default="diagnoses_icd.csv.gz")
    ap.add_argument("--chartevents", default="chartevents.csv.gz")
    ap.add_argument("--labevents", default="labevents.csv.gz")
    ap.add_argument("--max_len", type=int, default=256)
    ap.add_argument("--topk_labels", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--test_size", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    print(f"[INFO] Using {infer_device()}")
    print("[INFO] Loading CSVs...")
    admissions = read_gz_csv(args.admissions)
    diagnoses  = read_gz_csv(args.diagnoses)
    chartevents = read_gz_csv(args.chartevents)

    print("[INFO] Building labels (ICD categories)...")
    labels_df = pick_labels(diagnoses, topk=args.topk_labels)
    keep_subjects = set(labels_df["subject_id"].unique())
    chartevents = chartevents[chartevents["subject_id"].isin(keep_subjects)].copy()

    print("[INFO] Building sequences from chartevents...")
    X, sids = make_sequences(chartevents, max_len=args.max_len)

    # Align labels to X subjects
    sid2label = dict(zip(labels_df["subject_id"], labels_df["icd_cat"]))
    y_labels = [sid2label.get(s, None) for s in sids]
    mask = [lbl is not None for lbl in y_labels]
    X = X[mask]
    y_labels = [lbl for lbl in y_labels if lbl is not None]

    # Encode labels
    le = LabelEncoder()
    y = le.fit_transform(y_labels)
    num_classes = len(le.classes_)

    print(f"[INFO] Dataset: {len(y)} subjects, {num_classes} classes -> {list(le.classes_)}")

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.seed, stratify=y
    )

    # Model
    vocab_size = int(X.max()) + 1  # includes padding id 0
    model = Sequential([
        Embedding(input_dim=vocab_size, output_dim=128, mask_zero=True),
        LSTM(256, return_sequences=True, dropout=0.3),
        LSTM(128, dropout=0.3),
        Dense(num_classes, activation="softmax")
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])

    print("[INFO] Training...")
    model.fit(X_train, y_train,
              validation_data=(X_test, y_test),
              epochs=args.epochs, batch_size=args.batch_size, verbose=1)

    print("[INFO] Evaluating...")
    y_prob = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_prob, axis=1)

    acc = accuracy_score(y_test, y_pred)
    f1  = f1_score(y_test, y_pred, average="macro")
    try:
        auroc = roc_auc_score(pd.get_dummies(y_test), y_prob, multi_class="ovr" if num_classes > 2 else "raise")
    except Exception:
        auroc = np.nan

    ci_acc, ci_f1, ci_auc = bootstrap_ci(y_test, y_prob, y_pred, average="macro", n_iter=1000, ci=95)

    # Print summary
    print("\n=== LSTM Baseline (MIMIC-IV Demo) ===")
    print(f"Accuracy:  {acc:.4f} (95% CI: {ci_acc[0]:.4f}–{ci_acc[1]:.4f})")
    print(f"Macro F1:  {f1:.4f} (95% CI: {ci_f1[0]:.4f}–{ci_f1[1]:.4f})")
    print(f"AUROC:     {auroc:.4f} (95% CI: {ci_auc[0]:.4f}–{ci_auc[1]:.4f})")
    print("\nClasses:", list(le.classes_))

    # Save metrics
    os.makedirs("results", exist_ok=True)
    out = {
        "accuracy": float(acc),
        "macro_f1": float(f1),
        "auroc": float(auroc) if not np.isnan(auroc) else None,
        "ci_acc": [float(ci_acc[0]), float(ci_acc[1])],
        "ci_f1":  [float(ci_f1[0]),  float(ci_f1[1])],
        "ci_auc": [float(ci_auc[0]), float(ci_auc[1])],
        "classes": list(map(str, le.classes_))
    }
    with open("results/lstm_demo_metrics.json", "w") as f:
        import json; json.dump(out, f, indent=2)
    print("\nSaved metrics → results/lstm_demo_metrics.json")

if __name__ == "__main__":
    main()
