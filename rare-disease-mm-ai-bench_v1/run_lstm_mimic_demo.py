"""
This script serves two distinct purposes:
1.  **Image Encryption Demonstration:** It includes an `ImageEncryption` class that
    provides a comprehensive comparison between Triple DES (3DES) and AES
    encryption algorithms. This class can encrypt an image file using both
    ciphers, decrypt it, and perform a detailed analysis of the encrypted
    outputs, including size overhead, entropy, and byte-level patterns. It also
    generates visual representations of the encrypted data to demonstrate the
    randomness of the output.

2.  **Medical Time-Series Classification with LSTM:** The main part of the script
    trains and evaluates a Long Short-Term Memory (LSTM) neural network to
    predict primary diagnosis categories from patient clinical event sequences.
    It uses data from the MIMIC-IV clinical database.

The workflow for the machine learning pipeline is as follows:
-   **Data Loading:** Reads gzipped CSV files for admissions, diagnoses, and
    chartevents from the MIMIC-IV dataset.
-   **Label Engineering:** Selects the top N most frequent primary diagnosis
    categories (based on ICD codes) to create a multi-class classification task.
-   **Feature Engineering:** Converts time-ordered clinical events (chartevents)
    for each patient into integer sequences. These sequences are then padded or
    truncated to a fixed length.
-   **Model Training:** Defines and trains a stacked LSTM model using TensorFlow/Keras
    to classify the patient sequences into one of the diagnosis categories.
-   **Evaluation:** Evaluates the trained model on a held-out test set, calculating
    key metrics like accuracy, macro F1-score, and AUROC. It also computes 95%
    confidence intervals for these metrics using bootstrapping.
-   **Results:** Prints the final performance and saves the metrics to a JSON file.

The script can be configured and run from the command line, with arguments to
specify data paths, model hyperparameters, and other settings.

Example Usage:
    python run_lstm_mimic_demo.py --admissions path/to/admissions.csv.gz \
                                  --diagnoses path/to/diagnoses_icd.csv.gz \
                                  --chartevents path/to/chartevents.csv.gz \
                                  --epochs 10
"""
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
    """
    A class to demonstrate and compare Triple DES and AES encryption on an image file.
    It handles key generation, encryption, decryption, and analysis of the encrypted output.
    """
    def __init__(self):
        """
        Initializes the ImageEncryption class by generating random keys for 3DES and AES.
        """
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
        """Save bytes data to a binary file."""
        with open(filepath, 'wb') as file:
            file.write(data)
    
    def triple_des_encrypt(self, data):
        """
        Encrypts data using the Triple DES (3DES) algorithm in CBC mode.
        A random Initialization Vector (IV) is generated and prepended to the ciphertext.
        """
        cipher = DES3.new(self.des3_key, DES3.MODE_CBC)
        iv = cipher.iv
        
        # Pad data to be a multiple of 8 bytes (DES block size)
        padded_data = pad(data, DES3.block_size)
        encrypted_data = cipher.encrypt(padded_data)
        
        return iv + encrypted_data  # Prepend IV to encrypted data for use in decryption
    
    def triple_des_decrypt(self, encrypted_data):
        """
        Decrypts data using the Triple DES (3DES) algorithm.
        It expects the IV to be prepended to the encrypted data.
        """
        iv = encrypted_data[:DES3.block_size]  # Extract IV from the beginning
        ciphertext = encrypted_data[DES3.block_size:]
        
        cipher = DES3.new(self.des3_key, DES3.MODE_CBC, iv)
        padded_data = cipher.decrypt(ciphertext)
        
        # Remove padding to get the original data
        return unpad(padded_data, DES3.block_size)
    
    def aes_encrypt(self, data):
        """
        Encrypts data using the AES algorithm in CBC mode.
        A random Initialization Vector (IV) is generated and prepended to the ciphertext.
        """
        cipher = AES.new(self.aes_key, AES.MODE_CBC)
        iv = cipher.iv
        
        # Pad data to be a multiple of 16 bytes (AES block size)
        padded_data = pad(data, AES.block_size)
        encrypted_data = cipher.encrypt(padded_data)
        
        return iv + encrypted_data  # Prepend IV for use in decryption
    
    def aes_decrypt(self, encrypted_data):
        """
        Decrypts data using the AES algorithm.
        It expects the IV to be prepended to the encrypted data.
        """
        iv = encrypted_data[:AES.block_size]  # Extract IV from the beginning
        ciphertext = encrypted_data[AES.block_size:]
        
        cipher = AES.new(self.aes_key, AES.MODE_CBC, iv)
        padded_data = cipher.decrypt(ciphertext)
        
        # Remove padding to get the original data
        return unpad(padded_data, AES.block_size)
    
    def process_image(self, input_filepath):
        """
        Main function to perform a full encryption and decryption cycle on an image.
        It uses both 3DES and AES, saves the results, and verifies data integrity.
        """
        print(f"Processing image: {input_filepath}")
        
        # Read original image data
        original_data = self.read_image(input_filepath)
        if original_data is None:
            return
        
        print(f"Original image size: {len(original_data)} bytes")
        
        # --- Triple DES Encryption/Decryption ---
        print("\n=== Triple DES Processing ===")
        des3_encrypted = self.triple_des_encrypt(original_data)
        print(f"Encrypted size (3DES): {len(des3_encrypted)} bytes")
        
        # Save encrypted data to a binary file
        self.save_bytes_to_file(des3_encrypted, "passport_3des_encrypted.bin")
        print("3DES encrypted data saved as: passport_3des_encrypted.bin")
        
        # Decrypt the data and save it as a new image file
        des3_decrypted = self.triple_des_decrypt(des3_encrypted)
        self.save_bytes_to_file(des3_decrypted, "passport_3des_decrypted.jpg")
        print("3DES decrypted image saved as: passport_3des_decrypted.jpg")
        
        # --- AES Encryption/Decryption ---
        print("\n=== AES Processing ===")
        aes_encrypted = self.aes_encrypt(original_data)
        print(f"Encrypted size (AES): {len(aes_encrypted)} bytes")
        
        # Save encrypted data to a binary file
        self.save_bytes_to_file(aes_encrypted, "passport_aes_encrypted.bin")
        print("AES encrypted data saved as: passport_aes_encrypted.bin")
        
        # Decrypt the data and save it as a new image file
        aes_decrypted = self.aes_decrypt(aes_encrypted)
        self.save_bytes_to_file(aes_decrypted, "passport_aes_decrypted.jpg")
        print("AES decrypted image saved as: passport_aes_decrypted.jpg")
        
        # --- Verification ---
        print("\n=== Verification ===")
        # Compare hashes of original and decrypted data to ensure they are identical
        print(f"Original data hash: {hashlib.sha256(original_data).hexdigest()[:16]}...")
        print(f"3DES decrypted hash: {hashlib.sha256(des3_decrypted).hexdigest()[:16]}...")
        print(f"AES decrypted hash:  {hashlib.sha256(aes_decrypted).hexdigest()[:16]}...")
        
        if original_data == des3_decrypted == aes_decrypted:
            print("✅ All decryptions successful - data integrity verified!")
        else:
            print("❌ Data integrity check failed!")
        
        # Compare the properties of the encrypted data from both algorithms
        comparison_results = self.compare_encrypted_data(des3_encrypted, aes_encrypted, original_data)
        
        # Create a visual representation of the encrypted data
        self.visual_comparison(des3_encrypted, aes_encrypted)
        
        return comparison_results
    
    def compare_encrypted_data(self, des3_encrypted, aes_encrypted, original_data):
        """
        Compares the encrypted outputs from 3DES and AES based on several metrics
        like size, overhead, entropy, and byte patterns.
        """
        print("\n=== Encrypted Data Comparison ===")
        
        # Size comparison
        print(f"Original data size:    {len(original_data):,} bytes")
        print(f"3DES encrypted size:   {len(des3_encrypted):,} bytes")
        print(f"AES encrypted size:    {len(aes_encrypted):,} bytes")
        
        # Calculate encryption overhead (due to padding and IV)
        des3_overhead = len(des3_encrypted) - len(original_data)
        aes_overhead = len(aes_encrypted) - len(original_data)
        print(f"3DES overhead:         {des3_overhead} bytes")
        print(f"AES overhead:          {aes_overhead} bytes")
        
        # Entropy analysis to measure randomness
        def calculate_entropy(data):
            """Calculates the Shannon entropy of a byte string."""
            import math
            # Count occurrences of each byte value (0-255)
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
        
        print(f"\nEntropy Analysis (higher value indicates more randomness):")
        print(f"Original image entropy: {original_entropy:.3f}")
        print(f"3DES encrypted entropy: {des3_entropy:.3f}")
        print(f"AES encrypted entropy:  {aes_entropy:.3f}")
        
        # Byte frequency analysis
        print(f"\nByte Distribution (number of unique byte values):")
        print(f"Original unique bytes:  {len(set(original_data))}/256")
        print(f"3DES unique bytes:      {len(set(des3_encrypted))}/256")
        print(f"AES unique bytes:       {len(set(aes_encrypted))}/256")
        
        # Correlation test by comparing the first few bytes
        sample_size = min(100, len(original_data))
        print(f"\nFirst {sample_size} bytes comparison (hex representation):")
        print(f"Original: {original_data[:sample_size].hex()[:50]}...")
        print(f"3DES:     {des3_encrypted[8:8+sample_size].hex()[:50]}...")  # Skip 8-byte IV
        print(f"AES:      {aes_encrypted[16:16+sample_size].hex()[:50]}...")  # Skip 16-byte IV
        
        # Check for repeating patterns in the data
        def has_patterns(data, pattern_length=4):
            """Checks for repeating patterns of a given length."""
            patterns = {}
            for i in range(len(data) - pattern_length + 1):
                pattern = data[i:i+pattern_length]
                patterns[pattern] = patterns.get(pattern, 0) + 1
            
            max_repeats = max(patterns.values()) if patterns else 0
            return max_repeats, len(patterns)
        
        _, original_patterns = has_patterns(original_data)
        _, des3_patterns = has_patterns(des3_encrypted)
        _, aes_patterns = has_patterns(aes_encrypted)
        
        print(f"\n4-byte Pattern Analysis (total unique patterns found):")
        print(f"Original patterns:      {original_patterns}")
        print(f"3DES patterns:          {des3_patterns}")
        print(f"AES patterns:           {aes_patterns}")
        
        return {
            'sizes': {'original': len(original_data), '3des': len(des3_encrypted), 'aes': len(aes_encrypted)},
            'entropy': {'original': original_entropy, '3des': des3_entropy, 'aes': aes_entropy},
            'patterns': {'original': original_patterns, '3des': des3_patterns, 'aes': aes_patterns}
        }
    
    def visual_comparison(self, des3_encrypted, aes_encrypted):
        """
        Creates a grayscale image from the raw bytes of the encrypted data.
        For good encryption, the output should look like random noise.
        """
        try:
            from PIL import Image
            import numpy as np
            
            print("\n=== Visual Comparison ===")
            
            # Use a sample of the data for visualization
            sample_size = min(10000, len(des3_encrypted), len(aes_encrypted))
            
            # Calculate dimensions for a square image
            side_length = int(sample_size ** 0.5)
            actual_size = side_length * side_length
            
            # Prepare byte data as numpy arrays
            des3_sample = np.array(list(des3_encrypted[:actual_size]), dtype=np.uint8)
            aes_sample = np.array(list(aes_encrypted[:actual_size]), dtype=np.uint8)
            
            # Reshape arrays into 2D squares
            des3_square = des3_sample.reshape(side_length, side_length)
            aes_square = aes_sample.reshape(side_length, side_length)
            
            # Create grayscale images from the arrays
            des3_img = Image.fromarray(des3_square, mode='L')
            aes_img = Image.fromarray(aes_square, mode='L')
            
            # Save the generated images
            des3_img.save("3des_encrypted_visualization.png")
            aes_img.save("aes_encrypted_visualization.png")
            
            print(f"Visual comparison images saved:")
            print(f"- 3des_encrypted_visualization.png ({side_length}x{side_length})")
            print(f"- aes_encrypted_visualization.png ({side_length}x{side_length})")
            print("These images should resemble random noise if encryption is effective.")
            
        except ImportError:
            print("PIL/numpy not installed, skipping visual comparison.")
    
    def display_keys(self):
        """Displays the generated 3DES and AES keys in hexadecimal format."""
        print("\n=== Encryption Keys (for demonstration) ===")
        print(f"3DES Key: {self.des3_key.hex()}")
        print(f"AES Key:  {self.aes_key.hex()}")

# Example usage for the ImageEncryption class
if __name__ == "__main__":
    # Initialize the encryption system
    encryptor = ImageEncryption()
    
    # Display the generated keys
    encryptor.display_keys()
    
    # Process a sample image (ensure 'passport.jpg' exists in the same directory)
    encryptor.process_image("passport.jpg")
    
    print("\n=== Summary of Created Files ===")
    print("- passport_3des_encrypted.bin (Image encrypted with Triple DES)")
    print("- passport_3des_decrypted.jpg (Decrypted image, should match original)")
    print("- passport_aes_encrypted.bin (Image encrypted with AES)")
    print("- passport_aes_decrypted.jpg (Decrypted image, should match original)")

def read_gz_csv(path: str) -> pd.DataFrame:
    """Reads a gzipped CSV file into a pandas DataFrame."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file: {path}")
    with gzip.open(path, 'rt') as f:
        return pd.read_csv(f)

def infer_device():
    """Checks if a GPU is available for TensorFlow and returns 'GPU' or 'CPU'."""
    try:
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            return "GPU"
        else:
            return "CPU"
    except Exception:
        return "CPU"

def make_sequences(chartevents: pd.DataFrame, max_len: int = 256) -> (np.ndarray, np.ndarray):
    """
    Converts a DataFrame of chart events into sequences of item IDs for each subject.
    
    Args:
        chartevents: DataFrame with 'subject_id', 'itemid', and 'charttime'.
        max_len: The maximum length for each sequence (padding/truncating).

    Returns:
        A tuple containing:
        - Padded sequences of item IDs (X).
        - Corresponding subject IDs (sids).
    """
    # Keep necessary columns and sort to create ordered sequences
    cols = [c for c in ["subject_id", "itemid", "charttime"] if c in chartevents.columns]
    df = chartevents[cols].copy()
    if "charttime" in df.columns:
        df["charttime"] = pd.to_datetime(df["charttime"], errors="coerce")
        df = df.sort_values(["subject_id", "charttime"])
    
    # Group events by subject to form sequences of item IDs
    seqs = df.groupby("subject_id")["itemid"].apply(list)
    
    # Create a vocabulary that maps each unique item ID to an integer
    all_itemids = [iid for seq in seqs for iid in seq]
    vocab = {iid: i+1 for i, iid in enumerate(sorted(set(all_itemids)))} # Start at 1 to reserve 0 for padding
    
    # Convert sequences of item IDs to sequences of integers
    seqs_idx = [np.array([vocab[iid] for iid in seq], dtype=np.int32) for seq in seqs]
    
    # Pad or truncate sequences to ensure they all have the same length (max_len)
    X = pad_sequences(seqs_idx, maxlen=max_len, padding="post", truncating="post", value=0)
    sids = seqs.index.values
    return X, sids

def pick_labels(diagnoses: pd.DataFrame, topk: int = 5) -> pd.DataFrame:
    """
    Selects a single diagnostic label for each subject from the diagnoses data.
    It uses the first 3 characters of the ICD code (category) and filters for the
    top K most frequent categories to create a balanced classification problem.
    """
    d = diagnoses.dropna(subset=["icd_code"]).copy()
    d["icd_cat"] = d["icd_code"].astype(str).str[:3] # Extract ICD category
    
    # Get the primary (first) diagnosis for each subject
    d = d.sort_values(["subject_id"]).drop_duplicates(subset=["subject_id"], keep="first")
    
    # Identify the top K most frequent diagnostic categories
    counts = d["icd_cat"].value_counts()
    keep = set(counts.head(topk).index)
    
    # Filter the DataFrame to include only subjects with these top categories
    d = d[d["icd_cat"].isin(keep)].copy()
    return d[["subject_id", "icd_cat"]]

def bootstrap_ci(y_true, y_prob, y_pred, average="macro", n_iter=1000, ci=95):
    """
    Calculates confidence intervals for accuracy, F1-score, and AUROC using bootstrapping.
    Bootstrapping involves resampling the test set with replacement and recalculating metrics
    to estimate the uncertainty of the model's performance.
    """
    rng = np.random.default_rng(42) # Random number generator for reproducibility
    n = len(y_true)
    accs, f1s, aucs = [], [], []
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    y_pred = np.array(y_pred)

    # Determine if it's a multi-class problem
    num_classes = y_prob.shape[1] if y_prob.ndim == 2 else 2
    
    # Perform bootstrapping iterations
    for _ in range(n_iter):
        # Sample with replacement from the test set indices
        idx = rng.choice(n, size=n, replace=True)
        yt = y_true[idx]
        yp = y_pred[idx]
        ypr = y_prob[idx]
        
        # Calculate metrics for the bootstrapped sample
        try:
            # Use one-vs-rest for multi-class AUROC
            auc = roc_auc_score(pd.get_dummies(yt), ypr, multi_class="ovr" if num_classes > 2 else "raise")
        except Exception:
            auc = np.nan # Handle cases where AUROC cannot be computed
        accs.append(accuracy_score(yt, yp))
        f1s.append(f1_score(yt, yp, average=average))
        aucs.append(auc)
        
    # Calculate the lower and upper bounds of the confidence interval
    lo = (100 - ci) / 2.0
    hi = 100 - lo
    ci_acc = (np.nanpercentile(accs, lo), np.nanpercentile(accs, hi))
    ci_f1  = (np.nanpercentile(f1s,  lo), np.nanpercentile(f1s,  hi))
    ci_auc = (np.nanpercentile(aucs, lo), np.nanpercentile(aucs, hi))
    return ci_acc, ci_f1, ci_auc

def main():
    """
    Main function to run the LSTM model training and evaluation pipeline.
    """
    # --- Argument Parsing ---
    # Sets up command-line arguments to configure the script
    ap = argparse.ArgumentParser(description="Train an LSTM model on MIMIC-IV data to predict diagnosis categories.")
    ap.add_argument("--admissions", default="admissions.csv.gz", help="Path to admissions.csv.gz")
    ap.add_argument("--diagnoses", default="diagnoses_icd.csv.gz", help="Path to diagnoses_icd.csv.gz")
    ap.add_argument("--chartevents", default="chartevents.csv.gz", help="Path to chartevents.csv.gz")
    ap.add_argument("--labevents", default="labevents.csv.gz", help="Path to labevents.csv.gz")
    ap.add_argument("--max_len", type=int, default=256, help="Maximum sequence length for LSTM.")
    ap.add_argument("--topk_labels", type=int, default=5, help="Number of top diagnosis categories to use as labels.")
    ap.add_argument("--epochs", type=int, default=5, help="Number of training epochs.")
    ap.add_argument("--batch_size", type=int, default=32, help="Training batch size.")
    ap.add_argument("--test_size", type=float, default=0.2, help="Proportion of data to use for testing.")
    ap.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    args = ap.parse_args()

    print(f"[INFO] Using {infer_device()} for training.")
    
    # --- Data Loading ---
    print("[INFO] Loading MIMIC-IV CSVs...")
    admissions = read_gz_csv(args.admissions)
    diagnoses  = read_gz_csv(args.diagnoses)
    chartevents = read_gz_csv(args.chartevents)

    # --- Label and Feature Engineering ---
    print("[INFO] Building labels from ICD categories...")
    labels_df = pick_labels(diagnoses, topk=args.topk_labels)
    
    # Filter chartevents to include only subjects with valid labels
    keep_subjects = set(labels_df["subject_id"].unique())
    chartevents = chartevents[chartevents["subject_id"].isin(keep_subjects)].copy()

    print("[INFO] Building sequences from chartevents...")
    X, sids = make_sequences(chartevents, max_len=args.max_len)

    # Align labels with the generated sequences (X)
    sid2label = dict(zip(labels_df["subject_id"], labels_df["icd_cat"]))
    y_labels = [sid2label.get(s, None) for s in sids]
    
    # Ensure that only subjects with both sequences and labels are kept
    mask = [lbl is not None for lbl in y_labels]
    X = X[mask]
    y_labels = [lbl for lbl in y_labels if lbl is not None]

    # Encode string labels into integers
    le = LabelEncoder()
    y = le.fit_transform(y_labels)
    num_classes = len(le.classes_)

    print(f"[INFO] Dataset created: {len(y)} subjects, {num_classes} classes -> {list(le.classes_)}")

    # --- Train-Test Split ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.seed, stratify=y
    )

    # --- Model Definition ---
    # The LSTM model is defined using Keras Sequential API
    vocab_size = int(X.max()) + 1  # Vocabulary size is the max item ID + 1 (for padding)
    model = Sequential([
        Embedding(input_dim=vocab_size, output_dim=128, mask_zero=True), # Embedding layer to learn vector representations of item IDs
        LSTM(256, return_sequences=True, dropout=0.3), # First LSTM layer
        LSTM(128, dropout=0.3), # Second LSTM layer
        Dense(num_classes, activation="softmax") # Output layer with softmax for multi-class classification
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss="sparse_categorical_crossentropy", # Use sparse CE for integer labels
                  metrics=["accuracy"])
    model.summary() # Print model architecture

    # --- Model Training ---
    print("[INFO] Training LSTM model...")
    model.fit(X_train, y_train,
              validation_data=(X_test, y_test),
              epochs=args.epochs, batch_size=args.batch_size, verbose=1)

    # --- Model Evaluation ---
    print("[INFO] Evaluating model on the test set...")
    y_prob = model.predict(X_test, verbose=0) # Get class probabilities
    y_pred = np.argmax(y_prob, axis=1) # Get predicted class

    # Calculate performance metrics
    acc = accuracy_score(y_test, y_pred)
    f1  = f1_score(y_test, y_pred, average="macro")
    try:
        auroc = roc_auc_score(pd.get_dummies(y_test), y_prob, multi_class="ovr" if num_classes > 2 else "raise")
    except Exception:
        auroc = np.nan

    # Calculate 95% confidence intervals for the metrics
    ci_acc, ci_f1, ci_auc = bootstrap_ci(y_test, y_prob, y_pred, average="macro", n_iter=1000, ci=95)

    # --- Print and Save Results ---
    print("\n=== LSTM Baseline Performance (MIMIC-IV Demo) ===")
    print(f"Accuracy:  {acc:.4f} (95% CI: {ci_acc[0]:.4f}–{ci_acc[1]:.4f})")
    print(f"Macro F1:  {f1:.4f} (95% CI: {ci_f1[0]:.4f}–{ci_f1[1]:.4f})")
    print(f"AUROC:     {auroc:.4f} (95% CI: {ci_auc[0]:.4f}–{ci_auc[1]:.4f})")
    print("\nClasses:", list(le.classes_))

    # Save metrics to a JSON file
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
    print("\nSaved evaluation metrics to -> results/lstm_demo_metrics.json")

# This block executes the main function when the script is run directly
if __name__ == "__main__":
    main()
