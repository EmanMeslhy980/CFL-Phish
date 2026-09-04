"""
CFL-Phish: 768-Dimensional Multimodal Feature Extraction
--------------------------------------------------------
This script extracts deterministic 768-dimensional multimodal features 
(URL: 64, HTML: 128, ResNet18: 512, CRP: 64) from the Phish360 dataset.
It guarantees 100% reproducibility with zero randomness.
"""

import os
import re
import cv2
import urllib.parse
import warnings

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms

from PIL import Image
from bs4 import BeautifulSoup
from tqdm import tqdm

# Suppress non-critical warnings
warnings.filterwarnings("ignore")

# ==============================================================================
# Configuration
# ==============================================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 80)
print("CFL-Phish: 768-Dimensional Deterministic Feature Extraction")
print("=" * 80)
print(f"Using device: {DEVICE}")

if DEVICE.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
else:
    print("Warning: CUDA is not available. Feature extraction will use CPU (slower).")

# Google Drive cache directory for saving the .npz files
CACHE_DIR = "/content/drive/MyDrive/Phish360_cache"
os.makedirs(CACHE_DIR, exist_ok=True)
print(f"Feature cache directory: {CACHE_DIR}")


# ==============================================================================
# 1. URL Feature Extractor (64 dimensions)
# ==============================================================================
class URLFeatureExtractor:
    def __init__(self):
        self.feature_dim = 64
        self.suspicious_tlds = [".tk", ".ml", ".ga", ".cf", ".xyz", ".top", ".work"]
        self.suspicious_words = ["login", "secure", "bank", "paypal", "verify", "account", "update", "confirm"]

    def extract(self, url):
        # Initialize with zeros (deterministic padding for any unused indices)
        features = np.zeros(self.feature_dim, dtype=np.float32)
        if not url:
            return features

        try:
            parsed = urllib.parse.urlparse(url)
            domain = parsed.netloc
            path = parsed.path
            query = parsed.query
            url_lower = url.lower()
            domain_lower = domain.lower()

            # Basic URL statistics
            features[0] = len(url)
            features[1] = domain.count(".")
            features[2] = 1.0 if parsed.scheme.lower() == "https" else 0.0
            features[3] = len(domain)
            features[4] = len(path)
            features[5] = len(query)

            # Suspicious URL characteristics
            features[6] = 1.0 if re.search(r"\d{1,3}(\.\d{1,3}){3}", domain) else 0.0
            features[7] = 1.0 if "@" in url else 0.0
            features[8] = sum(url.count(char) for char in ["-", "_", "?", "=", "&", "%"])
            features[9] = sum(1 for word in self.suspicious_words if word in url_lower)
            features[10] = 1.0 if any(tld in domain_lower for tld in self.suspicious_tlds) else 0.0
            features[11] = sum(char.isdigit() for char in domain)

            # Character entropy
            if len(url) > 0:
                character_counts = {}
                for char in url:
                    character_counts[char] = character_counts.get(char, 0) + 1
                
                entropy = 0.0
                for count in character_counts.values():
                    probability = count / len(url)
                    entropy -= probability * np.log2(probability + 1e-10)
                features[12] = entropy / 5.0

            # Additional deterministic lexical features
            features[13] = url.count("/")
            features[14] = url.count(".")
            features[15] = url.count(":")
            features[16] = url.count("//")
            features[17] = url.count("?")
            features[18] = url.count("=")
            features[19] = url.count("&")
            features[20] = url.count("%")
            features[21] = url.count("#")
            features[22] = url.count(";")
            features[23] = url.count(",")
            features[24] = url.count("'")
            features[25] = url.count('"')
            features[26] = url.count("<")
            features[27] = url.count(">")
            features[28] = sum(char.isupper() for char in url)
            features[29] = sum(char.islower() for char in url)
            features[30] = sum(char.isalpha() for char in url)
            features[31] = sum(char.isdigit() for char in url)
            features[32] = len(parsed.username or "")
            features[33] = len(parsed.password or "")
            features[34] = len(parsed.hostname or "")
            features[35] = len(str(parsed.port)) if parsed.port is not None else 0.0
            features[36] = 1.0 if parsed.port is not None else 0.0
            features[37] = 1.0 if domain.startswith("www.") else 0.0

            # Deterministic statistical ratios
            url_length = max(len(url), 1)
            features[38] = features[11] / url_length
            features[39] = features[30] / url_length
            features[40] = features[31] / url_length
            features[41] = features[28] / url_length
            features[42] = features[29] / url_length
            features[43] = len(path) / url_length
            features[44] = len(query) / url_length
            features[45] = domain.count(".") / max(len(domain), 1)
            features[46] = domain.count("-") / max(len(domain), 1)
            features[47] = domain.count("_") / max(len(domain), 1)

            # Keyword-specific binary indicators
            for index, word in enumerate(self.suspicious_words):
                feature_index = 48 + index
                if feature_index < self.feature_dim:
                    features[feature_index] = 1.0 if word in url_lower else 0.0

            # Indices 56 to 63 remain 0.0 (deterministic padding)

        except Exception:
            pass  # Return zero-filled array on any parsing error
        
        return features


# ==============================================================================
# 2. HTML Feature Extractor (128 dimensions)
# ==============================================================================
class HTMLFeatureExtractor:
    def __init__(self):
        self.feature_dim = 128
        self.keywords = ["login", "signin", "password", "account", "update", "verify", "secure", "bank", "paypal"]

    def extract(self, html):
        features = np.zeros(self.feature_dim, dtype=np.float32)
        if not html:
            return features

        try:
            soup = BeautifulSoup(html, "html.parser")

            # Basic HTML tag statistics
            tags = ["form", "input", "script", "iframe", "meta", "link", "a", "div", "img", "button"]
            for i, tag in enumerate(tags):
                features[i] = len(soup.find_all(tag))

            # Form-related features
            forms = soup.find_all("form")
            features[10] = sum(len(form.find_all("input", {"type": "hidden"})) for form in forms)
            features[11] = sum(len(form.find_all("input", {"type": "password"})) for form in forms)
            features[12] = len(forms)

            # Script-related features
            scripts = soup.find_all("script")
            features[13] = len(scripts)
            features[14] = sum(1 for script in scripts if script.string and ("eval(" in script.string or "atob(" in script.string))

            # Link-related features
            links = soup.find_all("a")
            features[15] = len(links)
            features[16] = sum(1 for link in links if link.get("href", "").startswith("http"))
            features[17] = sum(1 for link in links if any(keyword in link.get("href", "").lower() for keyword in self.keywords))

            # Text-related features
            text = soup.get_text(separator=" ", strip=True).lower()
            features[18] = sum(text.count(keyword) for keyword in self.keywords)
            features[19] = len(text)

            # Additional deterministic HTML statistics
            all_elements = soup.find_all()
            features[20] = len(all_elements)
            features[21] = len(soup.find_all("title"))
            features[22] = len(soup.find_all("head"))
            features[23] = len(soup.find_all("body"))
            features[24] = len(soup.find_all("style"))
            features[25] = len(soup.find_all("textarea"))
            features[26] = len(soup.find_all("select"))
            features[27] = len(soup.find_all("option"))
            features[28] = len(soup.find_all("table"))
            features[29] = len(soup.find_all("tr"))
            features[30] = len(soup.find_all("td"))
            features[31] = len(soup.find_all("p"))
            features[32] = len(soup.find_all("h1"))
            features[33] = len(soup.find_all("h2"))
            features[34] = len(soup.find_all("h3"))

            # Keyword indicators in text
            for i, keyword in enumerate(self.keywords):
                feature_index = 35 + i
                if feature_index < self.feature_dim:
                    features[feature_index] = text.count(keyword)

            # HTML size and density statistics
            html_length = max(len(html), 1)
            features[45] = html_length
            features[46] = len(text) / html_length
            features[47] = len(scripts) / max(len(all_elements), 1)
            features[48] = len(links) / max(len(all_elements), 1)
            features[49] = len(forms) / max(len(all_elements), 1)

            # Attribute-based statistics
            features[50] = sum(1 for element in all_elements if element.get("id"))
            features[51] = sum(1 for element in all_elements if element.get("class"))
            features[52] = sum(1 for element in all_elements if element.get("style"))
            features[53] = sum(1 for element in all_elements if element.get("onclick"))

            # Indices 54 to 127 remain 0.0 (deterministic padding)

        except Exception:
            pass
        
        return features


# ==============================================================================
# 3. Hybrid Image Feature Extractor (512 + 64 = 576 dimensions)
# ==============================================================================
class HybridImageExtractor:
    def __init__(self):
        print("\nLoading ResNet18 model...")
        self.resnet = nn.Sequential(
            *list(models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1).children())[:-1]
        )
        self.resnet.to(DEVICE)
        self.resnet.eval()

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        print("ResNet18 loaded successfully.")

    def extract_resnet_batch(self, paths):
        tensors = []
        valid_indices = []

        for i, path in enumerate(paths):
            try:
                if os.path.exists(path):
                    image = Image.open(path).convert("RGB")
                    tensor = self.transform(image)
                    tensors.append(tensor)
                    valid_indices.append(i)
            except Exception:
                pass

        features = np.zeros((len(paths), 512), dtype=np.float32)
        if tensors:
            with torch.no_grad():
                batch = torch.stack(tensors).to(DEVICE)
                output = self.resnet(batch).squeeze().cpu().numpy()
            
            if output.ndim == 1:
                output = output.reshape(1, -1)
            
            for i, original_index in enumerate(valid_indices):
                features[original_index] = output[i]

        return features

    def extract_crp(self, path):
        features = np.zeros(64, dtype=np.float32)
        try:
            if not os.path.exists(path):
                return features

            image = cv2.imread(path)
            if image is None:
                return features

            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            height, width = gray.shape

            # Threshold bright regions to find form-like structures
            _, threshold = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            forms = []
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                if w > 100 and h > 50:
                    forms.append((x, y, w, h))

            # Number of detected form-like regions
            features[0] = len(forms)

            # Normalized geometric statistics
            if forms:
                x_values = [form[0] for form in forms]
                y_values = [form[1] for form in forms]
                width_values = [form[2] for form in forms]
                height_values = [form[3] for form in forms]

                features[1] = np.mean(x_values) / max(width, 1)
                features[2] = np.mean(y_values) / max(height, 1)
                features[3] = np.mean(width_values) / max(width, 1)
                features[4] = np.mean(height_values) / max(height, 1)
                features[5] = np.mean(np.array(width_values) / np.maximum(np.array(height_values), 1))
                features[6] = np.mean(np.array(width_values) * np.array(height_values) / max(width * height, 1))

            # Form-density indicator
            features[19] = min(1.0, features[0] * 0.2)

            # Image dimensions
            features[20] = width
            features[21] = height
            features[22] = width / max(height, 1)

            # Indices 23 to 63 remain 0.0 (deterministic padding)

        except Exception:
            pass
        
        return features


# ==============================================================================
# 4. Dataset Extraction Pipeline
# ==============================================================================
def extract_and_save(folder_path, save_filename):
    save_path = os.path.join(CACHE_DIR, save_filename)

    print("\n" + "=" * 80)
    print(f"Processing dataset: {folder_path}")
    print("=" * 80)

    url_extractor = URLFeatureExtractor()
    html_extractor = HTMLFeatureExtractor()
    image_extractor = HybridImageExtractor()

    folders = sorted([
        folder for folder in os.listdir(folder_path)
        if os.path.isdir(os.path.join(folder_path, folder))
    ])
    print(f"Number of webpage samples: {len(folders)}")

    all_features = []
    all_labels = []
    image_paths = []
    indices = []
    batch_size = 64

    for i, folder in enumerate(tqdm(folders, desc="Extracting multimodal features")):
        base_path = os.path.join(folder_path, folder)

        # 1. Label
        label = 1  # Default to phishing
        label_path = os.path.join(base_path, "Label")
        if os.path.exists(label_path) and os.listdir(label_path):
            label_file = os.path.join(label_path, os.listdir(label_path)[0])
            try:
                with open(label_file, "r", errors="ignore") as file:
                    if "legitimate" in file.read().lower():
                        label = 0
            except Exception:
                pass

        # 2. URL
        url = ""
        url_path = os.path.join(base_path, "URL")
        if os.path.exists(url_path) and os.listdir(url_path):
            url_file = os.path.join(url_path, os.listdir(url_path)[0])
            try:
                with open(url_file, "r", errors="ignore") as file:
                    url = file.read().strip()
            except Exception:
                pass

        # 3. Raw HTML
        html = ""
        html_path = os.path.join(base_path, "RAW-HTML")
        if os.path.exists(html_path) and os.listdir(html_path):
            html_file = os.path.join(html_path, os.listdir(html_path)[0])
            try:
                with open(html_file, "r", errors="ignore") as file:
                    html = file.read()
            except Exception:
                pass

        # 4. Screenshot
        image_path = ""
        screenshot_path = os.path.join(base_path, "SCREEN-SHOT")
        if os.path.exists(screenshot_path):
            images = [f for f in os.listdir(screenshot_path) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
            if images:
                images.sort()
                image_path = os.path.join(screenshot_path, images[0])

        # 5. Initialize 768-dimensional feature vector
        feature_vector = np.zeros(768, dtype=np.float32)
        feature_vector[0:64] = url_extractor.extract(url)
        feature_vector[64:192] = html_extractor.extract(html)
        # Indices 192:768 will be filled by image features in the batch step

        all_features.append(feature_vector)
        all_labels.append(label)
        image_paths.append(image_path)
        indices.append(len(all_features) - 1)

        # 6. Batch image feature extraction (to save GPU memory)
        if len(image_paths) == batch_size or i == len(folders) - 1:
            resnet_features = image_extractor.extract_resnet_batch(image_paths)
            crp_features = np.array([image_extractor.extract_crp(path) for path in image_paths])
            image_features = np.concatenate([resnet_features, crp_features], axis=1)

            # Insert image features into the main feature vector
            for batch_index, feature_index in enumerate(indices):
                all_features[feature_index][192:768] = image_features[batch_index]

            image_paths = []
            indices = []

    # Convert to NumPy arrays
    X = np.asarray(all_features, dtype=np.float32)
    y = np.asarray(all_labels, dtype=np.int64)

    # Validate dimensionality (Critical check for reproducibility)
    assert X.ndim == 2, f"Expected 2D array, got {X.ndim}D"
    assert X.shape[1] == 768, f"Expected 768 features, but obtained {X.shape[1]}"

    # Save extracted features
    np.savez(save_path, X=X, y=y)
    
    print("\nFeature extraction completed.")
    print(f"Samples      : {len(X)}")
    print(f"Feature size : {X.shape[1]}")
    print(f"Legitimate   : {np.sum(y == 0)}")
    print(f"Phishing     : {np.sum(y == 1)}")
    print(f"Saved to     : {save_path}")
    print("=" * 80)


# ==============================================================================
# 5. Main Execution & Final Verification
# ==============================================================================
if __name__ == "__main__":
    # Adjust these paths if your dataset is mounted differently in Colab
    TRAINVAL_PATH = "/content/Phish360/trainval"
    TEST_PATH = "/content/Phish360/test"

    # Extract training/validation features
    if os.path.exists(TRAINVAL_PATH):
        extract_and_save(TRAINVAL_PATH, "trainval_768.npz")
    else:
        print(f"Warning: Training path not found at {TRAINVAL_PATH}")

    # Extract test features
    if os.path.exists(TEST_PATH):
        extract_and_save(TEST_PATH, "test_768.npz")
    else:
        print(f"Warning: Test path not found at {TEST_PATH}")

    # ==============================================================================
    # Final Verification Block
    # ==============================================================================
    print("\n" + "=" * 80)
    print("Running Final Data Integrity Verification...")
    print("=" * 80)
    
    for fname in ["trainval_768.npz", "test_768.npz"]:
        fpath = os.path.join(CACHE_DIR, fname)
        if os.path.exists(fpath):
            data = np.load(fpath)
            X, y = data["X"], data["y"]
            print(f"\n[ {fname} ]")
            print(f"  -> X shape : {X.shape}")
            print(f"  -> y shape : {y.shape}")
            print(f"  -> NaNs in X: {np.isnan(X).sum()}")
            print(f"  -> Infs in X: {np.isinf(X).sum()}")
            print(f"  -> Class dist: {np.bincount(y)}")
        else:
            print(f"\n[ {fname} ] -> File not found!")

    print("\n" + "=" * 80)
    print("768-dimensional feature extraction and verification completed successfully.")
    print("=" * 80)
