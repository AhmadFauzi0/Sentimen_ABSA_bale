"""
==============================================================================
STEP 4: FINE-TUNING IndoBERT untuk ABSA
Title: Transforming User Feedback into Strategic Intelligence:
       An ABSA of balé by BTN Superapp using IndoBERT and E-S-QUAL Framework
==============================================================================
Tujuan  : Fine-tuning IndoBERT/IndoBERTweet untuk dua tugas:
          (a) Task 1 – Aspect Detection   : multi-label classification
          (b) Task 2 – Sentiment per Aspek: 3-class per aspek (Pos/Net/Neg)

Strategi: Joint training menggunakan shared encoder + dual classification head

Input   : data/labeled_reviews.csv  (sudah divalidasi pakar)
Output  :
  - models/aspect_detector/        (model deteksi aspek)
  - models/sentiment_classifier/   (model klasifikasi sentimen)
  - results/training_metrics.json

Dependensi:
    pip install transformers datasets torch scikit-learn pandas numpy tqdm
    (GPU sangat disarankan: NVIDIA dengan VRAM ≥ 8GB)
==============================================================================
"""

import json
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW

from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoConfig,
    get_linear_schedule_with_warmup,
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    f1_score,
    accuracy_score,
    confusion_matrix,
)
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("logs/step4_training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# KONFIGURASI TRAINING
# ─────────────────────────────────────────────
@dataclass
class TrainingConfig:
    # Model
    # model_name: str = "indolem/indobert-base-uncased"
    model_name: "indolem/indobertweet-base-uncased"
    # Alternatif: "w11wo/sundanese-roberta-base-posp-tagger" untuk dialek
    # Alternatif: "cahya/bert-base-indonesian-1.5G"

    # Path
    input_path: str = "data/labeled_reviews.csv"
    model_dir: str  = "models/"
    result_dir: str = "results/"

    # Aspek dan label
    aspects: List[str] = field(default_factory=lambda: [
        "EFFICIENCY", "SYSTEM_AVAILABILITY", "FULFILLMENT", "PRIVACY"
    ])
    sentiments: List[str] = field(default_factory=lambda: [
        "Positif", "Netral", "Negatif"
    ])
    sentiment2id: Dict[str, int] = field(default_factory=lambda: {
        "Positif": 0, "Netral": 1, "Negatif": 2
    })

    # Hyperparameter
    max_length: int     = 128       # Max token sequence length
    batch_size: int     = 32        # Kurangi jika OOM GPU
    learning_rate: float = 2e-5    # LR optimal untuk fine-tuning BERT
    num_epochs: int     = 5
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    dropout_rate: float = 0.3
    gradient_clip: float = 1.0

    # Split
    train_ratio: float  = 0.70
    val_ratio: float    = 0.15
    test_ratio: float   = 0.15
    random_seed: int    = 42

    # Class weights untuk handle imbalance
    use_class_weights: bool = True


CFG = TrainingConfig()


# ─────────────────────────────────────────────
# DATASET CLASS
# ─────────────────────────────────────────────
class ABSADataset(Dataset):
    """
    Dataset untuk ABSA multi-task:
    - Input  : teks ulasan (+ opsional aspect prompt)
    - Output : (aspect_labels[multi-label], sentiment_label[per-aspek])

    Menggunakan strategi "Aspect-Aware Encoding":
    Teks dikonkatenasi dengan nama aspek untuk konteks yang lebih kaya.
    Format: "[CLS] teks_ulasan [SEP] nama_aspek [SEP]"
    """

    def __init__(
        self,
        texts: List[str],
        aspect_labels: np.ndarray,    # (N, n_aspects) binary multi-label
        sentiment_labels: np.ndarray, # (N, n_aspects) 0=Pos, 1=Net, 2=Neg, -1=N/A
        tokenizer,
        max_length: int = CFG.max_length,
        aspects: List[str] = CFG.aspects,
    ):
        self.texts             = texts
        self.aspect_labels     = aspect_labels
        self.sentiment_labels  = sentiment_labels
        self.tokenizer         = tokenizer
        self.max_length        = max_length
        self.aspects           = aspects

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]

        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "input_ids"        : encoding["input_ids"].squeeze(0),
            "attention_mask"   : encoding["attention_mask"].squeeze(0),
            "token_type_ids"   : encoding.get("token_type_ids", torch.zeros(self.max_length, dtype=torch.long)).squeeze(0),
            "aspect_labels"    : torch.tensor(self.aspect_labels[idx], dtype=torch.float),
            "sentiment_labels" : torch.tensor(self.sentiment_labels[idx], dtype=torch.long),
        }


# ─────────────────────────────────────────────
# ARSITEKTUR MODEL ABSA
# ─────────────────────────────────────────────
class IndoBERTABSA(nn.Module):
    """
    Arsitektur joint model untuk ABSA:

    [IndoBERT Encoder]
           │
    ┌──────┴───────┐
    │              │
    ▼              ▼
[Aspect        [Sentiment
 Detector]      Classifier]
(Multi-label   (N_aspects ×
 binary)        3-class)

    Menggunakan CLS token sebagai representasi kalimat.
    """

    def __init__(
        self,
        model_name: str  = CFG.model_name,
        n_aspects: int   = len(CFG.aspects),
        n_sentiments: int = len(CFG.sentiments),
        dropout: float   = CFG.dropout_rate,
    ):
        super().__init__()
        self.n_aspects    = n_aspects
        self.n_sentiments = n_sentiments

        # Encoder backbone
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size  = self.encoder.config.hidden_size

        # Dropout layer
        self.dropout = nn.Dropout(dropout)

        # Task 1: Aspect Detection (multi-label binary classification)
        self.aspect_classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, n_aspects),
            # Sigmoid diterapkan di loss (BCEWithLogitsLoss)
        )

        # Task 2: Sentiment per Aspek (N_aspects × 3-class softmax)
        # Masing-masing aspek memiliki head klasifikasi sendiri
        self.sentiment_classifiers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 4),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size // 4, n_sentiments),
            )
            for _ in range(n_aspects)
        ])

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        Returns:
            aspect_logits    : (batch, n_aspects)        raw logits
            sentiment_logits : (batch, n_aspects, n_sent) raw logits
        """
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        cls_output = self.dropout(outputs.last_hidden_state[:, 0, :])  # [CLS] token

        # Task 1
        aspect_logits = self.aspect_classifier(cls_output)

        # Task 2: stack semua head
        sentiment_logits = torch.stack(
            [head(cls_output) for head in self.sentiment_classifiers],
            dim=1
        )  # (batch, n_aspects, n_sentiments)

        return aspect_logits, sentiment_logits


# ─────────────────────────────────────────────
# LOSS FUNCTION
# ─────────────────────────────────────────────
class ABSALoss(nn.Module):
    """
    Combined loss:
    L_total = α * L_aspect + β * L_sentiment

    L_aspect    : BCEWithLogitsLoss (multi-label)
    L_sentiment : CrossEntropyLoss per aspek (hanya aspek yang relevan)
    """

    def __init__(
        self,
        aspect_weight: float       = 1.0,
        sentiment_weight: float    = 1.0,
        class_weights: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.aspect_w    = aspect_weight
        self.sentiment_w = sentiment_weight

        self.aspect_loss = nn.BCEWithLogitsLoss()
        self.sent_loss   = nn.CrossEntropyLoss(
            weight=class_weights,
            ignore_index=-1  # -1 = aspek tidak relevan, diabaikan
        )

    def forward(
        self,
        aspect_logits: torch.Tensor,
        sentiment_logits: torch.Tensor,
        aspect_targets: torch.Tensor,
        sentiment_targets: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            total_loss, aspect_loss, sentiment_loss
        """
        l_aspect = self.aspect_loss(aspect_logits, aspect_targets)

        # Hitung sentiment loss per aspek
        batch, n_asp, n_sent = sentiment_logits.shape
        l_sentiment = self.sent_loss(
            sentiment_logits.view(batch * n_asp, n_sent),
            sentiment_targets.view(batch * n_asp),
        )

        total = self.aspect_w * l_aspect + self.sentiment_w * l_sentiment
        return total, l_aspect, l_sentiment


# ─────────────────────────────────────────────
# PERSIAPAN DATA
# ─────────────────────────────────────────────
def prepare_labels(df: pd.DataFrame, cfg: TrainingConfig = CFG):
    """
    Mengkonversi kolom 'aspects' dan 'sentiment_rule' menjadi
    matriks numerik yang siap digunakan model.

    Returns:
        aspect_matrix    : (N, n_aspects) binary
        sentiment_matrix : (N, n_aspects) 0/1/2/-1
    """
    n     = len(df)
    n_asp = len(cfg.aspects)

    aspect_matrix    = np.zeros((n, n_asp), dtype=np.float32)
    sentiment_matrix = np.full((n, n_asp), -1, dtype=np.int64)  # -1 = N/A

    for i, row in df.iterrows():
        row_aspects   = str(row.get("aspects", "GENERAL")).split("|")
        row_sentiment = cfg.sentiment2id.get(
            str(row.get("sentiment_rule", "Netral")), 1
        )

        for j, asp in enumerate(cfg.aspects):
            if asp in row_aspects:
                aspect_matrix[i, j]    = 1.0
                sentiment_matrix[i, j] = row_sentiment

    return aspect_matrix, sentiment_matrix


# ─────────────────────────────────────────────
# EVALUASI
# ─────────────────────────────────────────────
def evaluate_model(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    cfg: TrainingConfig = CFG,
) -> Dict:
    """Evaluasi lengkap: F1-score per aspek, confusion matrix."""
    model.eval()
    all_aspect_preds, all_aspect_targets   = [], []
    all_sent_preds, all_sent_targets       = [], []

    with torch.no_grad():
        for batch in data_loader:
            input_ids  = batch["input_ids"].to(device)
            att_mask   = batch["attention_mask"].to(device)
            tok_type   = batch["token_type_ids"].to(device)
            asp_labels = batch["aspect_labels"].cpu().numpy()
            snt_labels = batch["sentiment_labels"].cpu().numpy()

            asp_logits, snt_logits = model(input_ids, att_mask, tok_type)

            asp_preds = (torch.sigmoid(asp_logits) > 0.5).cpu().numpy().astype(int)
            snt_preds = torch.argmax(snt_logits, dim=-1).cpu().numpy()

            all_aspect_preds.append(asp_preds)
            all_aspect_targets.append(asp_labels.astype(int))
            all_sent_preds.append(snt_preds)
            all_sent_targets.append(snt_labels)

    all_ap = np.vstack(all_aspect_preds)
    all_at = np.vstack(all_aspect_targets)
    all_sp = np.vstack(all_sent_preds)
    all_st = np.vstack(all_sent_targets)

    # Aspect F1 (macro)
    aspect_f1 = f1_score(all_at, all_ap, average="macro", zero_division=0)

    # Sentiment F1 per aspek (ignore -1)
    sent_f1_per_aspect = {}
    for j, asp in enumerate(cfg.aspects):
        mask   = all_st[:, j] != -1
        if mask.sum() > 0:
            f1 = f1_score(
                all_st[mask, j], all_sp[mask, j],
                average="macro", zero_division=0
            )
            sent_f1_per_aspect[asp] = round(f1, 4)

    metrics = {
        "aspect_f1_macro"     : round(aspect_f1, 4),
        "sentiment_f1_per_aspect": sent_f1_per_aspect,
        "sentiment_f1_avg"    : round(np.mean(list(sent_f1_per_aspect.values())), 4),
    }
    return metrics


# ─────────────────────────────────────────────
# TRAINING LOOP
# ─────────────────────────────────────────────
def train_model(cfg: TrainingConfig = CFG) -> None:
    """Pipeline training lengkap."""
    Path(cfg.model_dir).mkdir(parents=True, exist_ok=True)
    Path(cfg.result_dir).mkdir(parents=True, exist_ok=True)

    # 1. Deteksi device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Training device: {device}")
    if device.type == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")

    # 2. Load data
    logger.info(f"Membaca dataset dari: {cfg.input_path}")
    df = pd.read_csv(cfg.input_path, encoding="utf-8-sig")
    df = df.dropna(subset=["clean_text"]).reset_index(drop=True)
    logger.info(f"Total data: {len(df):,}")

    # 3. Persiapan label
    aspect_matrix, sentiment_matrix = prepare_labels(df, cfg)

    # 4. Split data
    texts = df["clean_text"].tolist()
    (
        X_tr, X_tmp, ya_tr, ya_tmp, ys_tr, ys_tmp
    ) = train_test_split(
        texts, aspect_matrix, sentiment_matrix,
        test_size=(1 - cfg.train_ratio),
        random_state=cfg.random_seed
    )
    (
        X_val, X_test, ya_val, ya_test, ys_val, ys_test
    ) = train_test_split(
        X_tmp, ya_tmp, ys_tmp,
        test_size=cfg.test_ratio / (cfg.val_ratio + cfg.test_ratio),
        random_state=cfg.random_seed
    )

    logger.info(
        f"Split — Train: {len(X_tr):,} | Val: {len(X_val):,} | Test: {len(X_test):,}"
    )

    # 5. Tokenizer
    logger.info(f"Memuat tokenizer: {cfg.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)

    # 6. Dataset & DataLoader
    train_ds = ABSADataset(X_tr,   ya_tr,   ys_tr,   tokenizer, cfg.max_length)
    val_ds   = ABSADataset(X_val,  ya_val,  ys_val,  tokenizer, cfg.max_length)
    test_ds  = ABSADataset(X_test, ya_test, ys_test, tokenizer, cfg.max_length)

    train_dl = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,  num_workers=2, pin_memory=True)
    val_dl   = DataLoader(val_ds,   batch_size=cfg.batch_size, shuffle=False, num_workers=2, pin_memory=True)
    test_dl  = DataLoader(test_ds,  batch_size=cfg.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    # 7. Model
    logger.info(f"Memuat model: {cfg.model_name}")
    model = IndoBERTABSA(model_name=cfg.model_name).to(device)

    # 8. Class weights untuk mengatasi imbalance
    class_weights = None
    if cfg.use_class_weights:
        sent_flat = sentiment_matrix[sentiment_matrix != -1]
        counts    = np.bincount(sent_flat, minlength=len(cfg.sentiments)).astype(float)
        weights   = 1.0 / (counts + 1e-9)
        weights   /= weights.sum()
        class_weights = torch.tensor(weights, dtype=torch.float).to(device)
        logger.info(f"Class weights sentimen: {class_weights.cpu().numpy()}")

    # 9. Loss, Optimizer, Scheduler
    criterion = ABSALoss(class_weights=class_weights)
    optimizer = AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )
    total_steps = len(train_dl) * cfg.num_epochs
    warmup_steps = int(total_steps * cfg.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    # 10. Training loop
    best_val_f1    = 0.0
    training_log   = []

    for epoch in range(1, cfg.num_epochs + 1):
        model.train()
        total_loss, n_batches = 0.0, 0

        pbar = tqdm(train_dl, desc=f"Epoch {epoch}/{cfg.num_epochs}")
        for batch in pbar:
            input_ids  = batch["input_ids"].to(device)
            att_mask   = batch["attention_mask"].to(device)
            tok_type   = batch["token_type_ids"].to(device)
            asp_labels = batch["aspect_labels"].to(device)
            snt_labels = batch["sentiment_labels"].to(device)

            optimizer.zero_grad()
            asp_logits, snt_logits = model(input_ids, att_mask, tok_type)
            loss, l_asp, l_snt = criterion(asp_logits, snt_logits, asp_labels, snt_labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.gradient_clip)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            n_batches  += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}", asp=f"{l_asp.item():.4f}", snt=f"{l_snt.item():.4f}")

        avg_loss    = total_loss / n_batches
        val_metrics = evaluate_model(model, val_dl, device, cfg)

        logger.info(
            f"Epoch {epoch} | Loss: {avg_loss:.4f} | "
            f"Val Aspect F1: {val_metrics['aspect_f1_macro']:.4f} | "
            f"Val Sent F1 Avg: {val_metrics['sentiment_f1_avg']:.4f}"
        )

        training_log.append({
            "epoch": epoch,
            "train_loss": avg_loss,
            **{f"val_{k}": v for k, v in val_metrics.items() if not isinstance(v, dict)},
        })

        # Simpan model terbaik
        combined_f1 = (val_metrics["aspect_f1_macro"] + val_metrics["sentiment_f1_avg"]) / 2
        if combined_f1 > best_val_f1:
            best_val_f1 = combined_f1
            save_path   = Path(cfg.model_dir) / "best_model.pt"
            torch.save({
                "epoch"       : epoch,
                "model_state" : model.state_dict(),
                "optimizer"   : optimizer.state_dict(),
                "val_f1"      : combined_f1,
                "config"      : vars(cfg),
            }, save_path)
            logger.info(f"  → Model terbaik disimpan (F1={combined_f1:.4f})")
            tokenizer.save_pretrained(str(Path(cfg.model_dir) / "tokenizer"))

    # 11. Evaluasi akhir pada test set
    logger.info("\n" + "="*50)
    logger.info("EVALUASI FINAL PADA TEST SET")
    best_checkpoint = torch.load(Path(cfg.model_dir) / "best_model.pt", map_location=device)
    model.load_state_dict(best_checkpoint["model_state"])
    test_metrics = evaluate_model(model, test_dl, device, cfg)
    logger.info(f"Test Aspect F1 (macro): {test_metrics['aspect_f1_macro']:.4f}")
    logger.info(f"Test Sentiment F1 (avg): {test_metrics['sentiment_f1_avg']:.4f}")
    for asp, f1 in test_metrics["sentiment_f1_per_aspect"].items():
        logger.info(f"  {asp:<25}: F1 = {f1:.4f}")

    # 12. Simpan hasil
    results = {
        "training_log"  : training_log,
        "test_metrics"  : test_metrics,
        "config"        : vars(cfg),
    }
    result_path = Path(cfg.result_dir) / "training_metrics.json"
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info(f"Metrik training disimpan ke: {result_path}")


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import os
    os.makedirs("logs", exist_ok=True)

    train_model(CFG)

    print("\n" + "="*60)
    print("✅ STEP 4 SELESAI: Fine-tuning IndoBERT ABSA")
    print(f"   Model terbaik : models/best_model.pt")
    print(f"   Metrik hasil  : results/training_metrics.json")
    print("="*60)
