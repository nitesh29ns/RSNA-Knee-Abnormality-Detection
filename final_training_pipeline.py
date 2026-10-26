
# ============================================================
# DDP TRAINING PIPELINE
# 2 x GPU
#
# Configuration:
#   - Iterative multilabel train/validation split

#   - Full DenseNet fine-tuning
#   - 224 x 224
#   - 16 slices
#   - Batch size 1 / GPU
#   - Gradient accumulation 4
#   - AMP FP16
#   - Resume training
#   - Best model based on mean validation AUROC
# ============================================================


# model

import datetime

import glob
import os
import time
import subprocess
import random

import numpy as np
import torch
import torch.distributed as dist

from tqdm.auto import tqdm

from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from skmultilearn.model_selection import iterative_train_test_split
from sklearn.metrics import roc_auc_score, f1_score, average_precision_score


import gc
gc.collect()
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()

# ============================================================
# preprocessing
# ============================================================

import os
import pandas as pd
import cv2
import pydicom
import SimpleITK as sitk

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter1d, sobel
import matplotlib.pyplot as plt
from torch.utils.checkpoint import checkpoint





# ============================================================
# CONFIG
# ============================================================

N_GPUS = 2


NUM_EPOCHS = 50 # 50

BATCH_SIZE = 24 # 16 # 8

GRADIENT_ACCUMULATION_STEPS = 2 # 2  # 4

IMAGE_SIZE = (128, 128)

NUM_SLICES = 32

# Recommended for frozen RadImageNet
DENSE_CHUNK_SIZE = 64 # 32 # 32 # 16

# for redimagenet weights when trainable
# DENSE_CHUNK_SIZE = 8

NUM_WORKERS = 2

WEIGHT_DECAY = 1e-4

GRADIENT_CLIP = 1.0

USE_AMP = True

AMP_DTYPE = torch.float16

GPU_LOG_INTERVAL = 20

RANDOM_SEED = 42

FREEZE_BACKBONE = False

FREEZE_CHANNEL_ADAPTER = True

# ============================================================
# TRAINING CONTROL
# ============================================================

VAL_SPLIT = 0.20

LR_SCHEDULER_FACTOR = 0.5
LR_SCHEDULER_PATIENCE = 2
LR_SCHEDULER_MIN_LR = 1e-7

EARLY_STOPPING_PATIENCE = 8 # 5
EARLY_STOPPING_MIN_DELTA = 1e-4



# ============================================================
# LABELS
# ============================================================

LABEL_COLUMNS = [
    "ACL",
    "MCL",
    "Medial Meniscus",
    "Lateral Meniscus",
    "Medial OA",
    "Lateral OA",
    "PF OA",
    "Effusion",
    "Synovitis",
    "Baker's",
    "Contusion",
    "Fracture",
]


# ============================================================
# RADIMAGENET
# ============================================================

RADIMAGENET_PATH = "/kaggle/working/models/RadImageNet/DenseNet121.pt"



# ============================================================
# CHECKPOINT DIRECTORY
# ============================================================

CHECKPOINT_DIR = (
    "/kaggle/working/32_slices_redimagenet_checkpoints/"
)

SAVE_VAL_GRAPH_DIR = "/kaggle/working/32_slices_redimagenet_checkpoints/val_graphs"




def build_dataset(
    train_df: pd.DataFrame,
    cache_dirs
):
    """
    Build a lightweight study-level dataset index.

    The actual image tensors remain on disk and are loaded lazily
    by the PyTorch Dataset.__getitem__().

    Each returned entry contains:

        {
            "study_id": str,
            "pt_path": str,
            "label_source": "gold" or "llm",
            "labels": {
                "ACL": 0,
                ...
            }
        }

    The .pt file is loaded once here only to read metadata
    (label_source and labels). The tensors are NOT retained.
    """

    LABEL_COLUMNS = [
        "ACL",
        "MCL",
        "Medial Meniscus",
        "Lateral Meniscus",
        "Medial OA",
        "Lateral OA",
        "PF OA",
        "Effusion",
        "Synovitis",
        "Baker's",
        "Contusion",
        "Fracture",
    ]

    # =========================================================
    # 1. Find all preprocessed .pt files
    # =========================================================

    pt_map = {}

    for cache_dir in cache_dirs:

        if not os.path.isdir(cache_dir):
            print(
                f"WARNING: Cache directory not found: "
                f"{cache_dir}"
            )
            continue

        pt_files = glob.glob(
            os.path.join(cache_dir, "*.pt")
        )

        print(
            f"Found {len(pt_files)} .pt files in: "
            f"{cache_dir}"
        )

        for pt_path in pt_files:

            study_id = os.path.splitext(
                os.path.basename(pt_path)
            )[0]

            # Keep first file if duplicate StudyInstanceUID
            if study_id not in pt_map:
                pt_map[study_id] = pt_path

    print(
        f"\nTotal unique preprocessed .pt files: "
        f"{len(pt_map)}"
    )

    # =========================================================
    # 2. Build dataset index + metadata
    # =========================================================

    dataset = []
    missing_pt = []

    gold_count = 0
    llm_count = 0
    invalid_source = 0

    # Optional: prevent accidental duplicate studies
    seen_studies = set()

    for _, study_row in train_df.iterrows():

        study_uid = str(
            study_row["StudyInstanceUID"]
        )

        # ---------------------------------------------
        # Skip duplicate StudyInstanceUID
        # ---------------------------------------------

        if study_uid in seen_studies:
            continue

        seen_studies.add(study_uid)

        # ---------------------------------------------
        # Find preprocessed .pt
        # ---------------------------------------------

        pt_path = pt_map.get(study_uid)

        if pt_path is None:

            missing_pt.append(study_uid)
            continue

        # ---------------------------------------------
        # Load ONLY metadata from .pt
        # ---------------------------------------------

        try:

            pt_data = torch.load(
                pt_path,
                map_location="cpu"
            )

        except Exception as e:

            print(
                f"WARNING: Could not load metadata from "
                f"{pt_path}: {e}"
            )

            continue

        # ---------------------------------------------
        # Get label source
        # ---------------------------------------------

        label_source = pt_data.get(
            "label_source",
            None
        )

        if label_source not in {"gold", "llm"}:

            invalid_source += 1

            print(
                f"WARNING: Invalid label_source for "
                f"{study_uid}: {label_source}"
            )

            continue

        # ---------------------------------------------
        # Get labels
        # ---------------------------------------------

        labels = pt_data.get(
            "labels",
            None
        )

        if labels is None:

            print(
                f"WARNING: No labels found in "
                f"{pt_path}"
            )

            continue

        # ---------------------------------------------
        # Convert Tensor / numpy / dict to dictionary
        # ---------------------------------------------

        if isinstance(labels, torch.Tensor):

            labels = labels.detach().cpu().numpy()

            labels = labels.reshape(-1)

            if len(labels) != len(LABEL_COLUMNS):

                raise ValueError(
                    f"\nLabel count mismatch for "
                    f"{study_uid}\n"
                    f"Expected: {len(LABEL_COLUMNS)}\n"
                    f"Found: {len(labels)}"
                )

            labels = {
                label: float(value)
                for label, value in zip(
                    LABEL_COLUMNS,
                    labels
                )
            }

        elif isinstance(labels, dict):

            # Make a clean copy
            labels = {
                label: float(labels[label])
                for label in LABEL_COLUMNS
            }

        else:

            raise TypeError(
                f"\nUnsupported labels type for "
                f"{study_uid}: {type(labels)}"
            )

        # ---------------------------------------------
        # Count source
        # ---------------------------------------------

        if label_source == "gold":
            gold_count += 1

        elif label_source == "llm":
            llm_count += 1

        # ---------------------------------------------
        # Add lightweight dataset entry
        #
        # IMPORTANT:
        # Do NOT add volumes, mask, etc.
        # ---------------------------------------------

        dataset.append({
            "study_id": study_uid,
            "pt_path": pt_path,
            "label_source": label_source,
            "labels": labels,
        })

        # ---------------------------------------------
        # Free loaded .pt metadata
        # ---------------------------------------------

        del pt_data

    # =========================================================
    # 3. Dataset summary
    # =========================================================

    print("\n========================================")
    print("DATASET BUILDER SUMMARY")
    print("========================================")

    print(
        f"Studies in train_df : {len(train_df)}"
    )

    print(
        f"PT files found      : {len(pt_map)}"
    )

    print(
        f"Dataset entries     : {len(dataset)}"
    )

    print(
        f"Missing PT files    : {len(missing_pt)}"
    )

    print(
        f"Gold samples        : {gold_count}"
    )

    print(
        f"LLM samples         : {llm_count}"
    )

    print(
        f"Invalid label source: {invalid_source}"
    )

    # =========================================================
    # 4. Verify expected composition
    # =========================================================

    if gold_count != 58:

        print(
            f"\nWARNING: Expected 58 gold samples, "
            f"but found {gold_count}."
        )

    if gold_count + llm_count != len(dataset):

        print(
            "\nWARNING: Gold + LLM count does not "
            "match dataset size."
        )

    # =========================================================
    # 5. Missing files
    # =========================================================

    if missing_pt:

        print(
            f"\nWARNING: {len(missing_pt)} studies "
            f"do not have a corresponding .pt file."
        )

        print(
            "First 10 missing StudyInstanceUIDs:"
        )

        for study_id in missing_pt[:10]:
            print(study_id)

    # =========================================================
    # 6. Final structure validation
    # =========================================================

    required_keys = {
        "study_id",
        "pt_path",
        "label_source",
        "labels",
    }

    for i, item in enumerate(dataset):

        missing_keys = (
            required_keys - set(item.keys())
        )

        if missing_keys:

            raise RuntimeError(
                f"\nDataset entry {i} is missing: "
                f"{missing_keys}\n"
                f"Available keys: {item.keys()}"
            )

        missing_labels = [
            label
            for label in LABEL_COLUMNS
            if label not in item["labels"]
        ]

        if missing_labels:

            raise RuntimeError(
                f"\nDataset entry {i} is missing labels: "
                f"{missing_labels}"
            )

    # =========================================================
    # 7. Show first sample
    # =========================================================

    if dataset:

        print("\nFirst dataset entry:")

        print(
            {
                "study_id": dataset[0]["study_id"],
                "pt_path": dataset[0]["pt_path"],
                "label_source": dataset[0]["label_source"],
                "labels": dataset[0]["labels"],
            }
        )

    print("========================================\n")

    return dataset

class RSNAStudyDataset(Dataset):

    def __init__(self, dataset):
        self.dataset = dataset

        self.label_columns = [
            "ACL",
            "MCL",
            "Medial Meniscus",
            "Lateral Meniscus",
            "Medial OA",
            "Lateral OA",
            "PF OA",
            "Effusion",
            "Synovitis",
            "Baker's",
            "Contusion",
            "Fracture",
        ]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):

        study = self.dataset[idx]

        # ---------------------------------------------------------
        # Load preprocessed study from .pt cache
        # ---------------------------------------------------------

        data = torch.load(
            study["pt_path"],
            map_location="cpu"
        )

        # ---------------------------------------------------------
        # These are stored in the .pt file
        # ---------------------------------------------------------

        volumes = data["volumes"]
        plane = data["plane"]
        fluid = data["fluid"]
        fat = data["fat"]

        # ---------------------------------------------------------
        # IMPORTANT:
        # mask is NOT loaded here.
        #
        # rsna_collate_fn() creates mask after batching because
        # different studies can contain different numbers of series.
        # ---------------------------------------------------------

        output = {
            "study_id": study["study_id"],
            "volumes": volumes,
            "plane": plane,
            "fluid": fluid,
            "fat": fat,
            "label_source": study["label_source"],
        }

        # ---------------------------------------------------------
        # Labels come from the .pt file
        # ---------------------------------------------------------

        if "labels" in data:

            output["labels"] = data["labels"]

        else:

            # Fallback to metadata stored by build_dataset()
            if "labels" in study:

                output["labels"] = torch.tensor(
                    [
                        study["labels"][label]
                        for label in self.label_columns
                    ],
                    dtype=torch.float32
                )

        return output


def rsna_collate_fn(batch):

    batch_size = len(batch)

    # ---------------------------------------------------------
    # Maximum number of series in this batch
    # ---------------------------------------------------------

    max_series = max(
        sample["volumes"].shape[0]
        for sample in batch
    )

    # ---------------------------------------------------------
    # Tensor shape:
    # (series, channels, slices, height, width)
    # ---------------------------------------------------------

    _, channels, num_slices, height, width = (
        batch[0]["volumes"].shape
    )

    # ---------------------------------------------------------
    # KEEP FP16
    #
    # .pt files are already float16.
    # Do NOT convert the entire batch to float32.
    # ---------------------------------------------------------

    volumes = torch.zeros(
        batch_size,
        max_series,
        channels,
        num_slices,
        height,
        width,
        dtype=torch.float16
    )

    plane = torch.zeros(
        batch_size,
        max_series,
        dtype=torch.long
    )

    fluid = torch.zeros(
        batch_size,
        max_series,
        dtype=torch.long
    )

    fat = torch.zeros(
        batch_size,
        max_series,
        dtype=torch.long
    )

    mask = torch.zeros(
        batch_size,
        max_series,
        dtype=torch.bool
    )

    # ---------------------------------------------------------
    # Copy each study into padded batch
    # ---------------------------------------------------------

    for i, sample in enumerate(batch):

        n_series = sample["volumes"].shape[0]

        volumes[i, :n_series] = sample["volumes"]

        plane[i, :n_series] = sample["plane"]

        fluid[i, :n_series] = sample["fluid"]

        fat[i, :n_series] = sample["fat"]

        mask[i, :n_series] = True

    # ---------------------------------------------------------
    # Output
    # ---------------------------------------------------------

    output = {
        "study_id": [
            sample["study_id"]
            for sample in batch
        ],

        "volumes": volumes,

        "plane": plane,

        "fluid": fluid,

        "fat": fat,

        "mask": mask,

        "label_source": [
            sample["label_source"]
            for sample in batch
        ],

        "is_gold": torch.tensor(
            [
                1.0 if sample["label_source"] == "gold"
                else 0.0
                for sample in batch
            ],
            dtype=torch.float16
        ),
    }

    # ---------------------------------------------------------
    # Labels
    # ---------------------------------------------------------

    if "labels" in batch[0]:

        output["labels"] = torch.stack(
            [
                sample["labels"].float()
                for sample in batch
            ],
            dim=0
        )

    return output

def create_train_val_split(
    dataset,
    test_size=0.20,
    random_state=42
):
    """
    Create one study-level multilabel-stratified train/validation split
    using skmultilearn's iterative_train_test_split.
    """

    LABEL_COLUMNS = [
        "ACL",
        "MCL",
        "Medial Meniscus",
        "Lateral Meniscus",
        "Medial OA",
        "Lateral OA",
        "PF OA",
        "Effusion",
        "Synovitis",
        "Baker's",
        "Contusion",
        "Fracture",
    ]

    if len(dataset) < 2:
        raise ValueError("Dataset must contain at least 2 studies.")

    # 1. Build label matrix y of shape (N, num_classes)
    y = np.asarray(
        [
            [study["labels"][label] for label in LABEL_COLUMNS]
            for study in dataset
        ],
        dtype=np.float32
    )
    y_binary = (y >= 0.5).astype(int)

    # 2. Build 2D index matrix X where row i contains [i]
    X_indices = np.arange(len(dataset)).reshape(-1, 1)

    # 3. Perform iterative stratified split
    # skmultilearn returns (X_train, y_train, X_val, y_val)
    X_train, _, X_val, _ = iterative_train_test_split(
        X_indices,
        y_binary,
        test_size=test_size
    )

    # 4. Extract 1D integer indices by taking column 0
    train_indices = X_train[:, 0].astype(np.int64)
    val_indices = X_val[:, 0].astype(np.int64)

    # 5. Build dataset splits
    train_dataset = [dataset[i] for i in train_indices]
    val_dataset = [dataset[i] for i in val_indices]

    split = {
        "train": train_dataset,
        "val": val_dataset,
        "train_indices": train_indices,
        "val_indices": val_indices,
        "test_size": test_size,
        "random_state": random_state
    }

    rank = int(os.environ.get("RANK", 0))
 
    if rank == 0:

        train_gold = sum(
            x["label_source"] == "gold"
            for x in train_dataset
        )

        train_llm = sum(
            x["label_source"] == "llm"
            for x in train_dataset
        )

        val_gold = sum(
            x["label_source"] == "gold"
            for x in val_dataset
        )

        val_llm = sum(
            x["label_source"] == "llm"
            for x in val_dataset
        )

        train_ids = {
            x["study_id"]
            for x in train_dataset
        }

        val_ids = {
            x["study_id"]
            for x in val_dataset
        }

        overlap = train_ids & val_ids

        print("\n" + "=" * 60)
        print("TRAIN / VALIDATION SPLIT")
        print("=" * 60)

        print(f"Train: {len(train_dataset)}")
        print(f"  Gold: {train_gold}")
        print(f"  LLM : {train_llm}")

        print(f"\nValidation: {len(val_dataset)}")
        print(f"  Gold: {val_gold}")
        print(f"  LLM : {val_llm}")

        print(f"\nStudy overlap: {len(overlap)}")

        if len(overlap) == 0:
            print("✓ No study overlap")
        else:
            print("WARNING: Study overlap detected!")

        print("=" * 60)

    return split


def print_split_distribution(split):

    train = split["train"]
    val = split["val"]

    train_gold = sum(
        x["label_source"] == "gold"
        for x in train
    )

    train_llm = sum(
        x["label_source"] == "llm"
        for x in train
    )

    val_gold = sum(
        x["label_source"] == "gold"
        for x in val
    )

    val_llm = sum(
        x["label_source"] == "llm"
        for x in val
    )

    print(
        "\nTrain"
    )

    print(
        f"Train: {len(train)} "
        f"(Gold={train_gold}, "
        f"LLM={train_llm})"
    )

    print(
        f"Val:   {len(val)} "
        f"(Gold={val_gold}, "
        f"LLM={val_llm})"
    )


# ============================================================
# MODEL
# ============================================================

import torch
import torch.nn as nn
from torchvision.models import densenet121


# ============================================================
# DenseNet Series Encoder
# ============================================================

class DenseNetSeriesEncoder(nn.Module):
    def __init__(
        self,
        radimagenet_weights_path=None,
        proj_dim=512,
        dropout=0.2,
        num_heads=4,
        dense_chunk_size=16,
        freeze_backbone=True,
        freeze_channel_adapter=True,
        radimagenet_weights=None,
    ):
        super().__init__()

        self.proj_dim = proj_dim
        self.dense_chunk_size = dense_chunk_size
        self.freeze_backbone = freeze_backbone
        self.freeze_channel_adapter = freeze_channel_adapter

        # ============================================================
        # 1. DenseNet121 backbone
        # ============================================================
        base = densenet121(weights=None)

        self.features = base.features

        # ============================================================
        # 2. Load RadImageNet DenseNet121 weights
        # ============================================================
        
        if radimagenet_weights_path is not None:
            self._load_radimagenet_weights(
                radimagenet_weights_path
            )

        # ============================================================
        # 3. Grayscale MRI -> RGB adapter
        # ============================================================
        self.channel_adapter = nn.Conv2d(
            in_channels=1,
            out_channels=3,
            kernel_size=1,
            bias=False
        )

        # # Equivalent to grayscale -> RGB replication/averaging
        with torch.no_grad():
            self.channel_adapter.weight.fill_(1.0 / 3.0)

        # unfreeze only denseblock4
        # with torch.no_grad():
        #     self.channel_adapter.weight.fill_(1.0)

        # unfreeze only denseblock4

        # Freeze adapter if requested
        for param in self.channel_adapter.parameters():
            param.requires_grad = not freeze_channel_adapter

        # ============================================================
        # 4. Freeze DenseNet backbone (except denseblock4 and norm5)
        # ============================================================
        # for param in self.features.parameters():
        #     param.requires_grad = not freeze_backbone

        # # DenseNet121 output channels = 1024
        # self.feature_dim = 1024


        # unfreeze only denseblock4

        # ============================================================
        # Store configuration
        # ============================================================

        self.freeze_backbone = freeze_backbone

        # ============================================================
        # DenseNet backbone freezing
        #
        # freeze_backbone=True
        #     -> Entire DenseNet frozen
        #
        # freeze_backbone=False
        #     -> Only denseblock4 trainable
        # ============================================================

        # First freeze EVERYTHING
        for param in self.features.parameters():
            param.requires_grad = False

        # Only unfreeze denseblock4 when requested
        if not self.freeze_backbone:

            if hasattr(self.features, "denseblock4"):

                for param in self.features.denseblock4.parameters():
                    param.requires_grad = True

            else:
                raise AttributeError(
                    "DenseNet does not contain 'denseblock4'"
                )

        # DenseNet121 output channels = 1024
        self.feature_dim = 1024

        # unfreeze only denseblock4

        # ============================================================
        # 5. Projection
        # ============================================================
        # This remains trainable even when DenseNet is frozen.
        self.projection = nn.Sequential(
            nn.Linear(self.feature_dim, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # ============================================================
        # 6. Slice-level self attention
        # ============================================================
        self.slice_attn = nn.MultiheadAttention(
            embed_dim=proj_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        self.slice_query = nn.Parameter(
            torch.randn(1, 1, proj_dim) * 0.02
        )

        self.slice_norm = nn.LayerNorm(proj_dim)


    # unfreeze only denseblock4

    def forward_features(self, x):

        # ========================================================
        # Frozen DenseNet layers
        # ========================================================

        x = self.features.conv0(x)
        x = self.features.norm0(x)
        x = self.features.relu0(x)
        x = self.features.pool0(x)

        x = self.features.denseblock1(x)
        x = self.features.transition1(x)

        x = self.features.denseblock2(x)
        x = self.features.transition2(x)

        x = self.features.denseblock3(x)
        x = self.features.transition3(x)

        # ========================================================
        # DenseBlock4
        # ========================================================

        if self.freeze_backbone:

            x = self.features.denseblock4(x)

        else:

            x = checkpoint(
                self.features.denseblock4,
                x,
                use_reentrant=False
            )

        # ========================================================
        # Final frozen normalization
        # ========================================================

        x = self.features.norm5(x)

        return x

    # unfreeze only denseblock4


    # ========================================================
    # Load RadImageNet
    # ========================================================
    
    def _load_radimagenet_weights(self, path):

        print(
            f"\nLoading RadImageNet weights from:\n{path}"
        )

        raw = torch.load(
            path,
            map_location="cpu"
        )

        # ----------------------------------------------------
        # Extract state dict
        # ----------------------------------------------------

        if isinstance(raw, dict):

            ckpt_state = raw.get(
                "state_dict",
                raw
            )

        else:

            ckpt_state = raw

        target_state = self.features.state_dict()

        target_keys = set(
            target_state.keys()
        )

        remapped = {}

        # ----------------------------------------------------
        # Match checkpoint -> DenseNet features
        # ----------------------------------------------------

        for k, v in ckpt_state.items():

            k_clean = k.replace(
                "module.",
                ""
            )

            matched = None

            for target_key in target_keys:

                if k_clean.endswith(target_key):

                    matched = target_key
                    break

            if (
                matched is not None
                and target_state[matched].shape == v.shape
            ):

                remapped[matched] = v

        # ----------------------------------------------------
        # Load
        # ----------------------------------------------------

        missing, unexpected = (
            self.features.load_state_dict(
                remapped,
                strict=False
            )
        )

        print(
            f"[RadImageNet load] "
            f"matched={len(remapped)}/{len(target_keys)} | "
            f"missing={len(missing)} | "
            f"unexpected={len(unexpected)}"
        )

        # ----------------------------------------------------
        # Warning
        # ----------------------------------------------------

        if len(remapped) < len(target_keys) * 0.90:

            print(
                "\nWARNING:"
                " fewer than 90% of DenseNet weights matched."
            )

            print(
                "Sample checkpoint keys:"
            )

            print(
                list(ckpt_state.keys())[:5]
            )

            print(
                "Sample target keys:"
            )

            print(
                list(target_keys)[:5]
            )

        print("RadImageNet loading completed.\n")


    # ================================================================
    # Keep frozen DenseNet in eval mode
    # ================================================================

    # def train(self, mode=True):
    #         super().train(mode)
    
    #         if self.freeze_backbone:
    #             # Keep early layers in eval mode (frozen stats), but let denseblock4 train
    #             self.features.eval()
                
    #         return self

    # unfreeze only denseblock4

    def train(self, mode=True):

        # First put the entire model into the requested mode
        super().train(mode)

        # ========================================================
        # DenseNet backbone handling
        # ========================================================

        if self.freeze_backbone:

            # ----------------------------------------------------
            # COMPLETE BACKBONE FREEZE
            # ----------------------------------------------------
            #
            # Keep the entire DenseNet in eval mode.
            #
            # This is especially important for BatchNorm layers.
            # ----------------------------------------------------

            self.features.eval()

        else:

            # ----------------------------------------------------
            # ONLY denseblock4 IS TRAINABLE
            # ----------------------------------------------------
            #
            # Keep all other DenseNet layers in eval mode.
            # ----------------------------------------------------

            self.features.eval()

            # Put ONLY denseblock4 into training mode
            if hasattr(self.features, "denseblock4"):

                self.features.denseblock4.train(mode)

            else:
                raise AttributeError(
                    "DenseNet does not contain 'denseblock4'"
                )

        return self

    # unfreeze only denseblock4

    # ================================================================
    # Forward
    # ================================================================
    def forward(self, x):
        """
        x:
            [B, N, C, S, H, W]

        Example:
            [4, 13, 1, 32, 128, 128]

        Returns:
            [B, N, proj_dim]
        """

        B, N, C, S, H, W = x.shape

        # ------------------------------------------------------------
        # Flatten study/series/slice dimensions
        # [B, N, C, S, H, W] -> [B*N*S, C, H, W]
        # ------------------------------------------------------------
        x = x.permute(
            0, 1, 3, 2, 4, 5
        ).contiguous()

        x = x.view(
            B * N * S,
            C,
            H,
            W
        )

        # ------------------------------------------------------------
        # Frozen feature extraction
        # ------------------------------------------------------------

        # unfreeze only denseblock4

        # ============================================================
        # Channel adapter
        # ============================================================

        x = self.channel_adapter(x)

        # ============================================================
        # DenseNet feature extraction in chunks
        # ============================================================

        feature_chunks = []

        for chunk in torch.split(
            x,
            self.dense_chunk_size,
            dim=0
        ):

            # --------------------------------------------------------
            # Completely frozen DenseNet
            # --------------------------------------------------------

            if self.freeze_backbone:

                with torch.no_grad():

                    feat = self.forward_features(chunk)

            # --------------------------------------------------------
            # DenseBlock4 trainable
            # --------------------------------------------------------

            else:

                feat = self.forward_features(chunk)

            # --------------------------------------------------------
            # Global average pooling
            # --------------------------------------------------------

            feat = F.relu(
                feat,
                inplace=False
            )

            feat = F.adaptive_avg_pool2d(
                feat,
                (1, 1)
            )

            feat = torch.flatten(
                feat,
                start_dim=1
            )

            feature_chunks.append(feat)

        x = torch.cat(
            feature_chunks,
            dim=0
        )

        # unfreeze only denseblock4

        # if self.freeze_backbone and self.freeze_channel_adapter:

        #     with torch.no_grad():
        #         x = self.channel_adapter(x)

        #         feature_chunks = []
        #         for chunk in torch.split(
        #             x,
        #             self.dense_chunk_size,
        #             dim=0
        #         ):
        #             feat = self.features(chunk)
        #             feat = F.relu(feat, inplace=False)
        #             feat = F.adaptive_avg_pool2d(feat, (1, 1))
        #             feat = torch.flatten(feat, start_dim=1)
        #             feature_chunks.append(feat)

        #         x = torch.cat(feature_chunks, dim=0)

        # else:
        #     x = self.channel_adapter(x)

        #     feature_chunks = []
        #     for chunk in torch.split(
        #         x,
        #         self.dense_chunk_size,
        #         dim=0
        #     ):
        #         feat = self.features(chunk)
        #         feat = F.relu(feat, inplace=False)
        #         feat = F.adaptive_avg_pool2d(feat, (1, 1))
        #         feat = torch.flatten(feat, start_dim=1)
        #         feature_chunks.append(feat)

        #     x = torch.cat(feature_chunks, dim=0)

        # ------------------------------------------------------------
        # Project DenseNet features (Trainable)
        # ------------------------------------------------------------
        x = self.projection(x)

        # ------------------------------------------------------------
        # Restore: [B*N*S, proj_dim] -> [B, N, S, proj_dim]
        # ------------------------------------------------------------
        x = x.view(
            B,
            N,
            S,
            self.proj_dim
        )

        # ------------------------------------------------------------
        # Slice-level attention with Intermediate Pooling
        # [B, N, S, D] -> [B*N, S, D]
        # ------------------------------------------------------------
        x = x.view(
            B * N,
            S,
            self.proj_dim
        )

        # --- Intermediate Pooling: Reduce S slices (e.g., 32) down to S//2 (16) ---
        # F.avg_pool1d expects shape: [Batch, Channels, Length]
        x = x.permute(0, 2, 1)                  # [B*N, D, S]
        x = F.avg_pool1d(x, kernel_size=2, stride=2)  # [B*N, D, S//2]
        x = x.permute(0, 2, 1)                  # [B*N, S//2, D]

        query = self.slice_query.expand(
            B * N,
            -1,
            -1
        )

        attended, _ = self.slice_attn(
            query=query,
            key=x,
            value=x,
            need_weights=False
        )

        # ------------------------------------------------------------
        # Residual + normalization
        # ------------------------------------------------------------
        attended = self.slice_norm(
            attended + query
        )

        # ------------------------------------------------------------
        # One representation per series: [B*N, 1, D] -> [B, N, D]
        # ------------------------------------------------------------
        attended = attended.squeeze(1)

        attended = attended.view(
            B,
            N,
            self.proj_dim
        )

        return attended
    
# ============================================================
# Metadata Embedding
# ============================================================

class MetadataEmbedding(nn.Module):

    def __init__(
        self,
        plane_dim=16,
        fluid_dim=8,
        fat_dim=8
    ):
        super().__init__()

        self.plane_embedding = nn.Embedding(
            3,
            plane_dim
        )

        self.fluid_embedding = nn.Embedding(
            2,
            fluid_dim
        )

        self.fat_embedding = nn.Embedding(
            2,
            fat_dim
        )

    def forward(
        self,
        plane_idx,
        fluid_idx,
        fat_idx
    ):

        p = self.plane_embedding(
            plane_idx
        )

        f1 = self.fluid_embedding(
            fluid_idx
        )

        f2 = self.fat_embedding(
            fat_idx
        )

        return torch.cat(
            [
                p,
                f1,
                f2
            ],
            dim=-1
        )


# ============================================================
# Study-Level Attention Pool
# ============================================================

class AttentionPool(nn.Module):

    def __init__(
        self,
        dim=544,
        num_heads=4,
        dropout=0.1
    ):
        super().__init__()

        self.query = nn.Parameter(
            torch.randn(
                1,
                1,
                dim
            ) * 0.02
        )

        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        self.norm1 = nn.LayerNorm(
            dim
        )

        self.ffn = nn.Sequential(

            nn.Linear(
                dim,
                dim
            ),

            nn.GELU(),

            nn.Dropout(
                dropout
            ),

            nn.Linear(
                dim,
                dim
            )
        )

        self.norm2 = nn.LayerNorm(
            dim
        )

    def forward(
        self,
        series_tokens,
        key_padding_mask=None
    ):

        B = series_tokens.size(0)

        q = self.query.expand(
            B,
            -1,
            -1
        )

        pooled, attn_weights = self.attn(
            q,
            series_tokens,
            series_tokens,
            key_padding_mask=key_padding_mask
        )

        pooled = self.norm1(
            pooled.squeeze(1)
        )

        pooled = pooled + self.ffn(
            pooled
        )

        pooled = self.norm2(
            pooled
        )

        return pooled, attn_weights


# ============================================================
# Complete Knee Study Model
# ============================================================

class KneeStudyModel(nn.Module):

    def __init__(
        self,
        num_classes=12,
        radimagenet_weights_path=None,
        series_dim=512,
        metadata_dim=32,
        num_heads=4,
        classifier_hidden=256,
        dropout=0.3,
        dense_chunk_size=16,
        freeze_backbone=True,
        freeze_channel_adapter=True,
    ):
        super().__init__()

        # ----------------------------------------------------
        # MRI encoder
        # ----------------------------------------------------

        # self.series_encoder = DenseNetSeriesEncoder(

        #     radimagenet_weights_path=
        #         radimagenet_weights_path,

        #     proj_dim=series_dim,

        #     dropout=0.2,

        #     num_heads=num_heads,

        #     dense_chunk_size=
        #         dense_chunk_size,

        #     freeze_backbone=
        #         freeze_backbone
        # )


        self.series_encoder = DenseNetSeriesEncoder(
            radimagenet_weights_path=radimagenet_weights_path,
            proj_dim=series_dim,
            dropout=0.2,
            num_heads=num_heads,
            dense_chunk_size=dense_chunk_size,
            freeze_backbone=freeze_backbone,
            freeze_channel_adapter=freeze_channel_adapter
        )
        # ----------------------------------------------------
        # Metadata encoder
        # ----------------------------------------------------

        self.metadata_encoder = MetadataEmbedding()

        # ----------------------------------------------------
        # MRI + metadata dimension
        # ----------------------------------------------------

        token_dim = (
            series_dim +
            metadata_dim
        )

        # 512 + 32 = 544

        # ----------------------------------------------------
        # Study-level attention
        # ----------------------------------------------------

        self.attn_pool = AttentionPool(

            dim=token_dim,

            num_heads=num_heads,

            dropout=0.1
        )

        # ----------------------------------------------------
        # Classifier
        # ----------------------------------------------------

        self.classifier = nn.Sequential(

            nn.Linear(
                token_dim,
                classifier_hidden
            ),

            nn.LayerNorm(
                classifier_hidden
            ),

            nn.GELU(),

            nn.Dropout(
                dropout
            ),

            nn.Linear(
                classifier_hidden,
                num_classes
            )
        )

    # ========================================================
    # Forward
    # ========================================================

    def forward(
        self,
        slices,
        plane,
        fluid,
        fat,
        mask=None
    ):

        # ----------------------------------------------------
        # MRI series features
        # ----------------------------------------------------

        series_feats = self.series_encoder(
            slices
        )

        # [B,N,512]

        # ----------------------------------------------------
        # Metadata
        # ----------------------------------------------------

        meta_feats = self.metadata_encoder(
            plane,
            fluid,
            fat
        )

        # [B,N,32]

        # ----------------------------------------------------
        # Combine
        # ----------------------------------------------------

        tokens = torch.cat(
            [
                series_feats,
                meta_feats
            ],
            dim=-1
        )

        # [B,N,544]

        # ----------------------------------------------------
        # Study-level aggregation
        #
        # IMPORTANT:
        #
        # mask:
        # True  = real series
        # False = padding
        #
        # MultiheadAttention expects:
        # True  = ignore
        # False = valid
        #
        # Therefore use ~mask.
        # ----------------------------------------------------

        if mask is not None:

            study_vector, attn_weights = (
                self.attn_pool(
                    tokens,
                    key_padding_mask=~mask
                )
            )

        else:

            study_vector, attn_weights = (
                self.attn_pool(
                    tokens
                )
            )

        # [B,544]

        # ----------------------------------------------------
        # Classification
        # ----------------------------------------------------

        logits = self.classifier(
            study_vector
        )

        # [B,12]

        return logits



os.makedirs(
    CHECKPOINT_DIR,
    exist_ok=True
)


# ============================================================
# RANDOM SEED
# ============================================================

def set_seed(seed):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    torch.cuda.manual_seed_all(seed)


# ============================================================
# DDP SETUP
# ============================================================

def setup_ddp():

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    torch.cuda.set_device(local_rank)

    dist.init_process_group(
        backend="nccl",
        device_id=torch.device(
            f"cuda:{local_rank}"
        ),
        timeout=datetime.timedelta(minutes=2)  # Added timeout here
    )

    device = torch.device(
        f"cuda:{local_rank}"
    )

    return (
        rank,
        local_rank,
        world_size,
        device
    )    
# ============================================================
# DDP CLEANUP
# ============================================================

def cleanup_ddp():

    if dist.is_initialized():

        dist.destroy_process_group()


# ============================================================
# GPU INFO
# ============================================================

def print_gpu_info(
    rank,
    local_rank,
    world_size
):

    gpu_name = (
        torch.cuda.get_device_name(
            local_rank
        )
    )

    gpu_memory = (
        torch.cuda.get_device_properties(
            local_rank
        ).total_memory
        / 1024**3
    )

    print(
        f"[Rank {rank}] "
        f"GPU {local_rank}: "
        f"{gpu_name} | "
        f"{gpu_memory:.2f} GB | "
        f"World size: {world_size}"
    )


# ============================================================
# GPU MONITORING
# ============================================================

def get_gpu_usage(
    local_rank
):

    if not torch.cuda.is_available():

        return {
            "gpu_util": "N/A",
            "memory_used_mb": "N/A",
            "memory_total_mb": "N/A",
            "temperature": "N/A",
            "power_w": "N/A"
        }

    try:

        result = subprocess.check_output(

            [
                "nvidia-smi",

                "-i",
                str(local_rank),

                "--query-gpu="
                "utilization.gpu,"
                "memory.used,"
                "memory.total,"
                "temperature.gpu,"
                "power.draw",

                "--format=csv,noheader,nounits"
            ],

            encoding="utf-8"
        )

        values = [
            x.strip()
            for x in result.strip().split(",")
        ]

        return {

            "gpu_util": values[0],

            "memory_used_mb": values[1],

            "memory_total_mb": values[2],

            "temperature": values[3],

            "power_w": values[4]
        }

    except Exception as e:

        return {

            "gpu_util": "ERROR",

            "memory_used_mb": "ERROR",

            "memory_total_mb": "ERROR",

            "temperature": "ERROR",

            "power_w": str(e)
        }


# ============================================================
# GPU LOGGING
# ============================================================

def log_gpu_usage(
    log_file,
    epoch,
    batch,
    total_batches,
    loss,
    local_rank
):
    gpu = get_gpu_usage(local_rank)

    allocated = (
        torch.cuda.memory_allocated(local_rank)
        / 1024**3
    )

    reserved = (
        torch.cuda.memory_reserved(local_rank)
        / 1024**3
    )

    max_allocated = (
        torch.cuda.max_memory_allocated(local_rank)
        / 1024**3
    )

    max_reserved = (
        torch.cuda.max_memory_reserved(local_rank)
        / 1024**3
    )

    timestamp = time.strftime(
        "%Y-%m-%d %H:%M:%S"
    )

    line = (
        f"{timestamp} | "
        f"Rank {local_rank} | "
        f"Epoch {epoch} | "
        f"Batch {batch}/{total_batches} | "
        f"Loss {loss:.6f} | "

        f"GPU Util "
        f"{gpu['gpu_util']}% | "

        f"VRAM "
        f"{gpu['memory_used_mb']}/"
        f"{gpu['memory_total_mb']} MB | "

        f"Temp "
        f"{gpu['temperature']} C | "

        f"Power "
        f"{gpu['power_w']} W | "

        f"Torch Allocated "
        f"{allocated:.2f} GB | "

        f"Torch Reserved "
        f"{reserved:.2f} GB | "

        f"Max Allocated "
        f"{max_allocated:.2f} GB | "

        f"Max Reserved "
        f"{max_reserved:.2f} GB\n"
    )

    with open(log_file, "a") as f:
        f.write(line)

# ============================================================
# BUILD MODEL
# ============================================================

def build_model(
    device
):

    model = KneeStudyModel(

        num_classes=12,

        radimagenet_weights_path=
            RADIMAGENET_PATH,

        series_dim=512,

        metadata_dim=32,

        num_heads=4,

        classifier_hidden=256,

        dropout=0.3,

        dense_chunk_size=
            DENSE_CHUNK_SIZE,

        # IMPORTANT:
        # full fine-tuning
        # freeze_backbone=False

        # Freeze RadImageNet DenseNet121
        freeze_backbone=FREEZE_BACKBONE,
        freeze_channel_adapter=FREEZE_CHANNEL_ADAPTER,

    )

    model = model.to(
        device
    )

    return model


def print_model_info(
    model,
    rank
):

    # ========================================================
    # Only Rank 0 prints model information
    # ========================================================

    if rank != 0:
        return

    print(
        "\n"
        + "=" * 70
    )

    print(
        "MODEL PARAMETER INFORMATION"
    )

    print(
        "=" * 70
    )

    # ========================================================
    # DENSENET PARAMETER STATUS
    # ========================================================

    denseblock4_trainable = any(
        p.requires_grad
        for p in model.series_encoder.features.denseblock4.parameters()
    )

    other_densenet_trainable = []

    for name, param in model.series_encoder.features.named_parameters():

        if param.requires_grad and not name.startswith(
            "denseblock4."
        ):

            other_densenet_trainable.append(name)

    # ========================================================
    # Validate DenseNet configuration
    # ========================================================

    if FREEZE_BACKBONE:

        assert not denseblock4_trainable, (
            "FREEZE_BACKBONE=True, but DenseBlock4 "
            "contains trainable parameters!"
        )

        assert not other_densenet_trainable, (
            "FREEZE_BACKBONE=True, but other DenseNet "
            "parameters are trainable!"
        )

        print(
            "✓ RadImageNet DenseNet121: FROZEN"
        )

    else:

        assert denseblock4_trainable, (
            "FREEZE_BACKBONE=False, but DenseBlock4 "
            "has no trainable parameters!"
        )

        assert not other_densenet_trainable, (
            "Only DenseBlock4 should be trainable, "
            "but other DenseNet parameters are also trainable:\n"
            + "\n".join(
                other_densenet_trainable
            )
        )

        print(
            "✓ RadImageNet DenseNet121: "
            "DenseBlock4 FINE-TUNED"
        )

    # ========================================================
    # CHANNEL ADAPTER
    # ========================================================

    adapter_trainable = any(
        p.requires_grad
        for p in model.series_encoder.channel_adapter.parameters()
    )

    print(
        f"Channel adapter: "
        f"{'TRAINABLE' if adapter_trainable else 'FROZEN'}"
    )

    if FREEZE_CHANNEL_ADAPTER:

        assert not adapter_trainable, (
            "FREEZE_CHANNEL_ADAPTER=True, "
            "but channel adapter is trainable!"
        )

    else:

        assert adapter_trainable, (
            "FREEZE_CHANNEL_ADAPTER=False, "
            "but channel adapter is frozen!"
        )

    # ========================================================
    # DENSENET DETAILED STATUS
    # ========================================================

    print(
        "\nDenseNet status:"
    )

    for block_name in [
        "denseblock1",
        "denseblock2",
        "denseblock3",
        "denseblock4",
        "norm5"
    ]:

        if hasattr(
            model.series_encoder.features,
            block_name
        ):

            block = getattr(
                model.series_encoder.features,
                block_name
            )

            trainable = any(
                p.requires_grad
                for p in block.parameters()
            )

            print(
                f"  {block_name:<15}: "
                f"{'TRAINABLE' if trainable else 'FROZEN'}"
            )

    # ========================================================
    # PARAMETER COUNTS
    # ========================================================

    trainable_params = 0
    frozen_params = 0

    trainable_tensors = 0
    frozen_tensors = 0

    for param in model.parameters():

        if param.requires_grad:

            trainable_params += param.numel()

            trainable_tensors += 1

        else:

            frozen_params += param.numel()

            frozen_tensors += 1

    total_params = (
        trainable_params +
        frozen_params
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    print(
        "\nParameter summary:"
    )

    print(
        f"  Trainable parameters: "
        f"{trainable_params:,}"
    )

    print(
        f"  Frozen parameters: "
        f"{frozen_params:,}"
    )

    print(
        f"  Total parameters: "
        f"{total_params:,}"
    )

    print(
        f"  Trainable tensors: "
        f"{trainable_tensors:,}"
    )

    print(
        f"  Frozen tensors: "
        f"{frozen_tensors:,}"
    )

    print(
        "=" * 70
    )


# unfreeze only denseblock4 



# ============================================================
# OPTIMIZER
# ============================================================

# unfreeze only denseblock4

def build_optimizer(model):

    param_groups = [
        {
            "params": model.series_encoder.projection.parameters(),
            "lr": 1e-4
        },
        {
            "params": model.series_encoder.slice_attn.parameters(),
            "lr": 1e-4
        },
        {
            "params": [model.series_encoder.slice_query],
            "lr": 1e-4
        },
        {
            "params": model.series_encoder.slice_norm.parameters(),
            "lr": 1e-4
        },
        {
            "params": model.metadata_encoder.parameters(),
            "lr": 1e-4
        },
        {
            "params": model.attn_pool.parameters(),
            "lr": 1e-4
        },
        {
            "params": model.classifier.parameters(),
            "lr": 1e-4
        },
    ]

    # ============================================================
    # CHANNEL ADAPTER
    # ============================================================

    if not FREEZE_CHANNEL_ADAPTER:

        adapter_params = [
            p
            for p in model.series_encoder.channel_adapter.parameters()
            if p.requires_grad
        ]

        if adapter_params:
            param_groups.append({
                "params": adapter_params,
                "lr": 1e-5
            })

    # ============================================================
    # DENSENET BLOCK 4 ONLY
    # ============================================================

    denseblock4_params = [
        p
        for p in model.series_encoder.features.denseblock4.parameters()
        if p.requires_grad
    ]

    if denseblock4_params:

        param_groups.append({
            "params": denseblock4_params,
            "lr": 1e-5
        })

    # ============================================================
    # ADAMW
    # ============================================================

    return torch.optim.AdamW(
        param_groups,
        weight_decay=WEIGHT_DECAY
    )

# ============================================================
# DATA LOADERS
# ============================================================

def build_dataloaders(

    train_data,

    val_data,

    world_size,

    rank

):

    train_dataset = RSNAStudyDataset(

        train_data,

    )

    val_dataset = RSNAStudyDataset(

        val_data,

    )


    # --------------------------------------------------------
    # DISTRIBUTED SAMPLERS
    # --------------------------------------------------------

    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        seed=RANDOM_SEED,
        drop_last=True
        )

    val_sampler = DistributedSampler(

        val_dataset,

        num_replicas=world_size,

        rank=rank,

        shuffle=False,
        drop_last=True
        )


    # --------------------------------------------------------
    # TRAIN LOADER
    # --------------------------------------------------------

    train_loader = DataLoader(

        train_dataset,

        batch_size=BATCH_SIZE,

        sampler=train_sampler,

        num_workers=NUM_WORKERS,

        pin_memory=True,

        # persistent_workers=False, # Must be False when NUM_WORKERS == 0

        persistent_workers=(NUM_WORKERS > 0),
        prefetch_factor=2 if NUM_WORKERS > 0 else None,
        drop_last=False,
        
        collate_fn=rsna_collate_fn

    )


    # --------------------------------------------------------
    # VALIDATION LOADER
    # --------------------------------------------------------

    val_loader = DataLoader(

        val_dataset,

        batch_size=BATCH_SIZE,

        sampler=val_sampler,

        num_workers=NUM_WORKERS,

        pin_memory=True,

        # persistent_workers=False,

        persistent_workers=(NUM_WORKERS > 0),
        prefetch_factor=2 if NUM_WORKERS > 0 else None,
        drop_last=False,

        collate_fn=rsna_collate_fn

    )


    return (
        train_dataset,
        val_dataset,
        train_loader,
        val_loader,
        train_sampler,
        val_sampler
    )


# ============================================================
# DISTRIBUTED MEAN
# ============================================================

# def distributed_mean(
#     value,
#     device
# ):

#     tensor = torch.tensor(
#         value,
#         dtype=torch.float64,
#         device=device
#     )

#     dist.all_reduce(
#         tensor,
#         op=dist.ReduceOp.SUM
#     )

#     tensor /= dist.get_world_size()

#     return tensor.item()

def distributed_mean(value, device):
    rank = dist.get_rank()

    # print(
    #     f"[Rank {rank}] BEFORE distributed_mean ALLREDUCE "
    #     f"value={value}",
    #     flush=True
    # )

    tensor = torch.tensor(
        value,
        dtype=torch.float32,
        device=device
    )

    dist.all_reduce(
        tensor,
        op=dist.ReduceOp.SUM
    )

    result = (
        tensor / dist.get_world_size()
    ).item()

    # print(
    #     f"[Rank {rank}] AFTER distributed_mean ALLREDUCE "
    #     f"result={result}",
    #     flush=True
    # )

    return result



def evaluate_metrics(predictions, labels, threshold=0.5):

    predictions = np.asarray(predictions, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)

    if predictions.shape != labels.shape:
        raise ValueError(
            f"Shape mismatch: predictions={predictions.shape}, "
            f"labels={labels.shape}"
        )

    # Safety handling
    predictions = np.nan_to_num(
        predictions,
        nan=0.5,
        posinf=1.0,
        neginf=0.0
    )

    labels = np.nan_to_num(
        labels,
        nan=0.0,
        posinf=1.0,
        neginf=0.0
    )

    y_true = (labels >= 0.5).astype(np.int32)
    y_pred = (predictions >= threshold).astype(np.int32)

    per_label = {}

    auc_scores = []
    f1_scores = []
    ap_scores = []

    for label_idx, label_name in enumerate(LABEL_COLUMNS):

        true = y_true[:, label_idx]
        pred = y_pred[:, label_idx]
        prob = predictions[:, label_idx]

        # AUROC
        if len(np.unique(true)) >= 2:
            try:
                auc = roc_auc_score(true, prob)
            except ValueError:
                auc = np.nan
        else:
            auc = np.nan

        # F1
        try:
            f1 = f1_score(
                true,
                pred,
                zero_division=0
            )
        except ValueError:
            f1 = np.nan

        # Average Precision
        if len(np.unique(true)) >= 2:
            try:
                ap = average_precision_score(true, prob)
            except ValueError:
                ap = np.nan
        else:
            ap = np.nan

        per_label[label_name] = {
            "AUROC": auc,
            "F1": f1,
            "AP": ap,
            "Predicted_Positive_Rate": pred.mean(),
            "Actual_Positive_Rate": true.mean()
        }

        if not np.isnan(auc):
            auc_scores.append(auc)

        if not np.isnan(f1):
            f1_scores.append(f1)

        if not np.isnan(ap):
            ap_scores.append(ap)

    return {
        "mean_AUROC": (
            float(np.mean(auc_scores))
            if auc_scores else np.nan
        ),

        "mean_F1": (
            float(np.mean(f1_scores))
            if f1_scores else np.nan
        ),

        "mean_AP": (
            float(np.mean(ap_scores))
            if ap_scores else np.nan
        ),

        "per_label": per_label
    }

# ============================================================
# VALIDATION
# ============================================================

def validate(
    model,
    val_loader,
    criterion,
    device,
    rank,
    epoch=None,
    save_dir="./val_graph"
):
    model.eval()

    local_loss_sum = 0.0
    local_sample_count = 0

    local_predictions = []
    local_labels = []
    local_is_gold = []

    with torch.no_grad():

        for batch_idx, batch in enumerate(val_loader):

            volumes = batch["volumes"].to(
                device,
                non_blocking=True
            )

            mask = batch["mask"].to(
                device,
                non_blocking=True
            )

            plane = batch["plane"].to(
                device,
                non_blocking=True
            )

            fluid = batch["fluid"].to(
                device,
                non_blocking=True
            )

            fat = batch["fat"].to(
                device,
                non_blocking=True
            )

            labels = batch["labels"].float().to(
                device,
                non_blocking=True
            )

            # ------------------------------------------------
            # GOLD / LLM FLAG
            # ------------------------------------------------

            is_gold = batch["is_gold"].to(
                device,
                non_blocking=True
            )

            is_gold = is_gold.bool()

            if torch.isnan(volumes).any() or torch.isinf(volumes).any():
                print(
                    f"[Rank {rank}] WARNING: Validation volumes contain "
                    f"NaN/Inf at batch {batch_idx}. Skipping batch.",
                    flush=True
                )
                continue

            if torch.isnan(labels).any() or torch.isinf(labels).any():
                print(
                    f"[Rank {rank}] WARNING: Validation labels contain "
                    f"NaN/Inf at batch {batch_idx}. Skipping batch.",
                    flush=True
                )
                continue

            # ------------------------------------------------
            # FORWARD PASS
            # ------------------------------------------------

            with torch.autocast(
                device_type="cuda",
                dtype=AMP_DTYPE,
                enabled=USE_AMP
            ):

                logits = model(
                    slices=volumes,
                    mask=mask,
                    plane=plane,
                    fluid=fluid,
                    fat=fat
                )

                if torch.isnan(logits).any() or torch.isinf(logits).any():
                    print(
                        f"[Rank {rank}] WARNING: Validation logits contain "
                        f"NaN/Inf at batch {batch_idx}. Skipping batch.",
                        flush=True
                    )
                    continue

                loss_matrix = criterion(
                    logits,
                    labels
                )

                if (
                    torch.isnan(loss_matrix).any()
                    or torch.isinf(loss_matrix).any()
                ):
                    print(
                        f"[Rank {rank}] WARNING: Validation loss_matrix "
                        f"contains NaN/Inf at batch {batch_idx}. "
                        f"Skipping batch.",
                        flush=True
                    )
                    continue

                loss = loss_matrix.mean()

            if torch.isnan(loss) or torch.isinf(loss):
                print(
                    f"[Rank {rank}] WARNING: Validation loss is NaN/Inf "
                    f"at batch {batch_idx}. Skipping batch.",
                    flush=True
                )
                continue

            # ------------------------------------------------
            # LOSS
            # ------------------------------------------------

            batch_size = labels.shape[0]

            local_loss_sum += (
                loss.item() * batch_size
            )

            local_sample_count += batch_size

            # ------------------------------------------------
            # PROBABILITIES
            # ------------------------------------------------

            probabilities = torch.sigmoid(logits)

            if (
                torch.isnan(probabilities).any()
                or torch.isinf(probabilities).any()
            ):
                print(
                    f"[Rank {rank}] WARNING: Validation probabilities "
                    f"contain NaN/Inf at batch {batch_idx}. "
                    f"Skipping batch.",
                    flush=True
                )
                continue

            # ------------------------------------------------
            # STORE PREDICTIONS / LABELS / GOLD FLAG
            # ------------------------------------------------

            local_predictions.append(
                probabilities.detach().float().cpu()
            )

            local_labels.append(
                labels.detach().float().cpu()
            )

            local_is_gold.append(
                is_gold.detach().cpu()
            )

    # ========================================================
    # DISTRIBUTED VALIDATION LOSS
    # ========================================================

    local_loss_sum_tensor = torch.tensor(
        local_loss_sum,
        dtype=torch.float64,
        device=device
    )

    local_sample_count_tensor = torch.tensor(
        local_sample_count,
        dtype=torch.float64,
        device=device
    )

    dist.all_reduce(
        local_loss_sum_tensor,
        op=dist.ReduceOp.SUM
    )

    dist.all_reduce(
        local_sample_count_tensor,
        op=dist.ReduceOp.SUM
    )

    if local_sample_count_tensor.item() > 0:

        val_loss = (
            local_loss_sum_tensor
            / local_sample_count_tensor
        ).item()

    else:

        val_loss = float("nan")

        if rank == 0:
            print(
                "\nWARNING: No valid validation samples "
                "were processed.\n",
                flush=True
            )

    # ========================================================
    # LOCAL PREDICTIONS / LABELS / GOLD FLAGS
    # ========================================================

    if local_predictions:

        local_predictions = torch.cat(
            local_predictions,
            dim=0
        ).numpy()

        local_labels = torch.cat(
            local_labels,
            dim=0
        ).numpy()

        local_is_gold = torch.cat(
            local_is_gold,
            dim=0
        ).numpy().astype(bool)

    else:

        local_predictions = np.empty(
            (0, len(LABEL_COLUMNS)),
            dtype=np.float32
        )

        local_labels = np.empty(
            (0, len(LABEL_COLUMNS)),
            dtype=np.float32
        )

        local_is_gold = np.empty(
            (0,),
            dtype=bool
        )

    # ========================================================
    # GATHER FROM ALL GPUS
    # ========================================================

    world_size = dist.get_world_size()

    gathered_predictions = [None] * world_size
    gathered_labels = [None] * world_size
    gathered_is_gold = [None] * world_size

    dist.all_gather_object(
        gathered_predictions,
        local_predictions
    )

    dist.all_gather_object(
        gathered_labels,
        local_labels
    )

    dist.all_gather_object(
        gathered_is_gold,
        local_is_gold
    )

    # ========================================================
    # RANK 0 COMPUTES METRICS
    # ========================================================

    if rank == 0:

        predictions = np.concatenate(
            gathered_predictions,
            axis=0
        )

        labels = np.concatenate(
            gathered_labels,
            axis=0
        )

        is_gold = np.concatenate(
            gathered_is_gold,
            axis=0
        ).astype(bool)

        # ====================================================
        # MIXED VALIDATION
        # ====================================================

        mixed_metrics = evaluate_metrics(
            predictions=predictions,
            labels=labels,
            threshold=0.5
        )

        mixed_auc = mixed_metrics["mean_AUROC"]
        mixed_ap = mixed_metrics["mean_AP"]
        mixed_f1 = mixed_metrics["mean_F1"]

        # ====================================================
        # GOLD-ONLY VALIDATION
        # ====================================================

        gold_mask = is_gold

        gold_predictions = predictions[gold_mask]
        gold_labels = labels[gold_mask]

        gold_count = len(gold_labels)

        if gold_count > 0:

            gold_metrics = evaluate_metrics(
                predictions=gold_predictions,
                labels=gold_labels,
                threshold=0.5
            )

            gold_auc = gold_metrics["mean_AUROC"]
            gold_ap = gold_metrics["mean_AP"]
            gold_f1 = gold_metrics["mean_F1"]

        else:

            gold_metrics = None
            gold_auc = np.nan
            gold_ap = np.nan
            gold_f1 = np.nan

        # ====================================================
        # LLM-ONLY VALIDATION
        # ====================================================

        llm_mask = ~is_gold

        llm_predictions = predictions[llm_mask]
        llm_labels = labels[llm_mask]

        llm_count = len(llm_labels)

        if llm_count > 0:

            llm_metrics = evaluate_metrics(
                predictions=llm_predictions,
                labels=llm_labels,
                threshold=0.5
            )

            llm_auc = llm_metrics["mean_AUROC"]
            llm_ap = llm_metrics["mean_AP"]
            llm_f1 = llm_metrics["mean_F1"]

        else:

            llm_metrics = None
            llm_auc = np.nan
            llm_ap = np.nan
            llm_f1 = np.nan

        # ====================================================
        # PER-LABEL METRICS
        # ====================================================

        mixed_per_label = mixed_metrics["per_label"]

        gold_per_label = (
            gold_metrics["per_label"]
            if gold_metrics is not None
            else {}
        )

        llm_per_label = (
            llm_metrics["per_label"]
            if llm_metrics is not None
            else {}
        )

        # ====================================================
        # PRINT SUMMARY
        # ====================================================

        print(
            "\n" + "=" * 80,
            flush=True
        )

        print(
            f"Validation Metrics | Epoch {epoch + 1}",
            flush=True
        )

        print(
            "=" * 80,
            flush=True
        )

        print(
            f"Total Validation Samples : {len(labels)}",
            flush=True
        )

        print(
            f"Gold Samples             : {gold_count}",
            flush=True
        )

        print(
            f"LLM Samples              : {llm_count}",
            flush=True
        )

        print(
            "\nMIXED VALIDATION "
            "(Gold + LLM)",
            flush=True
        )

        print(
            f"  AUROC : {mixed_auc:.6f}",
            flush=True
        )

        print(
            f"  AP    : {mixed_ap:.6f}",
            flush=True
        )

        print(
            f"  F1    : {mixed_f1:.6f}",
            flush=True
        )

        print(
            "\nGOLD-ONLY VALIDATION",
            flush=True
        )

        print(
            f"  AUROC : {gold_auc:.6f}",
            flush=True
        )

        print(
            f"  AP    : {gold_ap:.6f}",
            flush=True
        )

        print(
            f"  F1    : {gold_f1:.6f}",
            flush=True
        )

        print(
            "\nLLM-ONLY VALIDATION",
            flush=True
        )

        print(
            f"  AUROC : {llm_auc:.6f}",
            flush=True
        )

        print(
            f"  AP    : {llm_ap:.6f}",
            flush=True
        )

        print(
            f"  F1    : {llm_f1:.6f}",
            flush=True
        )

        # ====================================================
        # TEXT LOG
        # ====================================================

        os.makedirs(
            save_dir,
            exist_ok=True
        )

        log_path = os.path.join(
            save_dir,
            "validation_metrics.txt"
        )

        with open(
            log_path,
            "a",
            encoding="utf-8"
        ) as f:

            f.write("\n")
            f.write("=" * 100 + "\n")
            f.write(
                f"Epoch {epoch + 1}\n"
            )
            f.write("=" * 100 + "\n")

            f.write(
                f"Total Validation Samples : {len(labels)}\n"
            )

            f.write(
                f"Gold Samples             : {gold_count}\n"
            )

            f.write(
                f"LLM Samples              : {llm_count}\n"
            )

            f.write("\n")

            # ------------------------------------------------
            # SUMMARY
            # ------------------------------------------------

            f.write(
                "SUMMARY\n"
            )

            f.write(
                "-" * 100 + "\n"
            )

            f.write(
                f"{'Dataset':<20}"
                f"{'Samples':>10}"
                f"{'AUROC':>15}"
                f"{'AP':>15}"
                f"{'F1':>15}\n"
            )

            f.write(
                "-" * 100 + "\n"
            )

            f.write(
                f"{'Mixed':<20}"
                f"{len(labels):>10}"
                f"{mixed_auc:>15.6f}"
                f"{mixed_ap:>15.6f}"
                f"{mixed_f1:>15.6f}\n"
            )

            f.write(
                f"{'Gold-only':<20}"
                f"{gold_count:>10}"
                f"{gold_auc:>15.6f}"
                f"{gold_ap:>15.6f}"
                f"{gold_f1:>15.6f}\n"
            )

            f.write(
                f"{'LLM-only':<20}"
                f"{llm_count:>10}"
                f"{llm_auc:>15.6f}"
                f"{llm_ap:>15.6f}"
                f"{llm_f1:>15.6f}\n"
            )

            f.write("\n")

            # ------------------------------------------------
            # PER LABEL
            # ------------------------------------------------

            f.write(
                "PER-LABEL AUROC\n"
            )

            f.write(
                "-" * 100 + "\n"
            )

            f.write(
                f"{'Label':<25}"
                f"{'Mixed':>15}"
                f"{'Gold':>15}"
                f"{'LLM':>15}\n"
            )

            f.write(
                "-" * 100 + "\n"
            )

            for label_name in LABEL_COLUMNS:

                mixed_value = mixed_per_label.get(
                    label_name,
                    {}
                ).get(
                    "AUROC",
                    np.nan
                )

                gold_value = gold_per_label.get(
                    label_name,
                    {}
                ).get(
                    "AUROC",
                    np.nan
                )

                llm_value = llm_per_label.get(
                    label_name,
                    {}
                ).get(
                    "AUROC",
                    np.nan
                )

                f.write(
                    f"{label_name:<25}"
                    f"{mixed_value:>15.6f}"
                    f"{gold_value:>15.6f}"
                    f"{llm_value:>15.6f}\n"
                )

            f.write("\n")

            # ------------------------------------------------
            # PER LABEL AP
            # ------------------------------------------------

            f.write(
                "PER-LABEL AVERAGE PRECISION\n"
            )

            f.write(
                "-" * 100 + "\n"
            )

            f.write(
                f"{'Label':<25}"
                f"{'Mixed':>15}"
                f"{'Gold':>15}"
                f"{'LLM':>15}\n"
            )

            f.write(
                "-" * 100 + "\n"
            )

            for label_name in LABEL_COLUMNS:

                mixed_value = mixed_per_label.get(
                    label_name,
                    {}
                ).get(
                    "AP",
                    np.nan
                )

                gold_value = gold_per_label.get(
                    label_name,
                    {}
                ).get(
                    "AP",
                    np.nan
                )

                llm_value = llm_per_label.get(
                    label_name,
                    {}
                ).get(
                    "AP",
                    np.nan
                )

                f.write(
                    f"{label_name:<25}"
                    f"{mixed_value:>15.6f}"
                    f"{gold_value:>15.6f}"
                    f"{llm_value:>15.6f}\n"
                )

            f.write("\n")

            # ------------------------------------------------
            # PER LABEL F1
            # ------------------------------------------------

            f.write(
                "PER-LABEL F1\n"
            )

            f.write(
                "-" * 100 + "\n"
            )

            f.write(
                f"{'Label':<25}"
                f"{'Mixed':>15}"
                f"{'Gold':>15}"
                f"{'LLM':>15}\n"
            )

            f.write(
                "-" * 100 + "\n"
            )

            for label_name in LABEL_COLUMNS:

                mixed_value = mixed_per_label.get(
                    label_name,
                    {}
                ).get(
                    "F1",
                    np.nan
                )

                gold_value = gold_per_label.get(
                    label_name,
                    {}
                ).get(
                    "F1",
                    np.nan
                )

                llm_value = llm_per_label.get(
                    label_name,
                    {}
                ).get(
                    "F1",
                    np.nan
                )

                f.write(
                    f"{label_name:<25}"
                    f"{mixed_value:>15.6f}"
                    f"{gold_value:>15.6f}"
                    f"{llm_value:>15.6f}\n"
                )

            f.write("\n")

        print(
            f"\nSaved validation metrics to: {log_path}",
            flush=True
        )

        # ====================================================
        # AUROC GRAPH - MIXED
        # ====================================================

        if (
            epoch is not None
            and (
                epoch < 10
                or epoch >= NUM_EPOCHS - 10
            )
        ):

            names = list(LABEL_COLUMNS)

            mixed_scores = []

            for label_name in names:

                value = mixed_per_label.get(
                    label_name,
                    {}
                ).get(
                    "AUROC",
                    np.nan
                )

                mixed_scores.append(
                    0.0 if np.isnan(value) else value
                )

            plt.figure(
                figsize=(10, 6)
            )

            plt.barh(
                names,
                mixed_scores,
                color="skyblue"
            )

            plt.axvline(
                x=0.5,
                color="red",
                linestyle="--",
                label="Random Guess (0.5)"
            )

            if not np.isnan(mixed_auc):

                plt.axvline(
                    x=mixed_auc,
                    color="green",
                    linestyle="-",
                    label=f"Mixed Mean AUROC ({mixed_auc:.3f})"
                )

            plt.xlabel("ROC-AUC")

            plt.title(
                f"Per-Label Mixed Validation AUROC "
                f"(Epoch {epoch + 1})"
            )

            plt.xlim(
                0.0,
                1.0
            )

            plt.legend(
                loc="lower right"
            )

            plt.tight_layout()

            mixed_plot_path = os.path.join(
                save_dir,
                f"val_auroc_mixed_epoch_{epoch + 1}.png"
            )

            plt.savefig(
                mixed_plot_path
            )

            plt.close()

            # =================================================
            # AUROC GRAPH - GOLD ONLY
            # =================================================

            gold_scores = []

            for label_name in names:

                value = gold_per_label.get(
                    label_name,
                    {}
                ).get(
                    "AUROC",
                    np.nan
                )

                gold_scores.append(
                    0.0 if np.isnan(value) else value
                )

            plt.figure(
                figsize=(10, 6)
            )

            plt.barh(
                names,
                gold_scores,
                color="gold"
            )

            plt.axvline(
                x=0.5,
                color="red",
                linestyle="--",
                label="Random Guess (0.5)"
            )

            if not np.isnan(gold_auc):

                plt.axvline(
                    x=gold_auc,
                    color="green",
                    linestyle="-",
                    label=f"Gold Mean AUROC ({gold_auc:.3f})"
                )

            plt.xlabel("ROC-AUC")

            plt.title(
                f"Per-Label Gold-Only AUROC "
                f"(Epoch {epoch + 1})"
            )

            plt.xlim(
                0.0,
                1.0
            )

            plt.legend(
                loc="lower right"
            )

            plt.tight_layout()

            gold_plot_path = os.path.join(
                save_dir,
                f"val_auroc_gold_epoch_{epoch + 1}.png"
            )

            plt.savefig(
                gold_plot_path
            )

            plt.close()

            print(
                f"Saved mixed AUROC graph to: "
                f"{mixed_plot_path}",
                flush=True
            )

            print(
                f"Saved gold AUROC graph to: "
                f"{gold_plot_path}",
                flush=True
            )

    else:

        mixed_metrics = None

        mixed_auc = np.nan
        mixed_ap = np.nan
        mixed_f1 = np.nan

    # ========================================================
    # BROADCAST MIXED METRICS
    # ========================================================

    metric_tensor = torch.tensor(
        [
            mixed_auc,
            mixed_ap,
            mixed_f1
        ],
        dtype=torch.float64,
        device=device
    )

    dist.broadcast(
        metric_tensor,
        src=0
    )

    mixed_auc = metric_tensor[0].item()
    mixed_ap = metric_tensor[1].item()
    mixed_f1 = metric_tensor[2].item()

    # ========================================================
    # RETURN
    # ========================================================

    return (
        val_loss,
        mixed_auc,
        mixed_ap,
        mixed_f1,
        mixed_metrics
    )


def train_one_epoch(
    model,
    train_loader,
    train_sampler,
    optimizer,
    criterion,
    scaler,
    device,
    epoch,
    rank,
    local_rank
):
    model.train()

    train_sampler.set_epoch(epoch)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)

    optimizer.zero_grad(set_to_none=True)

    running_loss = 0.0
    processed_batches = 0

    local_predictions = []
    local_labels = []

    GOLD_WEIGHT = 1.0
    LLM_WEIGHT = 0.50

    if rank == 0:
        progress_bar = tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}/{NUM_EPOCHS}",
            unit="batch"
        )
    else:
        progress_bar = train_loader

    # ========================================================
    # TRAINING LOOP
    # ========================================================

    for batch_idx, batch in enumerate(progress_bar):

        volumes = batch["volumes"].to(
            device,
            non_blocking=True
        )

        mask = batch["mask"].to(
            device,
            non_blocking=True
        )

        plane = batch["plane"].to(
            device,
            non_blocking=True
        )

        fluid = batch["fluid"].to(
            device,
            non_blocking=True
        )

        fat = batch["fat"].to(
            device,
            non_blocking=True
        )

        labels = batch["labels"].float().to(
            device,
            non_blocking=True
        )

        is_gold = batch["is_gold"].float().to(
            device,
            non_blocking=True
        )

        # ----------------------------------------------------
        # INPUT DATA SANITY CHECK
        # ----------------------------------------------------

        if torch.isnan(volumes).any() or torch.isinf(volumes).any():
            print(
                f"[Rank {rank}] WARNING: Input volumes contain "
                f"NaN or Inf at batch {batch_idx}. Skipping."
            )
            continue

        # ----------------------------------------------------
        # LABEL SANITY CHECK
        # ----------------------------------------------------

        if torch.isnan(labels).any() or torch.isinf(labels).any():
            print(
                f"[Rank {rank}] WARNING: Labels contain NaN or "
                f"Inf at batch {batch_idx}. Skipping."
            )
            continue

        # ----------------------------------------------------
        # FORWARD + LOSS
        # ----------------------------------------------------

        with torch.autocast(
            device_type="cuda",
            dtype=AMP_DTYPE,
            enabled=USE_AMP
        ):
            logits = model(
                slices=volumes,
                mask=mask,
                plane=plane,
                fluid=fluid,
                fat=fat
            )

            loss_matrix = criterion(
                logits,
                labels
            )

            sample_weights = (
                is_gold * GOLD_WEIGHT
                + (1.0 - is_gold) * LLM_WEIGHT
            )

            weighted_loss = (
                loss_matrix * sample_weights[:, None]
            )

            total_weight = sample_weights.sum()

            loss = (
                weighted_loss.sum()
                / (
                    total_weight
                    * labels.shape[1]
                )
            )

            loss_for_backward = (
                loss / GRADIENT_ACCUMULATION_STEPS
            )

        # ----------------------------------------------------
        # LOSS SANITY CHECK
        # ----------------------------------------------------

        if torch.isnan(loss) or torch.isinf(loss):
            print(
                f"[Rank {rank}] WARNING: Loss turned to NaN "
                f"or Inf at epoch {epoch + 1}, "
                f"batch {batch_idx}. Skipping batch."
            )

            optimizer.zero_grad(set_to_none=True)
            continue

        # ----------------------------------------------------
        # COLLECT PREDICTIONS FOR TRAIN METRICS
        # ----------------------------------------------------

        probabilities = torch.sigmoid(logits)

        local_predictions.append(
            probabilities.detach().float().cpu()
        )

        local_labels.append(
            labels.detach().float().cpu()
        )

        # ----------------------------------------------------
        # BACKWARD
        # ----------------------------------------------------

        scaler.scale(
            loss_for_backward
        ).backward()

        # ====================================================
        # GRADIENT SANITY CHECK
        # ====================================================

        if epoch == 0 and batch_idx == 0:

            print(
                "\n--- Checking Gradients on First Batch ---"
            )

            zero_grad_count = 0

            for name, param in model.named_parameters():

                if param.requires_grad:

                    if param.grad is not None:

                        if param.grad.abs().sum() == 0:

                            print(
                                f"WARNING: Zero gradient for {name}"
                            )

                            zero_grad_count += 1

                    else:

                        print(
                            f"WARNING: None gradient for {name}"
                        )

            print(
                "--- Gradient Check Complete. "
                f"Total zero-grad params: {zero_grad_count} ---\n"
            )

        # ----------------------------------------------------
        # OPTIMIZER STEP
        # ----------------------------------------------------

        should_step = (
            (batch_idx + 1) % GRADIENT_ACCUMULATION_STEPS == 0
            or (batch_idx + 1) == len(train_loader)
        )

        if should_step:

            scaler.unscale_(optimizer)

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=GRADIENT_CLIP
            )

            scaler.step(optimizer)
            scaler.update()

            optimizer.zero_grad(set_to_none=True)

        # ----------------------------------------------------
        # RUNNING LOSS
        # ----------------------------------------------------

        running_loss += loss.item()
        processed_batches += 1

        avg_loss = (
            running_loss / processed_batches
        )

        # ----------------------------------------------------
        # PROGRESS BAR
        # ----------------------------------------------------

        if rank == 0:

            if torch.cuda.is_available():

                current_vram = (
                    torch.cuda.memory_allocated(local_rank)
                    / 1024**3
                )

                vram_text = f"{current_vram:.2f}G"

            else:

                vram_text = "CPU"

            progress_bar.set_postfix(
                loss=f"{avg_loss:.4f}",
                vram=vram_text
            )

        # ----------------------------------------------------
        # GPU LOG
        # ----------------------------------------------------

        if (batch_idx + 1) % GPU_LOG_INTERVAL == 0:

            log_file = os.path.join(
                CHECKPOINT_DIR,
                f"gpu_rank_{local_rank}.txt"
            )

            log_gpu_usage(
                log_file,
                epoch=epoch + 1,
                batch=batch_idx + 1,
                total_batches=len(train_loader),
                loss=loss.item(),
                local_rank=local_rank
            )

    # ========================================================
    # DISTRIBUTED TRAIN LOSS
    # ========================================================

    local_mean_loss = (
        running_loss
        / max(processed_batches, 1)
    )

    train_loss = distributed_mean(
        local_mean_loss,
        device
    )

    # ========================================================
    # GATHER TRAINING PREDICTIONS
    # ========================================================

    if local_predictions:

        local_predictions = torch.cat(
            local_predictions,
            dim=0
        ).numpy()

    else:

        local_predictions = np.empty(
            (0, len(LABEL_COLUMNS)),
            dtype=np.float32
        )

    if local_labels:

        local_labels = torch.cat(
            local_labels,
            dim=0
        ).numpy()

    else:

        local_labels = np.empty(
            (0, len(LABEL_COLUMNS)),
            dtype=np.float32
        )

    world_size = dist.get_world_size()

    gathered_predictions = [
        None
    ] * world_size

    gathered_labels = [
        None
    ] * world_size

    dist.all_gather_object(
        gathered_predictions,
        local_predictions
    )

    dist.all_gather_object(
        gathered_labels,
        local_labels
    )

    # ========================================================
    # CALCULATE TRAIN METRICS
    # ========================================================

    if rank == 0:

        predictions = np.concatenate(
            gathered_predictions,
            axis=0
        )

        labels = np.concatenate(
            gathered_labels,
            axis=0
        )

        train_metrics = evaluate_metrics(
            predictions=predictions,
            labels=labels,
            threshold=0.5
        )

        train_auc = train_metrics[
            "mean_AUROC"
        ]

        train_ap = train_metrics[
            "mean_AP"
        ]

        train_f1 = train_metrics[
            "mean_F1"
        ]

    else:

        train_metrics = None
        train_auc = np.nan
        train_ap = np.nan
        train_f1 = np.nan

    # ========================================================
    # BROADCAST METRICS TO ALL RANKS
    # ========================================================

    metric_tensor = torch.tensor(
        [
            train_auc,
            train_ap,
            train_f1
        ],
        dtype=torch.float64,
        device=device
    )

    dist.broadcast(
        metric_tensor,
        src=0
    )

    train_auc = metric_tensor[0].item()
    train_ap = metric_tensor[1].item()
    train_f1 = metric_tensor[2].item()

    # ========================================================
    # RETURN
    # ========================================================

    return (
        train_loss,
        train_auc,
        train_ap,
        train_f1,
        train_metrics
    )

# ============================================================
# SAVE LAST CHECKPOINT
# ============================================================

def save_last_checkpoint(

    model,

    optimizer,

    scheduler,

    scaler,

    epoch,

    train_loss,

    val_loss,

    mean_auc,

    best_auc,

    early_stop_counter,

    checkpoint_path

):

    # --------------------------------------------------------
    # DDP model -> underlying model
    # --------------------------------------------------------

    model_state_dict = (
        model.module.state_dict()
    )


    torch.save(

        {

            "epoch":
                epoch + 1,

            "split":
                "train_val",

            "model_state_dict":
                model_state_dict,

            "optimizer_state_dict":
                optimizer.state_dict(),

            "scheduler_state_dict":
                scheduler.state_dict(),

            "scaler_state_dict":
                scaler.state_dict(),

            "train_loss":
                train_loss,

            "val_loss":
                val_loss,

            "mean_auc":
                mean_auc,

            "best_auc":
                best_auc,

            "early_stop_counter":
                early_stop_counter,

            "config": {

                "batch_size":
                    BATCH_SIZE,

                "gradient_accumulation_steps":
                    GRADIENT_ACCUMULATION_STEPS,

                "image_size":
                    IMAGE_SIZE,

                "num_slices":
                    NUM_SLICES,

                "dense_chunk_size":
                    DENSE_CHUNK_SIZE,

                "freeze_backbone":
                    FREEZE_BACKBONE,

                "amp":
                    USE_AMP,

                "num_epochs":
                    NUM_EPOCHS,

                "val_split":
                    VAL_SPLIT,

                "scheduler":
                    "ReduceLROnPlateau",

                "scheduler_factor":
                    LR_SCHEDULER_FACTOR,

                "scheduler_patience":
                    LR_SCHEDULER_PATIENCE,

                "early_stopping_patience":
                    EARLY_STOPPING_PATIENCE,

                "early_stopping_min_delta":
                    EARLY_STOPPING_MIN_DELTA

            }

        },

        checkpoint_path

    )


# ============================================================
# SAVE BEST MODEL
# ============================================================

def save_best_model(

    model,

    checkpoint_path

):

    # --------------------------------------------------------
    # ONLY MODEL WEIGHTS
    # --------------------------------------------------------

    torch.save(

        model.module.state_dict(),

        checkpoint_path

    )


# ============================================================
# LOAD CHECKPOINT
# ============================================================

def load_checkpoint(

    checkpoint_path,

    model,

    optimizer,

    scheduler,

    scaler,

    device

):

    checkpoint = torch.load(

        checkpoint_path,

        map_location=device

    )


    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model.load_state_dict(

        checkpoint[
            "model_state_dict"
        ]

    )


    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    optimizer.load_state_dict(

        checkpoint[
            "optimizer_state_dict"
        ]

    )


    # --------------------------------------------------------
    # Scheduler
    # --------------------------------------------------------

    if "scheduler_state_dict" in checkpoint:

        scheduler.load_state_dict(

            checkpoint[
                "scheduler_state_dict"
            ]

        )


    # --------------------------------------------------------
    # AMP scaler
    # --------------------------------------------------------

    scaler.load_state_dict(

        checkpoint[
            "scaler_state_dict"
        ]

    )


    start_epoch = checkpoint[
        "epoch"
    ]

    best_auc = checkpoint.get(
        "best_auc",
        checkpoint.get(
            "mean_auc",
            -np.inf
        )
    )

    early_stop_counter = checkpoint.get(
        "early_stop_counter",
        0
    )


    return (
        start_epoch,
        best_auc,
        early_stop_counter
    )


# ============================================================
# TRAIN ONE TRAIN/VALIDATION SPLIT
# ============================================================

# unfreeze only denseblock4

def train_split(

    split_info,

    rank,

    local_rank,

    world_size,

    device

):

    # ========================================================
    # DATA
    # ========================================================

    train_data = split_info["train"]
    val_data = split_info["val"]

    if rank == 0:

        print(
            "\n"
            + "=" * 70
        )

        print(
            "STARTING TRAIN / VALIDATION SPLIT"
        )

        print(
            "=" * 70
        )

        print(
            f"Train studies: "
            f"{len(train_data)}"
        )

        print(
            f"Validation studies: "
            f"{len(val_data)}"
        )

    (
        train_dataset,
        val_dataset,
        train_loader,
        val_loader,
        train_sampler,
        val_sampler

    ) = build_dataloaders(

        train_data,

        val_data,

        world_size,

        rank

    )

    # ========================================================
    # MODEL
    #
    # FREEZE_BACKBONE = False:
    #     Only DenseBlock4 is trainable.
    #
    # FREEZE_CHANNEL_ADAPTER = False:
    #     Channel adapter is trainable.
    #
    # Everything else in DenseNet remains frozen.
    # ========================================================

    model = build_model(
        device
    )

    if rank == 0:

        print(
            "\nRadImageNet loading completed."
        )

    print_model_info(
        model,
        rank
    )

    # ========================================================
    # DDP
    # ========================================================

    model = DDP(

        model,

        device_ids=[
            local_rank
        ],

        output_device=local_rank,

        find_unused_parameters=False

    )

    # ========================================================
    # OPTIMIZER
    #
    # Expected parameter groups:
    #
    #   Head              -> 1e-4
    #   Channel adapter   -> 1e-5
    #   DenseBlock4       -> 1e-5
    #
    # build_optimizer() only includes parameters where
    # requires_grad=True.
    # ========================================================

    optimizer = build_optimizer(
        model.module
    )

    # ========================================================
    # LR SCHEDULER
    #
    # Monitor validation AUROC.
    #
    # mode="max" because higher AUROC is better.
    # ========================================================

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(

        optimizer,

        mode="max",

        factor=LR_SCHEDULER_FACTOR,

        patience=LR_SCHEDULER_PATIENCE,

        threshold=EARLY_STOPPING_MIN_DELTA,

        threshold_mode="abs",

        min_lr=LR_SCHEDULER_MIN_LR

    )

    # ========================================================
    # LOSS
    # ========================================================

    criterion = nn.BCEWithLogitsLoss(
        reduction="none"
    )

    # ========================================================
    # AMP
    # ========================================================

    scaler = torch.amp.GradScaler(

        "cuda",

        enabled=USE_AMP

    )

    # ========================================================
    # SPLIT CHECKPOINT DIRECTORY
    # ========================================================

    split_dir = os.path.join(

        CHECKPOINT_DIR,

        "train_val_split"

    )

    if rank == 0:

        os.makedirs(
            split_dir,
            exist_ok=True
        )

    dist.barrier()

    last_checkpoint = os.path.join(

        split_dir,

        "last.pth"

    )

    best_checkpoint = os.path.join(

        split_dir,

        "best_model.pth"

    )

    # ========================================================
    # RESUME
    # ========================================================

    start_epoch = 0

    best_auc = -np.inf

    early_stop_counter = 0

    if os.path.exists(
        last_checkpoint
    ):

        if rank == 0:

            print(
                "\nResuming training from:"
            )

            print(
                last_checkpoint
            )

        (
            start_epoch,
            best_auc,
            early_stop_counter

        ) = load_checkpoint(

            last_checkpoint,

            model.module,

            optimizer,

            scheduler,

            scaler,

            device

        )

        if rank == 0:

            print(
                f"Resume epoch: "
                f"{start_epoch}"
            )

            print(
                f"Best validation AUROC: "
                f"{best_auc:.6f}"
            )

    else:

        if rank == 0:

            print(
                "\nStarting fold from "
                "RadImageNet weights."
            )

    # ========================================================
    # PRINT INITIAL LR GROUPS
    # ========================================================

    if rank == 0:

        print(
            "\nOptimizer parameter groups:"
        )

        for idx, group in enumerate(
            optimizer.param_groups
        ):

            print(
                f"  Group {idx}: "
                f"lr={group['lr']:.8g}, "
                f"params={len(group['params'])}"
            )

    # ========================================================
    # SPLIT TIMER
    # ========================================================

    split_start_time = time.time()

    # ========================================================
    # EPOCH LOOP
    # ========================================================

    for epoch in range(

        start_epoch,

        NUM_EPOCHS

    ):

        epoch_start = time.time()

        if torch.cuda.is_available():

            torch.cuda.reset_peak_memory_stats(
                device
            )

        # ====================================================
        # TRAIN
        # ====================================================

        (
            train_loss,
            train_auc,
            train_ap,
            train_f1,
            train_metrics

        ) = train_one_epoch(

            model=model,

            train_loader=train_loader,

            train_sampler=train_sampler,

            optimizer=optimizer,

            criterion=criterion,

            scaler=scaler,

            device=device,

            epoch=epoch,

            rank=rank,

            local_rank=local_rank

        )

        if torch.cuda.is_available():

            torch.cuda.synchronize(
                device
            )

            torch.cuda.empty_cache()

        # ====================================================
        # VALIDATION
        # ====================================================

        (
            val_loss,
            mean_auc,
            val_ap,
            val_f1,
            val_metrics

        ) = validate(

            model=model,

            val_loader=val_loader,

            criterion=criterion,

            device=device,

            rank=rank,

            epoch=epoch,

            save_dir=SAVE_VAL_GRAPH_DIR

        )

        # ====================================================
        # CHECK FOR NON-FINITE METRICS
        # ====================================================

        if rank == 0:

            if not np.isfinite(val_loss):

                print(
                    "\nWARNING: "
                    f"Validation loss is not finite: "
                    f"{val_loss}"
                )

            if not np.isfinite(mean_auc):

                print(
                    "\nWARNING: "
                    f"Validation AUROC is not finite: "
                    f"{mean_auc}"
                )

        # ====================================================
        # RANK 0 LOGGING
        # ====================================================

        if rank == 0:

            epoch_time = (
                time.time()
                - epoch_start
            )

            print(
                "\n"
                + "-" * 70
            )

            print(
                "Train/Val | "
                f"Epoch {epoch + 1}/{NUM_EPOCHS}"
            )

            print(
                f"Train Loss: "
                f"{train_loss:.6f}"
            )

            print(
                f"Train AUROC: "
                f"{train_auc:.6f}"
            )

            print(
                f"Train AP: "
                f"{train_ap:.6f}"
            )

            print(
                f"Train F1: "
                f"{train_f1:.6f}"
            )

            print(
                f"Val Loss: "
                f"{val_loss:.6f}"
            )

            print(
                f"Mean Val AUROC: "
                f"{mean_auc:.6f}"
            )

            print(
                f"Mean Val AP: "
                f"{val_metrics['mean_AP']:.6f}"
            )

            print(
                f"Mean Val F1: "
                f"{val_metrics['mean_F1']:.6f}"
            )

            print(
                f"Epoch time: "
                f"{epoch_time / 60:.2f} min"
            )

        # ====================================================
        # BEST MODEL / EARLY STOPPING
        # ====================================================

        is_best = False
        should_stop = False

        if rank == 0:

            if (

                np.isfinite(mean_auc)

                and

                mean_auc > (
                    best_auc
                    + EARLY_STOPPING_MIN_DELTA
                )

            ):

                best_auc = mean_auc

                early_stop_counter = 0

                is_best = True

                save_best_model(

                    model,

                    best_checkpoint

                )

                print(
                    "\nBEST MODEL UPDATED:"
                )

                print(
                    best_checkpoint
                )

                print(
                    f"Best AUROC: "
                    f"{best_auc:.6f}"
                )

            else:

                early_stop_counter += 1

            # ------------------------------------------------
            # Early stopping
            # ------------------------------------------------

            if (

                early_stop_counter
                >= EARLY_STOPPING_PATIENCE

            ):

                should_stop = True

                print(
                    "\nEARLY STOPPING TRIGGERED"
                )

                print(
                    "No validation AUROC improvement "
                    f"for {early_stop_counter} epochs."
                )

        # ====================================================
        # BROADCAST CONTROL STATE
        # ====================================================

        control_tensor = torch.tensor(

            [
                float(best_auc),

                float(early_stop_counter),

                float(should_stop)

            ],

            dtype=torch.float64,

            device=device

        )

        dist.broadcast(

            control_tensor,

            src=0

        )

        best_auc = control_tensor[0].item()

        early_stop_counter = int(
            control_tensor[1].item()
        )

        should_stop = bool(
            int(control_tensor[2].item())
        )

        # ====================================================
        # LR SCHEDULER
        #
        # IMPORTANT:
        #
        # scheduler.step() MUST receive validation AUROC.
        #
        # mode="max" means:
        #
        #     higher AUROC = better
        #
        # ====================================================

        if np.isfinite(mean_auc):

            scheduler.step(
                mean_auc
            )

        # ====================================================
        # PRINT ALL LEARNING RATES
        # ====================================================

        if rank == 0:

            print(
                "\nLearning rates:"
            )

            for idx, group in enumerate(
                optimizer.param_groups
            ):

                print(
                    f"  Group {idx}: "
                    f"{group['lr']:.8g}"
                )

        # ====================================================
        # SAVE LAST CHECKPOINT
        #
        # Scheduler has already been stepped, so the checkpoint
        # contains the current scheduler state.
        # ====================================================

        if rank == 0:

            save_last_checkpoint(

                model,

                optimizer,

                scheduler,

                scaler,

                epoch,

                train_loss,

                val_loss,

                mean_auc,

                best_auc,

                early_stop_counter,

                last_checkpoint

            )

            print(
                "\nLast checkpoint saved:"
            )

            print(
                last_checkpoint
            )

        # ====================================================
        # CHECKPOINT BARRIER
        # ====================================================

        dist.barrier()

        # ====================================================
        # PEAK GPU MEMORY
        # ====================================================

        if torch.cuda.is_available():

            torch.cuda.synchronize(
                device
            )

            peak_memory = (

                torch.cuda
                .max_memory_allocated(
                    device
                )
                / 1024**3

            )

            reserved_memory = (

                torch.cuda
                .max_memory_reserved(
                    device
                )
                / 1024**3

            )

            current_allocated = (

                torch.cuda
                .memory_allocated(
                    device
                )
                / 1024**3

            )

            current_reserved = (

                torch.cuda
                .memory_reserved(
                    device
                )
                / 1024**3

            )

            if rank == 0:

                print(

                    f"GPU {local_rank} | "

                    f"Current Allocated: "
                    f"{current_allocated:.2f} GB | "

                    f"Current Reserved: "
                    f"{current_reserved:.2f} GB | "

                    f"Peak Allocated: "
                    f"{peak_memory:.2f} GB | "

                    f"Peak Reserved: "
                    f"{reserved_memory:.2f} GB"

                )

        # ====================================================
        # EARLY STOP
        # ====================================================

        if should_stop:

            break

    # ========================================================
    # TRAINING SPLIT COMPLETE
    # ========================================================

    split_time = (

        time.time()
        - split_start_time

    )

    if rank == 0:

        print(
            "\n"
            + "=" * 70
        )

        print(
            "TRAIN / VALIDATION SPLIT COMPLETED"
        )

        print(
            "=" * 70
        )

        print(
            f"Best validation AUROC: "
            f"{best_auc:.6f}"
        )

        print(
            f"Split time: "
            f"{split_time / 60:.2f} minutes"
        )

        print(
            "Best model:"
        )

        print(
            best_checkpoint
        )

    return best_auc



# ========================================================
# Helper funcation
# ========================================================

def get_rank():
    return int(os.environ.get("RANK", 0))


def broadcast_object(obj, src=0):
    """
    Broadcast a Python object from src rank to all DDP ranks.
    """

    objects = [
        obj if get_rank() == src else None
    ]

    dist.broadcast_object_list(
        objects,
        src=src
    )

    return objects[0]

# ============================================================
# MAIN
# ============================================================
def main():

    # ========================================================
    # INPUT FILE
    # ========================================================

    train_df = pd.read_csv(
        "/kaggle/input/datasets/nitesh8527/without-nan/new_full_train_data.csv"
    )

    # ========================================================
    # PREPROCESSED CACHE DIRECTORIES
    # ========================================================

    CACHE_DIRS = [
        "/kaggle/input/datasets/nitesh8527/rsna-knee-preprocessed-data-224/preprocessed_data_224_1",
        "/kaggle/input/datasets/nitesh8527/rsna-knee-preprocessed-data-224/preprocessed_data_224_2",
        "/kaggle/input/datasets/nitesh8527/rsna-knee-preprocessed-data-224/preprocessed_data_224_3",
        "/kaggle/input/datasets/nitesh8527/rsna-knee-preprocessed-data-224/preprocessed_data_224_4",
    #     "/kaggle/input/datasets/nitesh8527/processed-rsna-knee-data/32_slices_rsna_preprocessed_cache_1/rsna_preprocessed_cache_1",
    #     "/kaggle/input/datasets/nitesh8527/processed-rsna-knee-data/32_slices_rsna_preprocessed_cache_2/rsna_preprocessed_cache_2"
        
    ]

    # ========================================================
    # DDP SETUP
    # ========================================================

    (
        rank,
        local_rank,
        world_size,
        device
    ) = setup_ddp()

    try:

        # ====================================================
        # SEED
        # ====================================================

        set_seed(RANDOM_SEED)

        # ====================================================
        # GPU INFO
        # ====================================================

        print_gpu_info(
            rank,
            local_rank,
            world_size
        )

        # ====================================================
        # VERIFY GPU COUNT
        # ====================================================

        if rank == 0:

            print(
                "\n"
                + "=" * 70
            )

            print(
                "TRAIN / VALIDATION DDP TRAINING"
            )

            print(
                "=" * 70
            )

            print(
                f"World size: "
                f"{world_size}"
            )

            print(
                f"Expected GPUs: "
                f"{N_GPUS}"
            )

            print(
                f"Epochs: "
                f"{NUM_EPOCHS}"
            )

            print(
                f"Batch / GPU: "
                f"{BATCH_SIZE}"
            )

            print(
                f"Gradient accumulation: "
                f"{GRADIENT_ACCUMULATION_STEPS}"
            )

            print(
                f"Effective batch size: "
                f"{BATCH_SIZE * world_size * GRADIENT_ACCUMULATION_STEPS}"
            )

            print(
                f"Image size: "
                f"{IMAGE_SIZE}"
            )

            print(
                f"Slices: "
                f"{NUM_SLICES}"
            )

            print(
                f"Dense chunk size: "
                f"{DENSE_CHUNK_SIZE}"
            )

            print(
                f"Validation split: "
                f"{VAL_SPLIT:.0%}"
            )

            print(
                f"Early stopping patience: "
                f"{EARLY_STOPPING_PATIENCE}"
            )

            print(
                "=" * 70
            )

        if world_size != N_GPUS:

            raise RuntimeError(
                f"Expected {N_GPUS} GPUs "
                f"but WORLD_SIZE={world_size}"
            )

        # ====================================================
        # BUILD DATASET + SPLIT ON ALL RANKS
        #
        # Both ranks run this independently and locally.
        # Since the random seed and dataframes are identical,
        # the resulting datasets and splits will match.
        # ====================================================

        print(
            f"\n[Rank {rank}] Building study-level dataset "
            "from preprocessed cache..."
        )

        data = build_dataset(
            train_df=train_df,
            cache_dirs=CACHE_DIRS
        )

        # ------------------------------------------------
        # Validate dataset size
        # ------------------------------------------------

        if len(data) < 2:

            raise ValueError(
                f"[Rank {rank}] Need at least 2 studies for "
                f"train/validation split, "
                f"but build_dataset returned "
                f"{len(data)}."
            )

        # ------------------------------------------------
        # Dataset information (Printed only on rank 0)
        # ------------------------------------------------

        if rank == 0:

            print(
                f"\nComplete study-level dataset: "
                f"{len(data)} studies"
            )

            data_gold = sum(
                x["label_source"] == "gold"
                for x in data
            )

            data_llm = sum(
                x["label_source"] == "llm"
                for x in data
            )

            print(
                f"Gold: {data_gold}"
            )

            print(
                f"LLM: {data_llm}"
            )

            print(
                f"Cache directories: "
                f"{len(CACHE_DIRS)}"
            )

        # ------------------------------------------------
        # Create ONE global train/validation split
        # ------------------------------------------------

        split = create_train_val_split(
            dataset=data,
            test_size=VAL_SPLIT,
            random_state=RANDOM_SEED
        )

        # ------------------------------------------------
        # Print split distribution (Printed only on rank 0)
        # ------------------------------------------------

        if rank == 0:
            print_split_distribution(
                split
            )

        # ====================================================
        # SYNCHRONIZE ALL RANKS
        # ====================================================
        dist.barrier()

        # ====================================================
        # VERIFY DATA ON EACH RANK
        # ====================================================

        print(
            f"[Rank {rank}] "
            f"Dataset={len(data)} | "
            f"Train={len(split['train'])} | "
            f"Val={len(split['val'])}"
        )

        # ====================================================
        # VERIFY DATA STRUCTURE (Rank 0 only)
        # ====================================================

        if rank == 0:

            sample = data[0]

            required_keys = {
                "study_id",
                "pt_path",
                "label_source",
                "labels"
            }

            missing_keys = (
                required_keys
                - set(sample.keys())
            )

            if missing_keys:

                raise RuntimeError(
                    f"Dataset sample is missing keys: "
                    f"{missing_keys}\n"
                    f"Available keys: "
                    f"{list(sample.keys())}"
                )

            print(
                "\nDataset structure verified."
            )

            print(
                f"First study: "
                f"{sample['study_id']}"
            )

            print(
                f"First label source: "
                f"{sample['label_source']}"
            )

            print(
                f"Number of labels: "
                f"{len(sample['labels'])}"
            )

        # ====================================================
        # SYNCHRONIZE BEFORE TRAINING
        # ====================================================

        dist.barrier()

        # ====================================================
        # TRAIN ONE TRAIN/VALIDATION SPLIT
        #
        # train_split() will create:
        #
        #   train_dataset
        #   val_dataset
        #   DistributedSampler
        #   DataLoader
        #
        # and train both GPUs using DDP.
        # ====================================================

        overall_start = time.time()

        best_auc = train_split(
            split_info=split,
            rank=rank,
            local_rank=local_rank,
            world_size=world_size,
            device=device
        )

        # ====================================================
        # SYNCHRONIZE AFTER TRAINING
        # ====================================================

        dist.barrier()

        # ====================================================
        # CLEAN CUDA MEMORY
        # ====================================================

        torch.cuda.empty_cache()

        # ====================================================
        # FINAL RESULTS
        # ====================================================

        if rank == 0:

            total_time = (
                time.time()
                - overall_start
            )

            print(
                "\n"
                + "=" * 70
            )

            print(
                "TRAINING COMPLETED"
            )

            print(
                "=" * 70
            )

            print(
                f"Best validation AUROC: "
                f"{best_auc:.6f}"
            )

            print(
                f"Total training time: "
                f"{total_time / 3600:.2f} hours"
            )

            split_dir = os.path.join(
                CHECKPOINT_DIR,
                "train_val_split"
            )

            print(
                "\nBest model:"
            )

            print(
                os.path.join(
                    split_dir,
                    "best_model.pth"
                )
            )

            print(
                "\nLast checkpoint:"
            )

            print(
                os.path.join(
                    split_dir,
                    "last.pth"
                )
            )

            print(
                "\n"
                + "=" * 70
            )

    finally:

        cleanup_ddp()


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()

    # raise RuntimeError(
    #     "Call main(train_df, train_series_df, train_series_root) "
    #     "after loading the raw RSNA training dataframes. "
    #     "The old folds file is intentionally not used."
    # )

    

# for running this in kaggle 

# %%time

# !TORCH_DISTRIBUTED_DEBUG=DETAIL torchrun \
#     --standalone \
#     --nproc_per_node=2 \
#     /kaggle/working/new_32_slices_training.py