%%time

import os

# 1. CRITICAL: Enforce strict single-threading for underlying C/C++ libraries
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = "1"

import gc
import cv2
import torch
import pandas as pd
import numpy as np
import SimpleITK as sitk
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter1d
from joblib import Parallel, delayed
from tqdm import tqdm

# Disable OpenCV multi-threading internally
cv2.setNumThreads(0)
torch.set_num_threads(1)


def get_informative_slice_range_fast(
    volume: np.ndarray,
    percentile: float = 20,
    smooth_sigma: float = 2.0,
    margin: float = 0.05,
    min_region_fraction: float = 0.20,
) -> tuple[int, int]:

    n_slices = volume.shape[0]

    if n_slices <= 4:
        return 0, n_slices

    # ---------------------------------------------------------
    # 1. Downsample only for slice detection
    # ---------------------------------------------------------
    small_h = 64
    small_w = 64

    volume_small = np.empty(
        (n_slices, small_h, small_w),
        dtype=np.float32
    )

    for i in range(n_slices):
        volume_small[i] = cv2.resize(
            volume[i],
            (small_w, small_h),
            interpolation=cv2.INTER_AREA
        )

    # ---------------------------------------------------------
    # 2. Normalize each slice robustly
    # ---------------------------------------------------------
    p1 = np.percentile(volume_small, 1, axis=(1, 2))
    p99 = np.percentile(volume_small, 99, axis=(1, 2))

    contrast = p99 - p1

    # ---------------------------------------------------------
    # 3. Foreground / anatomical-content score
    # ---------------------------------------------------------
    # Estimate background from border pixels
    border = np.concatenate([
        volume_small[:, 0, :],
        volume_small[:, -1, :],
        volume_small[:, :, 0],
        volume_small[:, :, -1],
    ], axis=1)

    background = np.median(border, axis=1)

    # Difference from border/background
    foreground_score = np.mean(
        np.abs(
            volume_small -
            background[:, None, None]
        ),
        axis=(1, 2)
    )

    # ---------------------------------------------------------
    # 4. Variance / texture score
    # ---------------------------------------------------------
    variance_score = volume_small.var(axis=(1, 2))

    # ---------------------------------------------------------
    # 5. Edge/content score
    # ---------------------------------------------------------
    edge_score = np.zeros(n_slices, dtype=np.float32)

    for i in range(n_slices):
        gx = cv2.Sobel(
            volume_small[i],
            cv2.CV_32F,
            1,
            0,
            ksize=3
        )

        gy = cv2.Sobel(
            volume_small[i],
            cv2.CV_32F,
            0,
            1,
            ksize=3
        )

        edge_score[i] = np.mean(
            np.sqrt(gx * gx + gy * gy)
        )

    # ---------------------------------------------------------
    # 6. Normalize each score across slices
    # ---------------------------------------------------------
    def normalize(x):
        x = np.asarray(x, dtype=np.float32)

        lo = np.percentile(x, 5)
        hi = np.percentile(x, 95)

        if hi <= lo:
            return np.zeros_like(x)

        return np.clip(
            (x - lo) / (hi - lo),
            0.0,
            1.0
        )

    foreground_score = normalize(foreground_score)
    variance_score = normalize(variance_score)
    edge_score = normalize(edge_score)
    contrast = normalize(contrast)

    # ---------------------------------------------------------
    # 7. Combined anatomical-content score
    # ---------------------------------------------------------
    score = (
        0.40 * foreground_score +
        0.25 * variance_score +
        0.20 * edge_score +
        0.15 * contrast
    )

    # Smooth across neighboring slices
    score = gaussian_filter1d(
        score,
        sigma=smooth_sigma
    )

    # ---------------------------------------------------------
    # 8. Determine anatomical region
    # ---------------------------------------------------------
    threshold = np.percentile(score, percentile)

    valid = score > threshold

    # ---------------------------------------------------------
    # 9. Find contiguous regions
    # ---------------------------------------------------------
    regions = []

    start = None

    for i, is_valid in enumerate(valid):

        if is_valid and start is None:
            start = i

        elif not is_valid and start is not None:
            regions.append((start, i))
            start = None

    if start is not None:
        regions.append((start, n_slices))

    if not regions:
        return 0, n_slices

    # ---------------------------------------------------------
    # 10. Select largest meaningful anatomical region
    # ---------------------------------------------------------
    min_region = max(
        3,
        int(n_slices * min_region_fraction)
    )

    valid_regions = [
        region
        for region in regions
        if (region[1] - region[0]) >= min_region
    ]

    if not valid_regions:
        valid_regions = regions

    # Score regions by both length and average signal
    best_region = max(
        valid_regions,
        key=lambda r: (
            (r[1] - r[0]) *
            np.mean(score[r[0]:r[1]])
        )
    )

    first, last = best_region

    # ---------------------------------------------------------
    # 11. Add margin so we don't cut the anatomy
    # ---------------------------------------------------------
    margin_slices = max(
        1,
        int(n_slices * margin)
    )

    first = max(
        0,
        first - margin_slices
    )

    last = min(
        n_slices,
        last + margin_slices
    )

    # Safety check
    if last - first < 4:
        return 0, n_slices

    return first, last


def process_single_study(args):
    sitk.ProcessObject_SetGlobalWarningDisplay(False)
    
    study_uid, study_series_rows, train_series_root, cache_dir, label_source, labels_dict, num_slices, spatial_size = args
    
    cache_path = os.path.join(cache_dir, f"{study_uid}.pt")
    if os.path.exists(cache_path):
        return {"study_id": study_uid, "label_source": label_source, "cache_path": cache_path}

    plane_map = {"Sagittal": 0, "Coronal": 1, "Axial": 2}
    volumes, planes, fluids, fats = [], [], [], []

    for _, row in study_series_rows.iterrows():
        series_uid = row["SeriesInstanceUID"]
        series_path = os.path.join(train_series_root, study_uid, series_uid)

        if not os.path.exists(series_path):
            continue

        try:
            dicom_files = list(sitk.ImageSeriesReader.GetGDCMSeriesFileNames(series_path))
        except Exception:
            dicom_files = []

        if not dicom_files:
            continue

        reader = sitk.ImageSeriesReader()
        reader.SetFileNames(dicom_files)
        try:
            image = reader.Execute()
        except Exception:
            continue
            
        volume = sitk.GetArrayFromImage(image).astype(np.float32)
        if volume.shape[0] == 0:
            continue

        # 1. Fast Informative Trim
        first, last = get_informative_slice_range_fast(
                volume,
                percentile=20,
                smooth_sigma=2.0,
                margin=0.08,
                min_region_fraction=0.20

        )
        volume = volume[first:last]

        # 2. Global Intensity Normalization
        p1, p99 = np.percentile(volume, (1, 99))
        volume = np.clip(volume, p1, p99)
        v_min, v_max = volume.min(), volume.max()
        if v_max > v_min:
            volume = (volume - v_min) / (v_max - v_min)
        else:
            volume = np.zeros_like(volume, dtype=np.float32)

        # 3. Resample via Trilinear Interpolation
        tensor_vol = torch.from_numpy(volume).unsqueeze(0).unsqueeze(0)
        resized_tensor = F.interpolate(
            tensor_vol,
            size=(num_slices, spatial_size[0], spatial_size[1]),
            mode="trilinear",
            align_corners=False
        )

        volumes.append(resized_tensor.squeeze(0).to(torch.float16))
        planes.append(plane_map.get(row["Anatomical_Plane"], 0))
        fluids.append(int(row["Fluid_Sensitive"]))
        fats.append(int(row["Fat_Suppression"]))

    if not volumes:
        return None

    payload = {
        "study_id": study_uid,
        "volumes": torch.stack(volumes, dim=0),
        "plane": torch.tensor(planes, dtype=torch.long),
        "fluid": torch.tensor(fluids, dtype=torch.long),
        "fat": torch.tensor(fats, dtype=torch.long),
        "label_source": label_source,
        "labels": torch.tensor(list(labels_dict.values()), dtype=torch.float32)
    }

    torch.save(payload, cache_path)
    
    # Clean memory explicitly
    del volumes, payload, volume, tensor_vol, resized_tensor
    gc.collect()

    return {"study_id": study_uid, "label_source": label_source, "cache_path": cache_path}


def generate_and_cache_dataset(
    train_df: pd.DataFrame,
    train_series_df: pd.DataFrame,
    train_series_root: str,
    cache_dir: str = "/tmp/rsna_preprocessed_cache",
    num_slices: int = 16,
    spatial_size: tuple = (224, 224),
    n_jobs: int = 2  # Set to 2 to guarantee memory stability on standard Kaggle instances
):
    os.makedirs(cache_dir, exist_ok=True)
    
    label_columns = [
        "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus",
        "Medial OA", "Lateral OA", "PF OA", "Effusion",
        "Synovitis", "Baker's", "Contusion", "Fracture",
    ]
    
    grouped_series = train_series_df.groupby("StudyInstanceUID")
    tasks = []

    for row_index, (_, study_row) in enumerate(train_df.iterrows()):
        study_uid = study_row["StudyInstanceUID"]
        if study_uid not in grouped_series.groups:
            continue

        label_source = "gold" if row_index < 58 else "llm"
        labels_dict = {col: float(study_row[col]) for col in label_columns}
        study_series_rows = grouped_series.get_group(study_uid)

        tasks.append((
            study_uid, study_series_rows, train_series_root, 
            cache_dir, label_source, labels_dict, num_slices, spatial_size
        ))

    print(f"Starting parallel execution across {len(tasks)} studies using {n_jobs} workers...")

    results = Parallel(n_jobs=n_jobs, backend="loky", max_nbytes=None)(
        delayed(process_single_study)(task) for task in tqdm(tasks)
    )

    dataset_records = [r for r in results if r is not None]
    print(f"Preprocessing complete! All files cached at: {cache_dir}")
    return dataset_records

# --- EXECUTE ---
train_df = pd.read_csv("/kaggle/input/datasets/nitesh8527/without-nan/new_full_train_data.csv")
train_series_df = pd.read_csv("/kaggle/input/competitions/rsna-knee-abnormality-detection/train_series.csv")
train_series_root = "/kaggle/input/competitions/rsna-knee-abnormality-detection/train_series"

all_records = generate_and_cache_dataset(
    train_df=train_df.iloc[3305:4407],
    train_series_df=train_series_df,
    train_series_root=train_series_root,
    cache_dir="/tmp/preprocessed_data_224_4",  # Stored safely in scratch space
    num_slices=32,
    n_jobs=2
)