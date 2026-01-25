import h5py
import numpy as np
import torch
from functools import partial
from torch.utils.data import Dataset, DataLoader


def pad_collate(batch, pad_value: float = -80.0):
    """
    Collate function to pad variable-length log-mel sequences to the max time in batch.
    Returns X shape (B, 1, Tmax, F), y tensor or None, lengths tensor.
    """
    xs, ys, lens = zip(*batch)
    B = len(xs)
    Tmax = max(int(x.shape[1]) for x in xs)
    F = xs[0].shape[2]

    X = xs[0].new_full((B, 1, Tmax, F), fill_value=pad_value)
    for i, x in enumerate(xs):
        T = x.shape[1]
        X[i, :, :T, :] = x

    y = None
    if ys[0] is not None:
        y = torch.stack(ys, dim=0)

    lengths = torch.stack(lens, dim=0)
    return X, y, lengths


class LMDataset(Dataset):
    """HDF5 log-mel dataset supporting (N, F, T) or (N, T, F) layouts with optional labels/lengths."""

    def __init__(
        self,
        h5_path: str,
        split: str = "train",
        feature_key: str = "logmel",
        label_key: str = "label",
        length_key: str = "length",
        time_dim: str = "F",  # "T" if feature dataset is (N, T, F); "F" if (N, F, T)
        dtype=np.float32,
        label_map=None,
    ):
        super().__init__()
        self.h5_path = h5_path
        self.split = split
        self.feature_key = feature_key
        self.label_key = label_key
        self.length_key = length_key
        self.time_dim = time_dim
        self.dtype = dtype
        self.label_map = label_map

        with h5py.File(self.h5_path, "r") as f:
            grp = f[self.split]
            self.N = grp[self.feature_key].shape[0]
            self.has_labels = self.label_key in grp
            self.has_lengths = self.length_key in grp

        self._h5 = None
        self._grp = None
        self._X = None
        self._Y = None
        self._L = None

    def __len__(self):
        return self.N

    def _ensure_open(self):
        if self._h5 is None:
            self._h5 = h5py.File(self.h5_path, "r", swmr=True, libver="latest")
            self._grp = self._h5[self.split]
            self._X = self._grp[self.feature_key]
            self._Y = self._grp[self.label_key] if self.has_labels else None
            self._L = self._grp[self.length_key] if self.has_lengths else None

    def __getitem__(self, idx):
        self._ensure_open()
        X = np.array(self._X[idx], dtype=self.dtype)
        if X.ndim != 2:
            raise ValueError(f"{self.split}/{self.feature_key} must be 2D per item; got {X.shape}")

        if self.time_dim == "F":  # data stored (F, T)
            X = X.T  # -> (T, F)

        X = torch.from_numpy(X[None, ...])  # (1, T, F) for Conv2D

        y_raw = self._Y[idx] if self.has_labels else None
        if y_raw is not None and self.label_map is not None:
            y_val = self.label_map[y_raw]
        else:
            y_val = y_raw
        y = torch.tensor(y_val, dtype=torch.long) if y_val is not None else None

        t_len = int(self._L[idx]) if self.has_lengths else X.shape[1]
        t_len = torch.tensor(t_len, dtype=torch.long)
        return X, y, t_len


def build_h5_loaders(
    h5_path: str,
    splits=("train", "val", "test"),
    feature_key: str = "logmel",
    label_key: str = "label",
    length_key: str = "length",
    time_dim: str = "F",
    batch_sizes=None,
    num_workers: int = 0,
    pad_value: float = -80.0,
    pin_memory: bool = True,
    persistent_workers: bool = True,
    shuffle_train: bool = True,
    remap_labels: bool = True,
    prefetch_factor: int = 2,
):
    """Build DataLoaders for available splits in the HDF5 file."""
    loaders = {}

    if batch_sizes is None:
        batch_sizes = {s: 32 for s in splits}

    label_map = None
    if remap_labels:
        all_labels = set()
        with h5py.File(h5_path, "r") as f:
            for s in f.keys():
                if s in ["train", "val", "test"] and label_key in f[s]:
                    all_labels.update(f[s][label_key][:])
        sorted_labels = sorted(list(all_labels))
        label_map = {old: new for new, old in enumerate(sorted_labels)}
        print(f"Remapping {len(label_map)} labels to dense range 0..{len(label_map)-1}")

    with h5py.File(h5_path, "r") as f:
        available = {k for k in f.keys()}

    for split in splits:
        if split not in available:
            continue

        dataset = LMDataset(
            h5_path=h5_path,
            split=split,
            feature_key=feature_key,
            label_key=label_key,
            length_key=length_key,
            time_dim=time_dim,
            label_map=label_map,
        )

        dataset_loader = DataLoader(
            dataset,
            batch_size=batch_sizes.get(split, 32),
            shuffle=(shuffle_train and split == "train"),
            collate_fn=partial(pad_collate, pad_value=pad_value),
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers if num_workers > 0 else False,
            # prefetch_factor is only valid when num_workers > 0
            prefetch_factor=prefetch_factor if num_workers > 0 else None,
        )
        loaders[split] = dataset_loader

    return loaders
