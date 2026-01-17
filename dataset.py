from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import StratifiedKFold


NUC_TO_ID: Dict[str, int] = {"A": 0, "C": 1, "G": 2, "T": 3, "N": 4}
ID_TO_COMP: Dict[int, int] = {0: 3, 1: 2, 2: 1, 3: 0, 4: 4}


def tokenize_sequence(seq: str, seq_len: int) -> np.ndarray:
    seq = (seq or "").upper()
    arr = np.frombuffer(seq.encode("ascii", errors="ignore"), dtype=np.uint8)
    # Map ASCII to tokens quickly via lookup table
    table = np.full(256, NUC_TO_ID["N"], dtype=np.int64)
    table[ord("A")] = NUC_TO_ID["A"]
    table[ord("C")] = NUC_TO_ID["C"]
    table[ord("G")] = NUC_TO_ID["G"]
    table[ord("T")] = NUC_TO_ID["T"]
    table[ord("N")] = NUC_TO_ID["N"]
    toks = table[arr]

    if len(toks) >= seq_len:
        return toks[:seq_len]
    out = np.full(seq_len, NUC_TO_ID["N"], dtype=np.int64)
    out[: len(toks)] = toks
    return out


def reverse_complement_tokens(tokens: np.ndarray) -> np.ndarray:
    comp = np.vectorize(ID_TO_COMP.get, otypes=[np.int64])(tokens)
    return comp[::-1]


def zscore_per_species(df: pd.DataFrame, species_col: str, value_col: str) -> pd.Series:
    grouped = df.groupby(species_col)[value_col]
    mean = grouped.transform("mean")
    std = grouped.transform("std").replace(0.0, 1.0)
    return (df[value_col] - mean) / std


@dataclass
class SplitData:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    species_to_id: Dict[str, int]


def _resolve_columns(df: pd.DataFrame) -> Tuple[str, str, str]:
    """Resolve column names for sequence, species, and expression value.

    Returns tuple: (seq_col, species_col, value_col)
    Accepts common aliases for the expression column (case-insensitive).
    """
    col_map = {c.lower(): c for c in df.columns}
    # required core columns
    for req in ["sequence", "species"]:
        if req not in col_map:
            raise ValueError(f"CSV 缺少必要列: {req}. 实际列: {list(df.columns)}")

    seq_col = col_map["sequence"]
    species_col = col_map["species"]

    expr_candidates = [
        "tpm",
        "tpm_max",
        "expr",
        "expression",
        "value",
        "fpkm",
        "rpkm",
    ]
    value_col = None
    for cand in expr_candidates:
        if cand in col_map:
            value_col = col_map[cand]
            break
    if value_col is None:
        raise ValueError(
            "CSV 未找到表达量列，请包含 'tpm' 或 'tpm_max'（或常见别名 expr/expression）。当前列: "
            + str(list(df.columns))
        )
    return seq_col, species_col, value_col


def load_and_prepare_csv(
    csv_path: str,
    seq_len: int,
    holdout_species: Optional[str] = None,
    split_ratio: Tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: int = 42,
) -> SplitData:
    df = pd.read_csv(csv_path)
    # Resolve columns with fallbacks
    seq_col, species_col, value_col = _resolve_columns(df)

    # log1p(tpm)
    # ensure numeric and non-negative before log1p
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
    df = df[df[value_col].notna()]
    df[value_col] = df[value_col].clip(lower=0)
    df["_y_log1p"] = np.log1p(df[value_col].astype(float))

    # Global z-score per species (避免数据泄漏到特征；这里仅作用于目标)
    tmp = df[[species_col, "_y_log1p"]].copy()
    tmp.rename(columns={species_col: "_sp"}, inplace=True)
    df["y"] = zscore_per_species(tmp, "_sp", "_y_log1p")

    # Species mapping
    all_species = sorted(df[species_col].astype(str).unique())
    species_to_id = {s: i for i, s in enumerate(all_species)}
    df["species_id"] = df[species_col].astype(str).map(species_to_id)

    # Split
    rng = np.random.RandomState(seed)
    if holdout_species is not None:
        holdout_mask = df[species_col].astype(str) == str(holdout_species)
        df_holdout = df.loc[holdout_mask].reset_index(drop=True)
        df_rest = df.loc[~holdout_mask].reset_index(drop=True)
        # split rest
        try:
            train_df, temp_df = train_test_split(
                df_rest, test_size=1 - split_ratio[0], random_state=seed, shuffle=True, stratify=df_rest["species_id"]
            )
        except ValueError:
            train_df, temp_df = train_test_split(
                df_rest, test_size=1 - split_ratio[0], random_state=seed, shuffle=True
            )
        val_rel = split_ratio[1] / (split_ratio[1] + split_ratio[2])
        try:
            val_df, test_df = train_test_split(
                temp_df, test_size=1 - val_rel, random_state=seed, shuffle=True, stratify=temp_df["species_id"]
            )
        except ValueError:
            val_df, test_df = train_test_split(
                temp_df, test_size=1 - val_rel, random_state=seed, shuffle=True
            )
        # holdout split equally for val/test
        if not df_holdout.empty:
            val_h, test_h = train_test_split(
                df_holdout, test_size=0.5, random_state=seed, shuffle=True
            )
            val_df = pd.concat([val_df, val_h], ignore_index=True)
            test_df = pd.concat([test_df, test_h], ignore_index=True)
    else:
        try:
            train_df, temp_df = train_test_split(
                df, test_size=1 - split_ratio[0], random_state=seed, shuffle=True, stratify=df["species_id"]
            )
        except ValueError:
            train_df, temp_df = train_test_split(
                df, test_size=1 - split_ratio[0], random_state=seed, shuffle=True
            )
        val_rel = split_ratio[1] / (split_ratio[1] + split_ratio[2])
        try:
            val_df, test_df = train_test_split(
                temp_df, test_size=1 - val_rel, random_state=seed, shuffle=True, stratify=temp_df["species_id"]
            )
        except ValueError:
            val_df, test_df = train_test_split(
                temp_df, test_size=1 - val_rel, random_state=seed, shuffle=True
            )

    # Keep only necessary columns
    keep_cols = [seq_col, "y", "species_id"]
    train_df = train_df[keep_cols].rename(columns={seq_col: "sequence"}).reset_index(drop=True)
    val_df = val_df[keep_cols].rename(columns={seq_col: "sequence"}).reset_index(drop=True)
    test_df = test_df[keep_cols].rename(columns={seq_col: "sequence"}).reset_index(drop=True)

    # Attach seq_len for downstream
    train_df["seq_len"] = seq_len
    val_df["seq_len"] = seq_len
    test_df["seq_len"] = seq_len

    return SplitData(train=train_df, val=val_df, test=test_df, species_to_id=species_to_id)


class SequenceDataset(Dataset):
    def __init__(
        self,
        frame: pd.DataFrame,
        split: str,
        seq_len: int,
        rc_prob: float = 0.5,
        jitter: int = 64,
        mask_ratio: float = 0.02,
        enable_aug: bool = True,
        seed: int = 42,
    ) -> None:
        self.frame = frame
        self.split = split
        self.seq_len = seq_len
        self.rc_prob = rc_prob
        self.jitter = jitter
        self.mask_ratio = mask_ratio
        self.enable_aug = enable_aug and split == "train"
        self.rng = np.random.RandomState(seed)

    def __len__(self) -> int:
        return len(self.frame)

    def _apply_augment(self, tokens: np.ndarray) -> np.ndarray:
        out = tokens.copy()
        # random jitter (cyclic roll within small range)
        if self.jitter > 0:
            shift = self.rng.randint(-self.jitter, self.jitter + 1)
            if shift != 0:
                out = np.roll(out, shift)
        # random N mask
        if self.mask_ratio > 0:
            num_mask = int(self.seq_len * self.mask_ratio)
            if num_mask > 0:
                idx = self.rng.choice(self.seq_len, size=num_mask, replace=False)
                out[idx] = NUC_TO_ID["N"]
        # reverse complement
        if self.rc_prob > 0 and self.rng.rand() < self.rc_prob:
            out = reverse_complement_tokens(out)
        return out

    def __getitem__(self, idx: int):
        row = self.frame.iloc[idx]
        seq = row["sequence"]
        y = float(row["y"])  # already z-scored per species
        species_id = int(row["species_id"])
        toks = tokenize_sequence(seq, self.seq_len)
        if self.enable_aug:
            toks = self._apply_augment(toks)
        tokens = torch.from_numpy(toks.astype(np.int64))
        return {"tokens": tokens, "y": torch.tensor(y, dtype=torch.float32), "species": torch.tensor(species_id, dtype=torch.long)}


def create_dataloaders(
    split: SplitData,
    seq_len: int,
    batch_size: int,
    num_workers: int = 0,
    seed: int = 42,
    **kwargs
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[int, float]]:
    train_ds = SequenceDataset(split.train, "train", seq_len=seq_len, enable_aug=True, seed=seed)
    val_ds = SequenceDataset(split.val, "val", seq_len=seq_len, enable_aug=False, seed=seed)
    test_ds = SequenceDataset(split.test, "test", seq_len=seq_len, enable_aug=False, seed=seed)

    # class weights from training distribution (for optional weighted CE)
    species_ids, counts = np.unique(split.train["species_id"].values, return_counts=True)
    total = counts.sum()
    weights = {int(k): float(total / (len(counts) * c)) for k, c in zip(species_ids, counts)}

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=False)
    
    return train_loader, val_loader, test_loader, weights