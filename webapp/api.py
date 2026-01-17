from __future__ import annotations

import os
import math
from typing import Any, Dict, Optional, List
import threading

from fastapi import FastAPI, HTTPException, APIRouter
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from fastapi.staticfiles import StaticFiles

# Ensure a usable temporary directory before importing torch/inference
# This prevents failures like "No usable temporary directory found" on some systems.
_base_dir = os.path.dirname(__file__)
_safe_tmp = os.path.abspath(os.path.join(_base_dir, 'tmp'))
try:
    os.makedirs(_safe_tmp, exist_ok=True)
    for _env in ('TMPDIR', 'TEMP', 'TMP'):
        os.environ.setdefault(_env, _safe_tmp)
except Exception:
    # If creating fallback temp dir fails, continue; system temp may still work.
    pass

ModelService = None  # type: ignore


MODEL_CKPT = os.getenv("MODEL_CKPT")
if MODEL_CKPT is None:
    # 尝试多个可能的路径（优先使用 evo2_mix/checkpoints 下的 Transformer+SSM 模型）
    base_dir = os.path.dirname(__file__)
    possible_paths = [
        os.path.join(base_dir, os.pardir, "evo2_mix", "checkpoints", "best.ckpt"),  # Transformer+SSM
        os.path.join(base_dir, os.pardir, "results", "best.ckpt"),
        os.path.join(base_dir, os.pardir, "results", "checkpoints", "best.ckpt"),
    ]
    for path in possible_paths:
        abs_path = os.path.abspath(path)
        if os.path.exists(abs_path):
            MODEL_CKPT = abs_path
            break
    else:
        # 如果都没找到，使用默认路径
        MODEL_CKPT = os.path.abspath(os.path.join(base_dir, os.pardir, "evo2_mix", "checkpoints", "best.ckpt"))

# 训练数据序列长度: TSS上游2000bp + TSS + 下游1000bp = 3001bp
SEQ_LEN = int(os.getenv("SEQ_LEN", "3001"))
EXPECTED_SEQ_LEN = 3001  # 推荐的输入序列长度

# 物种统计信息（用于将z-score转换回原始TPM值）
# 这些值来自训练数据的 log1p(tpm_max) 的均值和标准差
SPECIES_STATS = {
    "Arabidopsis thaliana": {"mean": 0.9546, "std": 0.4459, "id": 0},
    "Oryza sativa": {"mean": 1.1510, "std": 0.2696, "id": 1},
    "Solanum lycopersicum": {"mean": 0.9607, "std": 0.4442, "id": 2},
    "Zea mays": {"mean": 1.6292, "std": 0.6808, "id": 3},
}

# 按物种ID索引的统计信息
SPECIES_STATS_BY_ID = {v["id"]: {"name": k, **v} for k, v in SPECIES_STATS.items()}
SPECIES_NAMES = ["Arabidopsis thaliana", "Oryza sativa", "Solanum lycopersicum", "Zea mays"]


def zscore_to_tpm(zscore: float, species_id: int) -> float:
    """将z-score转换回原始TPM值。
    
    转换公式: TPM = exp(zscore * std + mean) - 1
    其中 mean 和 std 是该物种 log1p(TPM) 的统计值
    """
    if species_id not in SPECIES_STATS_BY_ID:
        # 如果物种未知，使用所有物种的平均统计值
        mean = sum(s["mean"] for s in SPECIES_STATS.values()) / len(SPECIES_STATS)
        std = sum(s["std"] for s in SPECIES_STATS.values()) / len(SPECIES_STATS)
    else:
        stats = SPECIES_STATS_BY_ID[species_id]
        mean = stats["mean"]
        std = stats["std"]
    
    # 反向转换: log1p_value = zscore * std + mean
    log1p_value = zscore * std + mean
    # TPM = exp(log1p_value) - 1 = expm1(log1p_value)
    tpm = math.expm1(log1p_value)
    return max(0.0, tpm)  # TPM不能为负


def get_tpm_estimates(zscore: float, class_probs: List[float]) -> Dict[str, Any]:
    """根据z-score和物种分类概率，估算各物种的TPM值。"""
    estimates = {}
    weighted_tpm = 0.0
    
    for species_id, prob in enumerate(class_probs):
        if species_id < len(SPECIES_NAMES):
            species_name = SPECIES_NAMES[species_id]
            tpm = zscore_to_tpm(zscore, species_id)
            estimates[species_name] = {
                "tpm": round(tpm, 4),
                "probability": round(prob, 4),
            }
            weighted_tpm += tpm * prob
    
    return {
        "per_species": estimates,
        "weighted_average": round(weighted_tpm, 4),
    }


class PredictRequest(BaseModel):
    sequence: str = Field(..., description="DNA sequence (ACGTN), 推荐长度3001bp (TSS上游2000bp + TSS + 下游1000bp)")
    rc_pool: bool = Field(True, description="Average forward and reverse-complement predictions")
    species_hint: Optional[int] = Field(None, description="物种ID提示 (0=拟南芥, 1=水稻, 2=番茄, 3=玉米)，用于TPM转换")


class ExplainRequest(BaseModel):
    sequence: str
    window: int = 15
    stride: int = 15
    batch_size: int = 64


class ExplainGIRequest(BaseModel):
    sequence: str
    rc_pool: bool = True


class ExplainISMRequest(BaseModel):
    sequence: str
    batch_size: int = 128
    rc_pool: bool = True


# GI-based CRE segment scanning
class CREScanGIRequest(BaseModel):
    sequence: str
    rc_pool: bool = True
    window: int = 21
    stride: int = 21
    top_k: int = 100


app = FastAPI(title="Transformer+SSM Genomics API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

service_lock = threading.Lock()
service: Optional[Any] = None
api = APIRouter(prefix="/api")


def get_service() -> Any:
    global service, ModelService
    if service is not None:
        return service
    with service_lock:
        if service is not None:
            return service
        if not os.path.exists(MODEL_CKPT):
            raise RuntimeError(f"MODEL_CKPT not found: {MODEL_CKPT}")
        # Lazy import to avoid importing torch at app import time
        from .inference import ModelService as _MS  # type: ignore
        ModelService = _MS  # type: ignore
        service = ModelService(ckpt_path=os.path.abspath(MODEL_CKPT), seq_len=SEQ_LEN)
        return service

# Register API router later after defining endpoints


@api.get("/health")
def health() -> Dict[str, Any]:
    return {"status": "ok"}


@api.get("/species")
def get_species_info() -> Dict[str, Any]:
    """Get supported species list and statistics."""
    return {
        "species": SPECIES_NAMES,
        "stats": SPECIES_STATS,
        "expected_seq_len": EXPECTED_SEQ_LEN,
        "seq_format": "2000bp upstream + TSS + 1000bp downstream = 3001bp",
    }


@api.post("/predict")
def predict(req: PredictRequest) -> Dict[str, Any]:
    """Predict promoter activity of a DNA sequence.
    
    Input requirements:
    - Recommended sequence length: 3001bp (2000bp upstream + TSS + 1000bp downstream)
    - Sequence format: ATCGN characters
    
    Return values:
    - y_pred: Predicted promoter activity (z-score normalized log1p(TPM))
      - Positive values indicate above-average activity
      - Negative values indicate below-average activity
      - Typical range: -4 to +2
    - tpm_estimates: Converted TPM estimates (per species)
    - class_probs: Species classification probabilities
    - predicted_species: Most likely predicted species
    """
    svc = get_service()
    if not req.sequence or not isinstance(req.sequence, str):
        raise HTTPException(status_code=400, detail="sequence is required")
    
    # Clean sequence
    clean_seq = req.sequence.upper().replace('\n', '').replace('\r', '').replace(' ', '')
    clean_seq = ''.join(c for c in clean_seq if c in 'ATCGN')
    seq_len = len(clean_seq)
    
    # Sequence length validation and warnings
    warnings = []
    if seq_len == 0:
        raise HTTPException(status_code=400, detail="Sequence is empty or contains no valid bases (ATCGN)")
    if seq_len != EXPECTED_SEQ_LEN:
        if seq_len < EXPECTED_SEQ_LEN * 0.5:
            warnings.append(f"Sequence length ({seq_len}bp) is much shorter than recommended ({EXPECTED_SEQ_LEN}bp), prediction accuracy may be reduced")
        elif seq_len < EXPECTED_SEQ_LEN:
            warnings.append(f"Sequence length ({seq_len}bp) is shorter than recommended ({EXPECTED_SEQ_LEN}bp)")
        else:
            warnings.append(f"Sequence length ({seq_len}bp) exceeds recommended ({EXPECTED_SEQ_LEN}bp), will be truncated")
    
    try:
        out = svc.predict(req.sequence, rc_pool=req.rc_pool)
        y_pred = out.get("y_pred", 0)
        class_probs = out.get("class_probs", [0.25, 0.25, 0.25, 0.25])
        
        # Determine species (use hint or prediction result)
        if req.species_hint is not None and 0 <= req.species_hint < len(SPECIES_NAMES):
            predicted_species_id = req.species_hint
            predicted_species = SPECIES_NAMES[predicted_species_id]
        else:
            predicted_species_id = class_probs.index(max(class_probs)) if class_probs else 0
            predicted_species = SPECIES_NAMES[predicted_species_id] if predicted_species_id < len(SPECIES_NAMES) else "Unknown"
        
        # Calculate TPM estimates
        tpm_estimates = get_tpm_estimates(y_pred, class_probs)
        primary_tpm = zscore_to_tpm(y_pred, predicted_species_id)
        
        # Activity level classification
        if y_pred > 1.0:
            activity_level = "High Activity"
        elif y_pred > 0:
            activity_level = "Medium Activity"
        elif y_pred > -1.0:
            activity_level = "Low Activity"
        else:
            activity_level = "Very Low Activity"
        
        # Build response
        out["activity_level"] = activity_level
        out["predicted_species"] = predicted_species
        out["predicted_species_id"] = predicted_species_id
        out["tpm_estimate"] = round(primary_tpm, 4)
        out["tpm_estimates"] = tpm_estimates
        out["expected_seq_len"] = EXPECTED_SEQ_LEN
        out["note"] = f"Estimated TPM based on predicted species ({predicted_species}): {primary_tpm:.2f}"
        
        if warnings:
            out["warnings"] = warnings
        
        return out
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@api.post("/explain/occlusion")
def explain(req: ExplainRequest) -> Dict[str, Any]:
    svc = get_service()
    try:
        out = svc.explain_occlusion(req.sequence, window=req.window, stride=req.stride, batch_size=req.batch_size)
        return out
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@api.post("/explain/gi")
def explain_gi(req: ExplainGIRequest) -> Dict[str, Any]:
    svc = get_service()
    try:
        out = svc.explain_gi(req.sequence, rc_pool=req.rc_pool)
        return out
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@api.post("/explain/ism")
def explain_ism(req: ExplainISMRequest) -> Dict[str, Any]:
    svc = get_service()
    try:
        seq_len = len(req.sequence)
        print(f"[API] ISM请求: seq_len={seq_len}, batch_size={req.batch_size}, rc_pool={req.rc_pool}")
        
        # 验证序列长度
        if seq_len > 10000:
            raise HTTPException(status_code=400, detail=f"序列过长: {seq_len}bp。最大支持10000bp")
        if seq_len == 0:
            raise HTTPException(status_code=400, detail="序列为空")
        
        out = svc.explain_ism(req.sequence, batch_size=req.batch_size, rc_pool=req.rc_pool)
        print(f"[API] ISM完成，返回数据: effects={len(out.get('effects', []))}, importance={len(out.get('importance', []))}")
        return out
    except Exception as e:
        import traceback
        print(f"[API] ISM错误: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@api.post("/cre/scan_gi")
def cre_scan_gi(req: CREScanGIRequest) -> Dict[str, Any]:
    svc = get_service()
    try:
        out = svc.explain_gi(req.sequence, rc_pool=req.rc_pool)
        imp = out.get('importance') or []
        if not isinstance(imp, (list, tuple)) or len(imp) == 0:
            return {"count": 0, "hits": []}
        
        import numpy as np
        imp_arr = np.array(imp, dtype=np.float32)
        L = len(imp_arr)
        
        # 智能峰值检测：基于重要性阈值识别变长CRE区域
        # 1. 计算阈值（使用百分位数或标准差）
        abs_imp = np.abs(imp_arr)
        threshold = max(np.percentile(abs_imp, 75), np.mean(abs_imp) + 0.5 * np.std(abs_imp))
        min_length = max(5, int(req.window) // 4)  # 最小CRE长度
        max_gap = 3  # 允许的最大间隔
        
        # 2. 识别高于阈值的位置
        above_threshold = abs_imp >= threshold
        
        # 3. 连续区域检测（合并间隔小的区域）
        segments = []
        in_region = False
        start = 0
        last_pos = -1
        
        for i in range(L):
            if above_threshold[i]:
                if not in_region or (i - last_pos > max_gap):
                    # 保存上一个区域
                    if in_region and (last_pos - start + 1) >= min_length:
                        segments.append((start, last_pos + 1))
                    # 开始新区域
                    start = i
                    in_region = True
                last_pos = i
            elif in_region and (i - last_pos > max_gap):
                # 区域结束
                if (last_pos - start + 1) >= min_length:
                    segments.append((start, last_pos + 1))
                in_region = False
        
        # 处理最后一个区域
        if in_region and (last_pos - start + 1) >= min_length:
            segments.append((start, last_pos + 1))
        
        # 4. 计算每个区域的统计量
        hits = []
        for start_idx, end_idx in segments:
            seg_imp = imp_arr[start_idx:end_idx]
            seg_abs = np.abs(seg_imp)
            mean_abs = float(np.mean(seg_abs))
            max_abs = float(np.max(seg_abs))
            peak_idx = start_idx + int(np.argmax(seg_abs))
            
            hits.append({
                'start': int(start_idx),
                'end': int(end_idx),
                'length': int(end_idx - start_idx),
                'mean_importance': round(mean_abs, 6),
                'max_importance': round(max_abs, 6),
                'peak_position': int(peak_idx),
            })
        
        # 5. 按平均重要性排序，取top_k
        hits.sort(key=lambda x: (-x['mean_importance'], -x['max_importance']))
        if req.top_k and req.top_k > 0:
            hits = hits[:req.top_k]
        
        return {
            "count": len(hits),
            "hits": hits,
            "importance": imp,
            "y_pred": out.get('y_pred'),
            "threshold": round(float(threshold), 6),
            "method": "adaptive_peak_detection"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
app.include_router(api)

# Serve static demo at root
static_dir = os.path.join(os.path.dirname(__file__), "static")
if os.path.isdir(static_dir):
    app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")

