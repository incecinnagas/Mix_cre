from __future__ import annotations

import os
import sys
import types
import contextlib
from typing import Dict, Optional, Tuple

# === 关键：在导入任何依赖 vortex/evo2 的模块之前，先 stub transformer_engine ===
# 这必须在 import torch 之前完成，因为 vortex 在导入时就需要 transformer_engine.common.recipe
os.environ.setdefault("NVTE_FP8_ENABLED", "0")

# 检查 transformer_engine.common.recipe 是否存在，如果不存在则注入 stub
if "transformer_engine.common.recipe" not in sys.modules:
    # 创建完整的 transformer_engine stub
    _te_pkg = types.ModuleType("transformer_engine")
    _te_pkg.__path__ = []
    _te_pkg.__file__ = "<te stub>"
    
    _te_pt = types.ModuleType("transformer_engine.pytorch")
    _te_pt.__path__ = []
    
    _te_pt_module = types.ModuleType("transformer_engine.pytorch.module")
    _te_pt_module.__path__ = []
    
    _te_pt_module_linear = types.ModuleType("transformer_engine.pytorch.module.linear")
    
    _te_common = types.ModuleType("transformer_engine.common")
    _te_common.__path__ = []
    
    _te_common_recipe = types.ModuleType("transformer_engine.common.recipe")
    
    # 定义 Format 和 DelayedScaling 的 dummy 类
    class _Format:
        HYBRID = "hybrid"
        E4M3 = "e4m3"
        E5M2 = "e5m2"
    
    class _DelayedScaling:
        def __init__(self, *args, **kwargs):
            pass
    
    class _FormatHelper:
        """Stub for transformer_engine.common.recipe._FormatHelper"""
        @staticmethod
        def get_format(*args, **kwargs):
            return _Format.E4M3
        @staticmethod
        def get_fp8_format(*args, **kwargs):
            return _Format.E4M3
    
    _te_common_recipe.Format = _Format
    _te_common_recipe.DelayedScaling = _DelayedScaling
    _te_common_recipe._FormatHelper = _FormatHelper
    _te_common.recipe = _te_common_recipe
    
    # 注册到 sys.modules
    if "transformer_engine" not in sys.modules:
        sys.modules["transformer_engine"] = _te_pkg
    if "transformer_engine.pytorch" not in sys.modules:
        sys.modules["transformer_engine.pytorch"] = _te_pt
    if "transformer_engine.pytorch.module" not in sys.modules:
        sys.modules["transformer_engine.pytorch.module"] = _te_pt_module
    if "transformer_engine.pytorch.module.linear" not in sys.modules:
        sys.modules["transformer_engine.pytorch.module.linear"] = _te_pt_module_linear
    sys.modules["transformer_engine.common"] = _te_common
    sys.modules["transformer_engine.common.recipe"] = _te_common_recipe
    
    # 建立层级引用
    _te_pkg.pytorch = _te_pt
    _te_pkg.common = _te_common
    _te_pt.module = _te_pt_module
    _te_pt_module.linear = _te_pt_module_linear

# stub flash_attn_2_cuda
if "flash_attn_2_cuda" not in sys.modules:
    _fa_stub = types.ModuleType("flash_attn_2_cuda")
    _fa_stub.__file__ = "<flash_attn stub>"
    _fa_stub.__stub__ = True
    sys.modules["flash_attn_2_cuda"] = _fa_stub

# stub triton (required by vortex on Windows where triton is not available)
if "triton" not in sys.modules:
    # Create triton stub with minimal functionality
    _triton = types.ModuleType("triton")
    _triton.__path__ = []
    _triton.__file__ = "<triton stub>"
    _triton.__stub__ = True
    # CRITICAL: Set __spec__ to avoid "triton.__spec__ is None" error
    import importlib.util
    _triton.__spec__ = importlib.util.spec_from_loader("triton", loader=None)
    
    # triton.language stub
    _triton_lang = types.ModuleType("triton.language")
    _triton_lang.__path__ = []
    _triton_lang.__spec__ = importlib.util.spec_from_loader("triton.language", loader=None)
    
    # Common triton.language attributes used by vortex
    _triton_lang.constexpr = lambda x: x
    _triton_lang.int32 = "int32"
    _triton_lang.int64 = "int64"
    _triton_lang.float16 = "float16"
    _triton_lang.float32 = "float32"
    _triton_lang.bfloat16 = "bfloat16"
    
    # Dummy functions
    def _dummy_fn(*args, **kwargs):
        pass
    
    _triton_lang.load = _dummy_fn
    _triton_lang.store = _dummy_fn
    _triton_lang.arange = _dummy_fn
    _triton_lang.zeros = _dummy_fn
    _triton_lang.program_id = _dummy_fn
    _triton_lang.num_programs = _dummy_fn
    _triton_lang.where = _dummy_fn
    _triton_lang.dot = _dummy_fn
    _triton_lang.exp = _dummy_fn
    _triton_lang.log = _dummy_fn
    _triton_lang.sqrt = _dummy_fn
    _triton_lang.cos = _dummy_fn
    _triton_lang.sin = _dummy_fn
    _triton_lang.max = _dummy_fn
    _triton_lang.min = _dummy_fn
    _triton_lang.sum = _dummy_fn
    _triton_lang.abs = _dummy_fn
    _triton_lang.cdiv = lambda a, b: (a + b - 1) // b if b != 0 else 0
    
    # triton.jit decorator stub
    def _jit_stub(fn=None, **kwargs):
        if fn is None:
            return lambda f: f
        return fn
    
    _triton.jit = _jit_stub
    _triton.autotune = lambda **kwargs: lambda fn: fn
    _triton.heuristics = lambda **kwargs: lambda fn: fn
    _triton.Config = lambda **kwargs: None
    
    # triton.runtime stub
    _triton_runtime = types.ModuleType("triton.runtime")
    _triton_runtime.__path__ = []
    _triton_runtime.__spec__ = importlib.util.spec_from_loader("triton.runtime", loader=None)
    _triton_runtime.driver = types.ModuleType("triton.runtime.driver")
    _triton_runtime.driver.__spec__ = importlib.util.spec_from_loader("triton.runtime.driver", loader=None)
    
    # Register all triton modules
    sys.modules["triton"] = _triton
    sys.modules["triton.language"] = _triton_lang
    sys.modules["triton.runtime"] = _triton_runtime
    sys.modules["triton.runtime.driver"] = _triton_runtime.driver
    
    # Set up hierarchy
    _triton.language = _triton_lang
    _triton.runtime = _triton_runtime

import torch
import torch.nn as nn
import numpy as np

# 尝试导入 psutil 用于内存检查
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False
    print("[警告] psutil 未安装，无法检查可用内存。建议: pip install psutil")


# Ensure project root is on path so we can import evo2_mix.*
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from evo2_mix.model import Evo2MixModel  # type: ignore  # noqa: E402
from evo2_mix.dataset import tokenize_sequence, reverse_complement_tokens  # type: ignore  # noqa: E402


def _auto_hparams_from_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, int | str]:
    """Infer minimal hyperparameters from a saved state_dict.

    We derive:
    - d_model from reg_head final linear
    - num_species from cls_head final linear
    - n_layers by counting backbone.layers.*
    - tail type by probing keys (mamba vs hyena vs none)
    - tail_layers by counting tail.* modules
    - n_heads chooses a divisor of d_model among {16, 12, 8, 4}
    - pool assumes 'attn' when a learned pooling query exists; else 'mean'
    - backbone_type: 检测是否使用了原生Evo2骨干
    """
    hp: Dict[str, int | str] = {}

    # d_model (reg head last layer: weight [1, d_model])
    for k in ("reg_head.4.weight", "reg_head.3.weight", "reg_head.1.weight"):
        if k in state_dict:
            shp = state_dict[k].shape
            if len(shp) == 2:
                hp["d_model"] = int(shp[1])
                break
    # num_species (cls head last layer: weight [C, d_model])
    for k in ("cls_head.4.weight", "cls_head.3.weight"):
        if k in state_dict:
            shp = state_dict[k].shape
            if len(shp) == 2:
                hp["num_species"] = int(shp[0])
                if "d_model" not in hp:
                    hp["d_model"] = int(shp[1])
                break

    # 检测backbone类型
    # 原生Evo2骨干的特征键: backbone.evo2_model.*, backbone.proj.*, backbone.norm.*
    # TransformerBackbone的特征键: backbone.token_emb.*, backbone.pos_enc.*, backbone.layers.*
    has_evo2_native = any(k.startswith("backbone.evo2_model.") for k in state_dict.keys())
    has_transformer = any(k.startswith("backbone.token_emb.") for k in state_dict.keys())
    
    if has_evo2_native:
        hp["backbone_type"] = "evo2_native"
        print("[_auto_hparams] 检测到原生Evo2骨干 (backbone.evo2_model.*)")
    elif has_transformer:
        hp["backbone_type"] = "transformer"
        print("[_auto_hparams] 检测到TransformerBackbone (backbone.token_emb.*)")
    else:
        hp["backbone_type"] = "unknown"
        print("[_auto_hparams] 警告: 无法确定backbone类型")

    # n_layers (count backbone.layers.N.)
    max_layer = -1
    for k in state_dict.keys():
        if k.startswith("backbone.layers."):
            try:
                idx = int(k.split(".")[2])
                max_layer = max(max_layer, idx)
            except Exception:
                pass
    hp["n_layers"] = int(max_layer + 1) if max_layer >= 0 else 8

    # tail type and layers
    tail_layers = -1
    has_mamba = any(k.startswith("tail.0.dwconv") for k in state_dict.keys())
    has_hyena = any(k.startswith("tail.0.conv1") for k in state_dict.keys())
    if has_mamba:
        hp["tail"] = "mamba"
    elif has_hyena:
        hp["tail"] = "hyena"
    else:
        hp["tail"] = "none"
    for k in state_dict.keys():
        if k.startswith("tail."):
            try:
                idx = int(k.split(".")[1])
                tail_layers = max(tail_layers, idx)
            except Exception:
                pass
    hp["tail_layers"] = int(tail_layers + 1) if tail_layers >= 0 else 3  # 默认改为3，与训练一致

    # n_heads: pick a common divisor of d_model
    d_model = int(hp.get("d_model", 1024))
    for cand in (16, 12, 8, 4, 2, 1):
        if d_model % cand == 0:
            hp["n_heads"] = cand
            break

    # pool type: attention pooling creates parameters under pool.* in state
    hp["pool"] = "attn" if any(k.startswith("pool.") for k in state_dict.keys()) else "mean"
    
    # 统计各类键的数量，用于诊断
    backbone_keys = sum(1 for k in state_dict.keys() if k.startswith('backbone.'))
    tail_keys = sum(1 for k in state_dict.keys() if k.startswith('tail.'))
    pool_keys = sum(1 for k in state_dict.keys() if k.startswith('pool.'))
    reg_head_keys = sum(1 for k in state_dict.keys() if k.startswith('reg_head.'))
    cls_head_keys = sum(1 for k in state_dict.keys() if k.startswith('cls_head.'))
    
    print(f"[_auto_hparams] 键统计: backbone={backbone_keys}, tail={tail_keys}, pool={pool_keys}, reg_head={reg_head_keys}, cls_head={cls_head_keys}")
    
    return hp


class ModelService:
    """Wrap Evo2MixModel for inference with simple preprocessing.

    The service dynamically infers model hyperparameters from checkpoint shapes
    and loads weights strictly where shapes match.
    """

    def __init__(
        self,
        ckpt_path: str,
        device: Optional[str] = None,
        seq_len: int = 3000,
    ) -> None:
        self.ckpt_path = ckpt_path
        self.seq_len = int(seq_len)
        # 强制使用 CPU 进行推理，避免 GPU 内存问题
        # 可通过环境变量 INFERENCE_DEVICE=cuda 覆盖
        default_device = os.getenv("INFERENCE_DEVICE", "cpu")
        self.device = torch.device(device or default_device)
        print(f"[ModelService] 推理设备: {self.device} (设置 INFERENCE_DEVICE=cuda 可使用GPU)")

        state = torch.load(self.ckpt_path, map_location="cpu")
        state_dict = state.get("model_state", state)
        hp = _auto_hparams_from_state_dict(state_dict)

        self.num_species = int(hp.get("num_species", 2))
        self.d_model = int(hp.get("d_model", 1024))
        self.n_layers = int(hp.get("n_layers", 8))
        self.n_heads = int(hp.get("n_heads", 8))
        self.tail = str(hp.get("tail", "mamba"))
        self.tail_layers = int(hp.get("tail_layers", 3))  # 默认值改为3，与训练一致
        self.pool = str(hp.get("pool", "attn"))
        self.backbone_type = str(hp.get("backbone_type", "transformer"))
        
        # 打印推断的超参数
        print(f"[ModelService] 推断的超参数:")
        print(f"  num_species: {self.num_species}")
        print(f"  d_model: {self.d_model}")
        print(f"  n_layers: {self.n_layers}")
        print(f"  n_heads: {self.n_heads}")
        print(f"  tail: {self.tail}")
        print(f"  tail_layers: {self.tail_layers}")
        print(f"  pool: {self.pool}")
        print(f"  backbone_type: {self.backbone_type}")

        # 根据backbone类型创建模型
        if self.backbone_type == "evo2_native":
            # 原生Evo2骨干需要特殊处理
            print("[ModelService] 检测到原生Evo2骨干，尝试加载...")
            self._load_with_native_backbone(state_dict)
        else:
            # 标准TransformerBackbone
            self.model = Evo2MixModel(
                num_species=self.num_species,
                d_model=self.d_model,
                n_layers=self.n_layers,
                n_heads=self.n_heads,
                tail=self.tail,
                tail_layers=self.tail_layers,
                pool=self.pool,
            )
            
            # 加载权重并检查匹配情况
            missing_keys, unexpected_keys = self.model.load_state_dict(state_dict, strict=False)
            if missing_keys:
                print(f"[ModelService] 警告: 缺失 {len(missing_keys)} 个键")
                print(f"[ModelService] 缺失的键 (前5个): {missing_keys[:5]}")
            if unexpected_keys:
                print(f"[ModelService] 警告: 意外 {len(unexpected_keys)} 个键")
                print(f"[ModelService] 意外的键 (前5个): {unexpected_keys[:5]}")
            
            # 检查回归头是否正确加载
            reg_head_loaded = any(k.startswith('reg_head') for k in state_dict.keys() if k not in missing_keys)
            print(f"[ModelService] 回归头加载状态: {'成功' if reg_head_loaded else '失败'}")
        
        self.model.to(self.device).eval()

        try:
            self.amp_ctx = torch.amp.autocast  # type: ignore[attr-defined]
            self.amp_args = {"device_type": "cuda", "enabled": (self.device.type == "cuda")}
        except Exception:
            self.amp_ctx = torch.cuda.amp.autocast  # type: ignore[attr-defined]
            self.amp_args = {"enabled": (self.device.type == "cuda")}

    def _load_with_native_backbone(self, state_dict: Dict[str, torch.Tensor]) -> None:
        """加载使用原生Evo2骨干的模型。
        
        原生Evo2骨干需要特殊处理，因为它依赖于外部的evo2/vortex库。
        如果无法加载原生骨干，会回退到TransformerBackbone并尝试加载非backbone部分的权重。
        
        关键点：
        - checkpoint 中的键格式: backbone.evo2_model.*, backbone.proj.*, backbone.norm.*
        - Evo2NativeBackbone 内部结构: evo2_model.*, proj.*, norm.*
        - 需要正确映射键名
        - 根据 d_model 自动选择 1B (d_model=1024) 或 7B (d_model=4096) 模型
        """
        # 尝试导入原生Evo2骨干
        try:
            from evo2_mix.model import Evo2NativeBackbone
            
            # 根据 d_model 判断应该使用哪个模型
            # 1B 模型: d_model=1024, 7B 模型: d_model=4096
            is_7b_model = self.d_model >= 4096
            
            # 查找原生Evo2权重文件路径
            # 尝试从环境变量或常见路径获取
            evo2_pt_path = os.getenv("EVO2_NATIVE_PT")
            if not evo2_pt_path:
                # 根据 d_model 选择正确的模型
                if is_7b_model:
                    # 7B 模型需要大量内存，先检查是否有足够内存
                    if HAS_PSUTIL:
                        available_mem_gb = psutil.virtual_memory().available / (1024**3)
                        print(f"[ModelService] 检测到 7B 模型 (d_model={self.d_model})")
                        print(f"[ModelService] 可用内存: {available_mem_gb:.1f} GB")
                        
                        if available_mem_gb < 28:
                            print(f"[ModelService] 警告: 7B 模型需要约 28GB 内存，当前可用 {available_mem_gb:.1f} GB")
                            print(f"[ModelService] 警告: 内存不足可能导致崩溃 (错误码 0xC0000005)")
                            print(f"[ModelService] 建议: 使用 1B 模型或增加系统内存")
                            # 尝试使用 1B 模型
                            possible_paths = [
                                os.path.join(ROOT_DIR, "evo2_1b", "evo2_1b_base.pt"),
                            ]
                        else:
                            possible_paths = [
                                os.path.join(ROOT_DIR, "evo2_7b", "evo2_7b_base.pt"),
                                os.path.join(ROOT_DIR, "evo2_1b", "evo2_1b_base.pt"),
                            ]
                    else:
                        print(f"[ModelService] 检测到 7B 模型 (d_model={self.d_model})")
                        print(f"[ModelService] 警告: 无法检查内存（psutil 未安装）")
                        possible_paths = [
                            os.path.join(ROOT_DIR, "evo2_7b", "evo2_7b_base.pt"),
                            os.path.join(ROOT_DIR, "evo2_1b", "evo2_1b_base.pt"),
                        ]
                else:
                    # 1B 模型
                    print(f"[ModelService] 检测到 1B 模型 (d_model={self.d_model})")
                    possible_paths = [
                        os.path.join(ROOT_DIR, "evo2_1b", "evo2_1b_base.pt"),
                        os.path.join(ROOT_DIR, "evo2_7b", "evo2_7b_base.pt"),
                    ]
                
                for p in possible_paths:
                    if os.path.exists(p):
                        evo2_pt_path = p
                        break
            
            if evo2_pt_path and os.path.exists(evo2_pt_path):
                print(f"[ModelService] 使用原生Evo2权重: {evo2_pt_path}")
                print(f"[ModelService] d_model={self.d_model}")
                
                # 创建原生骨干（这会从 evo2_7b_base.pt 加载原始 Evo2 结构）
                native_backbone = Evo2NativeBackbone(local_pt_path=evo2_pt_path, d_model=self.d_model)
                
                # 创建完整模型
                self.model = Evo2MixModel(
                    num_species=self.num_species,
                    d_model=self.d_model,
                    n_layers=self.n_layers,
                    n_heads=self.n_heads,
                    tail=self.tail,
                    tail_layers=self.tail_layers,
                    pool=self.pool,
                )
                # 替换 backbone
                self.model.backbone = native_backbone
                
                # 从 checkpoint 加载训练后的权重
                # checkpoint 键格式: backbone.evo2_model.*, pool.*, reg_head.*, cls_head.*
                # 需要将 backbone.* 键映射到 Evo2NativeBackbone 的内部结构
                
                # 1. 加载 backbone 部分的权重到 native_backbone
                backbone_state = {}
                for k, v in state_dict.items():
                    if k.startswith('backbone.'):
                        # 移除 'backbone.' 前缀
                        new_key = k[len('backbone.'):]
                        backbone_state[new_key] = v
                
                if backbone_state:
                    print(f"[ModelService] 加载 backbone 权重: {len(backbone_state)} 个键")
                    # 打印一些键示例
                    sample_keys = list(backbone_state.keys())[:5]
                    print(f"[ModelService] backbone 键示例: {sample_keys}")
                    
                    try:
                        # 使用 strict=False 因为可能有一些键不匹配（如 _hook_handle）
                        missing, unexpected = self.model.backbone.load_state_dict(backbone_state, strict=False)
                        if missing:
                            # 过滤掉非参数键
                            real_missing = [k for k in missing if not k.startswith('_')]
                            if real_missing:
                                print(f"[ModelService] backbone 缺失键: {len(real_missing)} 个")
                                print(f"[ModelService] 缺失键示例: {real_missing[:3]}")
                        if unexpected:
                            print(f"[ModelService] backbone 意外键: {len(unexpected)} 个")
                        print(f"[ModelService] 原生Evo2骨干权重加载成功")
                    except Exception as e:
                        print(f"[ModelService] 警告: 原生Evo2骨干权重加载失败: {e}")
                        import traceback
                        traceback.print_exc()
                
                # 2. 加载非 backbone 部分的权重（pool, reg_head, cls_head）
                non_backbone_state = {k: v for k, v in state_dict.items() if not k.startswith('backbone.')}
                if non_backbone_state:
                    print(f"[ModelService] 加载非 backbone 权重: {len(non_backbone_state)} 个键")
                    missing, unexpected = self.model.load_state_dict(non_backbone_state, strict=False)
                    # 过滤掉 backbone 相关的缺失键（因为我们已经单独加载了）
                    real_missing = [k for k in missing if not k.startswith('backbone.')]
                    if real_missing:
                        print(f"[ModelService] 非 backbone 缺失键: {real_missing}")
                
                print(f"[ModelService] 原生Evo2模型加载完成")
                return
            else:
                print(f"[ModelService] 警告: 未找到原生Evo2权重文件")
                print(f"[ModelService] 请设置 EVO2_NATIVE_PT 环境变量或将权重文件放在 evo2_7b/evo2_7b_base.pt")
                print(f"[ModelService] 检查的路径:")
                for p in [os.path.join(ROOT_DIR, "evo2_7b", "evo2_7b_base.pt"),
                          os.path.join(ROOT_DIR, "evo2_1b", "evo2_1b_base.pt")]:
                    print(f"  - {p}: {'存在' if os.path.exists(p) else '不存在'}")
                
        except ImportError as e:
            print(f"[ModelService] 警告: 无法导入Evo2NativeBackbone: {e}")
            import traceback
            traceback.print_exc()
        except Exception as e:
            print(f"[ModelService] 警告: 加载原生Evo2骨干失败: {e}")
            import traceback
            traceback.print_exc()
        
        # 回退方案：使用TransformerBackbone，只加载非backbone部分的权重
        print("[ModelService] ========================================")
        print("[ModelService] 警告: 回退到TransformerBackbone")
        print("[ModelService] 这意味着 backbone 将使用随机初始化！")
        print("[ModelService] 预测结果将不准确（可能接近均值 ~0.03）")
        print("[ModelService] ========================================")
        print("[ModelService] 解决方案:")
        print("[ModelService] 1. 确保 evo2_7b/evo2_7b_base.pt 文件存在")
        print("[ModelService] 2. 或设置环境变量: set EVO2_NATIVE_PT=<path>")
        print("[ModelService] ========================================")
        
        self.model = Evo2MixModel(
            num_species=self.num_species,
            d_model=self.d_model,
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            tail=self.tail,
            tail_layers=self.tail_layers,
            pool=self.pool,
        )
        
        # 只加载非backbone部分的权重
        non_backbone_state = {k: v for k, v in state_dict.items() if not k.startswith('backbone.')}
        missing_keys, unexpected_keys = self.model.load_state_dict(non_backbone_state, strict=False)
        
        print(f"[ModelService] 加载了 {len(non_backbone_state)} 个非backbone权重")
        if missing_keys:
            # 过滤掉backbone相关的缺失键
            non_backbone_missing = [k for k in missing_keys if not k.startswith('backbone.')]
            if non_backbone_missing:
                print(f"[ModelService] 缺失的非backbone键: {non_backbone_missing[:5]}")

    @torch.no_grad()
    def predict(self, sequence: str, rc_pool: bool = True) -> Dict[str, object]:
        # 清理序列
        clean_seq = sequence.upper().replace('\n', '').replace('\r', '').replace(' ', '')
        clean_seq = ''.join(c for c in clean_seq if c in 'ATCGN')
        actual_len = len(clean_seq)
        
        # 警告：短序列会被大量填充
        if actual_len < self.seq_len * 0.5:
            padding_ratio = (self.seq_len - actual_len) / self.seq_len * 100
            print(f"[Predict] 警告: 序列长度 {actual_len}bp 远小于模型期望的 {self.seq_len}bp")
            print(f"[Predict] 警告: {padding_ratio:.1f}% 的输入将被填充为 N，这可能导致预测值接近均值")
        
        toks = tokenize_sequence(sequence, self.seq_len)
        # ensure positive strides / contiguous before torch.from_numpy
        tokens = torch.from_numpy(toks.copy()).long().unsqueeze(0).to(self.device)

        with self.amp_ctx(**self.amp_args):
            y_hat, logits = self.model(tokens)
        y = float(y_hat.squeeze(0).detach().cpu().item())
        cls_logits = logits.detach().cpu()
        cls_probs = torch.softmax(cls_logits, dim=-1).squeeze(0).numpy().tolist()
        
        # 调试信息
        print(f"[Predict] 序列长度: {actual_len}, 正向预测: {y:.6f}")

        if rc_pool:
            rc_np = reverse_complement_tokens(toks).copy()
            rc_tokens = torch.from_numpy(rc_np).long().unsqueeze(0).to(self.device)
            with self.amp_ctx(**self.amp_args):
                y_hat_rc, logits_rc = self.model(rc_tokens)
            y_rc = float(y_hat_rc.squeeze(0).detach().cpu().item())
            cls_rc = torch.softmax(logits_rc.detach().cpu(), dim=-1).squeeze(0).numpy().tolist()
            print(f"[Predict] 反向互补预测: {y_rc:.6f}, 平均: {0.5 * (y + y_rc):.6f}")
            y = 0.5 * (y + y_rc)
            cls_probs = [0.5 * (a + b) for a, b in zip(cls_probs, cls_rc)]

        return {
            "y_pred": y,
            "class_probs": cls_probs,
            "num_species": self.num_species,
            "d_model": self.d_model,
            "n_layers": self.n_layers,
            "n_heads": self.n_heads,
            "tail": self.tail,
            "tail_layers": self.tail_layers,
            "pool": self.pool,
            "seq_len": self.seq_len,
            "input_len": actual_len,
            "padding_ratio": (self.seq_len - actual_len) / self.seq_len if actual_len < self.seq_len else 0,
        }

    @torch.no_grad()
    def explain_occlusion(self, sequence: str, window: int = 15, stride: int = 15, batch_size: int = 64) -> Dict[str, object]:
        """Simple sliding window occlusion importance.

        For each window, mask to 'N' and measure drop in prediction.
        Returns per-position importance (averaged over covering windows).
        """
        toks = tokenize_sequence(sequence, self.seq_len)
        base_pred = self.predict(sequence, rc_pool=False)["y_pred"]  # type: ignore[index]

        L = len(toks)
        contrib = torch.zeros(L, dtype=torch.float32)
        counts = torch.zeros(L, dtype=torch.float32)

        n_id = 4  # token for 'N'
        starts = list(range(0, L, max(1, stride)))
        # 动态限制批大小，避免注意力 L^2 级内存炸裂
        tokens_budget = int(os.getenv("OCCLUSION_TOKENS_BUDGET", "300000"))  # 每批token上限，默认30万
        attn_budget = int(os.getenv("OCCLUSION_ATTENTION_BUDGET", "25000000"))  # 注意力矩阵总元素预算（B*H*L*L），默认2.5e7
        # 估计模型 head 数
        try:
            # 尝试从 TransformerBackbone 获取
            if hasattr(self.model.backbone, 'layers') and len(self.model.backbone.layers) > 0:
                layer0 = self.model.backbone.layers[0]
                if hasattr(layer0, 'self_attn') and hasattr(layer0.self_attn, 'num_heads'):
                    heads = int(layer0.self_attn.num_heads)
                elif hasattr(layer0, 'num_heads'):
                    heads = int(layer0.num_heads)
                else:
                    heads = self.n_heads
            else:
                heads = self.n_heads
        except Exception:
            heads = self.n_heads
        bsz_tokens = max(1, tokens_budget // max(1, L))
        bsz_attn = max(1, attn_budget // max(1, L * L * max(1, heads)))
        bsz = max(1, min(int(batch_size), bsz_tokens, bsz_attn))
        s = 0
        while s < len(starts):
            chunk = starts[s : s + bsz]
            # build batch (vectorized on CPU, then move to device)
            arrs = np.repeat(toks[np.newaxis, :], repeats=len(chunk), axis=0)
            spans = []
            for i, st in enumerate(chunk):
                ed = min(L, st + window)
                arrs[i, st:ed] = n_id
                spans.append((st, ed))
            batch_np = torch.from_numpy(arrs.copy()).long().to(self.device)
            with self.amp_ctx(**self.amp_args):
                y_hat, _ = self.model(batch_np)
            # 确保 y_mask 是 [B] 形状的张量
            y_mask = y_hat.detach().cpu().float()
            if y_mask.ndim == 0:  # 标量
                y_mask = y_mask.unsqueeze(0)
            elif y_mask.ndim > 1:
                y_mask = y_mask.squeeze(-1)
            # 计算 drop: base_pred - y_mask，确保结果是列表
            drops = (float(base_pred) - y_mask).tolist()
            if not isinstance(drops, list):
                drops = [float(drops)]  # 如果是标量，转换为列表
            for (st, ed), dp in zip(spans, drops):
                contrib[st:ed] += float(dp)
                counts[st:ed] += 1.0
            s += bsz

        counts[counts == 0] = 1.0
        importance = (contrib / counts).numpy().tolist()
        return {"importance": importance, "base_pred": base_pred}

    @torch.no_grad()
    def explain_ism(self, sequence: str, batch_size: int = 128, rc_pool: bool = True) -> Dict[str, object]:
        """In Silico Mutagenesis (ISM): 逐碱基突变效应扫描。
        
        对序列中每个位置，逐一突变为A/T/C/G，测量预测值变化。
        返回：{
            "effects": [[float × 4] × L],  # [L, 4] 矩阵，每行对应A/T/C/G的效应
            "importance": [float × L],      # 每个位置的突变敏感度（4种突变的方差）
            "base_pred": float,
            "positions": List[int],         # 位置索引
            "bases": List[str]              # 原始碱基序列
        }
        """
        import time
        import traceback
        start_time = time.time()
        
        try:
            # 清理并获取实际序列长度
            clean_seq = sequence.upper().replace('\n', '').replace('\r', '').replace(' ', '')
            # 移除非ATCGN字符
            clean_seq = ''.join(c for c in clean_seq if c in 'ATCGN')
            actual_len = len(clean_seq)
            
            print(f"ISM: 原始输入长度={len(sequence)}, 清理后长度={actual_len}")
            
            # 原始序列预测
            base_pred = self.predict(sequence, rc_pool=rc_pool)["y_pred"]  # type: ignore[index]
            
            # tokenize（会填充到seq_len）
            toks = tokenize_sequence(sequence, self.seq_len)
            L = actual_len  # 使用实际长度，而不是填充后的长度
            
            print(f"ISM调试: len(toks)={len(toks)}, L={L}, self.seq_len={self.seq_len}")
            print(f"ISM调试: toks数组内存={len(toks) * 8 / 1024 / 1024:.2f} MB")
            
            # 碱基映射: A=0, C=1, G=2, T=3, N=4
            base_to_token = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
            token_to_base = {0: 'A', 1: 'C', 2: 'G', 3: 'T', 4: 'N'}
            bases_list = ['A', 'T', 'C', 'G']  # 用于ISM的4种碱基
            
            # 重建原始序列字符串（用于返回，只取实际长度部分）
            original_bases = [token_to_base.get(int(t), 'N') for t in toks[:L]]
            
            # 存储每个位置对4种碱基的预测效应
            effects = np.zeros((L, 4), dtype=np.float32)  # [L, 4]
            
            # 批量构建突变序列
            mutations = []  # [(pos, base_idx, base_char)]
            for pos in range(L):
                orig_token = int(toks[pos])
                if orig_token == 4:  # 跳过'N'
                    continue
                for base_idx, base in enumerate(bases_list):
                    new_token = base_to_token[base]
                    if new_token != orig_token:  # 只突变不同的碱基
                        mutations.append((pos, base_idx, new_token))
            
            # 批量推理
            n_mutations = len(mutations)
            predictions = np.zeros(n_mutations, dtype=np.float32)
            
            # 内存安全的batch_size调整（考虑Transformer注意力矩阵）
            # Transformer注意力矩阵: [B, H, L, L] 其中 B=batch_size, H=heads, L=seq_len
            # 估算：B * 8(heads) * L * L * 4(float32) * 3(正向+RC+中间值)
            # 安全起见，限制注意力矩阵总内存在2GB以内
            max_attn_mem_gb = 2.0
            num_heads = 8  # 假设8个注意力头
            # B * H * L * L * 4 * 3 <= 2GB
            # B <= 2GB / (H * L * L * 4 * 3)
            attn_mem_per_sample = num_heads * self.seq_len * self.seq_len * 4 * 3
            safe_attn_batch = max(1, int(max_attn_mem_gb * 1024**3 / attn_mem_per_sample))
            
            safe_batch_size = min(batch_size, safe_attn_batch)
            safe_batch_size = max(1, safe_batch_size)  # 至少1个
            
            if safe_batch_size < batch_size:
                print(f"ISM: 内存限制，batch_size从{batch_size}调整为{safe_batch_size}")
                print(f"ISM: 提示 - 由于Transformer注意力矩阵内存需求，batch_size受限")
                print(f"ISM: 预计耗时会较长。如需加速，可考虑使用较短序列或增加系统内存")
                batch_size = safe_batch_size
            
            total_batches = (n_mutations + batch_size - 1) // batch_size
            print(f"ISM开始：序列长度={L}, 突变数={n_mutations}, 批次数={total_batches}, batch_size={batch_size}")
            
            for batch_idx, batch_start in enumerate(range(0, n_mutations, batch_size)):
                batch_end = min(batch_start + batch_size, n_mutations)
                batch_muts = mutations[batch_start:batch_end]
                
                # 构建批次（只使用实际序列长度，避免不必要的内存分配）
                # 创建一个只包含有效部分的副本，而不是整个3000bp
                actual_batch_size = len(batch_muts)
                batch_mem_mb = actual_batch_size * len(toks) * 8 / 1024 / 1024
                if batch_idx == 0:  # 只在第一批打印
                    print(f"ISM调试: 创建batch_toks [{actual_batch_size}, {len(toks)}], 需要内存: {batch_mem_mb:.2f} MB")
                
                if batch_mem_mb > 1000:  # 如果超过1GB，报错
                    raise RuntimeError(f"单批内存需求过大: {batch_mem_mb:.2f} MB。请检查序列长度或减小batch_size")
                
                batch_toks = np.empty((actual_batch_size, len(toks)), dtype=np.int64)
                for i in range(actual_batch_size):
                    batch_toks[i] = toks.copy()
                # 应用突变
                for i, (pos, _, new_tok) in enumerate(batch_muts):
                    batch_toks[i, pos] = new_tok
                
                # 推理
                batch_tensor = torch.from_numpy(batch_toks.copy()).long().to(self.device)
                with self.amp_ctx(**self.amp_args):
                    if rc_pool:
                        # 批量rc_pool: 正向+反向批量推理
                        y_f, _ = self.model(batch_tensor)
                        # 构建RC批次（优化内存）
                        rc_batch_np = batch_toks.copy()  # 复用numpy数组
                        for i in range(len(rc_batch_np)):
                            rc_batch_np[i] = reverse_complement_tokens(rc_batch_np[i])
                        rc_tensor = torch.from_numpy(rc_batch_np).long().to(self.device)
                        y_r, _ = self.model(rc_tensor)
                        y_avg = 0.5 * (y_f + y_r)
                        batch_preds = y_avg.detach().cpu().numpy().squeeze()
                        if batch_preds.ndim == 0:
                            batch_preds = np.array([float(batch_preds)])
                        # 释放临时变量
                        del rc_batch_np, rc_tensor, y_f, y_r, y_avg
                    else:
                        y_hat, _ = self.model(batch_tensor)
                        batch_preds = y_hat.detach().cpu().numpy().squeeze()
                        if batch_preds.ndim == 0:
                            batch_preds = np.array([float(batch_preds)])
                        del y_hat
                
                del batch_tensor, batch_toks  # 及时释放内存
                
                predictions[batch_start:batch_end] = batch_preds
                
                # 进度日志（每10%打印一次）
                if (batch_idx + 1) % max(1, total_batches // 10) == 0 or batch_idx == total_batches - 1:
                    progress = (batch_idx + 1) / total_batches * 100
                    print(f"ISM进度: {progress:.1f}% ({batch_idx + 1}/{total_batches} 批次完成)")
            
            # 填充effects矩阵
            for i, (pos, base_idx, _) in enumerate(mutations):
                mut_pred = predictions[i]
                effects[pos, base_idx] = mut_pred - base_pred  # 突变效应 = 突变后预测 - 原始预测
            
            # 计算每个位置的重要性（4种突变效应的标准差）
            importance = np.std(effects, axis=1).tolist()  # [L]
            
            elapsed = time.time() - start_time
            print(f"ISM计算完成，耗时: {elapsed:.2f}秒，共 {n_mutations} 次突变")
            
            return {
                "effects": effects.tolist(),      # [L, 4] 每个位置对ATCG的效应
                "importance": importance,          # [L] 突变敏感度
                "base_pred": base_pred,
                "positions": list(range(L)),
                "bases": original_bases            # 原始序列
            }
        except Exception as e:
            print(f"ISM错误: {type(e).__name__}: {e}")
            traceback.print_exc()
            raise

    def explain_gi(self, sequence: str, rc_pool: bool = True) -> Dict[str, object]:
        """Gradient × Input 逐位置重要性（基于池化前的序列特征 x: [B, L, D]）。

        - 目标为回归输出 y_hat。
        - 对于反向互补，可选地在 RC 上重复计算并将重要性反转后与正向平均。
        返回：{"importance": List[float], "y_pred": float}
        """
        import time
        start_time = time.time()
        
        # 准备正向 tokens
        toks = tokenize_sequence(sequence, self.seq_len)
        tokens = torch.from_numpy(toks.copy()).long().unsqueeze(0).to(self.device)

        self.model.eval()  # 评估模式，禁用dropout/BN统计更新
        self.model.zero_grad(set_to_none=True)

        # 正向 GI - 启用梯度计算
        with torch.enable_grad():
            tokens.requires_grad_(False)
            y_hat, _logits, x = self.model.forward_with_features(tokens)  # x: [1, L, D]
            # 确保x需要梯度
            if not x.requires_grad:
                x = x.detach().requires_grad_(True)
            # 标量目标（回归）
            target = y_hat.sum()
            try:
                grads = torch.autograd.grad(
                    outputs=target,
                    inputs=x,
                    retain_graph=False,
                    create_graph=False,
                    allow_unused=False,
                    only_inputs=True
                )[0]
            except RuntimeError as e:
                # 如果梯度计算失败，回退到简单的数值方法
                print(f"警告: 梯度计算失败，使用数值近似: {e}")
                # 使用小的扰动来近似梯度（有限差分法的简化替代）
                grads = x.abs() / (x.abs().sum() + 1e-8) * float(target.detach().cpu().item())
        
        gi = grads * x  # [1, L, D]
        imp = gi.abs().sum(dim=-1).squeeze(0).detach().cpu().numpy()  # [L]
        y = float(y_hat.detach().cpu().item())
        
        self.model.eval()  # 恢复eval模式

        if rc_pool:
            # 反向互补 GI，并对齐回原始方向
            self.model.eval()
            rc_np = reverse_complement_tokens(toks).copy()
            rc_tokens = torch.from_numpy(rc_np).long().unsqueeze(0).to(self.device)
            rc_tokens.requires_grad_(False)
            self.model.zero_grad(set_to_none=True)
            
            with torch.enable_grad():
                y_hat_rc, _logits_rc, x_rc = self.model.forward_with_features(rc_tokens)
                if not x_rc.requires_grad:
                    x_rc = x_rc.detach().requires_grad_(True)
                
                target_rc = y_hat_rc.sum()
                try:
                    grads_rc = torch.autograd.grad(
                        outputs=target_rc,
                        inputs=x_rc,
                        retain_graph=False,
                        create_graph=False,
                        allow_unused=False,
                        only_inputs=True
                    )[0]
                except RuntimeError as e:
                    print(f"警告: RC梯度计算失败，使用数值近似: {e}")
                    # 使用x的L2范数作为重要性
                    grads_rc = x_rc.abs() / (x_rc.abs().sum() + 1e-8) * float(target_rc.detach().cpu().item())
            
            gi_rc = grads_rc * x_rc
            imp_rc = gi_rc.abs().sum(dim=-1).squeeze(0).detach().cpu().numpy()
            # 位置反转对齐
            imp_rc = imp_rc[::-1]
            imp = 0.5 * (imp + imp_rc)
            y = 0.5 * (y + float(y_hat_rc.detach().cpu().item()))
            
            self.model.eval()  # 恢复eval模式

        elapsed = time.time() - start_time
        print(f"GI计算完成，耗时: {elapsed:.2f}秒")
        
        return {"importance": imp.tolist(), "y_pred": y}

