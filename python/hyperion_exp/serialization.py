"""Сериализация моделей и результатов инференса.

Сохраняем/загружаем InferenceResult с полными сэмплами и диагностикой.
Формат: JAX-массивы через numpy, метаданные через JSON.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import jax.numpy as jnp
import numpy as np

from hyperion_inference.base import InferenceResult


def save_result(result: InferenceResult, path: str) -> None:
    """Сохранить результат инференса на диск.

    Создаёт директорию с:
    - samples/*.npy — массивы сэмплов для каждого латента
    - diagnostics.json — метрики и диагностика
    - metadata.json — метаинформация
    """
    out_dir = Path(path)
    out_dir.mkdir(parents=True, exist_ok=True)

    samples_dir = out_dir / "samples"
    samples_dir.mkdir(exist_ok=True)

    if result.samples:
        for name, arr in result.samples.items():
            np.save(str(samples_dir / f"{name}.npy"), np.asarray(arr))

    if result.samples_by_chain is not None:
        chain_dir = out_dir / "samples_by_chain"
        chain_dir.mkdir(exist_ok=True)
        for name, arr in result.samples_by_chain.items():
            np.save(str(chain_dir / f"{name}.npy"), np.asarray(arr))

    if result.log_probs is not None:
        np.save(str(out_dir / "log_probs.npy"), np.asarray(result.log_probs))

    diag = {}
    if result.diagnostics:
        for k, v in result.diagnostics.items():
            if isinstance(v, (jnp.ndarray, np.ndarray)):
                np.save(str(out_dir / f"diag_{k}.npy"), np.asarray(v))
                diag[k] = f"__array__:diag_{k}.npy"
            elif isinstance(v, (int, float, str, bool, type(None))):
                diag[k] = v
            elif isinstance(v, list):
                diag[k] = [float(x) if isinstance(x, (float, int, np.floating, jnp.floating)) else str(x) for x in v]
            else:
                diag[k] = str(v)

    with open(str(out_dir / "diagnostics.json"), "w", encoding="utf-8") as f:
        json.dump(diag, f, indent=2, ensure_ascii=False)

    meta = {
        "latent_names": list(result.samples.keys()) if result.samples else [],
        "num_samples": int(list(result.samples.values())[0].shape[0]) if result.samples else 0,
        "num_chains": result.num_chains,
    }
    if result.metadata:
        for k, v in result.metadata.items():
            if isinstance(v, (int, float, str, bool, type(None))):
                meta[k] = v

    with open(str(out_dir / "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)


def load_result(path: str) -> InferenceResult:
    """Загрузить результат инференса с диска.

    Обратная операция к save_result. Восстанавливает сэмплы и диагностику.
    """
    in_dir = Path(path)

    samples = {}
    samples_dir = in_dir / "samples"
    if samples_dir.exists():
        for npy_file in samples_dir.glob("*.npy"):
            name = npy_file.stem
            samples[name] = jnp.array(np.load(str(npy_file)))

    log_probs = None
    lp_path = in_dir / "log_probs.npy"
    if lp_path.exists():
        log_probs = jnp.array(np.load(str(lp_path)))

    diagnostics = {}
    diag_path = in_dir / "diagnostics.json"
    if diag_path.exists():
        with open(str(diag_path), "r", encoding="utf-8") as f:
            raw = json.load(f)
        for k, v in raw.items():
            if isinstance(v, str) and v.startswith("__array__:"):
                arr_file = v.split(":", 1)[1]
                arr_path = in_dir / arr_file
                if arr_path.exists():
                    diagnostics[k] = jnp.array(np.load(str(arr_path)))
            else:
                diagnostics[k] = v

    metadata = {}
    meta_path = in_dir / "metadata.json"
    if meta_path.exists():
        with open(str(meta_path), "r", encoding="utf-8") as f:
            metadata = json.load(f)

    samples_by_chain = None
    chain_dir = in_dir / "samples_by_chain"
    if chain_dir.exists():
        samples_by_chain = {}
        for npy_file in chain_dir.glob("*.npy"):
            name = npy_file.stem
            samples_by_chain[name] = jnp.array(np.load(str(npy_file)))

    num_chains = metadata.get("num_chains", 1)

    return InferenceResult(
        samples=samples,
        log_probs=log_probs,
        diagnostics=diagnostics,
        metadata=metadata,
        num_chains=num_chains,
        samples_by_chain=samples_by_chain,
    )
