#!/usr/bin/env python3
"""Select image-space ID-only proxies by frozen known-method rank fidelity."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.stats import kendalltau, spearmanr

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "msps_sota"))

from calibration import calibrate_candidates  # noqa: E402
from prototype_stats import ReferenceStats  # noqa: E402
from scores import (  # noqa: E402
    candidate_confidence,
    gather_candidate_support,
    original_msps_confidence,
    select_candidates,
    stage_similarities,
)

from audit_pulse_locked_proxies import cauchy_stage_confidence  # noqa: E402
from audit_symmetric_support_proxy import metric_row  # noqa: E402
from methods.compact_knn import mean_neighbor_confidence  # noqa: E402
from methods.pulse import PulseState  # noqa: E402
from reproduce_nnguide import assert_no_final_access  # noqa: E402


FAMILIES = ("patch_shuffle4", "center_cutmix", "phase_mix")
FINAL_DIM = 2048
K = 200
NPROBE = 64
FROZEN_LEGACY_MACRO_AUROC = {
    "PULSE": 0.9114263360092222,
    "DeepPrototype": 0.8899290047931596,
    "RC-MSPS": 0.8876363198137707,
    "MSPS": 0.8814656948374291,
    "CauchyStageTail": 0.8555907992525729,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--atom-dir", type=Path, required=True)
    parser.add_argument("--rc-state", type=Path, required=True)
    parser.add_argument("--rc-config", type=Path, required=True)
    parser.add_argument("--pulse-state", type=Path, required=True)
    parser.add_argument("--cauchy-state", type=Path, required=True)
    parser.add_argument("--compact-index", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--threads", type=int, default=24)
    return parser.parse_args()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_atoms(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        required = {"labels", "base_pooled", "shift_ratio", "localized", "meanview"}
        if not required.issubset(data.files):
            raise ValueError(f"missing atoms in {path}: {sorted(required - set(data.files))}")
        output = {name: data[name].astype(np.float32) for name in required - {"labels"}}
        output["labels"] = data["labels"].astype(np.int64)
    return output


def load_cauchy_references(path: Path) -> tuple[np.ndarray, ...]:
    with np.load(path, allow_pickle=False) as data:
        return tuple(data[f"stage_reference_{layer}"].astype(np.float64) for layer in range(4))


@torch.inference_mode()
def semantic_geometry(
    canonical: np.ndarray,
    localized: np.ndarray,
    prototypes: np.ndarray,
    *,
    device: str,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    prototype = F.normalize(torch.as_tensor(prototypes, dtype=torch.float32, device=device), dim=1)
    uniform, deep, local = [], [], []
    for begin in range(0, len(canonical), batch_size):
        whole = F.normalize(
            torch.as_tensor(canonical[begin : begin + batch_size, -FINAL_DIM:], device=device),
            dim=1,
        )
        crop = F.normalize(
            torch.as_tensor(localized[begin : begin + batch_size], device=device), dim=2
        )
        similarity = whole @ prototype.T
        rms = similarity.square().mean(dim=1).sqrt().clamp_min(1e-12)
        uniform.append((similarity.mean(dim=1) / rms).cpu().numpy())
        deep.append(similarity.amax(dim=1).cpu().numpy())
        local.append(torch.einsum("bvd,cd->bvc", crop, prototype).amax(dim=(1, 2)).cpu().numpy())
    return np.concatenate(uniform), np.concatenate(deep), np.concatenate(local)


def method_scores(
    atoms: dict[str, np.ndarray],
    *,
    stats: ReferenceStats,
    extra: dict[str, np.ndarray],
    rc_config: dict[str, object],
    pulse_state: PulseState,
    cauchy_references: tuple[np.ndarray, ...],
    compact_index,
    device: str,
    batch_size: int,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    similarity = stage_similarities(
        atoms["base_pooled"], stats.prototypes, device=device, batch_size=batch_size
    )
    known = np.ones(1000, dtype=bool)
    msps = original_msps_confidence(similarity, known)
    candidates = select_candidates(similarity, known, int(rc_config["k"]))
    support = gather_candidate_support(similarity, candidates)
    calibrated = calibrate_candidates(
        support,
        candidates,
        extra["class_sorted"],
        extra["global_sorted"],
        float(rc_config["class_shrinkage"]),
    )
    rc = candidate_confidence(
        calibrated,
        candidates,
        similarity.argmax(axis=2),
        weights=np.asarray(rc_config["weights"], dtype=np.float32),
        fusion=str(rc_config["fusion"]),
        variance_penalty=float(rc_config["lambda"]),
        consensus_reward=float(rc_config["eta"]),
    )
    uniform, deep, localized = semantic_geometry(
        atoms["base_pooled"], atoms["localized"], stats.prototypes[-1],
        device=device, batch_size=batch_size,
    )
    knn = mean_neighbor_confidence(
        compact_index, atoms["meanview"], k=K, nprobe=NPROBE, batch_size=1024
    )
    component = pulse_state.components(
        atoms["shift_ratio"], uniform, localized, knn
    )
    cauchy_input = {
        f"_stage_support_{layer}": similarity[:, layer, :].max(axis=1)
        for layer in range(4)
    }
    confidence = {
        "PULSE": -sum(component.values()),
        "DeepPrototype": deep,
        "RC-MSPS": rc,
        "MSPS": msps,
        "CauchyStageTail": cauchy_stage_confidence(cauchy_input, cauchy_references),
    }
    diagnostics = {
        **{f"ood_{name}": score for name, score in component.items()},
        "uniform_raw": uniform,
        "localized_raw": localized,
        "knn_raw": knn,
        "deep_raw": deep,
        **cauchy_input,
    }
    return confidence, diagnostics


def main() -> None:
    args = parse_args()
    atom_paths = {name: args.atom_dir / f"{name}.npz" for name in ("canonical", *FAMILIES)}
    paths = [
        *atom_paths.values(), args.rc_state, args.rc_config, args.pulse_state,
        args.cauchy_state, args.compact_index, args.output,
    ]
    assert_no_final_access(paths)
    source_manifest = (args.atom_dir / "canonical.jsonl").read_bytes()
    source_labels = load_atoms(atom_paths["canonical"])["labels"]
    for family in FAMILIES:
        if (args.atom_dir / f"{family}.jsonl").read_bytes() != source_manifest:
            raise ValueError(f"path alignment failed for {family}")
        if not np.array_equal(load_atoms(atom_paths[family])["labels"], source_labels):
            raise ValueError(f"label alignment failed for {family}")

    import faiss

    faiss.omp_set_num_threads(args.threads)
    compact_index = faiss.read_index(str(args.compact_index))
    stats, extra = ReferenceStats.load(args.rc_state)
    rc_config = json.loads(args.rc_config.read_text(encoding="utf-8"))["hyperparameters"]
    pulse_state = PulseState.load(args.pulse_state)
    cauchy_references = load_cauchy_references(args.cauchy_state)
    scores, diagnostics = {}, {}
    for family, path in atom_paths.items():
        scores[family], diagnostics[family] = method_scores(
            load_atoms(path), stats=stats, extra=extra, rc_config=rc_config,
            pulse_state=pulse_state, cauchy_references=cauchy_references,
            compact_index=compact_index, device=args.device, batch_size=args.batch_size,
        )

    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    for family in FAMILIES:
        payload = {"labels": source_labels}
        for method in FROZEN_LEGACY_MACRO_AUROC:
            rows.append(
                {
                    "proxy": family,
                    "method": method,
                    "legacy_macro_AUROC": FROZEN_LEGACY_MACRO_AUROC[method],
                    **metric_row(scores["canonical"][method], scores[family][method]),
                }
            )
            payload[f"id_confidence_{method}"] = scores["canonical"][method]
            payload[f"proxy_confidence_{method}"] = scores[family][method]
        for name, value in diagnostics["canonical"].items():
            payload[f"id_{name}"] = value
        for name, value in diagnostics[family].items():
            payload[f"proxy_{name}"] = value
        np.savez_compressed(args.output / f"{family}_scores.npz", **payload)
    table = pd.DataFrame(rows)
    fidelity_rows = []
    legacy = pd.Series(FROZEN_LEGACY_MACRO_AUROC)
    for family, group in table.groupby("proxy", sort=False):
        proxy = group.set_index("method")["AUROC"].reindex(legacy.index)
        fidelity_rows.append(
            {
                "proxy": family,
                "spearman": float(spearmanr(legacy, proxy).statistic),
                "kendall": float(kendalltau(legacy, proxy).statistic),
                "mean_proxy_AUROC": float(proxy.mean()),
                "worst_proxy_AUROC": float(proxy.min()),
                "pulse_proxy_AUROC": float(proxy["PULSE"]),
            }
        )
    fidelity = pd.DataFrame(fidelity_rows).sort_values(
        ["spearman", "kendall", "pulse_proxy_AUROC"], ascending=False
    )
    eligible = fidelity[(fidelity["spearman"] > 0.0) & (fidelity["kendall"] > 0.0)]
    selected = eligible.head(3)["proxy"].tolist()
    fidelity["selected"] = fidelity["proxy"].isin(selected)
    table.to_csv(args.output / "proxy_method_metrics.csv", index=False)
    fidelity.to_csv(args.output / "proxy_fidelity.csv", index=False)
    lock = {
        "selected_proxies": selected,
        "selection_rule": "top three image-space proxies with positive frozen-method Spearman and Kendall fidelity",
        "frozen_legacy_macro_AUROC": FROZEN_LEGACY_MACRO_AUROC,
        "legacy_values_used_for_proxy_infrastructure_only": True,
        "candidate_selection_may_read_legacy_ood": False,
        "source_manifest_sha256": hashlib.sha256(source_manifest).hexdigest(),
        "atom_sha256": {name: sha256(path) for name, path in atom_paths.items()},
        "final_benchmark_accessed": False,
    }
    serialized = json.dumps(lock, sort_keys=True, separators=(",", ":")).encode()
    lock["payload_sha256"] = hashlib.sha256(serialized).hexdigest()
    (args.output / "proxy_lock.json").write_text(
        json.dumps(lock, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(table.to_string(index=False))
    print(fidelity.to_string(index=False))
    print(json.dumps(lock, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
