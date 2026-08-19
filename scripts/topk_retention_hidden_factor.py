#!/usr/bin/env python3
"""ゲームscoreを説明する「見えていないファクター」= Top-K世界保持率の検証.

## 動機（ユーザーの直観）
piece accuracy とゲームscoreは厳密値で非単調（ResNet精度最高だがscore3位、GRUは精度平凡だがscore1位）。
「見えていないファクターがある」。候補: **TopK(k=5)アダプターが実際に使う量 = 真の色世界が上位K個に
入る率（Top-K保持率）**。盤面精度（駒ごとargmax）ではなく、70世界の順位を見る。

## 検証
holdout の各サンプル（捕獲済み駒-1を除外し active world 集合上で評価）について、
各推定器の good_probs から真の色世界のランクを求め、Top-5/Top-10 保持率・true_mass・実効世界数を測る。
importance_flags で「決定的 vs 通常」に層別する。
仮説: **HistoryGRU は Top-K保持率で他を上回る**（→ アダプターが正しい世界を掴む → score1位）。

## 再利用
run_paper_true_world_recall.py の _world_metrics（§7診断軸と同じ計算）をそのまま使う。
推定器ロードも同スクリプトの CheckpointColorEstimator を流用。

実行: cd backend && venv/bin/python scripts/topk_retention_hidden_factor.py
"""

from __future__ import annotations

import os
import pickle
import sys
from pathlib import Path

import numpy as np

QUAIC_ROOT = Path(__file__).resolve().parents[2]
BACKEND_ROOT = QUAIC_ROOT / "backend"
QUGEISTER = QUAIC_ROOT.parent / "Qugeister_clean"
for p in (BACKEND_ROOT, BACKEND_ROOT / "scripts", QUGEISTER, QUGEISTER / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
os.environ.setdefault("DEBUG", "true")

from run_paper_true_world_recall import PaperCheckpointEstimator, _world_metrics  # noqa: E402

HOLDOUT = Path(
    "/home/ks/Qugeister_clean/trajectories/"
    "color_gated_decision_sensitive_20260622_holdout_80.pkl"
)
EXP = BACKEND_ROOT / "data/experiments"
HISTORY_LEN = 8
TOP_K = 5


def load_samples():
    """(single_obs, history[T,448], colors[8], is_critical) を重複除去して返す。"""
    trajs = pickle.load(open(HOLDOUT, "rb"))  # noqa: S301
    out = []
    for t in trajs:
        for side in ("A", "B"):
            states = t.get(f"states_{side}", [])
            colors = t.get(f"true_colors_{side}", [])
            flags = t.get(f"importance_flags_{side}", [])
            hist: list[np.ndarray] = []
            prev = None
            for s, c, f in zip(states, colors, flags):
                flat = np.asarray(s, dtype=np.float32).reshape(-1)
                if flat.shape[0] != 448 or len(c) != 8:
                    continue
                if prev is not None and np.array_equal(prev, flat):
                    continue
                prev = flat
                hist.append(flat)
                window = hist[-HISTORY_LEN:]
                pad = [np.zeros_like(window[0])] * (HISTORY_LEN - len(window)) + window
                crit = bool(f.get("capture_available") or f.get("own_good_escape_available")
                            or f.get("enemy_escape_threat"))
                out.append((flat, np.stack(pad), np.asarray(c, dtype=np.int64), crit))
    return out


def eval_estimator(est, samples, mask):
    """mask=True のサンプルで Top-K保持率・true_mass・実効世界数・piece_acc を平均。"""
    acc = {"topk_recall": [], "top1_recall": [], "true_mass": [],
           "effective_worlds": [], "piece_accuracy": []}
    is_hist = (getattr(est, "model_class", "") == "HistoryGRUColorEstimator")
    for (single, history, colors, _c), m in zip(samples, mask):
        if not m:
            continue
        state = history if is_hist else single
        good = est.get_good_probs_from_state(state)
        wm = _world_metrics(good, colors, top_k=TOP_K, random_tie=False)
        if wm is None:
            continue
        for k in acc:
            acc[k].append(wm[k])
    return {k: (float(np.mean(v)) if v else float("nan")) for k, v in acc.items()}


def main() -> int:
    samples = load_samples()
    crit = np.array([s[3] for s in samples])
    print(f"holdout: {len(samples)} サンプル / 決定的 {int(crit.sum())} / 通常 {int((~crit).sum())}")

    roster = {
        "MLPColorEstimator": (EXP / "color_gated_mlp_color_estimator_20260622.pth", 0.778),
        "ResNetColorEstimator": (EXP / "color_gated_resnet_color_estimator_20260622.pth", 0.763),
        "QuantumColorEstimator": (EXP / "color_gated_quantum_color_estimator_20260622.pth", 0.719),
        "HistoryGRUColorEstimator": (EXP / "color_gated_history_gru_color_estimator_20260622.pth", 0.789),
    }

    print(f"\n■ 全体（Top-{TOP_K}保持率 = アダプターが真の世界を掴む率）と厳密ゲームscoreの対応")
    print(f"{'推定器':<17}{'piece_acc':>10}{'Top5保持':>10}{'true_mass':>11}{'N_eff':>8}{'ゲームscore':>12}")
    overall = {}
    for name, (ckpt, gscore) in roster.items():
        est = PaperCheckpointEstimator(ckpt, device="cpu", name=name)
        r = eval_estimator(est, samples, np.ones(len(samples), bool))
        overall[name] = (r, gscore)
        print(f"{name:<17}{r['piece_accuracy']*100:>9.2f}%{r['topk_recall']*100:>9.2f}%"
              f"{r['true_mass']*100:>10.2f}%{r['effective_worlds']:>8.2f}{gscore:>12.3f}")

    # 相関: 各診断量とゲームscoreの Spearman 順位一致を見る
    import itertools
    names = list(roster)
    gscores = {n: roster[n][1] for n in names}
    print("\n■ ゲームscore との順位一致（4推定器の順位が完全一致なら「その量がscoreを説明」）")
    for metric in ("piece_accuracy", "topk_recall", "true_mass", "effective_worlds"):
        by_metric = sorted(names, key=lambda n: -overall[n][0][metric])
        by_score = sorted(names, key=lambda n: -gscores[n])
        # ペアワイズ一致率
        agree = sum(1 for a, b in itertools.combinations(names, 2)
                    if (overall[a][0][metric] > overall[b][0][metric]) == (gscores[a] > gscores[b]))
        total = len(list(itertools.combinations(names, 2)))
        eff = "（効果小さいほど良い→符号反転）" if metric == "effective_worlds" else ""
        mark = " ★score順と完全一致" if by_metric == by_score else ""
        print(f"  {metric:<18} score順一致 {agree}/{total}  順位={[n.replace('Color','') for n in by_metric]}{mark}{eff}")

    print(f"\n■ 決定的 vs 通常での Top-{TOP_K}保持率")
    print(f"{'推定器':<17}{'決定的':>9}{'通常':>9}{'差':>10}")
    for name, (ckpt, _g) in roster.items():
        est = PaperCheckpointEstimator(ckpt, device="cpu", name=name)
        rc = eval_estimator(est, samples, crit)
        rn = eval_estimator(est, samples, ~crit)
        print(f"{name:<17}{rc['topk_recall']*100:>8.2f}%{rn['topk_recall']*100:>8.2f}%"
              f"{(rc['topk_recall']-rn['topk_recall'])*100:>+9.2f}pt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
