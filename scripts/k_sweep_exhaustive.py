# %% [markdown]
# # アダプター k のスイープ（分解能を最大化する k を探す・全列挙）
#
# ## 目的（ユーザーの意図）
# 攻撃版方策（引分罰）で帯の幅が 0.053→0.105 に広がった。さらに分解能を上げる要因として、
# 論文§4.3 自身が「Top-1=誤差増幅 / 全70=希釈、その中間が k」と書いている**アダプター k**を振る。
# k は再学習不要の評価側ノブなので、全列挙で厳密な「k → 帯の幅」曲線を引く。
#
# ## 仮説
# learned 推定器(58-62%)は score-精度曲線の飽和領域のすぐ上に固まっている。
# TopK の hedging（k>1）が「下手でも複数賭ければ当たる」を許して差を潰している。
# → **k を小さくすると誤差が直接効き、推定器の帯が広がる**（分解能↑）。
# 副次: 全員 k=1 にすると null control の交絡（Random=Top1 vs learned=TopK の床ズレ）も消える。
#
# ## 測る量
# 各 k で learned 4種の厳密 score を全列挙し、
#   帯の幅 := max(learned score) − min(learned score)
# が最大になる k を探す。上界 Oracle・下界 Random・null LowParam も併記。

# %% セットアップ
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from exhaustive_score import enumerate_score

# 攻撃版（color_aware + 引分罰 -0.7 + 時間罰 -0.005、非対称罰なし）= 今夜の最良方策
POLICY_REL = (
    "experiments/handyrl/color_aware/"
    "handyrl_perfect_info_color_aware_resnet_4blk_20260716_142543/best_model.pth"
)
LEARNED = ["HistoryGRUColorEstimator", "MLPColorEstimator", "ResNetColorEstimator", "QuantumColorEstimator"]
K_VALUES = [1, 3, 5, 10, 20, 30]  # §8主張点(5/10/20/30)検証のため調整
WORKERS = 14

# %% 実行
if __name__ == "__main__":
    # 参照点（k非依存）: Oracle上界・Random下界・LowParam null は Top1 相当で1回だけ測る
    print("■ 参照点（全列挙・厳密値）")
    ref = {}
    for name in ("Oracle",):
        r = enumerate_score(name, POLICY_REL, workers=WORKERS)
        ref[name] = r["score"]
        print(f"  {name:16s} score={r['score']:.4f}（上界）")
    # LowParam は null control。k で床が動くので k ごとに測る（下記ループ内）

    print("\n■ k スイープ（learned 4種の帯の幅を最大化する k を探す）")
    print(f"{'k':>4} | " + " ".join(f"{n.replace('Color',''):>9}" for n in LEARNED)
          + f" | {'LowParam':>9} {'帯の幅':>7}")
    results = []
    for k in K_VALUES:
        row = {"k": k, "scores": {}}
        t0 = time.time()
        for name in LEARNED:
            r = enumerate_score(name, POLICY_REL, workers=WORKERS, top_k=k)
            row["scores"][name] = r["score"]
        # null control もこの k で（アダプター床の位置を確認）
        rlp = enumerate_score("LowParamColorEstimator", POLICY_REL, workers=WORKERS, top_k=k)
        row["scores"]["LowParamColorEstimator"] = rlp["score"]
        vals = [row["scores"][n] for n in LEARNED]
        row["band_width"] = max(vals) - min(vals)
        row["elapsed"] = round(time.time() - t0, 0)
        results.append(row)
        print(f"{k:>4} | " + " ".join(f"{row['scores'][n]:>9.4f}" for n in LEARNED)
              + f" | {rlp['score']:>9.4f} {row['band_width']:>7.4f}  [{row['elapsed']:.0f}s]")

    best = max(results, key=lambda r: r["band_width"])
    print(f"\n→ 帯の幅が最大: k={best['k']} で {best['band_width']:.4f}"
          f"（現行 k=5 の {next(r['band_width'] for r in results if r['k']==5):.4f} と比較）")

    out = Path("/home/ks/QuAic/backend/data/experiments/k_sweep_offense.json")
    out.write_text(json.dumps({"policy": POLICY_REL, "oracle": ref.get("Oracle"),
                               "results": results}, indent=2, ensure_ascii=False),
                   encoding="utf-8")
    print(f"[write] {out}")
