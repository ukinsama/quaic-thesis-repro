# %% [markdown]
# # 方策バリアントの厳密比較（全列挙・統計誤差ゼロ）
#
# ## 目的（ユーザーの意図）
# 「強いAIの作り方」の報告として、**ある程度検証した値**を載せたい。
# 今夜作った固定方策のバリアントを、9,800局の全列挙（= 乱数空間の完全被覆）で
# 厳密に比較する。信頼区間は不要——これらは推定値ではなく真値である。
#
# ## 測る量（すべて Oracle vs RandomColorEstimator、同一方策を両者が共有）
# - score: 上界 Oracle が下界 Random に対して取る厳密スコア
# - 敗率: 「オラクルなのにランダムに負ける」率。今夜ずっと追いかけていた指標
# - 引分率: 決着力の裏返し
# これは方策の「色情報を勝ちに変換する能力」の厳密な測定になる。
#
# ## バリアント（今夜作った/見つけたもの）
# 1. recovered(pen=0.15)  : 論文の現行固定方策
# 2. recovered(pen=0.6)   : ゲート守備係数のみ調整（重みは同一・色感度は数学的に不変）
# 3. color_aware_124419   : 既存チェックポイント棚卸しで見つかった候補A
# 4. asym_defensive       : 守備シェーピング再学習（敗率は下がるが色感度が2割減で不採用）
# 5. asym_nodraw          : 守備シェーピング + 引分罰(-0.7)/時間罰(-0.005) ← 未監査
#
# 実行: PYTHONPATH=... python scripts/evaluation/policy_variants_exhaustive.py

# %% セットアップ
from __future__ import annotations

import json
import time
from pathlib import Path

from exhaustive_score import enumerate_score  # 同ディレクトリの全列挙エンジン

QUGEISTER_ROOT = Path(__file__).resolve().parents[2]
CA = "experiments/handyrl/color_aware"

VARIANTS = {
    "recovered(pen=0.15)  現行": (
        "experiments/handyrl/best/"
        "handyrl_perfect_info_color_aware_resnet_4blk_20260204_145150_RECOVERED/best_model.pth",
        None,
    ),
    "recovered(pen=0.6)   調整版": (
        "experiments/handyrl/best/"
        "handyrl_perfect_info_color_aware_resnet_4blk_20260204_145150_RECOVERED/best_model.pth",
        0.6,
    ),
    "color_aware_124419   候補A": (
        f"{CA}/handyrl_perfect_info_color_aware_resnet_4blk_20260204_124419/best_model.pth",
        None,
    ),
    "asym_defensive       守備版": (
        f"{CA}/handyrl_perfect_info_color_aware_asymmetric_resnet_4blk_20260716_000531/best_model.pth",
        None,
    ),
    "asym_nodraw          守備+引分罰": (
        f"{CA}/handyrl_perfect_info_color_aware_asymmetric_resnet_4blk_20260716_010420/best_model.pth",
        None,
    ),
}


# %% 実行
if __name__ == "__main__":
    rows = []
    print("■ 方策バリアントの厳密比較（Oracle vs RandomColorEstimator / 9,800局 全列挙）")
    print(f"{'方策':<32}{'score':>9}{'敗率':>9}{'引分率':>9}  W-L-D")
    for label, (rel, pen) in VARIANTS.items():
        if not (QUGEISTER_ROOT / rel).exists():
            print(f"{label:<32}  (checkpoint なし)")
            continue
        t0 = time.time()
        r = enumerate_score("Oracle", rel, pen=pen, workers=14)
        r["label"] = label
        rows.append(r)
        dr = r["draws"] / r["n_games"]
        print(f"{label:<32}{r['score']:>9.4f}{r['loss_rate']*100:>8.2f}%{dr*100:>8.2f}%  "
              f"{r['wins']}-{r['losses']}-{r['draws']}  [{time.time()-t0:.0f}s]")

    out = Path("/home/ks/QuAic/backend/data/experiments/exhaustive_policy_variants.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[write] {out}")
    print("\n注: これらは標本ではなく全列挙の厳密値。信頼区間は存在しない。")
