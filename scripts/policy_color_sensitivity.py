# %% [markdown]
# # 固定方策の色感度テスト（新旧モデルの比較）
#
# ## 目的（ユーザーの指摘）
# oracle_floor_tuning.py の再学習で敗率は半減したが、測ったのは Oracle vs Random の2点だけ。
# 論文が固定方策に要求するのは**色感度**——「色推定器を差し替えたときの差が対戦結果へ伝播すること」
# （背景§2.3.4、設計§4.4）である。守備シェーピング（「色が不確かなら捕獲するな」）は
# 捕獲＝色情報がゲームに効く主要経路を減らすため、**色感度を殺していないか**の検証が要る。
#
# ## 測るもの（論文の色感度根拠に対応）
# 1. **中間の分離**: learned estimator 4種が RandomColorEstimator(下界) と OracleColorEstimator(上界) の間に入るか（§7.5）
# 2. **null control**: LowParamColorEstimator（19パラメータ・盤面を見ない）が下界に張り付くか（§7.5）
# 3. **方向性コントロール**: good/bad を反転させると score が落ちるか（§7.7）
#
# ## 比較対象
# - OLD_recovered: 現行の固定方策（論文の実験で使用中）
# - NEW_asym: 守備シェーピング再学習モデル（敗率1.20%）
# 同一の推定器集合・同一アダプター・同一 seed で測る。

# %% セットアップ
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch

QUGEISTER_ROOT = Path(__file__).resolve().parents[2]
QUAIC_ROOT = QUGEISTER_ROOT.parent / "QuAic"
for cand in (QUGEISTER_ROOT, QUGEISTER_ROOT / "src", QUAIC_ROOT / "backend"):
    if str(cand) not in sys.path:
        sys.path.insert(0, str(cand))
import os

os.environ.setdefault("DEBUG", "true")

from qugeister.evaluation.color_pipeline import (  # noqa: E402
    OracleEstimator,
    RandomEstimator,
    Top1Agent,
    TopKAgent,
)
from scripts.evaluation.domain_shift_audit import load_policy  # noqa: E402
from scripts.evaluation.oracle_floor_tuning import play_game_with_reason  # noqa: E402

DEVICE = "cpu"
EXPDIR = QUAIC_ROOT / "backend/data/experiments"
LEARNED = {
    "MLPColorEstimator": EXPDIR / "color_gated_mlp_color_estimator_20260622.pth",
    "ResNetColorEstimator": EXPDIR / "color_gated_resnet_color_estimator_20260622.pth",
    "QuantumColorEstimator": EXPDIR / "color_gated_quantum_color_estimator_20260622.pth",
    "HistoryGRUColorEstimator": EXPDIR / "color_gated_history_gru_color_estimator_20260622.pth",
    "LowParamColorEstimator": EXPDIR / "color_gated_prior_only_color_estimator_20260623.pth",  # null control
}
POLICIES = {
    "OLD_recovered": QUGEISTER_ROOT / "experiments/handyrl/best/"
    "handyrl_perfect_info_color_aware_resnet_4blk_20260204_145150_RECOVERED/best_model.pth",
    "NEW_asym": QUGEISTER_ROOT / "experiments/handyrl/color_aware/"
    "handyrl_perfect_info_color_aware_asymmetric_resnet_4blk_20260716_000531/best_model.pth",
    # 攻撃シェーピング+引分罰（修論 表: 色感度 0.173 の再現用。roster 非採用）
    "NEW_offense_nodraw": QUGEISTER_ROOT / "experiments/handyrl/color_aware/"
    "handyrl_perfect_info_color_aware_resnet_4blk_20260716_142543/best_model.pth",
}

# QuAic 側の checkpoint ラッパを流用（論文リーグと同じロード経路）
sys.path.insert(0, str(QUAIC_ROOT / "backend/scripts"))
from run_paper_color_elo import QuAicCheckpointEstimator  # noqa: E402


# %% 対戦: 任意の推定器 vs RandomColorEstimator（同一方策を共有）
def score_vs_random(policy, estimator, name: str, invert: bool = False,
                    n_games: int = 100, seeds=(42, 43, 44)):
    """estimator を積んだエージェント vs RandomColorEstimator の score を測る。

    論文と同じく Random/Oracle は Top1、learned は TopK(k=5) を使う。
    """
    if invert:
        estimator.invert = True
    if isinstance(estimator, (OracleEstimator, RandomEstimator)):
        agent = Top1Agent(estimator, policy, DEVICE, name)
    else:
        agent = TopKAgent(estimator, policy, DEVICE, name=name, top_k=5)
    opponent = Top1Agent(RandomEstimator(), policy, DEVICE, "RandomColorEstimator")

    wins = losses = draws = 0
    for seed in seeds:
        np.random.seed(seed)
        torch.manual_seed(seed)
        import random as _random

        _random.seed(seed)
        for i in range(n_games):
            winner, _reason, _t = play_game_with_reason(agent, opponent, swap=(i % 2 == 1))
            if winner == "A":
                wins += 1
            elif winner == "B":
                losses += 1
            else:
                draws += 1
    n = wins + losses + draws
    return {"name": name, "W-L-D": (wins, losses, draws), "score": (wins + 0.5 * draws) / n, "n": n}


# %% 方策1つぶんの色感度プロファイルを測る
def sensitivity_profile(policy_path: Path, label: str, n_games: int = 100):
    policy = load_policy(policy_path, device=DEVICE, policy_arch="color-gated-resnet",
                         color_gate_scale=1.0)
    rows = []
    t0 = time.time()
    rows.append(score_vs_random(policy, OracleEstimator(), "OracleColorEstimator(上界)", n_games=n_games))
    for name, ckpt in LEARNED.items():
        est = QuAicCheckpointEstimator(ckpt, device=DEVICE)
        rows.append(score_vs_random(policy, est, name, n_games=n_games))
    print(f"\n===== {label}  ({time.time() - t0:.0f}s) =====")
    for r in rows:
        w, l, d = r["W-L-D"]
        print(f"  {r['name']:22s} score={r['score']:.3f}  W-L-D={w}-{l}-{d}")
    print(f"  {'RandomColorEstimator(下界)':22s} score=0.500  （定義）")
    return rows


# %% 実行: 新旧の色感度を比較
if __name__ == "__main__":
    for label, path in POLICIES.items():
        sensitivity_profile(path, label)

# %% [markdown]
# # 測定結果と結論（2026-07-16）
#
# ## 1. 色感度プロファイル（各300戦, seed 42-44, 対 RandomColorEstimator+Top1）
#
# | 推定器 | OLD_recovered | NEW_asym |
# |---|---|---|
# | OracleColorEstimator（上界, Top1） | 0.928 | 0.940 |
# | HistoryGRUColorEstimator (TopK) | 0.803 | 0.782 |
# | ResNetColorEstimator (TopK) | 0.775 | 0.732 |
# | MLPColorEstimator (TopK) | 0.772 | 0.737 |
# | QuantumColorEstimator (TopK) | 0.725 | 0.705 |
# | LowParamColorEstimator (TopK, null) | 0.492 | 0.548 |
#
# ## 2. ★重大な発見: null control には**アダプターの交絡**がある
# 論文プロトコルでは RandomColorEstimator=Top1、learned/null=TopK(k=5) とアダプターが違う。
# **推定器を同一（全駒0.5の無情報）にしてアダプターだけ変える**決定実験（各1500戦, seed 42-56）:
#
# | 対戦 | score | 95%CI |
# |---|---|---|
# | Random+Top1 vs Random+Top1（自己対戦） | 0.515 | [0.499, 0.531] ≒0.5 ✓ |
# | **Random+TopK5 vs Random+Top1（アダプター差のみ・色情報ゼロ）** | **0.556** | [0.537, 0.575] |
# | LowParamColorEstimator+TopK5（null control, 15seed） | 0.550 | [0.530, 0.570] |
#
# → **null control の「浮上」は色情報ではなくアダプター差で説明できる**。
# LowParamColorEstimator(0.550) は色情報ゼロの TopK 床(0.556) と統計的に区別できず、
# 19 パラメータから色情報を一切引き出していない＝**null control としては正しく機能している**。
#
# ## 3. ★論文の記述の問題（seed 依存）
# LowParamColorEstimator の seed 別 score（OLD方策・各100戦）: 論文seed(42-44)平均 **0.492** / 他12seed平均 **0.565**
# / 全15seed **0.550**（seed間SD 0.049）。論文§7.5 の「score 0.502 にとどまり…有意に浮き上がらなかった」は
# **たまたま低い3 seed を引いた結果**（2σ相当）。正しい主張は
# 「LowParamColorEstimator は**同一アダプターの床**（TopK床 0.556）と区別できない＝色情報を引き出していない」であり、
# 比較対象を 0.500（Top1床）ではなく TopK 床に取るべき。
#
# ## 4. 色感度は新モデルでも保たれるか → **保たれる**（ただしアダプター床で測ること）
#
# | | OLD_recovered | NEW_asym |
# |---|---|---|
# | TopK床（色情報ゼロ・1500戦） | 0.556 | 0.578 |
# | learned 帯 | 0.725–0.803 | 0.705–0.782 |
# | **床からの上乗せ（色情報の正味の価値）** | **+0.169〜+0.247** | **+0.127〜+0.204** |
# | learned 帯の幅（推定器間の識別幅） | 0.078 | 0.077 |
# | Oracle(Top1) − Random(Top1) | 0.428 | 0.440 |
#
# → 順序（Oracle > learned > 床）・識別幅（0.078）は維持。ただし
# **床からの上乗せは約2割縮む**（守備シェーピングで捕獲＝色情報の効く経路が減るため）。
# 上下界の間隔は逆に広がる（0.428→0.440）。
# 「敗率半減の代償として、中間帯の色感度がやや圧縮される」がトレードオフの正体。

# %% [markdown]
# # 追加: 方策そのものの色感度（因果テスト）
#
# ## 動機（ユーザー: 「色感度自体を確かめるテストはあるか？同等なら採用したい」）
# 定義が複数あり結論が割れるため、**推定器の質に依存しない純粋な方策の性質**を測る。
# 同じ方策に「真の色(Oracle)」と「完全に反転した色(AntiOracle)」を与え、その反応幅を見る:
#
#   色感度 := score(Oracle) − score(AntiOracle)
#
# 色を無視する方策なら Oracle = AntiOracle = 0.5（反応幅ゼロ）。
# 色に反応する方策ほど Oracle は上に、AntiOracle は下に開く。
# 論文§7.7の方向性コントロール（反転で score 低下）を、推定器誤差を挟まない最強の形にしたもの。

# %% 反転オラクル（完全に間違った色を与える上界の裏返し）
class AntiOracleEstimator(OracleEstimator):
    """真の色を反転して返す。good↔bad を取り違える「最悪の推定器」。"""

    def get_good_probs(self, env, player):
        return 1.0 - super().get_good_probs(env, player)


# %% 実行: 新旧の因果的色感度（1500戦 = seed 42-56 × 100）
if __name__ == "__main__":
    import math

    print("\n=== 方策の因果的色感度: Oracle vs AntiOracle（各1500戦, 相手は常に Random+Top1）===")
    for label, path in POLICIES.items():
        pol = load_policy(path, device=DEVICE, policy_arch="color-gated-resnet",
                          color_gate_scale=1.0)
        out = {}
        for tag, est in (("Oracle(真の色)", OracleEstimator()),
                         ("AntiOracle(反転色)", AntiOracleEstimator())):
            r = score_vs_random(pol, est, tag, n_games=100, seeds=tuple(range(42, 57)))
            w, l, d = r["W-L-D"]
            n = r["n"]
            ex2 = (w + 0.25 * d) / n
            se = math.sqrt(max(ex2 - r["score"] ** 2, 0) / n)
            out[tag] = (r["score"], se)
            print(f"  {label:14s} {tag:20s} score={r['score']:.3f} ±{1.96 * se:.3f}  W-L-D={w}-{l}-{d}")
        sens = out["Oracle(真の色)"][0] - out["AntiOracle(反転色)"][0]
        se_d = math.sqrt(out["Oracle(真の色)"][1] ** 2 + out["AntiOracle(反転色)"][1] ** 2)
        print(f"  → {label} の色感度 = {sens:.3f} ± {1.96 * se_d:.3f}\n")
