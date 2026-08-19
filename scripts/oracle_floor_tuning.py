# %% [markdown]
# # Oracle–Random 完全分離のためのゲートチューニング調査
#
# ## 目的（ユーザーの意図）
# 修論のベンチマークで、上界 OracleColorEstimator が下界 RandomColorEstimator に **1敗もしない**
# （score が敗北ゼロで分離する）固定方策構成を、ColorGatedResNetPolicy の
# ゲートパラメータのチューニングだけで作れるかを調べる。
# 現状（color_gate_scale=1.0）は score 0.925（264勝-9敗-27分/300戦）で、9敗が残る。
#
# ## 前提
# - リーグでは全エージェントが**同一の方策インスタンス**を共有する（固定 Strategy Bundle）。
#   したがってパラメータ変更は Oracle 側と Random 側の両方に同時に適用する。
# - RandomEstimator は全駒 0.5 を返す（色情報なし）。Random 側も自駒の色は見えているので、
#   自分の脱出・餌付けはフルに指せる。誤るのは相手駒の色評価だけ。
# - チューニング対象は ColorGatedResNetPolicy の 4 つの平文属性:
#   color_gate_scale / capture_good_bonus / capture_bad_penalty / own_good_safety_penalty
#   （ネットワーク重みは一切変えない）。
#
# ## 検算（トイ確認）
# ベースライン（scale=1.0, seed 42/43/44, 各100戦）が論文の 264-9-27 を再現するかを
# まず確認してから、スイープに進む。

# %% セットアップ
from __future__ import annotations

import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch

QUGEISTER_ROOT = Path(__file__).resolve().parents[2]
for cand in (QUGEISTER_ROOT, QUGEISTER_ROOT / "src"):
    if str(cand) not in sys.path:
        sys.path.insert(0, str(cand))

from qugeister.env.geister import GeisterEnvV2  # noqa: E402
from qugeister.evaluation.color_pipeline import (  # noqa: E402
    OracleEstimator,
    RandomEstimator,
    Top1Agent,
)
from qugeister.evaluation.game_runner import check_escape_victory  # noqa: E402
from scripts.evaluation.domain_shift_audit import load_policy  # noqa: E402

POLICY_PATH = (
    QUGEISTER_ROOT
    / "experiments/handyrl/best/"
    "handyrl_perfect_info_color_aware_resnet_4blk_20260204_145150_RECOVERED/"
    "best_model.pth"
)
DEVICE = "cpu"


# %% 勝敗理由つきの1ゲーム実行
# game_runner.play_single_game と同じ進行だが、終局時の env を見て
# 「どの勝利条件で決着したか」を分類して返す（敗因分析のため）。
def play_game_with_reason(agent_a, agent_b, max_turns: int = 200, swap: bool = False):
    env = GeisterEnvV2(max_turns=max_turns)
    env.reset()

    cur_a, cur_b = (agent_b, agent_a) if swap else (agent_a, agent_b)
    for ag in (cur_a, cur_b):
        if hasattr(ag, "reset"):
            ag.reset()

    while not env.game_over:
        player = env.current_player
        agent = cur_a if player == "A" else cur_b
        action = agent.select_action(env, player)
        if action is None:  # 合法手なし → 相手勝ち
            env.game_over = True
            env.winner = "B" if player == "A" else "A"
            break
        success, _ = env.step(action)
        if not success:  # 不正手 → 相手勝ち
            env.game_over = True
            env.winner = "B" if player == "A" else "A"
            break

    # --- 勝因の分類 ---
    if env.winner not in ("A", "B"):
        reason = "turn_limit_draw"
        winner_agent = None
    else:
        w = env.winner
        good = {"A": env._a_good_count, "B": env._b_good_count}
        bad = {"A": env._a_bad_count, "B": env._b_bad_count}
        total = {"A": len(env.pieces_a), "B": len(env.pieces_b)}
        loser = "B" if w == "A" else "A"
        if check_escape_victory(env, w):
            reason = "escape"  # 勝者の善玉が脱出
        elif good[loser] == 0:
            reason = "captured_all_good"  # 勝者が敗者の善玉4個を取り切った
        elif bad[w] == 0 and total[w] < 8:
            reason = "fed_all_bad"  # 勝者が自分の悪玉4個を取らせ切った
        else:
            reason = "forfeit"  # 合法手なし・不正手
        # swap している場合は agent_a / agent_b への割り当てを戻す
        winner_agent = ("B" if w == "A" else "A") if swap else w

    return winner_agent, reason, env.turn


# %% シリーズ実行（先後半々・seedごとに再現可能）
def run_series(policy_params: dict, n_games: int = 100, seeds=(42, 43, 44), max_turns: int = 200,
               policy_path: Path | None = None):
    """policy_params を適用した共有方策で Oracle(=A側) vs Random(=B側) を実行する。

    先後は league と同じく半々（swap を交互）。戻り値は集計辞書。
    policy_path を指定すると別チェックポイントの監査に使える（既定は論文の固定方策）。
    """
    policy = load_policy(
        policy_path or POLICY_PATH, device=DEVICE, policy_arch="color-gated-resnet",
        color_gate_scale=policy_params.get("color_gate_scale", 1.0),
    )
    # ゲートの残り3係数は平文属性なので、ロード後に上書きする（重みは不変）
    for key in ("capture_good_bonus", "capture_bad_penalty", "own_good_safety_penalty"):
        if key in policy_params:
            setattr(policy, key, float(policy_params[key]))

    oracle = Top1Agent(OracleEstimator(), policy, DEVICE, "OracleColorEstimator")
    randc = Top1Agent(RandomEstimator(), policy, DEVICE, "RandomColorEstimator")

    wins = losses = draws = 0
    loss_reasons: Counter = Counter()
    loss_turns: list[int] = []
    for seed in seeds:
        np.random.seed(seed)
        torch.manual_seed(seed)
        import random as _random

        _random.seed(seed)
        for i in range(n_games):
            winner, reason, turns = play_game_with_reason(
                oracle, randc, max_turns=max_turns, swap=(i % 2 == 1)
            )
            if winner == "A":
                wins += 1
            elif winner == "B":
                losses += 1
                loss_reasons[reason] += 1
                loss_turns.append(turns)
            else:
                draws += 1
    n = wins + losses + draws
    return {
        "params": policy_params,
        "n": n,
        "W-L-D": (wins, losses, draws),
        "score": (wins + 0.5 * draws) / n,
        "loss_reasons": dict(loss_reasons),
        "loss_turns": loss_turns,
    }


def show(res: dict) -> None:
    w, l, d = res["W-L-D"]
    print(
        f"params={res['params']}  score={res['score']:.3f}  "
        f"W-L-D={w}-{l}-{d}  敗因={res['loss_reasons']}"
    )


# %% ベースライン再現＋敗因分析（論文値 264-9-27 との突き合わせ）
if __name__ == "__main__":
    t0 = time.time()
    baseline = run_series({"color_gate_scale": 1.0})
    print(f"[baseline] {time.time() - t0:.0f}s")
    show(baseline)
    print("敗北時のターン数:", baseline["loss_turns"])

# %% [markdown]
# ## ベースラインの敗因（2026-07-15 実行結果）
# 267勝-10敗-23分（score 0.928）。敗因は 10 敗中 9 敗が captured_all_good
# （Oracle 側の善玉4個が取り切られる）、1 敗が forfeit。escape 負けはゼロ。
# 敗北ターンは 12〜30 が大半 = 序盤に善玉を突っ込んで取られている。
# → 仮説: own_good_safety_penalty（既定 0.15）が弱い。守備側の係数を上げるスイープを行う。

# %% スイープ: 守備係数とゲート倍率
if __name__ == "__main__":
    grid = [
        {"color_gate_scale": 1.0, "own_good_safety_penalty": 0.5},
        {"color_gate_scale": 1.0, "own_good_safety_penalty": 1.0},
        {"color_gate_scale": 1.0, "own_good_safety_penalty": 2.0},
        {"color_gate_scale": 2.0, "own_good_safety_penalty": 0.15},
        {"color_gate_scale": 2.0, "own_good_safety_penalty": 1.0},
        {"color_gate_scale": 4.0, "own_good_safety_penalty": 1.0},
        {"color_gate_scale": 1.0, "own_good_safety_penalty": 1.0, "capture_bad_penalty": 2.0},
    ]
    results = []
    for params in grid:
        t0 = time.time()
        res = run_series(params)
        print(f"[{time.time() - t0:.0f}s] ", end="")
        show(res)
        results.append(res)

# %% [markdown]
# ## スイープ第1弾の観察（2026-07-15）
# - own_good_safety_penalty だけが効く（0.15→1.0 で敗北 10→6、2.0 は引き分け・forfeit 過多で崩壊）
# - color_gate_scale と capture_bad_penalty は結果不変（ゲート差が既に Q 差を支配しており、
#   実質 (ゲート, Q) の辞書式順序で手が決まっているため飽和している）
# - ゲートには「善玉を取られる位置に置かない」守備項が存在しない
#   （own_good_risk は善玉で捕獲する手だけを罰する）→ 敗北ゼロには構造的な壁がある可能性
# 第2弾: penalty の細分と capture_good_bonus（相手善玉を速く取り切り試合を畳む）の組合せ。

# %% スイープ第2弾: penalty 細分 × 攻撃係数
if __name__ == "__main__":
    grid2 = [
        {"own_good_safety_penalty": p, "capture_good_bonus": g}
        for p in (0.6, 0.8, 1.0, 1.2, 1.5)
        for g in (1.0, 2.0)
    ]
    results2 = []
    for params in grid2:
        t0 = time.time()
        res = run_series(params)
        print(f"[{time.time() - t0:.0f}s] ", end="")
        show(res)
        results2.append(res)

# %% [markdown]
# ## スイープ第2弾の観察（2026-07-15）
# - 最良は pen=0.6, cgb=1.0 の 275-6-19（score 0.948）。ただし**敗北 6 が床**で、
#   pen をどう振っても captured_all_good を 4〜6 残す。
# - capture_good_bonus=2.0 は一貫して悪化（善玉で突っ込んで逆に取られる）。
# - 結論: 既存 4 係数のチューニングでは敗北ゼロに届かない。ゲートに守備項がないため。
#
# ## 第3弾: 訓練不要のゲート拡張（脅威マス回避項）の試作
# ColorGatedResNetPolicy を実験内でサブクラス化し、
# 「自善玉を、相手駒の隣接マス（次手番で取られ得るマス）へ動かす手」への罰則
# threat_penalty を追加する。相手駒の位置は色仮説に依存しない（ch2+ch3 の和）ので、
# この項は色感度を変えず、守備の事前分布だけを足す。重みは一切変えない。

# %% ゲート拡張のサブクラス
from qugeister.policies.resnet import ColorGatedResNetPolicy  # noqa: E402


class ThreatAwareColorGatedPolicy(ColorGatedResNetPolicy):
    """自善玉を相手駒の隣接マスへ進める手に罰則を足す、訓練不要のゲート拡張。"""

    def __init__(self, *args, threat_penalty: float = 1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.threat_penalty = float(threat_penalty)

    def _color_gate_bonus(self, state: torch.Tensor) -> torch.Tensor:
        bonus = super()._color_gate_bonus(state)
        if state.dim() == 1:
            state = state.unsqueeze(0)
        if state.dim() == 4:
            obs_hwc = state.permute(0, 2, 3, 1)
        else:
            obs_hwc = state.view(state.shape[0], 8, 8, self.n_channels)
        # 相手駒の位置（色仮説に依存しない: 仮説上の good/bad の和 = 相手駒マップ）
        opp = (obs_hwc[:, :, :, 2] + obs_hwc[:, :, :, 3]).clamp(max=1.0)  # [B,8,8]
        # 上下左右に1マス膨張 = 相手駒が次手番で到達（捕獲）できるマス
        threat = torch.zeros_like(opp)
        threat[:, 1:, :] += opp[:, :-1, :]
        threat[:, :-1, :] += opp[:, 1:, :]
        threat[:, :, 1:] += opp[:, :, :-1]
        threat[:, :, :-1] += opp[:, :, 1:]
        threat = threat.clamp(max=1.0).reshape(-1, 64)
        own_good_map = obs_hwc[:, :, :, 0].reshape(-1, 64)
        src_own_good = own_good_map[:, self.action_src_flat]
        # 相手駒マスへの侵入（=捕獲）は既存の own_good_risk が扱うので、隣接マスのみ罰する
        dest_threat = threat[:, self.action_dest_flat] * (
            1.0 - (obs_hwc.reshape(-1, 8, 8, self.n_channels)[:, :, :, 2]
                   + obs_hwc.reshape(-1, 8, 8, self.n_channels)[:, :, :, 3])
            .clamp(max=1.0).reshape(-1, 64)[:, self.action_dest_flat]
        )
        return bonus - self.color_gate_scale * self.threat_penalty * src_own_good * dest_threat


def run_series_threat(gate_params: dict, threat_penalty: float,
                      n_games: int = 100, seeds=(42, 43, 44)):
    """脅威回避項つき方策で Oracle vs Random を実行する（重みは同じ checkpoint）。"""
    base = load_policy(
        POLICY_PATH, device=DEVICE, policy_arch="color-gated-resnet",
        color_gate_scale=gate_params.get("color_gate_scale", 1.0),
    )
    policy = ThreatAwareColorGatedPolicy(
        n_channels=8, hidden_channels=128, n_blocks=4,
        color_gate_scale=gate_params.get("color_gate_scale", 1.0),
        threat_penalty=threat_penalty,
    )
    policy.load_state_dict(base.state_dict(), strict=False)
    policy.to(DEVICE).eval()
    for key in ("capture_good_bonus", "capture_bad_penalty", "own_good_safety_penalty"):
        if key in gate_params:
            setattr(policy, key, float(gate_params[key]))

    oracle = Top1Agent(OracleEstimator(), policy, DEVICE, "OracleColorEstimator")
    randc = Top1Agent(RandomEstimator(), policy, DEVICE, "RandomColorEstimator")
    wins = losses = draws = 0
    loss_reasons: Counter = Counter()
    for seed in seeds:
        np.random.seed(seed)
        torch.manual_seed(seed)
        import random as _random

        _random.seed(seed)
        for i in range(n_games):
            winner, reason, _turns = play_game_with_reason(
                oracle, randc, swap=(i % 2 == 1)
            )
            if winner == "A":
                wins += 1
            elif winner == "B":
                losses += 1
                loss_reasons[reason] += 1
            else:
                draws += 1
    n = wins + losses + draws
    return {
        "params": {**gate_params, "threat_penalty": threat_penalty},
        "n": n,
        "W-L-D": (wins, losses, draws),
        "score": (wins + 0.5 * draws) / n,
        "loss_reasons": dict(loss_reasons),
        "loss_turns": [],
    }


# %% 第3弾実行: 脅威回避項のスイープ
if __name__ == "__main__":
    for tp in (0.5, 1.0, 2.0):
        for pen in (0.15, 0.6):
            t0 = time.time()
            res = run_series_threat({"own_good_safety_penalty": pen}, threat_penalty=tp)
            print(f"[{time.time() - t0:.0f}s] ", end="")
            show(res)

# %% [markdown]
# ## 第3弾の観察（2026-07-15）
# 脅威回避項は守備には効く（captured_all_good 9→3）が、共有 Bundle のため
# Random 側も同じだけ守備的になり、試合が膠着する
# （引き分け 168〜174、forfeit 8〜11、score 0.66〜0.68）。総敗北はむしろ増える。
# → 対称に適用される訓練不要のゲート修正では、守備強化は分離をかえって悪化させる。

# %% 最良チューニング設定の検証（1500戦）
if __name__ == "__main__":
    t0 = time.time()
    res = run_series({"own_good_safety_penalty": 0.6}, n_games=100, seeds=tuple(range(42, 57)))
    print(f"[{time.time() - t0:.0f}s 1500戦] ", end="")
    show(res)

# %% [markdown]
# # 結論（2026-07-15）
#
# **既存4係数のチューニングだけでは「Oracle が Random に負けない」構成は作れない。**
#
# | 設定 | W-L-D | score | 敗率 |
# |------|-------|-------|------|
# | 現行（pen=0.15） | 267-10-23 /300 | 0.928 | 3.3% |
# | 最良（pen=0.6）  | 275-6-19 /300  | 0.948 | 2.0% |
# | 最良の1500戦検証 | 1358-30-112    | 0.943 | 2.0% |
# | 脅威回避項つき   | 120-12-168 /300| 0.68  | 4.0%（膠着崩壊） |
#
# - 敗率の床は約 2%。敗因はほぼ「序盤に善玉4個を取り切られる」で、
#   ゲートが捕獲時の色しか見ておらず（守備の語彙がない）、方策が探索なしの
#   1手読みであることに由来する構造的な限界。
# - 訓練不要の守備項追加は、共有 Strategy Bundle に対称に効くため膠着を招き逆効果。
# - 敗北ゼロに届く経路は (a) 1手先の被捕獲チェック付き行動選択（ミニ探索）か
#   (b) 守備報酬での再学習で、いずれも「チューニング」の範囲を超える。
#   また共有方策の変更は ELO リーグ・score 表・方向性コントロール・学習棋譜生成の
#   全面再実行を要する（棋譜も同方策の自己対局で生成しているため）。

# %% [markdown]
# ## 追記（2026-07-15 夜）: 既存チェックポイントの棚卸し監査
# 「新しい greedy 版」の最安ルートとして、既存の学習済み方策を同一 Bundle 条件で監査した。
# 実行は下のセル（run_series に policy_path を渡す）。主な結果（300戦、既定ゲート）:
#
# | checkpoint | W-L-D | score | 敗率 |
# |---|---|---|---|
# | recovered（現行） | 267-10-23 | 0.928 | 3.3% |
# | color_aware_124419/best | 276-6-18 | **0.950** | 2.0% |
# | plain_214224/best | 240-4-56 | 0.893 | 1.3% |
# | plain_215925/best | 218-2-80 | 0.860 | 0.7% |
# | plain_213321/best | 93-0-207 | 0.655 | 0%→**1500戦で0.93%** |
# | plain_215925/final | 97-0-203 | 0.662 | 0%→**1500戦で1.13%** |
# | contrastive_cgres 2種 | 引き分け過多 | 0.53-0.72 | - |
#
# ### 結論: greedy のフロンティア
# 決定力（score）と敗率はトレードオフのフロンティアを成す。
# 300戦でゼロ敗に見える個体も 1500戦では敗率約1%に収束し、
# **greedy（1手読み）の敗率の床は、引き分け型で約1%、決定力型で約2%**。
# 「無敗」は有限対戦数では証明不能（0/1500 でも 95% 上界 0.25%）であり、
# 経験的に主張できるのは「敗率 ≤ X%（CI付き）」まで。

# %% チェックポイント監査（再現用）
if __name__ == "__main__":
    CA = QUGEISTER_ROOT / "experiments/handyrl/color_aware"
    AUDIT = {
        "color_aware_124419/best": CA / "handyrl_perfect_info_color_aware_resnet_4blk_20260204_124419/best_model.pth",
        "plain_215925/best": CA / "handyrl_perfect_info_resnet_4blk_20260521_215925/best_model.pth",
        "plain_213321/best": CA / "handyrl_perfect_info_resnet_4blk_20260521_213321/best_model.pth",
        "plain_214224/best": CA / "handyrl_perfect_info_resnet_4blk_20260521_214224/best_model.pth",
    }
    for name, path in AUDIT.items():
        t0 = time.time()
        res = run_series({"color_gate_scale": 1.0}, policy_path=path)
        print(f"[{time.time() - t0:.0f}s] {name}: ", end="")
        show(res)

# %% [markdown]
# # 守備シェーピング再学習の結果（2026-07-16）
#
# ## レシピ
# ```bash
# cd /home/ks/Qugeister_clean && PYTHONPATH=/home/ks/Qugeister_clean/src \
#   .venv/bin/python scripts/training/train_handyrl_perfect_info.py \
#   --model-type resnet --n-blocks 4 --gpu \
#   --color-aware-rewards --asymmetric-color-rewards --color-reward-scale 0.3 \
#   --max-train-steps 20000
# ```
# 出力: `experiments/handyrl/color_aware/handyrl_perfect_info_color_aware_asymmetric_resnet_4blk_20260716_000531/`
# 所要 約50分（RTX 5070 Ti, 396 steps/分）。**必ず .venv + --gpu**（システムPythonはCPU専用torchで80倍遅い）。
#
# ## 1500戦での直接対決（seed 42-56 × 100、既定ゲート scale=1.0/pen=0.15）
#
# | 候補 | score | W-L-D | 敗率(95%CI) | 引分率 |
# |---|---|---|---|---|
# | **NEW/best_model（非対称レシピ）** | 0.934 | 1320-18-162 | **1.20% ±0.56** | 10.8% |
# | NEW/step7610 | 0.931 | 1315-23-162 | 1.53% ±0.63 | 10.8% |
# | OLD/124419_best（旧候補A） | 0.944 | 1374-42-84 | 2.80% ±0.85 | 5.6% |
#
# ## 結論
# - **敗率は半分以下（2.80%→1.20%）で、95%CIが重ならない＝統計的に有意な改善。**
# - score 差 0.010 は SE 0.007 に対し 1.4σ で**有意でない**（分離能は実質同等: 0.934 vs 下界 0.500）。
# - トレードオフは引き分け倍増（5.6%→10.8%）。守備シェーピングの代償で、
#   ゲート微調整（capture_good_bonus↑、pen↓）では引き分けを勝ちに変換できず既定値が最良だった。
# - 「敗率を減らす」という当初目的には成功。ただし敗率ゼロではない（1.20%）。

# %% 再学習モデルの監査（再現用）
if __name__ == "__main__":
    NEW = (
        QUGEISTER_ROOT / "experiments/handyrl/color_aware/"
        "handyrl_perfect_info_color_aware_asymmetric_resnet_4blk_20260716_000531"
    )
    for name in ("best_model.pth", "checkpoint_step7610.pth", "final_model.pth"):
        t0 = time.time()
        res = run_series({"color_gate_scale": 1.0}, policy_path=NEW / name)
        print(f"[{time.time() - t0:.0f}s] NEW/{name}: ", end="")
        show(res)
