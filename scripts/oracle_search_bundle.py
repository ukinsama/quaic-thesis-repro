# %% [markdown]
# # 浅い決定化探索を固定方策にした場合の Oracle–Random 分離調査
#
# ## 目的（ユーザーの意図）
# oracle_floor_tuning.py の結論: ゲート係数のチューニングでは Oracle の敗率 2% が床
# （敗因は序盤の善玉取り切られ＝1手読みの構造的限界）。
# そこで行動決定モジュールを「探索なし1手読み」から
# **浅い決定化探索（3手読み: 自手 → 相手応手 → 自手、葉は value head 評価）**
# に置き換えたとき、Oracle が Random に負けなくなるかを測る。
# これは ISMCTS の最小形（決定化 = Top-1 世界 1 本、ロールアウト = 方策 greedy、
# シミュレーション数 = 合法手数）に相当する。
#
# ## 設計
# - 決定化: Top1Agent と同じ最尤 4:4 配置（CONFIG_MASK 経由）で相手駒色を仮説に
#   差し替えた clone 環境を作る。捕獲済み駒の色は公開情報なので真の色を保持。
# - 探索: 各合法手 a について、仮説環境で a を指し、相手は gated policy の greedy 応手、
#   さらに自分も greedy で 1 手進め、葉を自分視点の value head で評価する。
#   途中で終局したら 勝ち=+2 / 負け=-2 / 引き分け=0（value の値域 [-1,1] を支配させる）。
# - 公平性: 従来どおり **両エージェントが同一の探索 Bundle を共有**する
#   （Random 側は誤った決定化世界の上で同じ 3 手読みを行う）。
# - ネットワーク重み・ゲート係数は現行のまま（scale=1.0, pen=0.15）。変えるのは行動決定のみ。

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

from qugeister.evaluation.color_pipeline import (  # noqa: E402
    CONFIG_COLORS,
    CONFIG_MASK,
    OracleEstimator,
    RandomEstimator,
    _select_best_action_for_player,
)
from qugeister.color.estimation import good_probs_from_output  # noqa: E402
from scripts.evaluation.domain_shift_audit import load_policy  # noqa: E402
from scripts.evaluation.oracle_floor_tuning import (  # noqa: E402
    POLICY_PATH,
    play_game_with_reason,
)

DEVICE = "cpu"


# %% 決定化: 相手駒色を仮説に差し替えた clone 環境を作る
def determinize(env, player: str, colors8: np.ndarray):
    """Top-1 仮説 colors8（0=good, 1=bad, 添字は base_id+i）で相手色を置き換えた clone を返す。

    生存している相手駒のみ仮説を適用し、捕獲済み駒（色は公開情報）は真の色を保つ。
    """
    sim = env.clone()
    if player == "A":
        base, alive, true_good = 100, sim.pieces_b, sim.good_pieces_b
    else:
        base, alive, true_good = 0, sim.pieces_a, sim.good_pieces_a
    new_good = set()
    for i in range(8):
        pid = base + i
        if pid in alive:
            if colors8[i] == 0:
                new_good.add(pid)
        elif pid in true_good:  # 捕獲済み: 真の色（公開情報）を維持
            new_good.add(pid)
    if player == "A":
        sim.good_pieces_b = new_good
        sim._b_good_count = sum(1 for pid in sim.pieces_b if pid in new_good)
        sim._b_bad_count = len(sim.pieces_b) - sim._b_good_count
    else:
        sim.good_pieces_a = new_good
        sim._a_good_count = sum(1 for pid in sim.pieces_a if pid in new_good)
        sim._a_bad_count = len(sim.pieces_a) - sim._a_good_count
    return sim


# %% 浅い決定化探索エージェント
class ShallowSearchAgent:
    """Top-1 決定化 + 3手読み（自手 → 相手greedy → 自分greedy → value評価）。"""

    def __init__(self, color_estimator, policy, device, name="Search"):
        self.estimator = color_estimator
        self.policy = policy
        self.device = device
        self.name = name

    def reset(self):
        self.estimator.reset()

    # --- 内部ヘルパ ---
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

    #: 相手モデルの観測モード:
    #:   "perfect"  = 決定化世界の完全情報（こちらの真の色が見える = paranoid）
    #:   "marginal" = こちらの駒を 0.5/0.5 のマージナルで見せる（色不明の相手 = 非パラノイド）
    opponent_model: str = "perfect"

    def _greedy_step(self, sim, side: str, is_opponent: bool = False):
        """決定化環境上で side が gated policy の greedy 手を1手指す。"""
        legal = sim.get_legal_actions(player=side)
        if not legal:
            sim.game_over = True
            sim.winner = "B" if side == "A" else "A"
            return
        if is_opponent and self.opponent_model == "marginal":
            from qugeister.evaluation.color_pipeline import construct_marginal_color_obs

            obs = construct_marginal_color_obs(sim, side, np.full(8, 0.5))
        else:
            obs = sim.get_observation(
                side, hide_opponent=False, rich_features=True, normalize_perspective=True
            )
        with torch.inference_mode():
            logits, _ = self.policy(
                torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            )
        action = _select_best_action_for_player(logits[0], sim, side, self.device)
        sim.step(action)

    def _leaf_value(self, sim, me: str) -> float:
        """自分視点の完全情報観測で value head を評価する。"""
        obs = sim.get_observation(
            me, hide_opponent=False, rich_features=True, normalize_perspective=True
        )
        with torch.inference_mode():
            _, value = self.policy(
                torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            )
        return float(value.reshape(-1)[0])

    def _terminal_score(self, sim, me: str) -> float | None:
        if not sim.game_over:
            return None
        if sim.winner == me:
            return 2.0
        if sim.winner in ("A", "B"):
            return -2.0
        return 0.0  # Draw

    def select_action(self, env, player, epsilon=0.0):
        legal_actions = env.get_legal_actions(player=player)
        if not legal_actions:
            raise ValueError("No legal actions!")

        # Top1Agent と同じ最尤 4:4 配置の選択
        good_probs = good_probs_from_output(self.estimator.get_good_probs(env, player))
        log_good = np.log(np.clip(good_probs, 1e-10, 1.0))
        log_bad = np.log(np.clip(1.0 - good_probs, 1e-10, 1.0))
        best_idx = np.argmax(CONFIG_MASK @ log_good + (1.0 - CONFIG_MASK) @ log_bad)
        colors = CONFIG_COLORS[best_idx]

        sim0 = determinize(env, player, colors)
        opp = "B" if player == "A" else "A"

        best_action, best_score = None, -np.inf
        for action in legal_actions:
            sim = sim0.clone()
            sim.step(action)  # ply1: 自手
            score = self._terminal_score(sim, player)
            if score is None:
                self._greedy_step(sim, opp, is_opponent=True)  # ply2: 相手の greedy 応手
                score = self._terminal_score(sim, player)
            if score is None:
                self._greedy_step(sim, player)  # ply3: 自分の greedy
                score = self._terminal_score(sim, player)
            if score is None:
                score = self._leaf_value(sim, player)  # 葉: value head（自分視点）
            if score > best_score:
                best_score, best_action = score, action
        return best_action


# %% シリーズ実行
def run_search_series(
    n_games: int = 100, seeds=(42, 43, 44), max_turns: int = 200,
    opponent_model: str = "perfect",
):
    policy = load_policy(
        POLICY_PATH, device=DEVICE, policy_arch="color-gated-resnet", color_gate_scale=1.0
    )
    oracle = ShallowSearchAgent(OracleEstimator(), policy, DEVICE, "OracleSearch")
    randc = ShallowSearchAgent(RandomEstimator(), policy, DEVICE, "RandomSearch")
    oracle.opponent_model = opponent_model
    randc.opponent_model = opponent_model

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
        "n": n,
        "W-L-D": (wins, losses, draws),
        "score": (wins + 0.5 * draws) / n,
        "loss_reasons": dict(loss_reasons),
        "loss_turns": loss_turns,
    }


# %% まずは小規模スモーク（速度と挙動の確認）
if __name__ == "__main__":
    t0 = time.time()
    smoke = run_search_series(n_games=10, seeds=(42,))
    dt = time.time() - t0
    w, l, d = smoke["W-L-D"]
    print(f"[smoke 10戦 {dt:.0f}s = {dt / 10:.1f}s/局] W-L-D={w}-{l}-{d} 敗因={smoke['loss_reasons']}")

# %% [markdown]
# ## 3手読み探索の結果（2026-07-15 スモーク）
# 1勝-3敗-6分（3.7s/局）。**大失敗**。原因は決定化探索の既知の病理:
# シミュレーション内の相手モデルが決定化世界の完全情報（=こちらの真の色）で応手するため、
# 「善玉を進める手はすべて次手で取られる」と評価され（paranoid opponent）、
# 探索が萎縮して膠着（引き分け6）と自己封鎖（forfeit 3敗）に陥る。
# これは determinization の strategy fusion / paranoia 問題そのもの。
#
# ## 方針転換: 2手読み安全フィルタ（veto 方式）
# 読みを「評価」ではなく「拒否」にだけ使う:
# gated policy の greedy 手を基本としつつ、
# 「指した後に相手の合法手の行き先に自分の善玉が存在する手」（=善玉が当たりに立つ手）を拒否し、
# 拒否されたら次点の手へ。全手が危険なら greedy にフォールバック。
# 拒否条件は相手の色知識に依存しない盤面機械判定なので萎縮しない。
# 罰則ゲート（oracle_floor_tuning 第3弾）との違いは、常時ペナルティではなく
# 「具体的に当たっている時だけの veto」である点。

# %% 2手読み安全フィルタ・エージェント
class SafeGreedyAgent:
    """gated policy greedy + 善玉ハング拒否（2手読み安全フィルタ）。"""

    def __init__(self, color_estimator, policy, device, name="SafeGreedy"):
        self.estimator = color_estimator
        self.policy = policy
        self.device = device
        self.name = name

    def reset(self):
        self.estimator.reset()

    def _my_good_positions(self, sim, player) -> set:
        pieces = sim.pieces_a if player == "A" else sim.pieces_b
        good = sim.good_pieces_a if player == "A" else sim.good_pieces_b
        return {pos for pid, pos in pieces.items() if pid in good}

    def _hangs_good(self, env, player, action) -> bool:
        """action を指した後、相手の合法手の行き先に自善玉があるか（色仮説不要の機械判定）。"""
        sim = env.clone()
        sim.step(action)
        if sim.game_over:
            return False  # 終局する手（勝ち/負け確定）は veto の対象外（勝ち手を殺さない）
        opp = "B" if player == "A" else "A"
        my_good = self._my_good_positions(sim, player)
        if not my_good:
            return False
        for fr, fc, dr, dc in sim.get_legal_actions(player=opp):
            if (fr + dr, fc + dc) in my_good:
                return True
        return False

    def select_action(self, env, player, epsilon=0.0):
        legal_actions = env.get_legal_actions(player=player)
        if not legal_actions:
            raise ValueError("No legal actions!")

        # Top1Agent と同じ最尤 4:4 配置 → gated policy の Q で全合法手を順位付け
        good_probs = good_probs_from_output(self.estimator.get_good_probs(env, player))
        log_good = np.log(np.clip(good_probs, 1e-10, 1.0))
        log_bad = np.log(np.clip(1.0 - good_probs, 1e-10, 1.0))
        best_idx = np.argmax(CONFIG_MASK @ log_good + (1.0 - CONFIG_MASK) @ log_bad)
        from qugeister.evaluation.color_pipeline import construct_perfect_info_obs

        obs = construct_perfect_info_obs(env, player, CONFIG_COLORS[best_idx])
        with torch.inference_mode():
            logits, _ = self.policy(
                torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            )
        # 合法手を Q 降順に並べ、善玉が当たりに立たない最初の手を選ぶ
        ranked = self._rank_actions(logits[0], env, player)
        for action in ranked:
            if not self._hangs_good(env, player, action):
                return action
        return ranked[0]  # 全手危険なら greedy にフォールバック

    def _rank_actions(self, policy_logits, env, player):
        """合法手を Q 降順の絶対座標で返す（Player B の正規化は既存ヘルパの規約に従う）。"""
        from qugeister.agents.action_decision import (
            action_indices,
            denormalize_action,
            get_normalized_legal_actions,
        )

        normalized = get_normalized_legal_actions(env, player)
        idx = action_indices(normalized)
        q = policy_logits[torch.tensor(idx, dtype=torch.long)].tolist()
        order = sorted(range(len(normalized)), key=lambda i: q[i], reverse=True)
        return [denormalize_action(env, player, normalized[i]) for i in order]


# %% SafeGreedy シリーズ実行
def run_safe_series(n_games: int = 100, seeds=(42, 43, 44), max_turns: int = 200):
    policy = load_policy(
        POLICY_PATH, device=DEVICE, policy_arch="color-gated-resnet", color_gate_scale=1.0
    )
    oracle = SafeGreedyAgent(OracleEstimator(), policy, DEVICE, "OracleSafe")
    randc = SafeGreedyAgent(RandomEstimator(), policy, DEVICE, "RandomSafe")

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
        "n": n,
        "W-L-D": (wins, losses, draws),
        "score": (wins + 0.5 * draws) / n,
        "loss_reasons": dict(loss_reasons),
        "loss_turns": loss_turns,
    }


# %% SafeGreedy スモーク → 本計測
if __name__ == "__main__":
    t0 = time.time()
    safe = run_safe_series(n_games=10, seeds=(42,))
    dt = time.time() - t0
    w, l, d = safe["W-L-D"]
    print(f"[safe smoke 10戦 {dt:.0f}s] W-L-D={w}-{l}-{d} 敗因={safe['loss_reasons']}")

# %% [markdown]
# ## 中間結果（2026-07-15）
# - SafeGreedy（veto 2手読み）: 300戦 70-55-175, score 0.525。**崩壊**。
#   一律の善玉リスク回避は、色知識が利得に変わる「交換」を消して優位ごと失わせる。
# - 非パラノイド探索（相手モデル=0.5マージナル観測）: 15戦 6-0-9。**無敗**だが引き分け6割。
#
# ## ハイブリッド: 大失着ガード付き greedy
# 通常は gated greedy の手をそのまま指し、greedy 手の3手読みスコアが
# 探索最良手より margin 以上悪い（=モデル上の大失着）ときだけ差し替える。
# 攻めの姿勢は greedy のまま、敗着だけを探索で拒否する狙い。

# %% 大失着ガード付きエージェント
class BlunderGuardedAgent(ShallowSearchAgent):
    """gated greedy を基本とし、3手読みで大失着と判定された時だけ差し替える。"""

    def __init__(self, color_estimator, policy, device, name="Guarded", margin: float = 1.0):
        super().__init__(color_estimator, policy, device, name)
        self.margin = float(margin)
        self.opponent_model = "marginal"  # 非パラノイド相手モデルを既定にする

    def select_action(self, env, player, epsilon=0.0):
        legal_actions = env.get_legal_actions(player=player)
        if not legal_actions:
            raise ValueError("No legal actions!")

        good_probs = good_probs_from_output(self.estimator.get_good_probs(env, player))
        log_good = np.log(np.clip(good_probs, 1e-10, 1.0))
        log_bad = np.log(np.clip(1.0 - good_probs, 1e-10, 1.0))
        best_idx = np.argmax(CONFIG_MASK @ log_good + (1.0 - CONFIG_MASK) @ log_bad)
        colors = CONFIG_COLORS[best_idx]

        from qugeister.evaluation.color_pipeline import construct_perfect_info_obs

        obs = construct_perfect_info_obs(env, player, colors)
        with torch.inference_mode():
            logits, _ = self.policy(
                torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            )
        greedy_action = _select_best_action_for_player(logits[0], env, player, self.device)

        # 全合法手の3手読みスコア（相手は marginal モデル）
        sim0 = determinize(env, player, colors)
        opp = "B" if player == "A" else "A"
        scores: dict = {}
        for action in legal_actions:
            sim = sim0.clone()
            sim.step(action)
            score = self._terminal_score(sim, player)
            if score is None:
                self._greedy_step(sim, opp, is_opponent=True)
                score = self._terminal_score(sim, player)
            if score is None:
                self._greedy_step(sim, player)
                score = self._terminal_score(sim, player)
            if score is None:
                score = self._leaf_value(sim, player)
            scores[action] = score

        best_action = max(scores, key=scores.get)
        if scores[greedy_action] >= scores[best_action] - self.margin:
            return greedy_action  # greedy が大失着でなければそのまま（攻め維持）
        return best_action


# %% [markdown]
# # 最終結果（2026-07-15、いずれも300戦 = seed 42/43/44 × 100、対 RandomColorEstimator）
#
# | 構成 | W-L-D | score | 敗北数 |
# |------|-------|-------|--------|
# | 現行 greedy（pen=0.15） | 267-10-23 | 0.928 | 10 |
# | チューニング最良 greedy（pen=0.6） | 275-6-19 | **0.948** | **6** |
# | 3手読み・paranoid相手モデル | （10戦 1-3-6 で棄却） | — | — |
# | veto 2手読み（SafeGreedy） | 70-55-175 | 0.525 | 55 |
# | 3手読み・marginal相手モデル | 159-50-91 | 0.682 | 50 |
# | 大失着ガード（margin=1.0） | 271-19-10 | 0.920 | 19 |
#
# ## 結論
# **浅い決定化探索は、どの変種でもチューニング済み greedy に勝てなかった。**
# 敗北ゼロ（完全分離）はいずれの構成でも未達。原因は三つ:
# 1. **相手モデルのジレンマ**: 完全情報モデルは paranoia（萎縮・膠着）、
#    色不明モデルは応手予測が外れて3手読みのバックアップ自体が誤る。
# 2. **value head の分布外評価**: HandyRL の価値関数は greedy 方策の自己対戦分布で
#    学習されており、探索が作る強制手順後の局面では信頼できない。
#    誤った葉評価が greedy より悪い「静かな手」を選ばせ、forfeit（自己封鎖）を生む。
# 3. **スモークの罠**: 10〜15戦ではテール事象（敗北2〜3%）は測れない。
#    guarded は 15戦全勝 → 300戦で19敗、marginal は 15戦無敗 → 300戦で50敗だった。
#
# 敗北ゼロに届き得る残りの経路は、信念からの複数決定化＋UCT＋十分なシミュレーション数を
# 持つ本格 ISMCTS（1手あたり数百回の展開）だが、Python 実装では対局時間が2〜3桁増え、
# 修論のリーグ運用（数千戦）には非現実的。議論章の「探索付き行動決定器は別設定」という
# 整理の実証的裏付けとして、この負の結果自体を記録する。

# %% ハイブリッドのシリーズ実行
def run_guarded_series(margin: float = 1.0, n_games: int = 100, seeds=(42, 43, 44)):
    policy = load_policy(
        POLICY_PATH, device=DEVICE, policy_arch="color-gated-resnet", color_gate_scale=1.0
    )
    oracle = BlunderGuardedAgent(OracleEstimator(), policy, DEVICE, "OracleGuarded", margin)
    randc = BlunderGuardedAgent(RandomEstimator(), policy, DEVICE, "RandomGuarded", margin)
    wins = losses = draws = 0
    loss_reasons: Counter = Counter()
    for seed in seeds:
        np.random.seed(seed)
        torch.manual_seed(seed)
        import random as _random

        _random.seed(seed)
        for i in range(n_games):
            winner, reason, _t = play_game_with_reason(oracle, randc, swap=(i % 2 == 1))
            if winner == "A":
                wins += 1
            elif winner == "B":
                losses += 1
                loss_reasons[reason] += 1
            else:
                draws += 1
    n = wins + losses + draws
    return {
        "margin": margin,
        "n": n,
        "W-L-D": (wins, losses, draws),
        "score": (wins + 0.5 * draws) / n,
        "loss_reasons": dict(loss_reasons),
    }
