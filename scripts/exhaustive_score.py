# %% [markdown]
# # 全列挙による厳密 score（統計誤差ゼロの評価）
#
# ## 目的（ユーザーの意図）
# 300〜1500戦の標本評価では learned 帯（幅0.08）が測定ノイズ（0.05〜0.08）に埋もれ、
# 推定器の順位が測り方で入れ替わってしまう（実測: ResNetColorEstimator が 0.700 と 0.775 の間で振れた）。
# このゲームは**乱数が有限**なので、全列挙すれば score は推定値ではなく**厳密値**になり、
# 信頼区間そのものが不要になる。
#
# ## 全列挙が成立する根拠（コードを読んで確認済み）
# GeisterEnvV2.reset() の乱数は駒色の割り当てのみ:
#   np.random.shuffle(a_piece_ids); good_pieces_a = set(a_piece_ids[:4])
#   np.random.shuffle(b_piece_ids); good_pieces_b = set(b_piece_ids[:4])
# 配置マスは固定（a_positions/b_positions はハードコード）、先手は常に A。
# 方策は greedy(argmax)、推定器・アダプターも決定的、epsilon=0。
# → **1局 = (Aの色割当, Bの色割当, 手番順) の決定的関数**
# → 乱数空間は C(8,4) × C(8,4) × 2 = 70 × 70 × 2 = 9,800 通りで**有限**
#
# ## 検算（このファイルの verify セル）
# (1) 同じ割当なら外側の乱数種を変えても結果が完全一致する（決定性）
# (2) 割当を変えれば結果が変わる（測定として意味がある）
# (3) 全列挙の平均が、従来のランダム標本評価の値と整合する（±標本誤差の範囲で）

# %% セットアップ
from __future__ import annotations

import argparse
import json
import os
import time
from itertools import combinations
from pathlib import Path

QUGEISTER_ROOT = Path(__file__).resolve().parents[2]
QUAIC_ROOT = QUGEISTER_ROOT.parent / "QuAic"

ALL_A = list(combinations(range(8), 4))          # 70 通り: A の善玉 4 個の選び方
ALL_B = list(combinations(range(100, 108), 4))   # 70 通り: B の善玉 4 個
N_TOTAL = len(ALL_A) * len(ALL_B) * 2            # 9,800

# worker プロセスごとの状態（fork 後に 1 回だけ構築）
_G: dict = {}


def _init_worker(policy_rel: str, est_name: str, pen: float | None, device: str,
                 top_k: int = 5) -> None:
    """各 worker で方策と推定器を1回だけ構築する（fork のたびのロードを避ける）。

    top_k: learned 推定器のアダプター k（分解能のノブ。Oracle/Random は Top1 固定）。
    """
    import sys

    for cand in (QUGEISTER_ROOT, QUGEISTER_ROOT / "src", QUAIC_ROOT / "backend",
                 QUAIC_ROOT / "backend/scripts"):
        if str(cand) not in sys.path:
            sys.path.insert(0, str(cand))
    os.environ.setdefault("DEBUG", "true")

    import torch

    torch.set_num_threads(1)  # worker 間で CPU を奪い合わせない

    from qugeister.evaluation.color_pipeline import (
        OracleEstimator,
        RandomEstimator,
        Top1Agent,
        TopKAgent,
    )
    from scripts.evaluation.domain_shift_audit import load_policy

    policy = load_policy(QUGEISTER_ROOT / policy_rel, device=device,
                         policy_arch="color-gated-resnet", color_gate_scale=1.0)
    if pen is not None:
        policy.own_good_safety_penalty = float(pen)

    # 論文プロトコル: Oracle/Random は Top1、learned は TopK(k=5)
    if est_name == "Oracle":
        agent = Top1Agent(OracleEstimator(), policy, device, "Oracle")
    elif est_name == "Random":
        agent = Top1Agent(RandomEstimator(), policy, device, "Random")
    else:
        from run_paper_color_elo import DEFAULT_NEW_CHECKPOINTS, QuAicCheckpointEstimator

        est = QuAicCheckpointEstimator(DEFAULT_NEW_CHECKPOINTS[est_name], device=device)
        agent = TopKAgent(est, policy, device, name=est_name, top_k=top_k)

    _G["agent"] = agent
    _G["opponent"] = Top1Agent(RandomEstimator(), policy, device, "RandomColorEstimator")


def _play_one(task: tuple) -> str:
    """1局を色割当を固定して実行する。乱数は一切引かない。"""
    from qugeister.env.geister import GeisterEnvV2

    a_good, b_good, swap = task
    env = GeisterEnvV2(max_turns=200)
    env.reset()
    # reset() の shuffle 結果を上書き = 特定の割当を再現する（配置マスは固定なので等価）
    env.good_pieces_a = set(a_good)
    env.good_pieces_b = set(b_good)
    env._a_good_count = env._b_good_count = 4
    env._a_bad_count = env._b_bad_count = 4

    me, opp = _G["agent"], _G["opponent"]
    cur_a, cur_b = (opp, me) if swap else (me, opp)
    me.reset()
    opp.reset()

    while not env.game_over:
        p = env.current_player
        act = (cur_a if p == "A" else cur_b).select_action(env, p)
        if act is None:
            env.game_over = True
            env.winner = "B" if p == "A" else "A"
            break
        ok, _ = env.step(act)
        if not ok:
            env.game_over = True
            env.winner = "B" if p == "A" else "A"
            break

    if env.winner not in ("A", "B"):
        return "draw"
    # swap 時は env の A/B と me/opp の対応が逆になる
    me_won = (env.winner == "B") if swap else (env.winner == "A")
    return "win" if me_won else "loss"


# %% 全列挙の実行
def enumerate_score(est_name: str, policy_rel: str, pen: float | None = None,
                    workers: int = 12, device: str = "cpu", limit: int | None = None,
                    top_k: int = 5):
    """全 9,800 局を列挙して厳密 score を返す（limit 指定時は先頭 limit 局のみ = 検証用）。"""
    import multiprocessing as mp

    tasks = [(a, b, s) for a in ALL_A for b in ALL_B for s in (False, True)]
    if limit:
        tasks = tasks[:limit]

    t0 = time.time()
    ctx = mp.get_context("fork")
    with ctx.Pool(workers, initializer=_init_worker,
                  initargs=(policy_rel, est_name, pen, device, top_k)) as pool:
        out = pool.map(_play_one, tasks, chunksize=32)
    w = out.count("win")
    lo = out.count("loss")
    d = out.count("draw")
    n = len(out)
    return {
        "estimator": est_name,
        "policy": policy_rel,
        "own_good_safety_penalty": pen if pen is not None else 0.15,
        "n_games": n,
        "exhaustive": limit is None,
        "top_k": top_k,
        "wins": w,
        "losses": lo,
        "draws": d,
        "score": (w + 0.5 * d) / n,
        "loss_rate": lo / n,
        "elapsed_sec": round(time.time() - t0, 1),
    }


def main() -> int:
    from run_paper_color_elo import POLICY  # 論文の固定方策（既定）

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--estimators", default="Oracle",
                    help="カンマ区切り。Oracle / MLPColorEstimator / ResNetColorEstimator / "
                         "QuantumColorEstimator / HistoryGRUColorEstimator / LowParamColorEstimator")
    ap.add_argument("--policy", default=None, help="qugeister-root からの相対パス（既定=論文の固定方策）")
    ap.add_argument("--pen", type=float, default=None, help="color gate の守備係数（既定=0.15）")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--limit", type=int, default=None, help="検証用: 先頭N局だけ")
    ap.add_argument("--output-json", type=Path, default=None)
    args = ap.parse_args()

    policy_rel = args.policy or POLICY
    rows = []
    for name in args.estimators.split(","):
        r = enumerate_score(name.strip(), policy_rel, pen=args.pen,
                            workers=args.workers, limit=args.limit)
        tag = "厳密値" if r["exhaustive"] else f"部分({r['n_games']}局)"
        print(f"[{r['elapsed_sec']:>6.1f}s] {r['estimator']:<16} score={r['score']:.4f} "
              f"({tag})  W-L-D={r['wins']}-{r['losses']}-{r['draws']}  敗率={r['loss_rate']*100:.2f}%")
        rows.append(r)

    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[write] {args.output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
