# %% [markdown]
# # 修論の数値検証（本文マクロ・表 ↔ 実験JSONの機械照合）
#
# **目的（ユーザーの意図）**: 回覧レビュー対応の最終盤で「論文がすべて正しいデータか」を
# 機械的に確かめる。きっかけは MK レビュー④で発覚した「図6.3 だけ差し替え前の旧データの
# まま」事件。本文数値の単一情報源 `texfile/numbers.tex` と、表に直書きされた数値を、
# `backend/data/experiments/` の実験 JSON・チェックポイントと突き合わせる。
#
# 判定は ✓一致 / ✗不一致 / △データ源未特定 の3値。丸めは論文の表示桁に合わせる。

# %% 0. 共通: パスとヘルパ
import json
import re
from pathlib import Path

QUAIC = Path("/home/ks/QuAic")
EXP = QUAIC / "backend/data/experiments"
THESIS = QUAIC / "paper/thesis"

results = []  # (項目, 論文値, データ値, 判定)


def check(label: str, paper_val, data_val, ok: bool | None = None):
    """論文値とデータ値を比べて記録する。ok を省略すると文字列一致で判定。"""
    if ok is None:
        ok = str(paper_val) == str(data_val)
    results.append((label, paper_val, data_val, "✓" if ok else "✗"))


def unknown(label: str, paper_val, note: str):
    results.append((label, paper_val, note, "△"))


def r(x, nd):
    """論文の表示桁への丸め（round-half-even は Python の round と同じ）。"""
    return round(float(x), nd)


def r_up(x, nd):
    """四捨五入（half-up）。論文の一部の帯は 0.125→0.13 の half-up で表示している。"""
    from decimal import Decimal, ROUND_HALF_UP
    return float(Decimal(str(x)).quantize(Decimal("1e-%d" % nd), rounding=ROUND_HALF_UP))


# %% 1. numbers.tex のマクロを読み込む
macro_src = (THESIS / "texfile/numbers.tex").read_text()
MACROS = dict(re.findall(r"\\newcommand\{\\(v\w+)\}\{([^}]*)\}", macro_src))
# LaTeX の桁区切り {,} を外して数値化しやすくする
MACROS = {k: v.replace("{,}", ",") for k, v in MACROS.items()}
print(f"マクロ {len(MACROS)} 個を読み込み")

# %% 2. トイ検算: パーサが既知値を正しく読めているか
assert MACROS["vTrainGames"] == "300", MACROS["vTrainGames"]
assert MACROS["vEloOracle"] == "1660.8", MACROS["vEloOracle"]
print("トイ検算 OK（vTrainGames=300, vEloOracle=1660.8）")

# %% 3. exhaustive_roundrobin.json ↔ Elo・score・W-L-D・派生値
rr = json.loads((EXP / "exhaustive_roundrobin.json").read_text())
elo = rr["elo"]
NAME = {
    "GRU": "HistoryGRUColorEstimator",
    "MLP": "MLPColorEstimator",
    "ResNet": "ResNetColorEstimator",
    "Quantum": "QuantumColorEstimator",
    "Oracle": "OracleColorEstimator",
    "LowParam": "LowParamColorEstimator",
    "Random": "RandomColorEstimator",
}
for short in ["Oracle", "GRU", "MLP", "ResNet", "Quantum", "LowParam", "Random"]:
    check(f"Elo {short}", MACROS[f"vElo{short}"], r(elo[NAME[short]], 1),
          float(MACROS[f"vElo{short}"]) == r(elo[NAME[short]], 1))

svr = rr["score_vs_random"]
for short in ["GRU", "MLP", "ResNet", "Quantum", "Oracle", "LowParam"]:
    check(f"score vs Random {short}", MACROS[f"vScore{short}"], r(svr[NAME[short]], 3),
          float(MACROS[f"vScore{short}"]) == r(svr[NAME[short]], 3))

# W-L-D は Random 視点の pair_results を推定器視点に反転する
for pr in rr["pair_results"]:
    if pr["a"] == "RandomColorEstimator" and pr["b"] != "OracleColorEstimator":
        short = [s for s, n in NAME.items() if n == pr["b"]]
        if short and f"vWLD{short[0]}" in MACROS:
            wld = f"{pr['losses']}-{pr['wins']}-{pr['draws']}"
            check(f"W-L-D {short[0]}", MACROS[f"vWLD{short[0]}"], wld)
    if pr["a"] == "RandomColorEstimator" and pr["b"] == "OracleColorEstimator":
        wld = f"{pr['losses']}-{pr['wins']}-{pr['draws']}"
        check("W-L-D Oracle", MACROS["vWLDOracle"], wld)

# 派生: LowParam-Random の Elo 差 / 学習型 Elo の密集帯 / 直接対戦の score 帯
gap = r(elo[NAME["LowParam"]] - elo[NAME["Random"]], 1)
check("Elo 差 LowParam−Random", MACROS["vEloGapLow"], gap,
      float(MACROS["vEloGapLow"]) == gap)
learned = ["GRU", "MLP", "ResNet", "Quantum"]
lo, hi = min(elo[NAME[s]] for s in learned), max(elo[NAME[s]] for s in learned)
check("学習型 Elo 帯", f"{MACROS['vEloClassLo']}–{MACROS['vEloClassHi']}",
      f"{int(lo)}–{int(hi)}",
      int(lo) == int(MACROS["vEloClassLo"]) and int(hi) == int(MACROS["vEloClassHi"]))
learned_full = {NAME[s] for s in learned}
pair_scores = [pr["score_a"] for pr in rr["pair_results"]
               if pr["a"] in learned_full and pr["b"] in learned_full]
check("学習型 直接対戦帯", f"{MACROS['vPairLearnedLo']}–{MACROS['vPairLearnedHi']}",
      f"{r(min(pair_scores),2)}–{r(max(pair_scores),2)}",
      r(min(pair_scores), 2) == float(MACROS["vPairLearnedLo"])
      and r(max(pair_scores), 2) == float(MACROS["vPairLearnedHi"]))

# %% 4. signal_controls*.json ↔ §6.5 反転コントロールと色感度
sc = json.loads((EXP / "signal_controls.json").read_text())
SC_NAME = {"MLP": "MLPColorEstimator", "ResNet": "ResNetColorEstimator",
           "Quantum": "QuantumColorEstimator", "GRU": "HistoryGRUColorEstimator"}
rows = {row["estimator"]: row for row in sc["results"]}
for short, full in SC_NAME.items():
    a, b = rows[full]["normal"]["score"], rows[full]["inverted"]["score"]
    check(f"反転 {short} normal", MACROS[f"vInv{short}a"], r(a, 3),
          float(MACROS[f"vInv{short}a"]) == r(a, 3))
    check(f"反転 {short} inverted", MACROS[f"vInv{short}b"], r(b, 3),
          float(MACROS[f"vInv{short}b"]) == r(b, 3))
drops = {s: rows[f]["normal"]["score"] - rows[f]["inverted"]["score"]
         for s, f in SC_NAME.items()}
check("反転低下幅の帯", f"{MACROS['vInvDropLo']}–{MACROS['vInvDropHi']}",
      f"{r(min(drops.values()),2)}–{r(max(drops.values()),2)}",
      r(min(drops.values()), 2) == float(MACROS["vInvDropLo"])
      and r(max(drops.values()), 2) == float(MACROS["vInvDropHi"]))

# 色感度 = 4 推定器の低下幅平均（表7.1: base/守備/守備+引分罰/攻撃+引分罰）
SENS_FILES = {"vSensBase": "signal_controls.json",
              "vSensDef": "signal_controls_pl_asym.json",
              "vSensDefDraw": "signal_controls_pl_asymdraw.json",
              "vSensOffDraw": "signal_controls_pl_offense.json"}
for key, fname in SENS_FILES.items():
    d = json.loads((EXP / fname).read_text())
    ds = [row["normal"]["score"] - row["inverted"]["score"] for row in d["results"]]
    mean_drop = r(sum(ds) / len(ds), 3)
    check(f"色感度 {key}（{fname}）", MACROS[key], mean_drop,
          float(MACROS[key]) == mean_drop)

# %% 5. temperature_sweep_wide.json ↔ 温度頑健性の帯（変動幅 0.08–0.13）
ts = json.loads((EXP / "temperature_sweep_wide.json").read_text())
spans = {}
for est, rows_ in ts["results"].items():
    scores = [row["score"] for row in rows_]
    temps = [row["temperature"] for row in rows_]
    spans[est] = max(scores) - min(scores)
    print(f"  {est}: T={min(temps)}..{max(temps)} 点数={len(rows_)} 変動幅={spans[est]:.4f}")
# 最大 0.1250 は half-up で 0.13 と表示している（banker's だと 0.12）
check("温度スイープ変動幅帯", f"{MACROS['vTempBandLo']}–{MACROS['vTempBandHi']}",
      f"{r_up(min(spans.values()),2)}–{r_up(max(spans.values()),2)}",
      r_up(min(spans.values()), 2) == float(MACROS["vTempBandLo"])
      and r_up(max(spans.values()), 2) == float(MACROS["vTempBandHi"]))

# %% 6. paper_benchmark_diagnostics.json ↔ piece acc・温度・診断改善幅
diag = json.loads((EXP / "paper_benchmark_diagnostics.json").read_text())
ckpt = diag["paper_checkpoints"]
for short, full in SC_NAME.items():
    acc = ckpt[full]["metrics"]["holdout"]["piece_accuracy"] * 100
    check(f"piece acc {short}", MACROS[f"vAcc{short}"], r(acc, 1),
          float(MACROS[f"vAcc{short}"]) == r(acc, 1))
    t = ckpt[full]["temperature"]
    check(f"温度 {short}", MACROS[f"vT{short}"], t, float(MACROS[f"vT{short}"]) == float(t))

# Neff 低下・Top-3 改善（§6.2 は active world 系の対Random指標を使う）
est_rows = diag["estimators"]
neff_drops, ret_gains = [], []
for full in SC_NAME.values():
    s = est_rows[full]["summary"]
    neff_drops.append(s["active_world_reduction_vs_random"])
    ret_gains.append(s["active_topk_lift_vs_random"] * 100)
print("  Neff低下(active):", [f"{x:.2f}" for x in neff_drops],
      " Top-3改善pt(active):", [f"{x:.2f}" for x in ret_gains])
check("Neff 低下幅帯", MACROS["vNeffDrop"].replace("--", "–"),
      f"{r(min(neff_drops),1)}–{r(max(neff_drops),1)}",
      MACROS["vNeffDrop"] == f"{r(min(neff_drops),1)}--{r(max(neff_drops),1)}")
check("Top-3 改善幅帯", MACROS["vRetGain"].replace("--", "–"),
      f"{r(min(ret_gains),1)}–{r(max(ret_gains),1)}",
      MACROS["vRetGain"] == f"{r(min(ret_gains),1)}--{r(max(ret_gains),1)}")

# %% 7. k_sweep_offense_pennylane.json ↔ 表A.1（tex 直書き）
ks = json.loads((EXP / "k_sweep_offense_pennylane.json").read_text())
print("k_sweep keys:", list(ks.keys())[:6])
appendix = (THESIS / "texfile/appendix.tex").read_text()
m = re.search(r"帯幅（学習型 max\$-\$min） & ([\d. &$]+)\\\\", appendix)
m2 = re.search(r"ヌル対照 score & ([\d. &$]+)\\\\", appendix)
def _texnums(mm):
    return [float(x.replace("$", "")) for x in mm.group(1).replace("\\", "").split("&")] if mm else []

tex_band = _texnums(m)
tex_null = _texnums(m2)
band_data, null_data = [], []
for row in sorted(ks["results"], key=lambda x: x["k"]):
    band_data.append(r(row["band_width"], 3))
    null_data.append(r(row["scores"]["LowParamColorEstimator"], 3))
check("表A.1 帯幅行", tex_band, band_data, tex_band == band_data)
check("表A.1 ヌル対照行", tex_null, null_data, tex_null == null_data)

# %% 8. exhaustive_policy_variants.json ↔ 表7.1（tex 直書き）
pv = json.loads((EXP / "exhaustive_policy_variants.json").read_text())
pv_rows = pv if isinstance(pv, list) else pv.get("results", [])
# 表7.1 の論文値（tex 直書き）: (score, 敗率%, 引き分け率%)
PAPER_71 = {
    "recovered(pen=0.15)": (0.9411, 2.14, 7.50),
    "recovered(pen=0.6)": (0.9410, 2.14, 7.51),
    "color_aware_124419": (0.9447, 2.70, 5.65),
    "守備シェーピング": (0.9228, 1.59, 12.27),
    "守備+引き分け罰": (0.9530, 2.57, 4.26),
    "攻撃+引き分け罰": (0.9629, 2.21, 3.00),
}
matched = set()
for obj in pv_rows:
    score = r(obj["score"], 4)
    loss_pct = r(obj["loss_rate"] * 100, 2)
    draw_pct = r(obj["draws"] / obj["n_games"] * 100, 2)
    hit = [k for k, v in PAPER_71.items()
           if v == (score, loss_pct, draw_pct)]
    label = obj.get("label", obj["policy"].split("/")[-2][-25:])
    if hit:
        matched.add(hit[0])
        check(f"表7.1 {hit[0]}", PAPER_71[hit[0]], (score, loss_pct, draw_pct))
    else:
        print(f"  JSON側で表7.1に一致しない行: {label} → ({score}, {loss_pct}%, {draw_pct}%)")
for k in PAPER_71:
    if k not in matched:
        if k == "攻撃+引き分け罰":
            # score は k_sweep_offense_pennylane.json の oracle 欄で照合できる
            check("表7.1 攻撃+引き分け罰 score", PAPER_71[k][0], r(ks["oracle"], 4),
                  r(ks["oracle"], 4) == PAPER_71[k][0])
            unknown("表7.1 攻撃+引き分け罰 敗率/引分率", PAPER_71[k][1:],
                    "W/L/D の記録がディスクに無い（scoreのみ照合済み）")
        else:
            unknown(f"表7.1 {k}", PAPER_71[k], "このJSONに該当行なし（別JSONの可能性）")

# %% 9. チェックポイント ↔ 表6.1 パラメータ数（tex 直書き）
import torch

PARAM_PAPER = {"MLP": 186_643, "Quantum": 122_938, "ResNet": 137_514,
               "GRU": 119_731, "LowParam": 19}
CKPT_FILES = {
    "MLP": "color_gated_mlp_color_estimator_20260622.pth",
    "Quantum": "pennylane_quantum_300_cnot.pth",
    "ResNet": "color_gated_resnet_color_estimator_20260622.pth",
    "GRU": "color_gated_history_gru_color_estimator_20260622.pth",
    "LowParam": "color_gated_prior_only_color_estimator_20260623.pth",
}
for short, fname in CKPT_FILES.items():
    try:
        cp = torch.load(EXP / fname, map_location="cpu", weights_only=False)
        sd = cp.get("model_state_dict", cp if isinstance(cp, dict) else {})
        n = sum(v.numel() for v in sd.values() if hasattr(v, "numel"))
        check(f"パラメータ数 {short}", f"{PARAM_PAPER[short]:,}", f"{n:,}",
              n == PARAM_PAPER[short])
    except Exception as e:
        unknown(f"パラメータ数 {short}", f"{PARAM_PAPER[short]:,}", f"読込失敗: {e}")

# %% 10. 学習棋譜のサンプル数 ↔ 付録B.2（Qugeister_clean のメタデータ）
meta_path = Path("/home/ks/Qugeister_clean/trajectories/color_gated_decision_sensitive_20260622.json")
if meta_path.exists():
    meta = json.loads(meta_path.read_text())
    print("trajectory meta keys:", list(meta.keys()))
    flat = json.dumps(meta)
    for label, key in [("学習棋譜サンプル 25,061", "25061"),
                       ("生サンプル 14,685", "14685"),
                       ("holdout サンプル 4,007", "4007")]:
        check(label, key, "メタデータ内に出現" if key in flat else "出現せず", key in flat)
else:
    unknown("学習棋譜サンプル数", "25,061 / 14,685 / 4,007", f"{meta_path} なし")

# %% 11. 表7.2 Top-5 保持率（22.1/21.0/20.6/18.2）のデータ源を探す
# データ源は rerun_hidden_factor_with_pennylane_quantum.py の出力ログ
# （repro_ext/logs/topk_retention.log、2026-08-03 に PennyLane 版で再実行済み）
log = (THESIS / "repro_ext/logs/topk_retention.log").read_text()
PAPER_72_RET = {"HistoryGRU": 22.1, "MLP": 21.0, "ResNet": 20.6, "Quantum": 18.2}
for short, paper_v in PAPER_72_RET.items():
    full = SC_NAME[short if short != "HistoryGRU" else "GRU"]
    mrow = re.search(rf"{full}\s+[\d.]+%\s+([\d.]+)%", log)
    logged = r(float(mrow.group(1)), 1) if mrow else None
    check(f"表7.2 Top-5保持率 {short}", paper_v, logged, logged == paper_v)
PAPER_72_RANK = {"piece_accuracy": "3/6", "topk_recall": "5/6",
                 "true_mass": "3/6", "effective_worlds": "2/6"}
for metric, paper_v in PAPER_72_RANK.items():
    mrow = re.search(rf"{metric}\s+(\d/6)", log)
    check(f"表7.2 順位一致 {metric}", paper_v, mrow.group(1) if mrow else None,
          bool(mrow) and mrow.group(1) == paper_v)

# %% 12. 結果サマリ
ok = sum(1 for x in results if x[3] == "✓")
ng = [x for x in results if x[3] == "✗"]
pend = [x for x in results if x[3] == "△"]
print(f"\n===== 検証サマリ: ✓{ok} / ✗{len(ng)} / △{len(pend)} =====")
for label, pv_, dv, mark in results:
    if mark != "✓":
        print(f"{mark} {label}: 論文={pv_} / データ={dv}")
print("\n（✓ の明細）")
for label, pv_, dv, mark in results:
    if mark == "✓":
        print(f"✓ {label}: {pv_}")
