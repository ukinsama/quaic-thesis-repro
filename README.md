# quaic-thesis-repro

外川貴翔 修士論文（慶應義塾大学大学院 理工学研究科, 2026年9月）
**「QuAic: 不完全情報ゲームにおける量子・古典隠れ状態推定モジュールの交換とベンチマーク評価のためのWebプラットフォーム」**
の評価（第6章・第7章・付録A）を再現・照合するためのスクリプト・結果・学習済み重み・学習棋譜の一式です。

Reproducibility archive (scripts, result JSONs, trained weights, and training
trajectories) for the master's thesis on QuAic, a web platform for swapping and
benchmarking quantum and classical hidden-state estimation modules in the
imperfect-information game Geister.

## 論文との対応

| 論文の表・図                                   | 結果ファイル                               | 生成スクリプト                                      |
| ---------------------------------------------- | ------------------------------------------ | --------------------------------------------------- |
| 表6.2・図6.2・図6.3（score / Elo / W-L-D）     | `results/exhaustive_roundrobin.json`       | `scripts/exhaustive_score.py`（全列挙エンジン）     |
| 図6.1・§6.2（診断軸: 実効世界数・Top-3保持率） | `results/paper_benchmark_diagnostics.json` | QuAic側 `run_paper_true_world_recall.py`            |
| 図6.4・§6.5（方向性コントロール）              | `results/signal_controls.json`             | QuAic側 `run_signal_controls.py`                    |
| §6.5（温度頑健性）                             | `results/temperature_sweep_wide.json`      | QuAic側 `rerun_tempsweep_with_pennylane_quantum.py` |
| 表7.1（固定方策候補の全列挙比較）              | `results/exhaustive_policy_variants.json`  | `scripts/policy_variants_exhaustive.py` ほか        |
| 表7.1 色感度列                                 | `results/signal_controls_pl_*.json`        | `scripts/policy_color_sensitivity.py`               |
| 表7.2（診断量と score の順位一致）             | `bundle/logs/topk_retention.log`           | `scripts/topk_retention_hidden_factor.py`           |
| 表A.1（アダプター解像度 K スイープ）           | `results/k_sweep_offense_pennylane.json`   | `scripts/k_sweep_exhaustive.py`                     |

本文・表の数値と本リポジトリの JSON の機械照合には
`scripts/verify_thesis_numbers.py`（69項目の照合ロジック）を用いた。

## 内容

- `scripts/` — 拡張実験スクリプト（論文 表B.1）と数値照合スクリプト。
  実行は QuAic / Qugeister_clean のコード基盤上で行ったアーカイブであり、
  実行コマンド・依存は各ファイル冒頭の docstring と `bundle/manifest.md` を参照
- `results/` — 全列挙・コントロール実験の結果 JSON（provenance にコミットハッシュを記録）
- `checkpoints/` — 凍結基本セットの色信念推定器 5 種（MLP / ResNet / HistoryGRU /
  LowParam / PennyLane 実回路 Quantum）と固定方策 `policy/best_model.pth`
  （ColorGatedResNetPolicy, recovered checkpoint）。SHA-256 は `bundle/manifest.md`
- `trajectories/` — 学習棋譜（train 300局）と holdout（80局）の pkl とメタデータ
- `bundle/` — 再現性バンドル（manifest と再実行ログ）

## 関連リンク

- 公開デモ: https://quaic.up.railway.app
- 学習ツールキット（Colab対応）: https://github.com/ukinsama/Qugeister
- 実験リポジトリ: https://github.com/ukinsama/Qugeister_clean

## License

MIT License（`LICENSE` を参照）
