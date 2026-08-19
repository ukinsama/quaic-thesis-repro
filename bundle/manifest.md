# 再現性バンドル：第8章の拡張実験

修論第8章（議論）で報告した拡張実験の再現材料一式。第7章の凍結基本セット（roster）とは
別条件（拡張1,200局データ・方策/アダプター/損失の変更）であり、ゲーム軸 score はすべて
全9,800局の全列挙による厳密値。第7章の roster バンドル（`backend/data/paper_runs/`、
`make paper-bundle` 生成物）と同じ趣旨で、コミットハッシュ・SHA-256・再実行コマンドを記録する。

## リポジトリのコミット

| リポジトリ      | コミット  |
| --------------- | --------- |
| QuAic           | `3c1d6a2` |
| Qugeister_clean | `34b73a3` |

- `venv` = QuAic backend（`/home/ks/QuAic/backend/venv`、pydantic-settings/torch/pennylane）
- `.venv` = Qugeister_clean（学習は `--gpu`）
- 全列挙評価は CPU（`exhaustive_score.py`、batch-1 逐次）。QuAic backend venv で実行する。
- 共通の全列挙エンジン: `Qugeister_clean/scripts/evaluation/exhaustive_score.py`（14並列・1構成約92秒）
- 学習データ（拡張1,200局）: `Qugeister_clean/trajectories/color_gated_decision_sensitive_merged_train_1200.pkl`
- holdout（凍結）: `Qugeister_clean/trajectories/color_gated_decision_sensitive_20260622_holdout_80.pkl`

## §8.2 固定方策の設計空間

| 内容                                     | スクリプト                                                                                                                                                           | 成果物 (sha256:16)                                                                                                                                 | 再実行                                                                                    |
| ---------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------- |
| 方策バリアント全列挙                     | `Qugeister_clean/scripts/evaluation/policy_variants_exhaustive.py`                                                                                                   | `QuAic/backend/data/experiments/exhaustive_policy_variants.json` (`2e105be7dd6134e0`)                                                              | `cd Qugeister_clean && .venv/bin/python scripts/evaluation/policy_variants_exhaustive.py` |
| ゲート係数の飽和                         | `Qugeister_clean/scripts/evaluation/oracle_floor_tuning.py`                                                                                                          | 同上（pen=0.15/0.6 行）                                                                                                                            | 同ディレクトリで実行                                                                      |
| 浅い探索の負の結果                       | `Qugeister_clean/scripts/evaluation/oracle_search_series_full.py`（bundle のシリーズ関数を本番 300 戦で実行）                                                        | `QuAic/backend/data/experiments/oracle_search_series_full.json`・`repro_ext/logs/oracle_search_series_full.log`                                    | `cd QuAic/backend && venv/bin/python .../oracle_search_series_full.py`                    |
| 色感度（方向性コントロール低下幅の平均） | `QuAic/backend/scripts/run_signal_controls.py`（`--policy`/`--own-good-safety-penalty` で方策を差し替え，normal−inverted の 4 推定器平均）                           | recovered 分は `signal_controls.json`（平均 0.177 = 低下幅 0.217/0.162/0.180/0.148 の平均）                                                        | 同スクリプトを各方策で実行                                                                |
| （参考）Oracle−AntiOracle 版の色感度     | `Qugeister_clean/scripts/evaluation/policy_color_sensitivity.py`（修論の色感度とは別定義。2026-07-22 に 3 方策で実測: recovered 0.632 / asym 0.700 / offense 0.554） | `repro_ext/logs/policy_color_sensitivity.log`                                                                                                      | 同ディレクトリで実行                                                                      |
| 攻撃版方策 (score 0.9629)                | 学習済みモデル                                                                                                                                                       | `Qugeister_clean/experiments/handyrl/color_aware/handyrl_perfect_info_color_aware_resnet_4blk_20260716_142543/best_model.pth` (`9afc6e033a28334c`) | roster 非採用                                                                             |

## §8.3 診断から改良・量子診断

| 内容                            | スクリプト                                                                                                                                                                          | 成果物 (sha256:16)                                                                                                                                                                                 | 再実行                                                                                         |
| ------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------- |
| Top-K世界保持率の対応           | `QuAic/backend/scripts/topk_retention_hidden_factor.py`                                                                                                                             | `repro_ext/logs/topk_retention.log`（Quantum を実 PennyLane 回路へ差し替えた再実行値: Top-5 保持率 22.1/21.0/20.6/18.2・順位一致 3/6・5/6・3/6・2/6。2026-08-03 の再実行で同値を確認しログを更新） | `cd QuAic/backend && venv/bin/python scripts/topk_retention_hidden_factor.py`                  |
| 履歴長スイープ                  | `QuAic/backend/scripts/train_world_aware_gru.py`                                                                                                                                    | `histlen_gru_h16.pth` (`230845b783d43170`)                                                                                                                                                         | `venv/bin/python scripts/train_world_aware_gru.py --history-length 16 --lam 0`                 |
| world損失スイープ               | `QuAic/backend/scripts/train_world_aware_gru.py`                                                                                                                                    | `worldaware_gru_lam030.pth` (`73d77de055ddabd1`)                                                                                                                                                   | `venv/bin/python scripts/train_world_aware_gru.py --lam 0.3`                                   |
| 構成・データ拡張                | `Qugeister_clean/scripts/evaluation/quantum_config_and_data_expansion.py`                                                                                                           | `expanded_quantum_q6l3.pth` (`8b7b7c65a54ca1bd`), `expanded_mlp.pth` (`c69e9fe01982dd40`)                                                                                                          | Qugeister_clean 側で実行                                                                       |
| ゲート種（単一seed）            | `QuAic/backend/scripts/quantum_gate_type_sweep.py`                                                                                                                                  | `gate_type_sweep_results.json` (`a5b2b806a7973028`)                                                                                                                                                | `cd QuAic/backend && venv/bin/python scripts/quantum_gate_type_sweep.py`                       |
| ゲート種（3-seed集計）          | `QuAic/backend/scripts/quantum_gate_type_sweep.py`                                                                                                                                  | `gate_multiseed_results.json` (`598240118c317ab1`)                                                                                                                                                 | 同上（seed 20260622/7/99）                                                                     |
| 履歴×量子HNN                    | `QuAic/backend/scripts/train_quantum_history_gru.py`                                                                                                                                | `quantum_history_gru_cz_h16.pth` (`33236ebce1a11dfd`)                                                                                                                                              | `venv/bin/python scripts/train_quantum_history_gru.py --gate cz --lam 0.3 --history-length 16` |
| kスイープ（帯幅）               | `Qugeister_clean/scripts/evaluation/k_sweep_exhaustive.py`（K_VALUES=[1,3,5,10,20,30]）                                                                                             | `k_sweep_offense_pennylane.json`（Quantum 差し替え後。旧 `k_sweep_offense.json` は代替実装時の記録）                                                                                               | backend venv で実行（PYTHONPATH に Qugeister_clean/src と QuAic/backend を通す）               |
| 拡張ckptのゲーム軸score一括検証 | `Qugeister_clean/scripts/evaluation/extension_checkpoint_scores.py`（履歴長・λ・ゲート種・履歴×量子・データ拡張の各 .pth を roster 方策＋TopK(k=5) で全列挙採点し本文期待値と突合） | `QuAic/backend/data/experiments/extension_checkpoint_scores.json`・`repro_ext/logs/extension_checkpoint_scores.log`（2026-07-22 実行。期待値のある 12 本すべて一致）                               | `cd QuAic/backend && venv/bin/python .../extension_checkpoint_scores.py`                       |

## 注意

- データ拡張とゲーム軸のゲート種の学習は単一 seed（CZ 優位のみ 3-seed で確認済み）。
- ゲート種 3-seed 平均（CZ 0.810 / CNOT 0.776）のうち基準 seed(20260622) の .pth は未保存。s7/s99 の 2 seed 分は `extension_checkpoint_scores.json` に再採点済み（cz 0.821/0.807，cnot 0.761/0.794 で 3-seed 平均と整合）。
- 色感度は Quantum 差し替え後に 4 方策すべて再測定した: recovered 0.177 / 守備 0.135 / 守備+引分罰 0.141 / 攻撃+引分罰 0.196。結果 JSON は `signal_controls_pennylane.json` および `signal_controls_pl_{asym,asymdraw,offense}.json`（再実行は `rerun_controls_with_pennylane_quantum.py --policy ...`）。
- **2026-07-27 の Quantum 差し替え**: 色信念推定器 Quantum を Torch-native の代替実装から本物の PennyLane 回路（`default.qubit`・RY 角度埋め込み・linear CNOT・6 量子ビット×3 層）へ変更した。チェックポイントは `pennylane_quantum_300_cnot.pth`、学習は `train_pennylane_quantum_300.py`。Elo・K スイープ・色感度・診断量は `rerun_*_with_pennylane_quantum.py` で再実行し、Quantum を含まない測定は既存の全列挙結果を再利用している。
- 成果物 `.pth`/`.json` は `QuAic/backend/data/experiments/` 配下（gitignore対象。SHA-256 で同一性を確認する）。
- 将来は `export_paper_bundle.py` を拡張し、本バンドルを `make paper-bundle` で自動生成する予定。
