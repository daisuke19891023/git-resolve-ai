# GOAPGit 実装タスクリスト（Responses API 強化版）

本ドキュメントは LLM による git conflict 解消を効率化するための追加改修計画（LZ01-LZ31）を管理する。すべてのタスクは 1〜3 時間規模を想定し、完了時には `uv run nox -s lint` と `uv run nox -s typing` を通過させること。

---

## フェーズ A: LLM クライアント基盤（LZ01-LZ04）

| ID   | タスク | 目的 | 主要アウトプット | 受け入れ基準 (AC) | 備考 |
|------|--------|------|------------------|--------------------|------|
| LZ01 | Provider 切替実装 | `GOAPGIT_LLM_PROVIDER` で OpenAI / Azure を自動切替 | `src/goapgit/llm/client.py` | 環境変数だけでプロバイダ切替が動作（HTTP モック） | `AzureOpenAI` を優先利用 |
| LZ02 | Responses API ラッパ | `complete_json()` で instructions 再送と `previous_response_id` チェーンを管理 | 同上 | 2 連続呼び出しで直前 ID だけを渡して文脈継承を確認 | `response.output_text` も露出 |
| LZ03 | Schema サニタイザ | Pydantic Schema を Strict JSON Schema に整形 | `src/goapgit/llm/schema.py` | PatchSet/StrategyAdvice/PlanHint/MessageDraft のスキーマが API に受理される | `additionalProperties=false` を保証 |
| LZ04 | Instructions コンポーザ | 役割別テンプレート生成と再送 | `src/goapgit/llm/instructions.py` | 全呼び出しログに instructions が含まれる | Realtime の継承注意を README に記載 |

---

## フェーズ B: 競合解決ワークフロー適用（LZ10-LZ13）

| ID   | タスク | 目的 | 主要アウトプット | 受け入れ基準 (AC) | 備考 |
|------|--------|------|------------------|--------------------|------|
| LZ10 | LLM:ProposePatch | 競合ハンク最小抜粋→PatchSet 生成 | `src/goapgit/llm/patch.py` | json/md/lock の 3 種で `git apply --check` が成功、失敗時に previous ID で再提案可能 | 失敗フィードバックのみ再送 |
| LZ11 | StrategyAdvice | ours/theirs/manual/merge-driver 推奨 JSON | `src/goapgit/llm/advice.py` | lock→theirs、json→merge-driver、md→manual のゴールデン一致 | merge-tree 要約を few-shot |
| LZ12 | PlanHint | 代替アクションと±20% コスト補正を返す | `src/goapgit/llm/plan.py` | A* のログに adjusted_cost が出力される | 補正は ±0.2 にクランプ |
| LZ13 | MessageDraft | commit/PR 用メッセージの JSON 出力 | `src/goapgit/llm/message.py` | タイトル 72 文字以内、本文は 4 章構成、`response.output_text` と一致 | range-diff 要約を入力 |

---

## フェーズ C: セーフティと運用（LZ20-LZ21）

| ID   | タスク | 目的 | 主要アウトプット | 受け入れ基準 (AC) | 備考 |
|------|--------|------|------------------|--------------------|------|
| LZ20 | Redactor & Budget | サニタイズとコスト制御 | `src/goapgit/llm/safety.py` | 疑似鍵がマスクされ、予算超過時に LLM フェーズが停止 | 秘密検出は正規表現 |
| LZ21 | LLM Telemetry | Responses チェーンの追跡 | `src/goapgit/llm/telemetry.py` | JSON Lines に `response.id` 鎖とトークン使用量が記録される | 応答本文は保存しない |

---

## フェーズ D: CLI 統合（LZ30-LZ31）

| ID   | タスク | 目的 | 主要アウトプット | 受け入れ基準 (AC) | 備考 |
|------|--------|------|------------------|--------------------|------|
| LZ30 | `goapgit llm doctor` | 環境診断とチェーン擬似実行 | `src/goapgit/cli/llm_doctor.py` | OpenAI/Azure 双方で環境未整備・整備の判定が正しく表示 | ネット未接続時はモック |
| LZ31 | `goapgit run --llm` 強化 | モード/セーフティ設定を Responses 版へ適用 | `src/goapgit/cli/run.py` | `off|explain|suggest|auto` が期待通り動作し、自動適用は安全条件を満たす場合のみ | 既存ログと互換性維持 |

---

## 進行順とマイルストーン

1. **基盤構築** — LZ01-LZ04 を完了し、Responses API で最小チェーンが成立すること。
2. **ワークフロー実装** — LZ10-LZ13 を順次実装し、git apply / ゴールデンケース / コスト補正 / メッセージ生成の AC を満たすこと。
3. **セーフティ強化** — LZ20-LZ21 を実装し、秘密マスキングとトークン予算の監査を確立すること。
4. **CLI 統合** — LZ30-LZ31 を実装し、開発者が環境診断と `run` モード拡張を利用できるようにすること。

各マイルストーン後に README と仕様書を更新し、`docs/adr/` に主要判断を記録する。
