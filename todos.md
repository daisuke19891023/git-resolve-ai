# GOAPGit 実装タスクリスト（詳細版）

本ドキュメントは v0.1 で想定している 61 個のタスクをフェーズごとに整理し、目的・アウトプット・受け入れ基準 (AC)・依存関係・備考を明記する。各タスクは 1〜3 時間規模を目安に設計している。

---

## フェーズ 1: 基盤整備（T01-T03）

| ID | タスク | 内容 | 出力 | AC | 依存/備考 |
|----|--------|------|------|----|-----------|
| T01 | pydantic モデル定義 | §3 のモデルを `goapgit/core/models.py` に実装。`RepoState` と `Plan` の JSON round-trip を保証し、モデルは immutability を維持する。 | `src/goapgit/core/models.py` | ruff/mypy 合格。`pytest` で round-trip とデフォルト値テスト通過。 | 先行で `tests/unit/core/test_models.py` を作成。 |
| T02 | 設定ローダ | TOML からのロード関数を `goapgit.io.config.load_config` に実装。`overrides` での上書きにも対応する。 | `src/goapgit/io/config.py` | 不正キー・型で `ValidationError`。例示 TOML がロード可。ファイル未存在で `FileNotFoundError`。 | `tomllib` を利用し、深い辞書マージを実装。 |
| T03 | 構造化ログ | JSON Lines ベースのロガーを `goapgit.io.logging.StructuredLogger` として実装。 | `src/goapgit/io/logging.py` | `--json` モードで 1 行 1 JSON。INFO/ERROR 等のレベル切替と付加フィールドが動作。 | 既定はテキストモード。CI では JSON モードを利用。 |

---

## フェーズ 2: Git ラッパ／観測（T10-T14）

| ID | タスク | 内容 | 主出力 | AC | 依存/備考 |
|----|--------|------|--------|----|-----------|
| T10 | GitFacade.run | subprocess ラッパを実装。`timeout`, `cwd`, `dry_run` の制御と例外化を行う。 | `src/goapgit/git/facade.py` | 正常/異常終了の扱いが pytest モックで確認できる。dry-run 時は履歴に記録するのみ。 | フェーズ1のロガーを利用しコマンド出力を記録。✅ 完了 |
| T11 | fetch/rebase/push API | `fetch()`, `rebase()`, `rebase_continue()/abort()`, `push_with_lease()` を提供。 | 同上 | 所定の引数で git を呼び出す。`subprocess` モックで検証。 | `GitFacade.run` に依存。✅ 完了 |
| T12 | 状態観測 | `git status --porcelain=v2` を解析し `RepoState` を構築。 | `src/goapgit/git/observe.py` | 変更あり/なし/競合ありのケースがテストデータと一致。 | `core.models.RepoState` を利用。✅ 完了 |
| T13 | 競合詳細解析 | zdiff3 マーカーを解析し `ConflictDetail.hunk_count` と `ctype` を推定。 | `src/goapgit/git/parse.py` | サンプル競合ファイルで期待値が得られる。 | `.json`/`.yaml` 拡張子の判定ルールを実装。✅ 完了 |
| T14 | merge-tree 衝突予測 | `git merge-tree --write-tree` の出力を解析して競合集合を推定。 | 同上 | 擬似リポで実マージと結果が一致。 | 重いテストは `pytest.mark.integration`。✅ 完了 |

---

## フェーズ 3: プランナー／実行（T20-T22）

| ID | タスク | 内容 | 主出力 | AC | 依存/備考 |
|----|--------|------|--------|----|-----------|
| T20 | ヒューリスティクス h(s) | 競合数・乖離・進行中作業・陳腐化度を重み付きで評価する関数。 | `src/goapgit/core/cost.py` | 指標が増えると h(s) が単調増加。単体テストで係数調整。 | 係数は設定値として `Config` から注入可能に。 |
| T21 | A* プランナー | Action グラフから最短プランを生成。 | `src/goapgit/core/planner.py` | 3〜5 個のダミーアクションで期待経路を返す pytest を通過。 | `Plan`/`ActionSpec` を活用。 |
| T22 | Executor | 実行→観測→再計画ループを構築。 | `src/goapgit/core/executor.py` | 故意に Action を失敗させると再計画分岐に入るテストが通過。 | `GitFacade`, `planner`, `logging` に依存。 |

---

## フェーズ 4: アクション実装（T30-T39）

| ID | タスク | 内容 | 主出力 | AC | 依存/備考 |
|----|--------|------|--------|----|-----------|
| T30 | Sync:FetchAll | `git fetch --prune --tags` の呼び出しとログ出力。 | `src/goapgit/actions/sync.py` | ログに fetch コマンドが記録される。 | `GitFacade` を利用。 |
| T31 | Safety:CreateBackupRef | `refs/backup/goap/<ts>` に HEAD を保存。 | `src/goapgit/actions/safety.py` | `git show-ref` でバックアップが確認できる。 | `timestamp` 生成に `datetime` を使用。 |
| T32 | Safety:EnsureCleanOrStash | 未コミット変更を stash。 | 同上 | 変更ありで stash され、無い場合は no-op。 | Stash 名に `goap/<ts>` を付与。 |
| T33 | Rebase:RebaseOntoUpstream | `--update-refs` オプションをサポート。 | `src/goapgit/actions/rebase.py` | 軽量ブランチで参照更新が確認できる。 | integration テストを `pytest.mark.slow`。 |
| T34 | Conflict:AutoTrivialResolve | rerere による既知衝突の自動解決。 | `src/goapgit/actions/conflict.py` | 再発ケースで diff が空。 | `rerere.enabled` 設定を確認。 |
| T35 | Conflict:ApplyPathStrategy | パターンごとの ours/theirs 適用。 | 同上 | lock→theirs、md whitespace→ours。 | StrategyRule を利用。 |
| T36 | Conflict:UseMergeDriver(JSON) | JSON 用マージドライバの起動。 | `src/goapgit/plugins/json_merge.py` | 成功で 0、失敗で非 0。 | `ruamel.yaml` 等を optional dep とする。 |
| T37 | Rebase:ContinueOrAbort | rebase 継続/スキップ/中断。 | `src/goapgit/actions/rebase.py` | 競合解消後に成功、未解消で失敗を検知。 | エラー時はバックアップから復元。 |
| T38 | Sync:PushWithLease | `git push --force-with-lease`。 | `src/goapgit/actions/sync.py` | 想定と異なる追跡先で拒否される。 | 擬似リモート環境をセットアップ。 |
| T39 | Explain:RangeDiff | `git range-diff` の結果をログ/ファイルに出力。 | `src/goapgit/actions/quality.py` | rebase 前後の差分要約が確認できる。 | 構造化ログで要約を残す。 |

---

## フェーズ 5: 戦略／診断／CLI（T40-T53）

| ID | タスク | 内容 | 主出力 | AC | 依存/備考 |
|----|--------|------|--------|----|-----------|
| T40 | 推奨 Git 設定の提示 | `goapgit diagnose` で `conflictStyle`, `rerere`, `pull.rebase` を評価。 | `src/goapgit/cli/main.py` | 推奨＆検出結果が JSON で出力。 | OS 非依存の `git config --get` を利用。 |
| T41 | 大規模 repo 対応 | ファイル数や履歴深度が閾値を超えた際に sparse/worktree を助言。 | 同上 | diagnose の JSON に提案が含まれる。 | 解析ロジックは `git count-objects` 等。 |
| T50 | CLI:plan | プランの JSON/人可読表示を実装。 | 同上 | RepoState→Plan が表示される。`--json` で構造化出力。 | `planner`, `logging` を利用。 |
| T51 | CLI:run | `--confirm` 有無で dry-run/実実行を切り替え。 | 同上 | `--confirm` 無しで状態不変、指定でアクション実行。 | Executor を呼び出す。 |
| T52 | CLI:explain | アクションの理由・代替案・コストを提示。 | 同上 | `Plan.notes` と `Explain` モジュールを使用。 | `goapgit/core/explain.py` を整備。 |
| T53 | CLI:dry-run | 実世界変更なしで実行手順を一覧。 | 同上 | ファイル/refs が変化しない。 | `StructuredLogger` を JSON モードで利用。 |

---

## フェーズ 6: QA／CI（T60-T61）

| ID | タスク | 内容 | 主出力 | AC | 依存/備考 |
|----|--------|------|--------|----|-----------|
| T60 | pytest シナリオ群 | lock/JSON/手動解決の 3 シナリオを含む統合テスト。 | `tests/integration/...` | シナリオ全成功。`pytest -m integration` で実行。 | 一時ディレクトリにリポ構築。 |
| T61 | ruff + mypy + coverage | CI ターゲットの静的解析とカバレッジ 80% 以上。 | `.github/workflows/ci.yml` など | `uv run nox -s ci` がグリーン。 | カバレッジ閾値は `pytest-cov` で管理。 |

---

## 共通受け入れ基準

1. すべての編集は `uv run nox -s lint` と `uv run nox -s typing` を通過すること。
2. 該当タスクに応じた単体テスト・結合テストが追加され、`uv run nox -s test` が成功すること。
3. 設定追加時は `.env.example` とドキュメントに反映すること。
4. ログは `StructuredLogger` を利用し、重要な操作に因果 ID を付与すること。
5. 重大な設計判断は `docs/adr/` に記録すること。

---

## 進行状況トラッキング

- **Phase 1**: プロジェクト基盤。ここが完了すると GOAP モデル・設定・ロギングの骨格が整う。
- **Phase 2**: Git ラッパと観測。実リポに触れる実装のため、モックを中心とした単体テストと擬似リポによる統合テストを併用。
- **Phase 3**: プランナーと Executor。A* の計算量に注意し、ヒューリスティクス調整を `Config` で管理する。
- **Phase 4**: 行動群の具象化。rerere や merge-tree といった Git の先進機能を組み込み、Explainability を強化。
- **Phase 5**: CLI と診断。ユーザ向けインターフェースと推奨設定の提示で運用性を高める。
- **Phase 6**: QA/CI。自動化された品質ゲートとカバレッジ確保でリリース準備を整える。

進行状況は GitHub Projects 等で管理し、各タスク完了時にドキュメントを更新すること。

