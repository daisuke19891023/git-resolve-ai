# GOAPGit 仕様書 v0.1 草案（詳細版）

本書は GOAPGit パッケージの v0.1 に向けた完全仕様書です。GOAP（Goal-Oriented Action Planning）を核に、最新の Git プラクティスと pydantic v2 を組み合わせて「ゴール＝コンフリクト解消」に到達するまでの計画・実行・観測・再計画ループを自動化します。

---

## 0. パッケージの位置づけ

- **名称（案）**: `goapgit`
- **目的**: Git リポジトリの衝突解消とブランチ鮮度維持（小まめな rebase/pull）を GOAP によって半自動化する。
- **非目的**: 意味理解を伴う最終意思決定の完全自動化。難所では人間オペレーターにフォールバックする。

---

## 1. 全体アーキテクチャ

```
goapgit/
  core/            # GOAP 中核
    models.py      # pydantic データ型（State/Goal/Plan/ActionSpec…）
    planner.py     # A* プランナー + ヒューリスティクス
    executor.py    # 実行・観測・再計画ループ
    cost.py        # コスト・リスクモデル
    explain.py     # Explainability（根拠/代替案）
  git/
    facade.py      # Git コマンド安全ラッパ（dry-run/timeout/戻り値正規化）
    observe.py     # 状態観測（status/merge-tree 解析/競合抽出）
    parse.py       # porcelain v2・conflict markers・range-diff 解析
    strategies.py  # ours/theirs/zdiff3/rerere/merge driver 等の戦術
  actions/         # 原子的 Action 実装
    sync.py
    rebase.py
    conflict.py
    quality.py
    safety.py
  plugins/
    json_merge.py  # JSON/YAML 専用マージドライバの例
  cli/
    main.py        # Typer ベース CLI
  llm/
    client.py      # Responses API ラッパ（OpenAI/Azure 切替）
    instructions.py  # 役割別テンプレート生成
  io/
    config.py      # 設定（pydantic）読み込み/検証
    logging.py     # 構造化ログ
    snapshot.py    # ステートスナップショット
  tests/           # pytest（unit/integration）
```

付随ディレクトリとして `docs/adr/` にアーキテクチャ判断を記録し、`docs/reference/` に CLI/データモデルの API 仕様を整備する。

---

## 2. 仕様（機能要求）

### 2.1 ゴール（Goal）

最終ゴールは次の条件を満たすリポジトリ状態である。

1. 競合ファイルが 0 件で、進行中の rebase/merge が存在しない。
2. 作業ツリーがクリーンで、ステージ済みファイルも無い。
3. 追跡ブランチとの差分が解消され、fast-forward もしくは rebase が完了している。
4. オプション: テストが成功し、`git push --force-with-lease` が完了している。

### 2.2 コア機能

1. **計画**: RepoState をノードとし、Action をエッジにした A* で最短手順を合成。
2. **予測**: `git merge-tree --write-tree` を用いて非破壊の擬似マージを行い、競合リスクと難度を推定。
3. **実行**: GitFacade が timeout/dry-run/リトライを含む安全なコマンド実行を行う。
4. **観測**: `git status --porcelain=v2` と zdiff3 競合マーカー解析により新しい RepoState を構築する。
5. **再計画**: 予測と観測の差異や失敗を検知し、必要に応じてプランを再生成する。

### 2.3 Git ベストプラクティスの組み込み

- `git push --force-with-lease` をデフォルトにし、`--force` は明示時のみ許可。
- `git rebase --update-refs` と `rebase.updateRefs=true` を活用し、スタックブランチの参照を自動更新。
- `merge.conflictStyle=zdiff3` を推奨。diff3 より文脈が多く、競合解決の効率が向上。
- `rerere.enabled=true` と `rerere.autoupdate=true`（任意）で既知解決の自動適用を実現。
- デフォルトマージ戦略は `ort`。必要に応じて `-X diff-algorithm=histogram` や `-X patience` を切り替える。
- `git pull --rebase` / `--ff-only` を状況に応じて切り替え、直線履歴を維持。
- 大規模リポは `git sparse-checkout`（cone モード）や `git worktree` を活用し、操作を分離・高速化。
- 説明可能性のため `git range-diff` を用いた差分提示を組み込む。

### 2.4 LLM 支援ワークフロー（Responses API 統一）

- 競合パッチ提案、戦術助言、プラン補正、メッセージ生成に OpenAI Python SDK の `client.responses.create(...)` を採用する。
- 直前ターンのみを `previous_response_id` で連結し、入力には今回必要な最小情報（競合ハンク抜粋、失敗要約など）だけを送信する。
- すべてのターンで `instructions` を明示送信し、ガードレールと出力形式を毎回固定する。
- Structured Outputs（JSON Schema strict モード）で Patch/Advice/PlanHint/Message を Pydantic から生成し、`additionalProperties=false` に正規化する。
- テレメトリに `response.id`, `previous_response_id`, トークン使用量を記録し、本文は保存しない。

---

## 3. データモデル（pydantic v2）

### 3.1 基本 Enum

```python
class RiskLevel(str, Enum):
    low = "low"
    med = "med"
    high = "high"

class ConflictType(str, Enum):
    text = "text"
    json = "json"
    yaml = "yaml"
    lock = "lock"
    binary = "binary"

class GoalMode(str, Enum):
    resolve_only = "resolve_only"
    rebase_to_upstream = "rebase_to_upstream"
    push_with_lease = "push_with_lease"
```

### 3.2 リポジトリ状態モデル

```python
class RepoRef(BaseModel):
    branch: str
    tracking: str | None = None
    sha: str | None = None
    model_config = ConfigDict(frozen=True, extra="forbid")

class ConflictDetail(BaseModel):
    path: str
    hunk_count: int = 0
    ctype: ConflictType = ConflictType.text
    trivial_ratio: float = 0.0
    preferred_strategy: str | None = None
    model_config = ConfigDict(frozen=True, extra="forbid")

class RepoState(BaseModel):
    repo_path: Path
    ref: RepoRef
    diverged_local: int = 0
    diverged_remote: int = 0
    working_tree_clean: bool = True
    staged_changes: bool = False
    ongoing_rebase: bool = False
    ongoing_merge: bool = False
    stash_entries: int = 0
    conflicts: tuple[ConflictDetail, ...] = Field(default_factory=tuple)
    conflict_difficulty: float = 0.0
    tests_last_result: bool | None = None
    has_unpushed_commits: bool = False
    staleness_score: float = 0.0
    risk_level: RiskLevel = RiskLevel.low
    model_config = ConfigDict(frozen=True, extra="forbid")
```

### 3.3 計画モデル

```python
class GoalSpec(BaseModel):
    mode: GoalMode = GoalMode.rebase_to_upstream
    tests_must_pass: bool = False
    push_with_lease: bool = False
    model_config = ConfigDict(extra="forbid")

class ActionSpec(BaseModel):
    name: str
    params: Mapping[str, str] | None = None
    cost: float
    rationale: str | None = None
    model_config = ConfigDict(frozen=True, extra="forbid")

class Plan(BaseModel):
    actions: list[ActionSpec]
    estimated_cost: float
    notes: list[str] = Field(default_factory=list)
    model_config = ConfigDict(frozen=True, extra="forbid")
```

### 3.4 設定モデル

```python
class StrategyRule(BaseModel):
    pattern: str
    resolution: str
    when: str | None = None
    model_config = ConfigDict(extra="forbid")

class Config(BaseModel):
    goal: GoalSpec
    strategy_rules: list[StrategyRule] = Field(default_factory=list)
    enable_rerere: bool = True
    conflict_style: Literal["merge", "diff3", "zdiff3"] = "zdiff3"
    allow_force_push: bool = False
    dry_run: bool = True
    max_test_runtime_sec: int = 600
    model_config = ConfigDict(extra="forbid")
```

---

### 3.5 LLM 連携モデル（Structured Outputs）

Responses API の Structured Outputs 制約に合わせ、以下の Pydantic モデルを厳格化する。すべて `ConfigDict(extra="forbid")` を設定し、
Schema サニタイザで `additionalProperties=false` と必須フィールド列挙を保証する。

```python
class PatchSet(BaseModel):
    patches: tuple[str, ...]
    confidence: Literal["low", "med", "high"]
    rationale: str
    model_config = ConfigDict(extra="forbid")

class StrategyAdvice(BaseModel):
    resolution: Literal["ours", "theirs", "manual", "merge-driver"]
    reason: str
    confidence: Literal["low", "med", "high"]
    model_config = ConfigDict(extra="forbid")

class PlanHint(BaseModel):
    action: str
    cost_adjustment_pct: float  # ±20% にクランプ
    note: str | None = None
    model_config = ConfigDict(extra="forbid")

class MessageDraft(BaseModel):
    title: str  # 72 文字以内
    body: str   # 目的→変更→影響→ロールバックの章立て
    model_config = ConfigDict(extra="forbid")
```

Azure/OpenAI 双方で同一スキーマを利用し、JSON Schema は `model_json_schema()` をサニタイズした上で `strict=true` で送信する。

---

## 4. アクション設計（最新 Git の取り込み）

各 Action は「前提条件」「効果」「コスト」「失敗時の巻き戻し方針」を持つ。ここでは主なカテゴリを示す。

### 4.1 準備・安全系

- **Safety:CreateBackupRef** — `git update-ref refs/backup/goap/<ts> HEAD`
  - 効果: 任意時点で HEAD を復元可能。
- **Safety:EnsureCleanOrStash** — `git stash push --include-untracked -m goap/<ts>`（必要時）
  - 効果: 作業ツリーを一時退避しクリーンに保つ。

### 4.2 同期系

- **Sync:FetchAll** — `git fetch --prune --tags`
  - 効果: リモートとの差異を同期し、削除ブランチを検知。

### 4.3 Rebase / Merge 系

- **Rebase:RebaseOntoUpstream** — `git rebase <tracking>`（`--update-refs` や `--rebase-merges` を状況に応じて付与）。
- **Merge:PreviewWithMergeTree** — `git merge-tree --write-tree <ours> <theirs>` で非破壊衝突検出。

### 4.4 競合解決系

- **Conflict:AutoTrivialResolve** — rerere による既知解決の再適用。
- **Conflict:ApplyPathStrategy** — パターンに応じて ours/theirs/merge-driver を自動適用。
- **Conflict:UseMergeDriver(JSON)** — `.gitattributes` とカスタムドライバで JSON/YAML の競合を解決。
- **Conflict:MergetoolDeferred** — 人間による確認を許す GUI/CLI mergetool の起動。

### 4.5 仕上げ系

- **Rebase:ContinueOrAbort** — `git rebase --continue | --skip | --abort`。
- **Quality:RunTests** — `pytest -q` 等ユーザ設定のテストコマンド。
- **Sync:PushWithLease** — `git push --force-with-lease`。
- **Explain:RangeDiff** — `git range-diff -- <before> <after>` を提示して説明責任を果たす（`--` セパレータでオプション注入を防止）。

---

## 5. GOAP プランナー

- **状態表現**: ノード = `RepoState`。
- **アクション表現**: エッジ = `ActionSpec` + 実装クラス。
- **コスト計算**: `Action.cost` にリスクペナルティ、推定所要時間を加算。
- **ヒューリスティクス**: `α * conflicts + β * diverged + γ * ongoing_rebase + δ * staleness` を採用し、過小推定を保証。
- **停止条件**: `GoalSpec` を充足したら探索終了。
- **Explainability**: `Plan.notes` にヒューリスティクス根拠や代替案を記録。

---

## 6. CLI 仕様

```
$ goapgit plan        # 乾式：merge-tree で衝突予測＋最短プラン提示
$ goapgit run         # 実行：1 アクションずつ観測しつつ再計画
$ goapgit dry-run     # 実世界変更なしで手順と影響を一覧
$ goapgit explain     # 意思決定の根拠・代替案・range-diff の提示
$ goapgit diagnose    # リポの健全性/推奨 Git 設定を提示
```

共通オプション: `--repo PATH`, `--config FILE`, `--json`, `--verbose`, `--confirm`。`--json` 指定時は構造化ログで 1 行 1 JSON を保証する。

---

## 7. 設定（pydantic + TOML）

`goapgit.io.config.load_config` が TOML をロードし、`Config` モデルにバリデートする。セクション構造は以下を想定。

```toml
[goal]
mode = "rebase_to_upstream"
tests_must_pass = false
push_with_lease = true

[strategy]
enable_rerere = true
conflict_style = "zdiff3"
rules = [
  { pattern = "**/*.lock", resolution = "theirs" },
  { pattern = "**/*.json", resolution = "merge-driver:json" }
]

[safety]
dry_run = true
allow_force_push = false
max_test_runtime_sec = 600

[llm]
provider = "env"          # env → openai/azure を環境変数で指定
model = "gpt-4o-mini"      # Azure はデプロイ名
mode = "suggest"           # off|explain|suggest|auto
safety = "balanced"
max_tokens = 1200
max_cost_usd = 1.0
```

優先順位は「環境変数 > TOML > デフォルト」。`GOAPGIT_LLM_PROVIDER` が `azure` の場合、`AzureOpenAI` クライアントを選び `AZURE_OPENAI_API_KEY`
もしくは `AZURE_OPENAI_AD_TOKEN`、`AZURE_OPENAI_ENDPOINT`、`OPENAI_API_VERSION` を要求する。`openai` の場合は `OpenAI` クライアントを使い
`OPENAI_API_KEY` と任意の `OPENAI_BASE_URL` を解決する。CLI フラグやプロファイルは `overrides` でマージし、ValidationError を以て型不整合
を通知する。

---

## 8. プロジェクト管理（uv + pyproject.toml）

- `uv` を標準ツールとし、`uv add`, `uv lock`, `uv sync` を基礎コマンドとする。
- `tool.uv.package = true` を指定し、パッケージとしての扱いを固定。
- `pyproject.toml` では `pydantic>=2.6`, `typer>=0.12`, `rich>=13.7` を主依存とし、`pytest`, `ruff`, `mypy` 等を dev 依存に登録。
- Responses API を扱う optional extra `llm` を設け、`openai>=2.6`, `httpx>=0.27` を含める。
- `uv.lock` を VCS にコミットし再現性を確保する。

---

## 9. ログ/テレメトリ

- JSON Lines 形式を採用。各レコードは `timestamp`, `level`, `logger`, `message`, `action_id` などのキーを含む。
- `--json` オプション指定時は構造化ログのみを出力し、CI/ダッシュボード連携を容易にする。
- テキストモードでは `[timestamp] LEVEL logger: message | key=value` 形式で人間可読なログを提供する。

---

## 10. セキュリティ/安全

- 既定は dry-run。`--confirm` で実実行に切り替える。
- `--force-with-lease` 以外の強制更新は `allow_force_push=false` が解除されない限り禁止。
- 認証トークンや URL はログからサニタイズする。

---

## 11. テスト戦略

1. **単体テスト**: パーサ、観測器、プランナーのヒューリスティクス関数などをモックで検証。
2. **結合テスト**: 一時ディレクトリで `git init` → 2 ブランチ作成 → 競合再現 → Action 実行まで確認。
3. **シナリオテスト**: lock ファイルの theirs 自動解決、JSON プラグインによる成功/失敗、`--update-refs` の挙動を確認。
4. **説明可能性**: `git range-diff` 出力の存在・最小要素を確認するテストを追加。

---

## 12. 実装タスク一覧（LZ01〜LZ31）

Responses API ベースの LLM 拡張に向けた最終タスクリスト。各タスクは 1〜3 時間規模で、受け入れ基準 (AC) を満たすこと。

### A. クライアント層

- **[LZ01] Provider 切替実装** — `GOAPGIT_LLM_PROVIDER` に基づき OpenAI/Azure のクライアントを自動選択。HTTP モックで双方向を検証。
- **[LZ02] Responses API ラッパ（最小履歴）** — `complete_json()` が `instructions` の再送と `previous_response_id` チェーンを管理。
- **[LZ03] Schema サニタイザ** — Pydantic schema を Strict JSON Schema に整形し、Structured Outputs で受理されることを確認。
- **[LZ04] Instructions コンポーザ** — Resolver/Messenger/Planner 役割に応じたテンプレートを提供し、各呼び出しで必ず添付。

### B. 競合解決ワークフロー

- **[LZ10] LLM:ProposePatch（Responses 版）** — 最小抜粋入力で PatchSet を生成し、失敗時は `previous_response_id` で鎖状に再提案。
- **[LZ11] StrategyAdvice（Responses 版）** — ours/theirs/manual/merge-driver の推奨 JSON を返し、ゴールデンケースを一致させる。
- **[LZ12] PlanHint（コスト補正）** — ±20% クランプ済みのコスト補正を返し、A* の見積もりログに反映する。
- **[LZ13] MessageDraft（commit/pr）** — 72 文字以内タイトルと章立て本文を生成し、`response.output_text` と整合。

### C. セーフティ／運用

- **[LZ20] Redactor & Budget** — 機密サニタイズと `--llm-max-tokens` / `--llm-max-cost` による早期停止を実装。
- **[LZ21] LLM Telemetry（Responses 版）** — `response.id` 鎖とトークン統計を JSON Lines に記録。

### D. CLI

- **[LZ30] llm doctor** — 環境変数チェック、Structured Outputs のパース、`previous_response_id` 連鎖の健全性を診断。
- **[LZ31] run --llm モード拡張** — `off|explain|suggest|auto` の各モードとセーフティレベルを Responses 版で再構築。

推奨実装順は `[LZ01] → [LZ04] → [LZ10]/[LZ12] → [LZ30]`。各フェーズ完了後に README / docs を更新し、`uv run nox -s lint` と `uv run nox -s typing` を通過させること。

---

## 13. LLM 運用ガイドライン

1. **最小履歴ポリシー** — 直前の `response.id` のみを `previous_response_id` に指定し、入力は最新の差分や抜粋に絞る。履歴全文や秘匿情報を送らない。
2. **instructions の再送** — Responses API は instructions を継承しない前提で、各ターンでテンプレートを明示送信する。
3. **Structured Outputs の厳格化** — サニタイザで `additionalProperties=false`、必須項目列挙、深さ/項目数の制約を守る。逸脱時はリトライと軽微な自動修復で対処。
4. **セキュリティ** — サニタイズ対象（API キー、URL パラメータ等）を送信前にマスクし、ログにも非表示で記録する。
5. **コスト管理** — `--llm-max-tokens` と `--llm-max-cost` を超過したら LLM 呼び出しを停止し、従来の GOAP 手順へフォールバックする。
6. **テレメトリ** — `response.id`, `previous_response_id`, token usage, mode, 成否のみを JSON Lines に記録し、応答本文は保持しない。
7. **モデル切替** — チェーン中にモデルやデプロイを変えると継承が不安定になるため、同一モデルでチェーンを完結させる。切替が必要な場合は新規チェーンを開始する。

---

## 13. 代表ユーザーフロー

1. **goapgit plan**: `merge-tree` で競合を予測し、コスト付きの最短プランを提示。
2. **goapgit run --confirm**: `CreateBackupRef` → `FetchAll` → `RebaseOntoUpstream --update-refs` → 各種競合解決 → テスト → `PushWithLease`。
3. **goapgit explain**: `git range-diff` による before/after の差分説明を提示し、採用しなかった代替案を `Plan.notes` で説明。

---

## 14. リスクと対策

- Rebase 中は ours/theirs の意味が逆転する点を抽象化層で吸収する。
- `merge=ours/theirs` の過剰適用は破壊的なため、StrategyRule で限定しログに根拠を残す。
- 乾式計画は `merge-tree` を第一選択とし、作業ツリーを汚さない。

---

## 15. 推奨 Git 設定支援 (`goapgit diagnose`)

- `git config --global merge.conflictStyle zdiff3`
- `git config --global rerere.enabled true`
- `git config --global pull.rebase true`

診断コマンドは既存設定との差異を JSON で提示し、必要に応じて sparse-checkout/worktree の利用を助言する。

---

## 16. まとめ

GOAPGit は pydantic v2 による厳格なデータモデル、uv による再現性の高い環境、最新 Git 機能（`--update-refs`, `merge-tree`, `zdiff3`, `rerere`, `--force-with-lease` など）を Action と Strategy に取り込み、GOAP による計画策定と説明可能性を備えた Git オペレーション自動化基盤を目指す。ここに記載したタスク群 (T01〜T61) を順次実装することで、最小プロトタイプから実運用レベルへ段階的に成熟させることができる。

