# GOAPGit User Manual / ユーザーマニュアル

## English

### Overview
GOAPGit is a command-line assistant that leverages Goal Oriented Action Planning (GOAP) to restore a Git repository to a healthy state. It inspects the current repository, predicts the safest recovery path, and can explain each planned action before execution.

### Prerequisites
- Git 2.40 or later
- Python 3.13 or later
- [uv](https://github.com/astral-sh/uv) for dependency management

Ensure you have access to the target repository and the permission to run Git commands. For actions that modify remote branches, confirm that you can push using `--force-with-lease` when necessary.

### Installation
```bash
# Clone the repository
git clone https://github.com/<your-org>/goapgit.git
cd goapgit

# Install dependencies
uv sync
```

If you are contributing to GOAPGit, create a virtual environment before running the commands above or rely on uv's isolated execution (`uv run`).

### Configuration
GOAPGit reads settings from TOML files validated via Pydantic. By default it loads `.goapgit.toml` in the repository root, but you can point to another file using `--config`.

A minimal configuration looks like this:
```toml
[goal]
mode = "rebase_to_upstream"
tests_must_pass = false

[strategy]
enable_rerere = true
conflict_style = "zdiff3"

[safety]
dry_run = true
allow_force_push = false
```
Override individual keys at runtime using CLI options (for example `--confirm` or `--json`).

### Basic Commands
| Command | Description |
| --- | --- |
| `goapgit plan` | Inspect repository state, simulate merges, and output the shortest action plan. |
| `goapgit run` | Execute the planned actions. Requires `--confirm` to apply real Git changes. |
| `goapgit dry-run` | Simulate execution and list Git commands without modifying the repository. |
| `goapgit explain` | Provide a rationale for each action, alternative strategies, and estimated costs. |
| `goapgit diagnose` | Report recommended Git settings and repository health checks. |

Common flags:
- `--repo PATH`: work against another repository
- `--config FILE`: load a specific TOML configuration
- `--json`: emit machine-readable JSON instead of human-readable text
- `--confirm`: allow state-changing operations (otherwise runs in dry mode)
- `--verbose`: increase logging detail for troubleshooting

### Typical Workflow
1. Run `goapgit plan` to understand the repository status and suggested steps.
2. Inspect the output. If the plan looks safe, run `goapgit dry-run` to review the exact Git commands.
3. Execute `goapgit run --confirm` to apply the plan. GOAPGit will re-evaluate after each action and replan if needed.
4. Optional: `goapgit explain --json` to archive a structured explanation for code review or incident records.

### Example Session
```bash
# Check the current plan
uv run goapgit plan --json > plan.json

# Preview Git commands without changing the repository
uv run goapgit dry-run

# Execute once you are satisfied
uv run goapgit run --confirm
```

### Troubleshooting
- **Plan fails due to merge conflicts:** Review the conflict list in the plan output. Update strategy rules in the configuration (e.g., specify custom merge drivers) and rerun `plan`.
- **`--confirm` skipped actions:** Ensure you have the required permissions and that safety settings (such as `allow_force_push`) permit the operation.
- **Unexpected repository state:** Use `goapgit diagnose` to identify missing Git configuration, then adjust your local settings.

### Quality and Safety Practices
- Always inspect plans before running with `--confirm`.
- Keep your repository clean: GOAPGit expects no untracked or unstaged files unless the plan explicitly covers them.
- Enable `rerere` and `zdiff3` in Git to benefit from automatic conflict resolution guidance.

### Getting Help
Consult `README.md` for development practices, explore `spec.md` for detailed architecture, or file an issue in the repository tracker if you encounter a defect.

---

## 日本語

### 概要
GOAPGit は Goal Oriented Action Planning（GOAP）を活用し、Git リポジトリを安全な状態に戻すためのコマンドライン支援ツールです。現在のリポジトリを観測し、最も安全で短い回復手順を提案し、各アクションの根拠も説明できます。

### 前提条件
- Git 2.40 以上
- Python 3.13 以上
- 依存関係管理に [uv](https://github.com/astral-sh/uv)

対象リポジトリにアクセスでき、Git コマンドを実行する権限があることを確認してください。リモートブランチに変更を加える場合は、必要に応じて `--force-with-lease` でプッシュできるか事前に確認しておきます。

### インストール手順
```bash
# リポジトリを取得
git clone https://github.com/<your-org>/goapgit.git
cd goapgit

# 依存関係をインストール
uv sync
```

GOAPGit にコントリビュートする場合は、上記コマンドを実行する前に仮想環境を作成するか、`uv run` による分離実行を利用してください。

### 設定
GOAPGit は TOML ファイルの設定を Pydantic で検証して読み込みます。既定ではリポジトリ直下の `.goapgit.toml` を使用しますが、`--config` で別ファイルを指定できます。

最小構成の例:
```toml
[goal]
mode = "rebase_to_upstream"
tests_must_pass = false

[strategy]
enable_rerere = true
conflict_style = "zdiff3"

[safety]
dry_run = true
allow_force_push = false
```

CLI オプション（例: `--confirm` や `--json`）で個別の設定を上書きできます。

### 基本コマンド
| コマンド | 説明 |
| --- | --- |
| `goapgit plan` | リポジトリ状態を観測し、仮想マージを通じて最短プランを提示します。 |
| `goapgit run` | プランを実行します。実際に変更するには `--confirm` が必要です。 |
| `goapgit dry-run` | リポジトリを変更せずに Git コマンドの一覧を表示します。 |
| `goapgit explain` | 各アクションの根拠、代替案、コスト見積もりを説明します。 |
| `goapgit diagnose` | 推奨 Git 設定とリポジトリの健全性をレポートします。 |

主なフラグ:
- `--repo PATH`: 別リポジトリを対象にする
- `--config FILE`: 指定した TOML 設定を読み込む
- `--json`: 人間向けテキストではなく JSON を出力
- `--confirm`: 実際に状態を変更する操作を許可
- `--verbose`: トラブルシュート用にログ詳細度を上げる

### 典型的な利用手順
1. `goapgit plan` を実行してリポジトリの現状と提案ステップを確認します。
2. 出力を精査し、安全だと判断したら `goapgit dry-run` で実行予定の Git コマンドを確認します。
3. 納得したら `goapgit run --confirm` を実行し、アクションを適用します。GOAPGit は各アクション後に再観測し、必要であれば再計画します。
4. オプション: `goapgit explain --json` でレビューや事後共有のための構造化レポートを保存します。

### 利用例
```bash
# 現在のプランを確認
uv run goapgit plan --json > plan.json

# 変更を加えずに Git コマンドを確認
uv run goapgit dry-run

# 準備が整ったら実行
uv run goapgit run --confirm
```

### トラブルシューティング
- **マージ競合でプランが失敗する:** プラン出力の競合一覧を確認し、設定の戦略ルール（例: カスタムマージドライバ）を調整して再度 `plan` を実行してください。
- **`--confirm` でもアクションがスキップされる:** 権限があるか、設定（`allow_force_push` など）が操作を許可しているか確認します。
- **想定外のリポジトリ状態が検出された:** `goapgit diagnose` を使って必要な Git 設定を特定し、ローカル環境を整備してください。

### 品質と安全のベストプラクティス
- `--confirm` で実行する前に必ずプラン内容を確認します。
- 余分な変更を残さないようリポジトリをクリーンに保ちます。計画に含まれていない未追跡ファイルやステージングはエラーの原因になります。
- Git の `rerere` と `zdiff3` を有効化し、自動競合解決の恩恵を受けましょう。

### サポート
開発に関する情報は `README.md` を、詳細なアーキテクチャは `spec.md` を参照してください。問題が発生した場合はリポジトリの issue トラッカーで報告してください。
