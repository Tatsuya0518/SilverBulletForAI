# SBFA - マルチAIチーム + RAG + MCP 実装プラン (v2 修正版)

## Context

SilverBulletForAI (SBFA) は「AIを万能薬として使う」ためのシステム。異種AIがチームを組み、互いの弱点を補完しながら協調動作する。全エージェントが共有RAG（SurrealDB）を読み書きし、A2Aプロトコルでエージェント間通信を行う。ローカルLLM対応の拡張性も確保する。

**v2修正**: v1プランのレビューで発見された13件の欠陥を全て反映。

---

## 技術選定

| 領域 | 技術 |
|---|---|
| 言語 | Python 3.12+ (アプリケーション層) + Rust (高速化コア) |
| DB | SurrealDB v2.x (KVストア + ベクトル検索) |
| エージェント間通信 | A2A Protocol v0.3 (JSON-RPC 2.0 / gRPC / HTTP/REST から選択) |
| RAGベクトル検索 | SurrealDB HNSW インデックス (TYPE F32, DEFER対応) |
| Embedding | OpenAI `text-embedding-3-small` (デフォルト、切替可能) |
| AI SDK | `anthropic`, `google-genai`, `openai` |
| API層 | FastAPI (Python) + Axum (Rust、パフォーマンスクリティカルなエンドポイント) |
| Python-Rust連携 | PyO3 + maturin (PythonからRust呼び出し) |
| MCP | `mcp` Python SDK |
| パッケージ管理 | `uv` + `pyproject.toml` (Python) / `cargo` (Rust) |

### AIモデル構成 (2026年3月時点の実在モデル)

| エージェント | モデルID | 得意分野 |
|---|---|---|
| Claude Sonnet 4.6 | `claude-sonnet-4-6` | コーディング、論理推論、長文分析 |
| Gemini 2.5 Flash | `gemini-2.5-flash` | マルチモーダル、高速応答、大コンテキスト |
| GPT-5.4 | `gpt-5.4` | 汎用タスク、関数呼び出し、構造化出力 |
| ローカルLLM (拡張) | 設定可能 | プライバシー重視、オフライン処理 |

> **注**: モデルIDは `src/sbfa/config.py` で環境変数から設定し、ハードコードしない。モデル更新時は環境変数のみ変更。

---

## プロジェクト構造

```
SilverBulletForAI/
├── pyproject.toml
├── README.md
├── CLAUDE.md
├── .env.example
├── docker-compose.yml          # SurrealDB起動用 (永続化・認証設定込み)
│
├── src/
│   └── sbfa/
│       ├── __init__.py
│       ├── main.py             # FastAPIエントリポイント
│       ├── config.py           # 設定管理 (env読み込み、モデルID設定)
│       │
│       ├── agents/
│       │   ├── __init__.py
│       │   ├── base.py         # BaseAgent 抽象クラス (MCPツール結果対応)
│       │   ├── claude_agent.py
│       │   ├── gemini_agent.py
│       │   ├── openai_agent.py
│       │   └── local_agent.py  # ローカルLLM (Ollama等) 拡張用
│       │
│       ├── a2a/
│       │   ├── __init__.py
│       │   ├── agent_card.py   # AgentCard (camelCase準拠 + JWS署名)
│       │   ├── registry.py     # エージェント登録・発見 (SurrealDB)
│       │   ├── protocol.py     # JSON-RPC メッセージング
│       │   └── security.py     # AgentCard署名検証 (JWS/RFC7515)
│       │
│       ├── rag/
│       │   ├── __init__.py
│       │   ├── ingestion.py    # ドキュメント取り込み (DEFER対応)
│       │   ├── embeddings.py   # Embedding生成 (プロバイダ切替可能)
│       │   ├── retrieval.py    # 類似検索・取得
│       │   └── store.py        # SurrealDB RAGストア操作
│       │
│       ├── orchestrator/
│       │   ├── __init__.py
│       │   ├── router.py       # タスクルーティング (スコアリング)
│       │   └── coordinator.py  # マルチエージェント協調制御
│       │
│       ├── mcp/
│       │   ├── __init__.py
│       │   ├── manager.py      # MCPサーバー管理・接続
│       │   └── types.py        # ToolResult型定義
│       │
│       └── db/
│           ├── __init__.py
│           ├── client.py       # SurrealDB接続 (認証対応)
│           └── schema.py       # テーブル定義・マイグレーション・REBUILD INDEX
│
├── rust/
│   ├── Cargo.toml
│   └── sbfa-core/
│       ├── Cargo.toml
│       ├── src/
│       │   ├── lib.rs          # PyO3バインディング
│       │   ├── chunker.rs      # 高速テキストチャンキング
│       │   ├── similarity.rs   # ベクトル類似度計算 (SIMD)
│       │   └── task_router.rs  # タスク分類・ルーティングエンジン
│       └── benches/
│           └── benchmarks.rs
│
└── tests/
    ├── __init__.py
    ├── test_agents.py
    ├── test_rag.py
    ├── test_a2a.py
    └── test_rust_core.py
```

---

## Phase 1: 基盤構築（DB + プロジェクト骨格）

### 1-1. プロジェクト初期化
- `pyproject.toml`（依存: `surrealdb`, `fastapi`, `uvicorn`, `anthropic`, `google-genai`, `openai`, `python-dotenv`, `pyjwt[crypto]`）
- `docker-compose.yml`: SurrealDB コンテナ（**認証・永続化設定込み**）
- `.env.example`: 環境変数テンプレート（モデルID含む）
- `src/sbfa/config.py`: 環境変数読み込み
- `rust/sbfa-core/Cargo.toml`: PyO3依存を最初から定義（CI/CD統合のため）

### 1-2. docker-compose.yml (永続化・認証)

```yaml
services:
  surrealdb:
    image: surrealdb/surrealdb:v2
    command: start --user root --pass ${SURREAL_PASS:-root} rocksdb:/data/srdb
    ports:
      - "8000:8000"
    volumes:
      - surreal_data:/data/srdb
    environment:
      - SURREAL_CAPS_ALLOW_ALL=true
volumes:
  surreal_data:
```

### 1-3. SurrealDB スキーマ定義

**ファイル:** `src/sbfa/db/schema.py`

```sql
-- ===========================================
-- エージェントカード (A2A v0.3 camelCase準拠)
-- ===========================================
-- 注: SurrealDB内部はsnake_caseで保存し、
--     Python側のPydanticモデルでcamelCase変換する (model_config alias_generator)
-- ===========================================
DEFINE TABLE agent_card SCHEMAFULL;
DEFINE FIELD name ON agent_card TYPE string;
DEFINE FIELD description ON agent_card TYPE string;
DEFINE FIELD url ON agent_card TYPE string;
DEFINE FIELD provider ON agent_card TYPE string;
DEFINE FIELD model ON agent_card TYPE string;
DEFINE FIELD version ON agent_card TYPE string DEFAULT '0.3.0';
DEFINE FIELD skills ON agent_card TYPE array<object>;
DEFINE FIELD capabilities ON agent_card TYPE object;
DEFINE FIELD default_input_modes ON agent_card TYPE array<string> DEFAULT ['text'];
DEFINE FIELD default_output_modes ON agent_card TYPE array<string> DEFAULT ['text'];
DEFINE FIELD auth ON agent_card TYPE option<object>;
DEFINE FIELD signature ON agent_card TYPE option<object>;  -- JWS署名 (RFC 7515)
DEFINE FIELD status ON agent_card TYPE string DEFAULT 'active';
DEFINE FIELD created_at ON agent_card TYPE datetime DEFAULT time::now();
DEFINE FIELD updated_at ON agent_card TYPE datetime DEFAULT time::now();

-- ===========================================
-- RAGドキュメント
-- ===========================================
DEFINE TABLE rag_document SCHEMAFULL;
DEFINE FIELD title ON rag_document TYPE string;
DEFINE FIELD source ON rag_document TYPE string;
DEFINE FIELD content_hash ON rag_document TYPE string;
DEFINE FIELD metadata ON rag_document TYPE option<object>;
DEFINE FIELD created_at ON rag_document TYPE datetime DEFAULT time::now();

-- ===========================================
-- RAGチャンク (ベクトル埋め込み付き)
-- TYPE F32明示、EFC/Mパラメータ指定、DEFER
-- ===========================================
DEFINE TABLE rag_chunk SCHEMAFULL;
DEFINE FIELD document ON rag_chunk TYPE record<rag_document>;
DEFINE FIELD content ON rag_chunk TYPE string;
DEFINE FIELD embedding ON rag_chunk TYPE array<float>;
DEFINE FIELD chunk_index ON rag_chunk TYPE int;
DEFINE FIELD metadata ON rag_chunk TYPE option<object>;
DEFINE INDEX idx_chunk_embedding ON rag_chunk FIELDS embedding
  HNSW DIMENSION 1536 TYPE F32 DIST COSINE EFC 500 M 16 DEFER;

-- ===========================================
-- タスク履歴 (要約のみ保存、フルテキスト非保存)
-- ===========================================
DEFINE TABLE task_history SCHEMAFULL;
DEFINE FIELD task ON task_history TYPE string;
DEFINE FIELD assigned_agent ON task_history TYPE record<agent_card>;
DEFINE FIELD input_summary ON task_history TYPE string;
DEFINE FIELD input_token_count ON task_history TYPE int;
DEFINE FIELD output_summary ON task_history TYPE option<string>;
DEFINE FIELD output_token_count ON task_history TYPE option<int>;
DEFINE FIELD status ON task_history TYPE string;
DEFINE FIELD cost ON task_history TYPE option<float>;
DEFINE FIELD latency_ms ON task_history TYPE option<int>;
DEFINE FIELD created_at ON task_history TYPE datetime DEFAULT time::now();
```

**起動時REBUILD処理:**
```python
async def rebuild_indexes_if_needed(db):
    """サーバー再起動後にHNSWインデックスを再構築"""
    await db.query("REBUILD INDEX idx_chunk_embedding ON rag_chunk")
```

---

## Phase 2: エージェント基盤 + A2A

### 2-1. BaseAgent抽象クラス (MCP ToolResult対応)
**ファイル:** `src/sbfa/agents/base.py`

```python
@dataclass
class ToolResult:
    """MCPツール実行結果の構造化型"""
    tool_name: str
    result: Any
    is_error: bool = False

class BaseAgent(ABC):
    name: str
    provider: str
    model: str
    skills: list[Skill]

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        context: list[str] | None = None,
        tool_results: list[ToolResult] | None = None,
    ) -> str: ...

    @abstractmethod
    async def stream(
        self,
        prompt: str,
        context: list[str] | None = None,
        tool_results: list[ToolResult] | None = None,
    ) -> AsyncIterator[str]: ...

    def to_agent_card(self) -> AgentCard:
        """A2A AgentCard (camelCase) へ変換"""
        ...
```

### 2-2. 各AIエージェント実装
- **`claude_agent.py`**: Anthropic SDK。モデルID: `config.CLAUDE_MODEL` (デフォルト `claude-sonnet-4-6`)
- **`gemini_agent.py`**: Google GenAI SDK。モデルID: `config.GEMINI_MODEL` (デフォルト `gemini-2.5-flash`)
- **`openai_agent.py`**: OpenAI SDK。モデルID: `config.OPENAI_MODEL` (デフォルト `gpt-5.4`)
- **`local_agent.py`**: OpenAI互換API (Ollama等)。`config.LOCAL_MODEL_ENDPOINT` + `config.LOCAL_MODEL_NAME`

### 2-3. A2Aエージェントカード管理 (camelCase + 署名)
**ファイル:** `src/sbfa/a2a/agent_card.py`

```python
class AgentCard(BaseModel):
    model_config = ConfigDict(
        alias_generator=to_camel,
        populate_by_name=True,
    )

    name: str
    description: str
    url: str
    version: str = "0.3.0"
    protocol_version: str = "0.3.0"
    provider: str
    model: str
    skills: list[Skill]
    capabilities: Capabilities
    default_input_modes: list[str] = ["text"]
    default_output_modes: list[str] = ["text"]
    auth: AuthConfig | None = None
    signature: AgentCardSignature | None = None
```

**エンドポイント**: `GET /.well-known/agent-card.json` (v0.3準拠パス)

### 2-4. AgentCardセキュリティ
**ファイル:** `src/sbfa/a2a/security.py`

- JWS署名生成 (RFC 7515) + JCS正規化 (RFC 8785)
- 署名検証: レジストリ登録時に必須検証
- 内部ネットワークでもなりすまし防止のため署名を要求

---

## Phase 3: RAGシステム

### 3-1. ドキュメント取り込み (DEFER対応)
**ファイル:** `src/sbfa/rag/ingestion.py`
- テキスト分割: 固定サイズ + オーバーラップ
- 対応形式: テキスト、Markdown、PDF（拡張可能）
- **バッチインジェスト**: DEFERインデックスにより高ボリューム並行書き込みに対応
- 元文書は`content_hash`で参照

### 3-2. Embedding生成
**ファイル:** `src/sbfa/rag/embeddings.py`
- デフォルト: OpenAI `text-embedding-3-small` (1536次元)
- プロバイダ切替可能 (Ollama, Gemini Embedding等)

### 3-3. SurrealDB RAGストア
**ファイル:** `src/sbfa/rag/store.py`
- チャンク + F32ベクトルの保存
- HNSW DEFER インデックスによるKNN検索
- `REBUILD INDEX` のスケジュール実行サポート

### 3-4. 検索・取得
**ファイル:** `src/sbfa/rag/retrieval.py`
- コサイン類似度ベースの検索
- Top-K取得 → コンテキストとして各エージェントに注入

---

## Phase 4: オーケストレーター (ルーティングロジック具体化)

### 4-1. タスクルーター
**ファイル:** `src/sbfa/orchestrator/router.py`

**ルーティングアルゴリズム**:
1. タスクをキーワード + 意図分類で分類 (code/multimodal/general/reasoning)
2. 各エージェントのスキルタグとマッチングスコアを計算
3. 同一スコアの場合の優先順位: **レイテンシ > コスト > 最終使用時刻**
4. 加重スコア: `score = skill_match * 0.5 + (1/latency_avg) * 0.3 + (1/cost_per_token) * 0.2`

**フォールバック条件**:
- APIエラー (4xx/5xx) → 次スコアのエージェントへ
- タイムアウト (設定可能、デフォルト30s) → 次スコアのエージェントへ
- レート制限 (429) → 別プロバイダへ即時切替
- 全エージェント失敗 → エラー返却 + タスク履歴に記録

### 4-2. 協調コーディネーター
**ファイル:** `src/sbfa/orchestrator/coordinator.py`
- シーケンシャル協調: Agent A の出力 → Agent B のコンテキスト
- パラレル協調: 複数エージェント同時実行 → 結果統合
- RAGコンテキスト自動注入

---

## Phase 5: MCP統合 (インターフェース設計)

**ファイル:** `src/sbfa/mcp/manager.py`, `src/sbfa/mcp/types.py`

MCPはエージェントの「ツールアクセス」の主要経路。BaseAgentの`tool_results`パラメータと連携:

```python
class MCPManager:
    async def connect(self, server_config: MCPServerConfig) -> None: ...
    async def call_tool(self, tool_name: str, args: dict) -> ToolResult: ...
    async def list_tools(self) -> list[ToolInfo]: ...
    async def disconnect(self) -> None: ...

class AgentMCPBinding:
    agent_name: str
    mcp_servers: list[MCPServerConfig]
    allowed_tools: list[str]
```

---

## Phase 6: FastAPI エンドポイント

**ファイル:** `src/sbfa/main.py`
- `POST /task` - タスク投入（ルーティング → エージェント実行）
- `GET /agents` - 登録済みエージェント一覧
- `POST /rag/ingest` - ドキュメント取り込み
- `POST /rag/query` - RAG検索
- `GET /.well-known/agent-card.json` - A2Aエージェントカード発見エンドポイント (v0.3準拠)
- `GET /agents/{id}/card` - 個別エージェントカード取得

---

## Rust高速化コア (`rust/sbfa-core/`)

| モジュール | 役割 | 高速化対象 |
|---|---|---|
| `chunker.rs` | テキストチャンキング | RAG取り込み時の大量テキスト分割 |
| `similarity.rs` | ベクトル類似度計算 | Embedding比較・リランキング (SIMD活用) |
| `task_router.rs` | タスク分類エンジン | スキルマッチングの高速スコアリング |
| `lib.rs` | PyO3バインディング | Python `import sbfa_core` で利用可能 |

**CI/CD統合**:
- `Cargo.toml` は Phase 1 で作成し、PyO3依存を最初から定義
- GitHub Actions で `maturin build --release` を Linux/macOS 向けに実行
- Rustビルド失敗時は Python フォールバックで自動切替 + テストで両パスを検証

---

## 実装順序

| 順序 | 内容 | 依存 |
|---|---|---|
| 1 | プロジェクト初期化 (`pyproject.toml`, `docker-compose.yml`, `config.py`, `Cargo.toml`) | なし |
| 2 | SurrealDB接続 + スキーマ + REBUILD INDEX (`db/`) | 1 |
| 3 | MCP型定義 (`mcp/types.py`) + BaseAgent (tool_results対応) | 1 |
| 4 | 3つのAIエージェント実装 | 3 |
| 5 | A2Aエージェントカード (camelCase + JWS署名) + レジストリ | 2, 4 |
| 6 | RAGシステム (ingestion → embedding → store → retrieval) | 2 |
| 7 | オーケストレーター (スコアリングルーター + coordinator) | 4, 5, 6 |
| 8 | FastAPIエンドポイント (/.well-known/agent-card.json含む) | 2-7 |
| 9 | MCPマネージャー実装 | 3 |
| 10 | Rustコア (`sbfa-core`) + CI/CD統合 | 1 |
| 11 | テスト + ベンチマーク | 全体 |

**戦略**: MCP型定義を Phase 3 に前倒しし、エージェントIF安定化を優先。Rustは最後だがCargo.tomlは最初から。

---

## v1からの修正対応表

| # | 欠陥 | 修正内容 |
|---|---|---|
| 1 | A2Aトランスポート | JSON-RPC / gRPC / HTTP/REST から選択と明記 |
| 2 | AgentCard URLパス | `/.well-known/agent-card.json` に修正 |
| 3 | snake_case非準拠 | Pydantic `alias_generator=to_camel` でcamelCase変換 |
| 4 | HNSW TYPE未指定 | `TYPE F32 EFC 500 M 16 DEFER` 追加 |
| 5 | モデル名架空 | `gemini-2.5-flash`, `gpt-5.4` に修正。環境変数化 |
| 6 | AgentCardセキュリティ | `security.py` 新設、JWS署名 (RFC 7515) |
| 7 | 並行インジェスト | HNSW `DEFER` で最終一貫性対応 |
| 8 | Rustビルド未計画 | Phase 1でCargo.toml作成、CI/CD統合手順追記 |
| 9 | ルーティング未定義 | 加重スコアリング + フォールバック条件を具体化 |
| 10 | MCPスタブ問題 | MCP型定義をPhase 3に前倒し、BaseAgentのtool_results対応 |
| 11 | タスク履歴肥大化 | `input_summary` / `output_summary` に変更、トークン数記録 |
| 12 | docker永続化なし | ボリュームマウント + 認証 + rocksdb 指定 |
| 13 | REBUILD INDEX | 起動時自動REBUILD処理を追加 |

---

## 検証方法

1. `docker compose up -d` → SurrealDB起動 → スキーマ適用確認
2. 各エージェント単体テスト: 実在モデルIDでAPI呼び出し → 応答確認
3. A2Aカード: `GET /.well-known/agent-card.json` → camelCaseレスポンス検証
4. AgentCard署名: 署名生成 → 検証 → 改ざん検出テスト
5. RAG: テキスト取り込み → F32ベクトル保存 → `REBUILD INDEX` → 類似検索
6. ルーター: 複数タスク種別 → 期待エージェントへのルーティング確認
7. `pytest tests/` で全テスト実行
