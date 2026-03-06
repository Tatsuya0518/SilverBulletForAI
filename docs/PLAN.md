# SBFA - マルチAIチーム + RAG + MCP 実装プラン

## Context

SilverBulletForAI (SBFA) は「AIを万能薬として使う」ためのシステム。異種AI（Claude Sonnet 4.6, Gemini 3 Flash, GPT-5.2）がチームを組み、互いの弱点を補完しながら協調動作する。全エージェントが共有RAG（SurrealDB）を読み書きし、A2Aプロトコルでエージェント間通信を行う。ローカルLLM対応の拡張性も確保する。

---

## 技術選定

| 領域 | 技術 |
|---|---|
| 言語 | Python 3.12+ (アプリケーション層) + Rust (高速化コア) |
| DB | SurrealDB v2.x (KVストア + ベクトル検索) |
| エージェント間通信 | A2A Protocol v0.3 (JSON-RPC 2.0 over HTTP) |
| RAGベクトル検索 | SurrealDB HNSW インデックス |
| Embedding | OpenAI `text-embedding-3-small` (デフォルト、切替可能) |
| AI SDK | `anthropic`, `google-genai`, `openai` |
| API層 | FastAPI (Python) + Axum (Rust、パフォーマンスクリティカルなエンドポイント) |
| Python-Rust連携 | PyO3 + maturin (PythonからRust呼び出し) |
| MCP | `mcp` Python SDK (後日選定) |
| パッケージ管理 | `uv` + `pyproject.toml` (Python) / `cargo` (Rust) |

---

## プロジェクト構造

```
SilverBulletForAI/
├── pyproject.toml
├── README.md
├── .env.example
├── docker-compose.yml          # SurrealDB起動用
│
├── src/
│   └── sbfa/                   # Python アプリケーション層
│       ├── __init__.py
│       ├── main.py             # FastAPIエントリポイント
│       ├── config.py           # 設定管理 (env読み込み)
│       │
│       ├── agents/             # エージェント定義
│       │   ├── __init__.py
│       │   ├── base.py         # BaseAgent 抽象クラス
│       │   ├── claude_agent.py # Claude Sonnet 4.6
│       │   ├── gemini_agent.py # Gemini 3 Flash
│       │   ├── openai_agent.py # GPT-5.2
│       │   └── local_agent.py  # ローカルLLM (Ollama等) 拡張用
│       │
│       ├── a2a/                # A2Aプロトコル層
│       │   ├── __init__.py
│       │   ├── agent_card.py   # AgentCard スキーマ & CRUD
│       │   ├── registry.py     # エージェント登録・発見 (SurrealDB)
│       │   └── protocol.py     # JSON-RPC メッセージング
│       │
│       ├── rag/                # RAGシステム
│       │   ├── __init__.py
│       │   ├── ingestion.py    # ドキュメント取り込みパイプライン
│       │   ├── embeddings.py   # Embedding生成 (プロバイダ切替可能)
│       │   ├── retrieval.py    # 類似検索・取得
│       │   └── store.py        # SurrealDB RAGストア操作
│       │
│       ├── orchestrator/       # タスク振り分け・協調
│       │   ├── __init__.py
│       │   ├── router.py       # タスクルーティング (強み判定)
│       │   └── coordinator.py  # マルチエージェント協調制御
│       │
│       ├── mcp/                # MCP統合 (後日拡充)
│       │   ├── __init__.py
│       │   └── manager.py      # MCPサーバー管理・接続
│       │
│       └── db/                 # SurrealDB接続
│           ├── __init__.py
│           ├── client.py       # SurrealDB接続クライアント
│           └── schema.py       # テーブル定義・マイグレーション
│
├── rust/                       # Rust 高速化コア
│   ├── Cargo.toml
│   └── sbfa-core/
│       ├── Cargo.toml
│       ├── src/
│       │   ├── lib.rs          # PyO3エントリポイント (Pythonバインディング)
│       │   ├── chunker.rs      # 高速テキストチャンキング
│       │   ├── similarity.rs   # ベクトル類似度計算 (SIMD最適化)
│       │   └── task_router.rs  # タスク分類・ルーティングエンジン
│       └── benches/
│           └── benchmarks.rs   # パフォーマンスベンチマーク
│
└── tests/
    ├── __init__.py
    ├── test_agents.py
    ├── test_rag.py
    ├── test_a2a.py
    └── test_rust_core.py       # Rust コア統合テスト
```

### Rust高速化コア (`rust/sbfa-core/`)

PythonからPyO3経由で呼び出すRustネイティブモジュール。以下の処理を高速化:

| モジュール | 役割 | 高速化対象 |
|---|---|---|
| `chunker.rs` | テキストチャンキング | RAG取り込み時の大量テキスト分割 |
| `similarity.rs` | ベクトル類似度計算 | Embedding比較・リランキング (SIMD活用) |
| `task_router.rs` | タスク分類エンジン | スキルマッチングの高速スコアリング |
| `lib.rs` | PyO3バインディング | Python `import sbfa_core` で利用可能 |

**Python側からの利用例:**
```python
# Rustで高速チャンキング (フォールバック: Python実装)
try:
    from sbfa_core import fast_chunk_text
except ImportError:
    from sbfa.rag.ingestion import chunk_text as fast_chunk_text
```

**ビルド:** `maturin develop` でPythonパッケージとしてインストール

---

## Phase 1: 基盤構築（DB + プロジェクト骨格）

### 1-1. プロジェクト初期化
- `pyproject.toml` 作成（依存: `surrealdb`, `fastapi`, `uvicorn`, `anthropic`, `google-genai`, `openai`, `python-dotenv`）
- `docker-compose.yml` で SurrealDB コンテナ定義
- `.env.example` に必要な環境変数テンプレート
- `src/sbfa/config.py` で環境変数読み込み

### 1-2. SurrealDB接続・スキーマ定義
**ファイル:** `src/sbfa/db/client.py`, `src/sbfa/db/schema.py`

```sql
-- エージェントカード (A2A準拠)
DEFINE TABLE agent_card SCHEMAFULL;
DEFINE FIELD name ON agent_card TYPE string;
DEFINE FIELD description ON agent_card TYPE string;
DEFINE FIELD url ON agent_card TYPE string;
DEFINE FIELD provider ON agent_card TYPE string;        -- "anthropic" | "google" | "openai" | "local"
DEFINE FIELD model ON agent_card TYPE string;            -- "claude-sonnet-4-6" etc.
DEFINE FIELD skills ON agent_card TYPE array<object>;    -- [{name, description, tags}]
DEFINE FIELD capabilities ON agent_card TYPE object;     -- {streaming, multimodal, ...}
DEFINE FIELD auth ON agent_card TYPE option<object>;
DEFINE FIELD status ON agent_card TYPE string DEFAULT 'active';
DEFINE FIELD created_at ON agent_card TYPE datetime DEFAULT time::now();
DEFINE FIELD updated_at ON agent_card TYPE datetime DEFAULT time::now();

-- RAGドキュメント
DEFINE TABLE rag_document SCHEMAFULL;
DEFINE FIELD title ON rag_document TYPE string;
DEFINE FIELD source ON rag_document TYPE string;
DEFINE FIELD content ON rag_document TYPE string;
DEFINE FIELD metadata ON rag_document TYPE option<object>;
DEFINE FIELD created_at ON rag_document TYPE datetime DEFAULT time::now();

-- RAGチャンク (ベクトル埋め込み付き)
DEFINE TABLE rag_chunk SCHEMAFULL;
DEFINE FIELD document ON rag_chunk TYPE record<rag_document>;
DEFINE FIELD content ON rag_chunk TYPE string;
DEFINE FIELD embedding ON rag_chunk TYPE array<float>;
DEFINE FIELD chunk_index ON rag_chunk TYPE int;
DEFINE FIELD metadata ON rag_chunk TYPE option<object>;
DEFINE INDEX idx_chunk_embedding ON rag_chunk FIELDS embedding
  HNSW DIMENSION 1536 DIST COSINE;

-- タスク履歴
DEFINE TABLE task_history SCHEMAFULL;
DEFINE FIELD task ON task_history TYPE string;
DEFINE FIELD assigned_agent ON task_history TYPE record<agent_card>;
DEFINE FIELD input ON task_history TYPE string;
DEFINE FIELD output ON task_history TYPE option<string>;
DEFINE FIELD status ON task_history TYPE string;
DEFINE FIELD created_at ON task_history TYPE datetime DEFAULT time::now();
```

---

## Phase 2: エージェント基盤 + A2A

### 2-1. BaseAgent抽象クラス
**ファイル:** `src/sbfa/agents/base.py`

```python
class BaseAgent(ABC):
    """全エージェント共通インターフェース"""
    name: str
    provider: str
    model: str
    skills: list[Skill]

    @abstractmethod
    async def generate(self, prompt: str, context: list[str] = None) -> str: ...

    @abstractmethod
    async def stream(self, prompt: str, context: list[str] = None) -> AsyncIterator[str]: ...

    def to_agent_card(self) -> dict:
        """A2A AgentCardへ変換"""
        ...
```

### 2-2. 各AIエージェント実装
- **`claude_agent.py`**: Anthropic SDK使用。得意分野: コーディング、論理推論、長文分析
- **`gemini_agent.py`**: Google GenAI SDK使用。得意分野: マルチモーダル、高速応答、大コンテキスト
- **`openai_agent.py`**: OpenAI SDK使用。得意分野: 汎用タスク、関数呼び出し、構造化出力
- **`local_agent.py`**: OpenAI互換API (Ollama等) 使用。ローカルLLM拡張ポイント

### 2-3. A2Aエージェントカード管理
**ファイル:** `src/sbfa/a2a/agent_card.py`, `src/sbfa/a2a/registry.py`

- AgentCard Pydanticモデル（A2A v0.3準拠フィールド）
- SurrealDBへのCRUD操作
- `registry.py`: エージェント登録・発見・スキルベース検索

---

## Phase 3: RAGシステム

### 3-1. ドキュメント取り込み
**ファイル:** `src/sbfa/rag/ingestion.py`
- テキスト分割（チャンキング）: 固定サイズ + オーバーラップ
- 対応形式: テキスト、Markdown、PDF（拡張可能）

### 3-2. Embedding生成
**ファイル:** `src/sbfa/rag/embeddings.py`
- デフォルト: OpenAI `text-embedding-3-small` (1536次元)
- プロバイダ切替可能な設計（Ollama, Gemini Embedding等）

### 3-3. SurrealDB RAGストア
**ファイル:** `src/sbfa/rag/store.py`
- チャンク + ベクトルの保存
- HNSW インデックスによるKNN検索

### 3-4. 検索・取得
**ファイル:** `src/sbfa/rag/retrieval.py`
- コサイン類似度ベースの検索
- Top-K取得 → コンテキストとして各エージェントに注入

---

## Phase 4: オーケストレーター

### 4-1. タスクルーター
**ファイル:** `src/sbfa/orchestrator/router.py`
- タスク分類 → 最適エージェント選定
- スキルタグマッチング + エージェントカードの `skills` フィールド参照
- フォールバック戦略: プライマリ失敗時に別エージェントへ

### 4-2. 協調コーディネーター
**ファイル:** `src/sbfa/orchestrator/coordinator.py`
- シーケンシャル協調: Agent A の出力 → Agent B のコンテキスト
- パラレル協調: 複数エージェント同時実行 → 結果統合
- RAGコンテキスト自動注入

---

## Phase 5: MCP統合（スタブ）

**ファイル:** `src/sbfa/mcp/manager.py`
- MCPサーバー接続管理のインターフェース定義
- 各エージェントにMCPツールを紐付ける拡張ポイント
- 具体的なMCPサーバー選定は後日

---

## Phase 6: FastAPI エンドポイント

**ファイル:** `src/sbfa/main.py`
- `POST /task` - タスク投入（ルーティング → エージェント実行）
- `GET /agents` - 登録済みエージェント一覧
- `POST /rag/ingest` - ドキュメント取り込み
- `POST /rag/query` - RAG検索
- `GET /agents/{id}/card` - A2Aエージェントカード取得 (`.well-known/agent.json`互換)

---

## 実装順序

| 順序 | 内容 | 依存 |
|---|---|---|
| 1 | プロジェクト初期化 (`pyproject.toml`, `docker-compose.yml`, `config.py`, `Cargo.toml`) | なし |
| 2 | SurrealDB接続 + スキーマ (`db/client.py`, `db/schema.py`) | 1 |
| 3 | BaseAgent + 3つのAIエージェント実装 | 1 |
| 4 | A2Aエージェントカード + レジストリ | 2, 3 |
| 5 | RAGシステム (ingestion → embedding → store → retrieval) - Python版 | 2 |
| 6 | Rustコア (`sbfa-core`: chunker, similarity, task_router) + PyO3バインディング | 1 |
| 7 | RAG + オーケストレーターにRust高速化を統合 | 5, 6 |
| 8 | オーケストレーター (router + coordinator) | 3, 4, 5 |
| 9 | FastAPIエンドポイント | 2-8 |
| 10 | MCPスタブ | 3 |
| 11 | テスト + ベンチマーク | 全体 |

**戦略**: まずPython版で全機能を動かし、次にRustで性能ボトルネックを置き換える。Rustが無くてもPythonフォールバックで動作する設計。

---

## 検証方法

1. **SurrealDB接続テスト**: `docker compose up -d` → Pythonからテーブル作成・CRUD確認
2. **エージェント単体テスト**: 各エージェントに簡単なプロンプト送信 → 応答確認
3. **A2Aカード登録テスト**: エージェント起動 → SurrealDBにカード保存 → 取得で検証
4. **RAGテスト**: テキスト取り込み → チャンク化 → Embedding保存 → 類似検索で関連チャンク取得
5. **統合テスト**: タスク投入 → ルーティング → RAGコンテキスト付きでエージェント実行 → 結果返却
6. **`pytest`**: `tests/` ディレクトリのテスト実行
