# SilverBulletForAI

通称SBFAとする。

## Overview

SilverBulletForAI (SBFA) は「AIを万能薬として使う」ためのシステム。
異種AI（Claude Sonnet 4.6, Gemini 3 Flash, GPT-5.2）がチームを組み、互いの弱点を補完しながら協調動作する。

- 全エージェントが共有RAG（SurrealDB）を読み書き
- A2Aプロトコルでエージェント間通信
- ローカルLLM対応の拡張性を確保
- Rust高速化コアによるパフォーマンス最適化

## 技術選定

| 領域 | 技術 |
|---|---|
| 言語 | Python 3.12+ (アプリケーション層) + Rust (高速化コア) |
| DB | SurrealDB v2.x (KVストア + ベクトル検索) |
| エージェント間通信 | A2A Protocol v0.3 (JSON-RPC 2.0 over HTTP) |
| RAGベクトル検索 | SurrealDB HNSW インデックス |
| Embedding | OpenAI `text-embedding-3-small` (デフォルト、切替可能) |
| AI SDK | `anthropic`, `google-genai`, `openai` |
| API層 | FastAPI (Python) + Axum (Rust) |
| Python-Rust連携 | PyO3 + maturin |
| MCP | `mcp` Python SDK (後日選定) |
| パッケージ管理 | `uv` + `pyproject.toml` (Python) / `cargo` (Rust) |

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
│       ├── main.py             # FastAPIエントリポイント
│       ├── config.py           # 設定管理
│       ├── agents/             # エージェント定義 (Claude, Gemini, GPT, Local)
│       ├── a2a/                # A2Aプロトコル層 (AgentCard, Registry)
│       ├── rag/                # RAGシステム (ingestion, embedding, retrieval)
│       ├── orchestrator/       # タスク振り分け・協調
│       ├── mcp/                # MCP統合
│       └── db/                 # SurrealDB接続
│
├── rust/                       # Rust 高速化コア
│   └── sbfa-core/              # PyO3バインディング (chunker, similarity, task_router)
│
└── tests/
```

## エージェント役割分担

| エージェント | 得意分野 |
|---|---|
| Claude Sonnet 4.6 | コーディング、論理推論、長文分析 |
| Gemini 3 Flash | マルチモーダル、高速応答、大コンテキスト |
| GPT-5.2 | 汎用タスク、関数呼び出し、構造化出力 |
| ローカルLLM (拡張) | プライバシー重視、オフライン処理 |

## 実装プラン

詳細な実装プランは [`docs/PLAN.md`](docs/PLAN.md) を参照。
