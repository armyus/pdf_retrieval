"""
PDF Semantic Retrieval & RAG System — Web Dashboard

Provides a fast, local Web Dashboard & REST API built with Flask.
Allows users to visually explore documents, run multi-strategy search,
ask grounded RAG questions with interactive citation cards, and run benchmarks.

Usage:
    python -m src.dashboard --folder ./data --port 8000
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Any, List

from flask import Flask, request, jsonify, render_template_string

from src.pdf_scanner import PDFScanner
from src.pdf_parser import PDFParser, ParsedPDF
from src.config import RetrievalConfig
from src.retrievers import get_retriever, BaseRetriever
from src.rag import RAGPipeline, RAGResult
from src.evaluation import Evaluator

logger = logging.getLogger(__name__)

# Global state for the server
STATE: Dict[str, Any] = {
    "folder_path": "./data",
    "pdf_files": [],
    "documents": [],
    "retrievers": {},
    "rag_pipeline": None,
    "total_chunks": 0,
}


def initialize_system(folder_path: str):
    """Scans and indexes the target PDF folder into all retrieval engines."""
    resolved_path = str(Path(folder_path).resolve())
    STATE["folder_path"] = resolved_path

    scanner = PDFScanner(resolved_path)
    pdf_files = scanner.scan()
    STATE["pdf_files"] = pdf_files

    if not pdf_files:
        logger.warning("No PDF files found in %s", resolved_path)
        return

    parser = PDFParser()
    documents = parser.parse_many([pdf.full_path for pdf in pdf_files])
    STATE["documents"] = documents

    config = RetrievalConfig.default()

    # Pre-index retrievers
    for strat in ["hybrid", "dense", "bm25", "keyword"]:
        cfg = RetrievalConfig.default()
        cfg.strategy = strat
        retriever = get_retriever(strat, cfg)
        total = retriever.index_documents(documents)
        STATE["retrievers"][strat] = retriever
        if strat == "hybrid":
            STATE["total_chunks"] = total

    # Setup RAG pipeline with Hybrid retriever
    hybrid_retriever = STATE["retrievers"]["hybrid"]
    STATE["rag_pipeline"] = RAGPipeline(retriever=hybrid_retriever, llm_backend="auto")
    logger.info("Initialized system with %d documents and %d chunks.", len(documents), STATE["total_chunks"])


HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>PDF Knowledge Retrieval & RAG System</title>
  <style>
    :root {
      --bg-primary: #0f172a;
      --bg-secondary: #1e293b;
      --bg-card: #24344d;
      --accent: #38bdf8;
      --accent-glow: rgba(56, 189, 248, 0.25);
      --text-main: #f8fafc;
      --text-muted: #94a3b8;
      --border: #334155;
      --success: #34d399;
      --warning: #fbbf24;
      --font-sans: system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
    }

    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      background-color: var(--bg-primary);
      color: var(--text-main);
      font-family: var(--font-sans);
      line-height: 1.6;
      min-height: 100vh;
      display: flex;
      flex-direction: column;
    }

    header {
      background-color: var(--bg-secondary);
      border-bottom: 1px solid var(--border);
      padding: 1rem 2rem;
      display: flex;
      justify-content: space-between;
      align-items: center;
    }

    .brand {
      display: flex;
      align-items: center;
      gap: 0.75rem;
    }

    .brand h1 {
      font-size: 1.25rem;
      font-weight: 700;
      color: var(--text-main);
    }

    .badge {
      background: var(--accent-glow);
      color: var(--accent);
      border: 1px solid var(--accent);
      padding: 0.2rem 0.6rem;
      border-radius: 9999px;
      font-size: 0.75rem;
      font-weight: 600;
    }

    .stats-bar {
      display: flex;
      gap: 1.5rem;
      font-size: 0.85rem;
      color: var(--text-muted);
    }

    .stats-bar span strong {
      color: var(--text-main);
    }

    .nav-tabs {
      background-color: var(--bg-secondary);
      display: flex;
      gap: 0.5rem;
      padding: 0 2rem;
      border-bottom: 1px solid var(--border);
    }

    .tab-btn {
      background: transparent;
      border: none;
      color: var(--text-muted);
      padding: 0.85rem 1.25rem;
      font-size: 0.9rem;
      font-weight: 600;
      cursor: pointer;
      border-bottom: 2px solid transparent;
      transition: all 0.2s ease;
    }

    .tab-btn:hover {
      color: var(--text-main);
    }

    .tab-btn.active {
      color: var(--accent);
      border-bottom-color: var(--accent);
    }

    main {
      flex: 1;
      padding: 2rem;
      max-width: 1200px;
      margin: 0 auto;
      width: 100%;
    }

    .tab-content { display: none; }
    .tab-content.active { display: block; }

    .card {
      background-color: var(--bg-secondary);
      border: 1px solid var(--border);
      border-radius: 0.75rem;
      padding: 1.5rem;
      margin-bottom: 1.5rem;
    }

    .input-group {
      display: flex;
      gap: 0.75rem;
      margin-bottom: 1rem;
    }

    input[type="text"], select {
      flex: 1;
      background-color: var(--bg-primary);
      border: 1px solid var(--border);
      color: var(--text-main);
      padding: 0.75rem 1rem;
      border-radius: 0.5rem;
      font-size: 0.95rem;
      outline: none;
    }

    input[type="text"]:focus, select:focus {
      border-color: var(--accent);
      box-shadow: 0 0 0 2px var(--accent-glow);
    }

    button.btn {
      background-color: var(--accent);
      color: #0f172a;
      border: none;
      padding: 0.75rem 1.5rem;
      border-radius: 0.5rem;
      font-size: 0.95rem;
      font-weight: 600;
      cursor: pointer;
      transition: opacity 0.2s ease;
      display: flex;
      align-items: center;
      gap: 0.5rem;
    }

    button.btn:hover { opacity: 0.9; }

    .controls-row {
      display: flex;
      gap: 1rem;
      align-items: center;
      flex-wrap: wrap;
      margin-top: 0.5rem;
      font-size: 0.85rem;
      color: var(--text-muted);
    }

    .answer-box {
      background-color: var(--bg-card);
      border-left: 4px solid var(--accent);
      padding: 1.25rem;
      border-radius: 0.5rem;
      margin-top: 1rem;
      white-space: pre-wrap;
    }

    .sources-list {
      margin-top: 1.25rem;
      display: flex;
      flex-direction: column;
      gap: 0.75rem;
    }

    .source-card {
      background-color: var(--bg-primary);
      border: 1px solid var(--border);
      border-radius: 0.5rem;
      padding: 1rem;
    }

    .source-header {
      display: flex;
      justify-content: space-between;
      font-size: 0.85rem;
      color: var(--accent);
      margin-bottom: 0.4rem;
      font-weight: 600;
    }

    .source-snippet {
      font-size: 0.9rem;
      color: var(--text-muted);
    }

    table {
      width: 100%;
      border-collapse: collapse;
      margin-top: 1rem;
    }

    th, td {
      padding: 0.75rem 1rem;
      text-align: left;
      border-bottom: 1px solid var(--border);
    }

    th {
      background-color: var(--bg-primary);
      color: var(--accent);
      font-weight: 600;
      font-size: 0.85rem;
    }

    td { font-size: 0.9rem; }

    .loading {
      display: none;
      color: var(--accent);
      font-size: 0.9rem;
      margin-top: 0.5rem;
    }

    .grid-2 {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 1.5rem;
    }

    @media (max-width: 768px) {
      .grid-2 { grid-template-columns: 1fr; }
      .stats-bar { display: none; }
    }
  </style>
</head>
<body>

  <header>
    <div class="brand">
      <h1>Personal PDF Knowledge Retrieval</h1>
      <span class="badge">V6 Hybrid + RAG</span>
    </div>
    <div class="stats-bar">
      <span>Folder: <strong>{{ folder }}</strong></span>
      <span>PDFs: <strong>{{ doc_count }}</strong></span>
      <span>Chunks: <strong>{{ total_chunks }}</strong></span>
    </div>
  </header>

  <nav class="nav-tabs">
    <button class="tab-btn active" onclick="switchTab('rag')">💬 Grounded RAG Assistant</button>
    <button class="tab-btn" onclick="switchTab('search')">🔍 Multi-Strategy Search</button>
    <button class="tab-btn" onclick="switchTab('compare')">⚖️ Strategy Comparison</button>
    <button class="tab-btn" onclick="switchTab('eval')">📊 Evaluation Benchmark</button>
    <button class="tab-btn" onclick="switchTab('docs')">📁 Document Repository</button>
  </nav>

  <main>
    <!-- TAB 1: RAG Assistant -->
    <div id="rag-tab" class="tab-content active">
      <div class="card">
        <h2 style="font-size: 1.15rem; margin-bottom: 0.75rem;">Ask Questions About Your Documents</h2>
        <p style="color: var(--text-muted); font-size: 0.9rem; margin-bottom: 1rem;">
          The system retrieves evidence from your PDFs and synthesizes grounded answers with citations.
        </p>

        <div class="input-group">
          <input type="text" id="rag-query" placeholder="e.g., What are the main graphical techniques used in exploratory data analysis?" value="What are the main graphical techniques used in exploratory data analysis?" onkeydown="if(event.key==='Enter') runRAG()">
          <button class="btn" onclick="runRAG()">Ask Question</button>
        </div>

        <div class="controls-row">
          <label>Strategy:
            <select id="rag-strategy" style="padding: 0.3rem 0.6rem; width: auto;">
              <option value="hybrid" selected>Hybrid (Dense + BM25 with RRF)</option>
              <option value="dense">Dense Vector Embeddings</option>
              <option value="bm25">BM25 Sparse Lexical</option>
            </select>
          </label>
          <label>Top-K Passages:
            <select id="rag-topk" style="padding: 0.3rem 0.6rem; width: auto;">
              <option value="3" selected>3</option>
              <option value="5">5</option>
              <option value="8">8</option>
            </select>
          </label>
        </div>

        <div id="rag-loading" class="loading">⏳ Retrieving passages and synthesizing grounded answer...</div>

        <div id="rag-output" style="display: none; margin-top: 1.5rem;">
          <h3 style="font-size: 1rem; color: var(--accent);">Synthesized Answer</h3>
          <div id="rag-answer" class="answer-box"></div>
          <h3 style="font-size: 1rem; color: var(--accent); margin-top: 1.5rem;">Sources & Citations</h3>
          <div id="rag-sources" class="sources-list"></div>
        </div>
      </div>
    </div>

    <!-- TAB 2: Multi-Strategy Search -->
    <div id="search-tab" class="tab-content">
      <div class="card">
        <h2 style="font-size: 1.15rem; margin-bottom: 0.75rem;">Search Passages</h2>
        <div class="input-group">
          <input type="text" id="search-query" placeholder="Search concept or keywords..." value="RSA prime numbers encryption" onkeydown="if(event.key==='Enter') runSearch()">
          <button class="btn" onclick="runSearch()">Search</button>
        </div>

        <div class="controls-row">
          <label>Strategy:
            <select id="search-strategy" style="padding: 0.3rem 0.6rem; width: auto;">
              <option value="hybrid" selected>Hybrid (Dense + BM25)</option>
              <option value="dense">Dense Semantic Vector</option>
              <option value="bm25">BM25 Lexical</option>
              <option value="keyword">Exact Keyword</option>
            </select>
          </label>
          <label>Top-K:
            <select id="search-topk" style="padding: 0.3rem 0.6rem; width: auto;">
              <option value="5" selected>5</option>
              <option value="10">10</option>
              <option value="15">15</option>
            </select>
          </label>
        </div>

        <div id="search-loading" class="loading">⏳ Searching index...</div>
        <div id="search-results" class="sources-list" style="margin-top: 1.5rem;"></div>
      </div>
    </div>

    <!-- TAB 3: Strategy Comparison -->
    <div id="compare-tab" class="tab-content">
      <div class="card">
        <h2 style="font-size: 1.15rem; margin-bottom: 0.75rem;">Side-by-Side Strategy Comparison</h2>
        <p style="color: var(--text-muted); font-size: 0.9rem; margin-bottom: 1rem;">
          Compare results retrieved simultaneously by Dense Vector Search vs BM25 vs Hybrid (RRF).
        </p>

        <div class="input-group">
          <input type="text" id="compare-query" placeholder="Query to compare across retrievers..." value="cryptographic public key RSA prime factors" onkeydown="if(event.key==='Enter') runComparison()">
          <button class="btn" onclick="runComparison()">Compare All</button>
        </div>

        <div id="compare-loading" class="loading">⏳ Running multi-retriever comparison...</div>

        <div class="grid-2" id="compare-grid" style="margin-top: 1.5rem;">
          <div>
            <h3 style="color: var(--accent); margin-bottom: 0.5rem;">Dense Semantic (Embeddings)</h3>
            <div id="compare-dense" class="sources-list"></div>
          </div>
          <div>
            <h3 style="color: var(--warning); margin-bottom: 0.5rem;">BM25 Lexical (Okapi)</h3>
            <div id="compare-bm25" class="sources-list"></div>
          </div>
        </div>
      </div>
    </div>

    <!-- TAB 4: Evaluation Benchmark -->
    <div id="eval-tab" class="tab-content">
      <div class="card">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem;">
          <div>
            <h2 style="font-size: 1.15rem;">Information Retrieval Benchmark Suite (V6)</h2>
            <p style="color: var(--text-muted); font-size: 0.9rem;">
              Computes Precision@K, Recall@K, Mean Reciprocal Rank (MRR), MAP, and nDCG across test queries.
            </p>
          </div>
          <button class="btn" onclick="runBenchmark()">▶ Run Benchmark</button>
        </div>

        <div id="eval-loading" class="loading">⏳ Running evaluation benchmark across all retrieval strategies...</div>

        <div id="eval-results" style="margin-top: 1rem;">
          <p style="color: var(--text-muted);">Click "Run Benchmark" to execute the evaluation suite.</p>
        </div>
      </div>
    </div>

    <!-- TAB 5: Document Repository -->
    <div id="docs-tab" class="tab-content">
      <div class="card">
        <h2 style="font-size: 1.15rem; margin-bottom: 1rem;">Indexed PDF Documents</h2>
        <table>
          <thead>
            <tr>
              <th>#</th>
              <th>Filename</th>
              <th>Pages</th>
              <th>Status</th>
              <th>Full Path</th>
            </tr>
          </thead>
          <tbody>
            {% for doc in documents %}
            <tr>
              <td>{{ loop.index }}</td>
              <td><strong>{{ doc.filename }}</strong></td>
              <td>{{ doc.total_pages }}</td>
              <td><span style="color: var(--success);">✓ Extracted</span></td>
              <td style="color: var(--text-muted); font-size: 0.8rem;">{{ doc.full_path }}</td>
            </tr>
            {% endfor %}
          </tbody>
        </table>
      </div>
    </div>
  </main>

  <script>
    function switchTab(tabId) {
      document.querySelectorAll('.tab-content').forEach(el => el.classList.remove('active'));
      document.querySelectorAll('.tab-btn').forEach(el => el.classList.remove('active'));
      document.getElementById(tabId + '-tab').classList.add('active');
      event.target.classList.add('active');
    }

    async function runRAG() {
      const query = document.getElementById('rag-query').value.trim();
      const strategy = document.getElementById('rag-strategy').value;
      const top_k = parseInt(document.getElementById('rag-topk').value);
      if (!query) return;

      document.getElementById('rag-loading').style.display = 'block';
      document.getElementById('rag-output').style.display = 'none';

      try {
        const res = await fetch('/api/rag', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({ query, strategy, top_k })
        });
        const data = await res.json();

        document.getElementById('rag-loading').style.display = 'none';
        document.getElementById('rag-output').style.display = 'block';
        document.getElementById('rag-answer').innerText = data.answer;

        const sourcesContainer = document.getElementById('rag-sources');
        sourcesContainer.innerHTML = '';

        if (data.sources && data.sources.length > 0) {
          data.sources.forEach(s => {
            const card = document.createElement('div');
            card.className = 'source-card';
            card.innerHTML = `
              <div class="source-header">
                <span>[${s.index}] ${s.filename} — Page ${s.page_number}/${s.total_pages}</span>
                <span>Score: ${s.score.toFixed(4)}</span>
              </div>
              <div class="source-snippet">"${s.snippet}"</div>
            `;
            sourcesContainer.appendChild(card);
          });
        }
      } catch (err) {
        document.getElementById('rag-loading').style.display = 'none';
        alert('Error: ' + err);
      }
    }

    async function runSearch() {
      const query = document.getElementById('search-query').value.trim();
      const strategy = document.getElementById('search-strategy').value;
      const top_k = parseInt(document.getElementById('search-topk').value);
      if (!query) return;

      document.getElementById('search-loading').style.display = 'block';
      const container = document.getElementById('search-results');
      container.innerHTML = '';

      try {
        const res = await fetch('/api/search', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({ query, strategy, top_k })
        });
        const data = await res.json();

        document.getElementById('search-loading').style.display = 'none';

        if (data.results.length === 0) {
          container.innerHTML = '<p style="color: var(--text-muted);">No passages found.</p>';
          return;
        }

        data.results.forEach((r, idx) => {
          const card = document.createElement('div');
          card.className = 'source-card';
          card.innerHTML = `
            <div class="source-header">
              <span>#${idx+1}. ${r.filename} — Page ${r.page_number} (${r.chunk_id})</span>
              <span>Score: ${r.score.toFixed(4)}</span>
            </div>
            <div class="source-snippet">"${r.text}"</div>
          `;
          container.appendChild(card);
        });
      } catch (err) {
        document.getElementById('search-loading').style.display = 'none';
        alert('Error: ' + err);
      }
    }

    async function runComparison() {
      const query = document.getElementById('compare-query').value.trim();
      if (!query) return;

      document.getElementById('compare-loading').style.display = 'block';
      const denseBox = document.getElementById('compare-dense');
      const bm25Box = document.getElementById('compare-bm25');
      denseBox.innerHTML = '';
      bm25Box.innerHTML = '';

      try {
        const [resDense, resBM25] = await Promise.all([
          fetch('/api/search', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({ query, strategy: 'dense', top_k: 4 })
          }).then(r => r.json()),
          fetch('/api/search', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({ query, strategy: 'bm25', top_k: 4 })
          }).then(r => r.json())
        ]);

        document.getElementById('compare-loading').style.display = 'none';

        resDense.results.forEach((r, idx) => {
          denseBox.innerHTML += `
            <div class="source-card">
              <div class="source-header"><span>#${idx+1}. ${r.filename} (p. ${r.page_number})</span><span>Score: ${r.score.toFixed(4)}</span></div>
              <div class="source-snippet">"${r.text.substring(0, 180)}..."</div>
            </div>
          `;
        });

        resBM25.results.forEach((r, idx) => {
          bm25Box.innerHTML += `
            <div class="source-card">
              <div class="source-header"><span>#${idx+1}. ${r.filename} (p. ${r.page_number})</span><span>Score: ${r.score.toFixed(4)}</span></div>
              <div class="source-snippet">"${r.text.substring(0, 180)}..."</div>
            </div>
          `;
        });
      } catch (err) {
        document.getElementById('compare-loading').style.display = 'none';
        alert('Error: ' + err);
      }
    }

    async function runBenchmark() {
      document.getElementById('eval-loading').style.display = 'block';
      const container = document.getElementById('eval-results');

      try {
        const res = await fetch('/api/evaluate');
        const data = await res.json();
        document.getElementById('eval-loading').style.display = 'none';

        let html = `
          <table>
            <thead>
              <tr>
                <th>Strategy</th>
                <th>Precision@1</th>
                <th>Precision@3</th>
                <th>Precision@5</th>
                <th>MRR (Mean Reciprocal Rank)</th>
                <th>MAP (Mean Avg Precision)</th>
                <th>nDCG@5</th>
              </tr>
            </thead>
            <tbody>
        `;

        for (const [strat, m] of Object.entries(data)) {
          html += `
            <tr>
              <td><strong>${strat.toUpperCase()}</strong></td>
              <td>${m.mean_precision_at_k['1'].toFixed(4)}</td>
              <td>${m.mean_precision_at_k['3'].toFixed(4)}</td>
              <td>${m.mean_precision_at_k['5'].toFixed(4)}</td>
              <td><span style="color: var(--accent); font-weight: 600;">${m.mean_reciprocal_rank.toFixed(4)}</span></td>
              <td>${m.mean_average_precision.toFixed(4)}</td>
              <td>${m.mean_ndcg_at_k['5'].toFixed(4)}</td>
            </tr>
          `;
        }

        html += `</tbody></table>`;
        container.innerHTML = html;
      } catch (err) {
        document.getElementById('eval-loading').style.display = 'none';
        alert('Error: ' + err);
      }
    }
  </script>
</body>
</html>
"""


def create_app(folder_path: str = "./data") -> Flask:
    """Creates and configures the Flask dashboard application."""
    app = Flask(__name__)
    initialize_system(folder_path)

    @app.route("/")
    def index():
        return render_template_string(
            HTML_TEMPLATE,
            folder=STATE["folder_path"],
            doc_count=len(STATE["documents"]),
            total_chunks=STATE["total_chunks"],
            documents=STATE["documents"],
        )

    @app.route("/api/status", methods=["GET"])
    def status():
        return jsonify({
            "folder": STATE["folder_path"],
            "documents_count": len(STATE["documents"]),
            "total_chunks": STATE["total_chunks"],
            "available_strategies": list(STATE["retrievers"].keys()),
        })

    @app.route("/api/search", methods=["POST"])
    def search():
        data = request.get_json() or {}
        query = data.get("query", "").strip()
        strategy = data.get("strategy", "hybrid")
        top_k = int(data.get("top_k", 5))

        if not query:
            return jsonify({"results": []})

        retriever: BaseRetriever = STATE["retrievers"].get(strategy, STATE["retrievers"]["hybrid"])
        results = retriever.search(query=query, top_k=top_k, deduplicate=True)

        serialized = [
            {
                "chunk_id": r.chunk.chunk_id,
                "filename": r.chunk.filename,
                "page_number": r.chunk.page_number,
                "total_pages": r.chunk.total_pages,
                "text": r.chunk.text,
                "score": r.score,
            }
            for r in results
        ]
        return jsonify({"results": serialized})

    @app.route("/api/rag", methods=["POST"])
    def rag():
        data = request.get_json() or {}
        query = data.get("query", "").strip()
        strategy = data.get("strategy", "hybrid")
        top_k = int(data.get("top_k", 3))

        if not query:
            return jsonify({
                "answer": "Please enter a valid question.",
                "sources": [],
                "has_sufficient_context": False,
            })

        retriever = STATE["retrievers"].get(strategy, STATE["retrievers"]["hybrid"])
        pipeline = RAGPipeline(retriever=retriever, llm_backend="auto")
        result = pipeline.answer(query=query, top_k=top_k)

        return jsonify({
            "query": result.query,
            "answer": result.answer,
            "sources": result.sources,
            "has_sufficient_context": result.has_sufficient_context,
        })

    @app.route("/api/evaluate", methods=["GET"])
    def evaluate():
        evaluator = Evaluator(k_values=[1, 3, 5])
        benchmarks = evaluator.run_comparative_benchmark(
            documents=STATE["documents"],
            strategies=["bm25", "dense", "hybrid"],
        )

        serialized = {
            strat: {
                "strategy": res.strategy,
                "total_queries": res.total_queries,
                "mean_precision_at_k": {str(k): v for k, v in res.mean_precision_at_k.items()},
                "mean_recall_at_k": {str(k): v for k, v in res.mean_recall_at_k.items()},
                "mean_reciprocal_rank": res.mean_reciprocal_rank,
                "mean_average_precision": res.mean_average_precision,
                "mean_ndcg_at_k": {str(k): v for k, v in res.mean_ndcg_at_k.items()},
            }
            for strat, res in benchmarks.items()
        }
        return jsonify(serialized)

    return app


def main():
    parser = argparse.ArgumentParser(description="PDF Retrieval & RAG Web Dashboard")
    parser.add_argument("--folder", type=str, default="./data", help="Path to PDF folder (default: ./data)")
    parser.add_argument("--port", type=int, default=8000, help="Port to run web dashboard on (default: 8000)")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host address (default: 127.0.0.1)")
    args = parser.parse_args()

    app = create_app(folder_path=args.folder)
    print("\n" + "=" * 70)
    print(f"  🚀 PDF Semantic Retrieval & RAG Web Dashboard is Running!")
    print(f"  👉 Open in your browser: http://localhost:{args.port}")
    print("=" * 70 + "\n")
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
