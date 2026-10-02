import React, { useState } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { UploadCloud, Search, FileText, Clock, Sparkles } from 'lucide-react';

const API_BASE = import.meta.env.VITE_API_BASE;

export default function App() {
  const [file, setFile] = useState(null);
  const [uploadStatus, setUploadStatus] = useState(null);
  const [docId, setDocId] = useState(null);
  const [chunksCount, setChunksCount] = useState(0);
  const [isUploading, setIsUploading] = useState(false);

  const [query, setQuery] = useState('');
  const [answer, setAnswer] = useState('');
  const [results, setResults] = useState([]);
  const [isSearching, setIsSearching] = useState(false);

  const handleUpload = async () => {
    if (!file) return;
    setIsUploading(true);
    setUploadStatus('PENDING');

    const formData = new FormData();
    formData.append('file', file);

    try {
      const res = await fetch(`${API_BASE}/documents/upload`, {
        method: 'POST',
        body: formData,
      });
      const data = await res.json();
      setDocId(data.document_id);
      pollStatus(data.document_id);
    } catch (err) {
      console.error(err);
      setUploadStatus('FAILED');
      setIsUploading(false);
    }
  };

  const pollStatus = (id) => {
    const interval = setInterval(async () => {
      try {
        const res = await fetch(`${API_BASE}/documents/${id}/status`);
        const data = await res.json();
        setUploadStatus(data.status);
        if (data.status === 'COMPLETED') {
          setChunksCount(data.total_chunks);
          setIsUploading(false);
          clearInterval(interval);
        } else if (data.status === 'FAILED') {
          setIsUploading(false);
          clearInterval(interval);
        }
      } catch (e) {
        clearInterval(interval);
        setIsUploading(false);
      }
    }, 1000);
  };

  const handleSearch = async (e) => {
    e.preventDefault();
    if (!query.trim()) return;
    setIsSearching(true);
    setAnswer('');

    try {
      const res = await fetch(`${API_BASE}/query`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query: query,
          top_candidates: 10,
          final_top_k: 3,
        }),
      });
      const data = await res.json();
      setAnswer(data.answer || '');
      setResults(data.sources || []);
    } catch (err) {
      console.error(err);
    } finally {
      setIsSearching(false);
    }
  };

  return (
    <div className="container">
      <header className="navbar">
        <div className="navbar-copy">
          <h1>Distributed Hybrid RAG Engine</h1>
          <p>Two-Stage Information Retrieval: pgvector Dense + BM25 Sparse + Cross-Encoder Re-Ranking + Groq LLM</p>
        </div>
        <a className="github-link" href="https://github.com/DevDreamer26/Distributed_Hybrid_RAG_Engine" target="_blank" rel="noreferrer" aria-label="Visit GitHub" title="GitHub">
          <svg className="github-mark" viewBox="0 0 19 19" aria-hidden="true">
            <use href="/icons.svg#github-icon" />
          </svg>
          <span>GitHub</span>
        </a>
      </header>

      <div className="grid">
        {/* Left Column: Fixed, Non-stretching Ingestion Sidebar */}
        <aside className="card sidebar">
          <h2><UploadCloud size={20} /> Ingestion Pipeline</h2>
          <div className="upload-box" onClick={() => document.getElementById('pdf-input').click()}>
            <FileText size={32} color="#94a3b8" />
            <p>{file ? file.name : "Click to select a PDF document"}</p>
            <input
              id="pdf-input"
              type="file"
              accept=".pdf"
              style={{ display: 'none' }}
              onChange={(e) => setFile(e.target.files[0])}
            />
          </div>

          <button
            className="btn"
            onClick={handleUpload}
            disabled={!file || isUploading}
          >
            {isUploading ? <Clock size={16} /> : <UploadCloud size={16} />}
            {isUploading ? 'Ingesting via Celery...' : 'Upload & Process'}
          </button>

          {uploadStatus && (
            <div style={{ marginTop: '1.25rem', borderTop: '1px solid var(--border)', paddingTop: '1rem' }}>
              <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>Status:</p>
              <div style={{ marginTop: '0.35rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                <span className={`status-badge status-${uploadStatus}`}>{uploadStatus}</span>
                {uploadStatus === 'COMPLETED' && (
                  <span style={{ fontSize: '0.8rem', color: 'var(--success)' }}>
                    ({chunksCount} chunks indexed)
                  </span>
                )}
              </div>
            </div>
          )}
        </aside>

        {/* Right Column: Grounded AI Search */}
        <main className="card">
          <h2><Search size={20} /> Grounded AI Search</h2>
          <form onSubmit={handleSearch} className="search-input-group">
            <input
              type="text"
              placeholder="Ask a question about your uploaded documents..."
              value={query}
              onChange={(e) => setQuery(e.target.value)}
            />
            <button
              className="btn"
              type="submit"
              style={{ width: 'auto', marginTop: 0 }}
              disabled={isSearching}
            >
              {isSearching ? <Clock size={16} /> : <Sparkles size={16} />}
              {isSearching ? 'Thinking...' : 'Ask AI'}
            </button>
          </form>

          {/* Synthesized Answer Box */}
          {answer && (
            <div className="ai-answer-card">
              <h3>Synthesized Answer (Grounded Context)</h3>
              <ReactMarkdown remarkPlugins={[remarkGfm]}>{answer}</ReactMarkdown>
            </div>
          )}

          {/* Context Snippets */}
          <div>
            {results.length > 0 && (
              <h4 style={{ fontSize: '0.85rem', color: 'var(--text-muted)', marginBottom: '0.75rem' }}>
                Retrieved Context Sources ({results.length}):
              </h4>
            )}
            {results.length > 0 ? (
              results.map((res, i) => (
                <div key={res.chunk_id} className="result-item">
                  <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.4rem' }}>
                    <span style={{ fontWeight: 600, color: 'var(--accent)', fontSize: '0.85rem' }}>
                      Source #{i + 1}
                    </span>
                    <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
                      Chunk Index: {res.chunk_index}
                    </span>
                  </div>
                  <p className="content-snippet">{res.content}</p>

                  <div className="metric-pills">
                    <span className="pill">Re-rank: {res.cross_encoder_score}</span>
                    <span className="pill">RRF: {res.rrf_score}</span>
                    <span className="pill">Dense Rank: {res.dense_rank ?? 'N/A'}</span>
                    <span className="pill">Sparse Rank: {res.sparse_rank ?? 'N/A'}</span>
                  </div>
                </div>
              ))
            ) : (
              !answer && (
                <p style={{ color: 'var(--text-muted)', fontSize: '0.9rem', textAlign: 'center', padding: '3rem 0' }}>
                  Ask any question about your uploaded documents to view synthesized answers with source citations.
                </p>
              )
            )}
          </div>
        </main>
      </div>
      <footer className="footer">
            <p>&copy; 2026 | Bidyasagar Hazarika. All rights reserved.</p>
          </footer>
    </div>
  );
}



