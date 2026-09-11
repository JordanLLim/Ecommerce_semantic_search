import { useMemo, useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { Search, Sparkles, BrainCircuit, Network, Radar, Shuffle, Trophy, Clock3 } from 'lucide-react'

const API_URL = import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000'

const examples = [
  'quiet cooling fan for gaming pc under desk',
  'wireless keyboard for office',
  'noise cancelling headphones',
]

const stageCopy = [
  ['Understanding intent', 'Turning words into search intent'],
  ['Embedding query', 'Mapping meaning into vector space'],
  ['Dense retrieval', 'Searching the FAISS index'],
  ['Sparse retrieval', 'Matching exact terms with BM25'],
  ['RRF fusion', 'Combining semantic + keyword rankings'],
  ['AI re-ranking', 'CrossEncoder compares the strongest candidates'],
]

function tokeniseIntent(query) {
  return query
    .split(/\s+/)
    .filter(Boolean)
    .slice(0, 7)
}

function Universe({ active, results }) {
  const points = useMemo(() => {
    return Array.from({ length: 90 }, (_, i) => ({
      id: i,
      left: `${8 + ((i * 37) % 84)}%`,
      top: `${8 + ((i * 53) % 82)}%`,
      delay: (i % 9) * 0.08,
      size: 2 + (i % 4),
    }))
  }, [])

  return (
    <div className="universe-panel glass">
      <div className="panel-kicker"><Radar size={16} /> Retrieval universe</div>
      <div className="universe">
        <div className="orbit orbit-a" />
        <div className="orbit orbit-b" />
        {points.map((point) => (
          <motion.span
            key={point.id}
            className="star"
            style={{ left: point.left, top: point.top, width: point.size, height: point.size }}
            animate={active ? { opacity: [0.25, 0.9, 0.3], scale: [1, 1.9, 1] } : { opacity: 0.25 }}
            transition={{ duration: 1.8, repeat: Infinity, delay: point.delay }}
          />
        ))}
        <motion.div
          className="query-core"
          animate={active ? { scale: [1, 1.15, 1], boxShadow: ['0 0 25px #7c5cff', '0 0 70px #38bdf8', '0 0 25px #7c5cff'] } : {}}
          transition={{ duration: 1.4, repeat: Infinity }}
        >
          AI
        </motion.div>
        {results.slice(0, 5).map((item, idx) => (
          <motion.div
            key={item.product_id || idx}
            className={`candidate candidate-${idx + 1}`}
            initial={{ opacity: 0, scale: 0.7 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ delay: idx * 0.12 }}
            title={item.title}
          >
            #{idx + 1}
          </motion.div>
        ))}
      </div>
      <div className="universe-footer">
        <span>1.3M catalog</span>
        <span>Dense + sparse retrieval</span>
      </div>
    </div>
  )
}

function Metric({ label, value, icon: Icon }) {
  return (
    <div className="metric glass">
      <Icon size={18} />
      <div><strong>{value}</strong><span>{label}</span></div>
    </div>
  )
}

function App() {
  const [query, setQuery] = useState(examples[0])
  const [loading, setLoading] = useState(false)
  const [stage, setStage] = useState(-1)
  const [payload, setPayload] = useState(null)
  const [error, setError] = useState('')

  const intent = tokeniseIntent(query)

  async function runSearch(event) {
    event?.preventDefault()
    if (!query.trim() || loading) return

    setLoading(true)
    setPayload(null)
    setError('')
    setStage(0)

    const timers = stageCopy.map((_, i) => setTimeout(() => setStage(i), 350 + i * 500))

    try {
      const response = await fetch(`${API_URL}/search`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query,
          top_k: 10,
          candidate_k: 50,
          hybrid: true,
          rerank: true,
          rerank_candidate_k: 20,
        }),
      })
      if (!response.ok) throw new Error(`Search failed (${response.status})`)
      const data = await response.json()
      setPayload(data)
      setStage(stageCopy.length)
    } catch (err) {
      setError(err.message || 'Search failed')
    } finally {
      timers.forEach(clearTimeout)
      setLoading(false)
    }
  }

  const results = payload?.results || []

  return (
    <main className="app-shell">
      <div className="aurora aurora-one" />
      <div className="aurora aurora-two" />

      <nav className="nav">
        <div className="brand"><div className="brand-mark"><Sparkles size={17} /></div>SearchVerse</div>
        <div className="nav-note">Watch AI think before you shop.</div>
      </nav>

      <section className="hero">
        <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }}>
          <div className="eyebrow">AI PRODUCT DISCOVERY · 1.3M PRODUCTS</div>
          <h1>One vague query.<br /><span>One intelligent search journey.</span></h1>
          <p>See semantic search, keyword retrieval, fusion and AI re-ranking happen as one visual pipeline.</p>
        </motion.div>

        <form className="search-box glass" onSubmit={runSearch}>
          <Search size={22} />
          <input value={query} onChange={(e) => setQuery(e.target.value)} aria-label="Search products" />
          <button type="submit" disabled={loading}>{loading ? 'Thinking…' : 'Search'}</button>
        </form>

        <div className="examples">
          {examples.map((item) => <button key={item} onClick={() => setQuery(item)}>{item}</button>)}
        </div>
      </section>

      <section className="intent-row glass">
        <div>
          <div className="panel-kicker"><BrainCircuit size={16} /> Intent map</div>
          <h2>{query || 'Type a query'}</h2>
        </div>
        <div className="chips">
          {intent.map((item, idx) => (
            <motion.span key={`${item}-${idx}`} initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: idx * 0.05 }}>{item}</motion.span>
          ))}
        </div>
      </section>

      <section className="visual-grid">
        <Universe active={loading} results={results} />

        <div className="pipeline glass">
          <div className="panel-kicker"><Network size={16} /> Live retrieval pipeline</div>
          <div className="pipeline-list">
            {stageCopy.map(([title, subtitle], idx) => {
              const done = stage > idx || Boolean(payload)
              const current = stage === idx && loading
              return (
                <motion.div key={title} className={`stage ${done ? 'done' : ''} ${current ? 'current' : ''}`} animate={current ? { x: [0, 5, 0] } : {}} transition={{ repeat: Infinity, duration: 0.8 }}>
                  <div className="stage-index">{done ? '✓' : idx + 1}</div>
                  <div><strong>{title}</strong><span>{subtitle}</span></div>
                </motion.div>
              )
            })}
          </div>
          <div className="fusion-strip">
            <div><span className="dense-dot" />FAISS semantic</div>
            <Shuffle size={17} />
            <div><span className="sparse-dot" />BM25 keyword</div>
          </div>
        </div>
      </section>

      <AnimatePresence>
        {error && <motion.div className="error glass" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}>{error}. Make sure FastAPI is running at {API_URL}.</motion.div>}
      </AnimatePresence>

      {payload && (
        <motion.section className="results-section" initial={{ opacity: 0, y: 30 }} animate={{ opacity: 1, y: 0 }}>
          <div className="results-heading">
            <div><div className="eyebrow">FINAL RESULTS</div><h2>From 1.3M products to your top {results.length}</h2></div>
            <div className="metrics-row">
              <Metric label="backend" value={`${payload.latency_ms} ms`} icon={Clock3} />
              <Metric label="candidates" value="50" icon={Network} />
              <Metric label="reranked" value={payload.rerank_candidate_k || 0} icon={Trophy} />
            </div>
          </div>

          <div className="result-grid">
            {results.map((item, idx) => (
              <motion.article className="product-card glass" key={item.product_id || idx} initial={{ opacity: 0, y: 18 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: idx * 0.06 }}>
                <div className="rank">#{idx + 1}</div>
                <div className="product-art">{(item.brand || item.title || '?').slice(0, 1).toUpperCase()}</div>
                <div className="product-meta">
                  <span>{item.brand || 'Unbranded'} · {(item.locale || 'US').toUpperCase()}</span>
                  <h3>{item.title}</h3>
                  <div className="signal-row">
                    {item.score != null && <span>Dense {Number(item.score).toFixed(3)}</span>}
                    {item.bm25_score != null && <span>BM25 {Number(item.bm25_score).toFixed(2)}</span>}
                    {item.fusion_score != null && <span>RRF {Number(item.fusion_score).toFixed(4)}</span>}
                  </div>
                </div>
              </motion.article>
            ))}
          </div>

          <div className="latency glass">
            <div><span>Embedding</span><strong>{payload.embedding_latency_ms} ms</strong></div>
            <div><span>FAISS</span><strong>{payload.faiss_latency_ms} ms</strong></div>
            <div><span>BM25</span><strong>{payload.bm25_latency_ms} ms</strong></div>
            <div><span>CrossEncoder</span><strong>{payload.rerank_latency_ms} ms</strong></div>
          </div>
        </motion.section>
      )}

      <footer>SearchVerse · Real retrieval signals, visualized for demo clarity.</footer>
    </main>
  )
}

export default App
