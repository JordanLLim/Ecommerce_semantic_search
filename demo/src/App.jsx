import { useMemo, useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import {
  Search, Sparkles, BrainCircuit, Network, Radar, Shuffle, Trophy,
  Clock3, Boxes, Zap, ArrowRight, Layers3, Gauge,
} from 'lucide-react'

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

const stageNumbers = ['1.3M', '1 vector', '50 dense', '50 sparse', '50 fused', '20 reranked']

function tokeniseIntent(query) {
  return query.split(/\s+/).filter(Boolean).slice(0, 7)
}

function Universe({ active, results }) {
  const points = useMemo(() => Array.from({ length: 120 }, (_, i) => ({
    id: i,
    left: `${5 + ((i * 37) % 90)}%`,
    top: `${5 + ((i * 53) % 88)}%`,
    delay: (i % 12) * 0.06,
    size: 2 + (i % 4),
  })), [])

  return (
    <div className="universe-panel glass">
      <div className="panel-kicker"><Radar size={16} /> Retrieval universe</div>
      <div className="universe">
        <div className="grid-floor" />
        <div className="orbit orbit-a" />
        <div className="orbit orbit-b" />
        <div className="orbit orbit-c" />
        {points.map((point) => (
          <motion.span
            key={point.id}
            className="star"
            style={{ left: point.left, top: point.top, width: point.size, height: point.size }}
            animate={active ? { opacity: [0.18, 0.95, 0.24], scale: [1, 2.2, 1] } : { opacity: 0.25 }}
            transition={{ duration: 1.7, repeat: Infinity, delay: point.delay }}
          />
        ))}
        {active && <motion.div className="scan-ring" initial={{ scale: .15, opacity: 1 }} animate={{ scale: 7, opacity: 0 }} transition={{ duration: 2.2, repeat: Infinity, ease: 'easeOut' }} />}
        <motion.div
          className="query-core"
          animate={active ? { scale: [1, 1.16, 1], boxShadow: ['0 0 25px #7c5cff', '0 0 85px #38bdf8', '0 0 25px #7c5cff'] } : {}}
          transition={{ duration: 1.4, repeat: Infinity }}
        >
          AI
        </motion.div>
        {results.slice(0, 5).map((item, idx) => (
          <motion.div
            key={item.product_id || idx}
            className={`candidate candidate-${idx + 1}`}
            initial={{ opacity: 0, scale: 0.3, x: 0, y: 0 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ delay: idx * 0.12, type: 'spring' }}
            title={item.title}
          >#{idx + 1}</motion.div>
        ))}
      </div>
      <div className="universe-footer"><span>1,300,000 indexed products</span><span>Semantic space · live query</span></div>
    </div>
  )
}

function Metric({ label, value, icon: Icon }) {
  return <div className="metric glass"><Icon size={18} /><div><strong>{value}</strong><span>{label}</span></div></div>
}

function RetrievalRace({ active, results }) {
  const dense = results.slice(0, 4)
  const sparse = [...results].sort((a, b) => (b.bm25_score || 0) - (a.bm25_score || 0)).slice(0, 4)
  return (
    <div className="race glass">
      <div className="panel-kicker"><Shuffle size={16} /> Retrieval race</div>
      <div className="race-grid">
        <div className="lane dense-lane">
          <div className="lane-title"><span className="dense-dot" />FAISS · meaning</div>
          {(dense.length ? dense : Array.from({ length: 4 }, (_, i) => ({ title: `semantic candidate ${i + 1}` }))).map((item, idx) => (
            <motion.div className="mini-card" key={`d-${item.product_id || idx}`} animate={active ? { x: [0, 8, 0] } : {}} transition={{ repeat: Infinity, duration: 1.3, delay: idx * .11 }}>
              <span>#{idx + 1}</span><p>{item.title}</p>
            </motion.div>
          ))}
        </div>

        <div className="fusion-core">
          <motion.div animate={active ? { rotate: 360, scale: [1, 1.08, 1] } : { rotate: 0 }} transition={{ rotate: { duration: 6, repeat: Infinity, ease: 'linear' }, scale: { duration: 1.2, repeat: Infinity } }}>
            <Layers3 size={28} />
          </motion.div>
          <strong>RRF</strong><span>fusion</span>
        </div>

        <div className="lane sparse-lane">
          <div className="lane-title"><span className="sparse-dot" />BM25 · words</div>
          {(sparse.length ? sparse : Array.from({ length: 4 }, (_, i) => ({ title: `keyword candidate ${i + 1}` }))).map((item, idx) => (
            <motion.div className="mini-card" key={`s-${item.product_id || idx}`} animate={active ? { x: [0, -8, 0] } : {}} transition={{ repeat: Infinity, duration: 1.3, delay: idx * .11 }}>
              <span>#{idx + 1}</span><p>{item.title}</p>
            </motion.div>
          ))}
        </div>
      </div>
    </div>
  )
}

function Cinematic({ open, stage, query, onSkip }) {
  if (!open) return null
  const safeStage = Math.max(0, Math.min(stage, stageCopy.length - 1))
  const [title, subtitle] = stageCopy[safeStage]
  return (
    <AnimatePresence>
      <motion.div className="cinematic" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}>
        <div className="cinematic-stars" />
        <button className="skip-demo" onClick={onSkip}>Skip animation</button>
        <motion.div className="cinematic-copy" key={safeStage} initial={{ opacity: 0, y: 30, scale: .96 }} animate={{ opacity: 1, y: 0, scale: 1 }}>
          <div className="scene-count">0{safeStage + 1} / 06</div>
          <div className="scene-number">{stageNumbers[safeStage]}</div>
          <h2>{title}</h2>
          <p>{subtitle}</p>
          <div className="query-pill">“{query}”</div>
        </motion.div>
        <div className="cinematic-track">
          {stageCopy.map((item, idx) => <span key={item[0]} className={idx <= safeStage ? 'active' : ''} />)}
        </div>
      </motion.div>
    </AnimatePresence>
  )
}

function App() {
  const [query, setQuery] = useState(examples[0])
  const [loading, setLoading] = useState(false)
  const [stage, setStage] = useState(-1)
  const [payload, setPayload] = useState(null)
  const [error, setError] = useState('')
  const [cinematic, setCinematic] = useState(false)

  const intent = tokeniseIntent(query)

  async function runSearch(event) {
    event?.preventDefault()
    if (!query.trim() || loading) return
    setLoading(true); setPayload(null); setError(''); setStage(0); setCinematic(true)

    const timers = stageCopy.map((_, i) => setTimeout(() => setStage(i), 450 + i * 650))
    try {
      const response = await fetch(`${API_URL}/search`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query, top_k: 10, candidate_k: 50, hybrid: true, rerank: true, rerank_candidate_k: 20 }),
      })
      if (!response.ok) throw new Error(`Search failed (${response.status})`)
      const data = await response.json()
      const minimumShowtime = new Promise(resolve => setTimeout(resolve, 3900))
      await minimumShowtime
      setPayload(data); setStage(stageCopy.length); setCinematic(false)
      setTimeout(() => document.getElementById('results')?.scrollIntoView({ behavior: 'smooth', block: 'start' }), 120)
    } catch (err) {
      setError(err.message || 'Search failed'); setCinematic(false)
    } finally {
      timers.forEach(clearTimeout); setLoading(false)
    }
  }

  const results = payload?.results || []

  return (
    <main className="app-shell">
      <Cinematic open={cinematic} stage={stage} query={query} onSkip={() => setCinematic(false)} />
      <div className="aurora aurora-one" /><div className="aurora aurora-two" />

      <nav className="nav">
        <div className="brand"><div className="brand-mark"><Sparkles size={17} /></div>SearchVerse</div>
        <div className="nav-note"><span className="live-dot" /> 1.3M-product AI search</div>
      </nav>

      <section className="hero">
        <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }}>
          <div className="eyebrow">AI PRODUCT DISCOVERY · VISUALIZED LIVE</div>
          <h1>Watch 1.3 million products<br /><span>collapse into the right answer.</span></h1>
          <p>One search triggers semantic retrieval, keyword matching, rank fusion and AI re-ranking — then shows you how it happened.</p>
        </motion.div>

        <form className="search-box glass" onSubmit={runSearch}>
          <Search size={22} /><input value={query} onChange={(e) => setQuery(e.target.value)} aria-label="Search products" />
          <button type="submit" disabled={loading}>{loading ? 'Searching…' : <><Zap size={16} /> Run AI search</>}</button>
        </form>
        <div className="examples">{examples.map((item) => <button key={item} onClick={() => setQuery(item)}>{item}</button>)}</div>

        <div className="hero-stats">
          <div><strong>1.3M</strong><span>products</span></div><ArrowRight size={16} />
          <div><strong>50</strong><span>retrieved</span></div><ArrowRight size={16} />
          <div><strong>20</strong><span>AI reranked</span></div><ArrowRight size={16} />
          <div><strong>10</strong><span>shown</span></div>
        </div>
      </section>

      <section className="intent-row glass">
        <div><div className="panel-kicker"><BrainCircuit size={16} /> Intent map</div><h2>{query || 'Type a query'}</h2></div>
        <div className="chips">{intent.map((item, idx) => <motion.span key={`${item}-${idx}`} initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: idx * 0.05 }}>{item}</motion.span>)}</div>
      </section>

      <section className="visual-grid">
        <Universe active={loading} results={results} />
        <div className="pipeline glass">
          <div className="panel-kicker"><Network size={16} /> Live retrieval pipeline</div>
          <div className="pipeline-list">
            {stageCopy.map(([title, subtitle], idx) => {
              const done = stage > idx || Boolean(payload); const current = stage === idx && loading
              return <motion.div key={title} className={`stage ${done ? 'done' : ''} ${current ? 'current' : ''}`} animate={current ? { x: [0, 5, 0] } : {}} transition={{ repeat: Infinity, duration: .8 }}>
                <div className="stage-index">{done ? '✓' : idx + 1}</div><div><strong>{title}</strong><span>{subtitle}</span></div>
              </motion.div>
            })}
          </div>
          <div className="fusion-strip"><div><span className="dense-dot" />FAISS semantic</div><Shuffle size={17} /><div><span className="sparse-dot" />BM25 keyword</div></div>
        </div>
      </section>

      <RetrievalRace active={loading} results={results} />

      <AnimatePresence>{error && <motion.div className="error glass" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}>{error}. Make sure FastAPI is running at {API_URL}.</motion.div>}</AnimatePresence>

      {payload && <motion.section id="results" className="results-section" initial={{ opacity: 0, y: 30 }} animate={{ opacity: 1, y: 0 }}>
        <div className="results-heading">
          <div><div className="eyebrow">FINAL RESULTS</div><h2>1.3M → {results.length} products</h2><p className="results-sub">Real products returned by the backend. Real component timings. No pre-scripted search results.</p></div>
          <div className="metrics-row">
            <Metric label="backend" value={`${payload.latency_ms} ms`} icon={Clock3} />
            <Metric label="candidates" value="50" icon={Boxes} />
            <Metric label="reranked" value={payload.rerank_candidate_k || 0} icon={Trophy} />
          </div>
        </div>

        <div className="result-grid">{results.map((item, idx) => <motion.article className={`product-card glass ${idx < 3 ? 'podium-card' : ''}`} key={item.product_id || idx} initial={{ opacity: 0, y: 28, rotateX: -8 }} animate={{ opacity: 1, y: 0, rotateX: 0 }} transition={{ delay: idx * .07 }}>
          <div className="rank">#{idx + 1}</div><div className="product-art">{(item.brand || item.title || '?').slice(0, 1).toUpperCase()}</div>
          <div className="product-meta"><span>{item.brand || 'Unbranded'} · {(item.locale || 'US').toUpperCase()}</span><h3>{item.title}</h3><div className="signal-row">
            {item.score != null && <span>Dense {Number(item.score).toFixed(3)}</span>}{item.bm25_score != null && <span>BM25 {Number(item.bm25_score).toFixed(2)}</span>}{item.fusion_score != null && <span>RRF {Number(item.fusion_score).toFixed(4)}</span>}
          </div></div>
        </motion.article>)}</div>

        <div className="latency glass">
          <div><span>Embedding</span><strong>{payload.embedding_latency_ms} ms</strong></div>
          <div><span>FAISS</span><strong>{payload.faiss_latency_ms} ms</strong></div>
          <div><span>BM25</span><strong>{payload.bm25_latency_ms} ms</strong></div>
          <div><span>CrossEncoder</span><strong>{payload.rerank_latency_ms} ms</strong></div>
        </div>
        <div className="final-banner glass"><Gauge size={22} /><div><strong>Search complete.</strong><span>Semantic retrieval + sparse retrieval + RRF + CrossEncoder, surfaced as one explainable journey.</span></div></div>
      </motion.section>}

      <footer>SearchVerse · Real retrieval signals, visualized for demo clarity.</footer>
    </main>
  )
}

export default App
