import { useCallback, useEffect, useRef, useState } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import './App.css'

const ACCEPT = 'image/jpeg,image/png,image/webp,image/gif'

function App() {
  const [file, setFile] = useState(null)
  const [previewUrl, setPreviewUrl] = useState(null)
  const [recentImages, setRecentImages] = useState([])
  const [draggedRecentFile, setDraggedRecentFile] = useState(null)
  const [text, setText] = useState('')
  const [model, setModel] = useState('')
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)
  const [dragOver, setDragOver] = useState(false)
  const [healthOk, setHealthOk] = useState(null)
  const inputRef = useRef(null)

  useEffect(() => {
    fetch('/health')
      .then((r) => {
        setHealthOk(r.ok)
        return r.json().catch(() => ({}))
      })
      .then((data) => {
        if (data.model) setModel(data.model)
      })
      .catch(() => setHealthOk(false))
  }, [])

  useEffect(() => {
    if (!file) {
      setPreviewUrl(null)
      return
    }
    const url = URL.createObjectURL(file)
    setPreviewUrl(url)
    return () => URL.revokeObjectURL(url)
  }, [file])

  const onFile = useCallback((f) => {
    if (!f || !f.type.startsWith('image/')) {
      setError('Choose a JPEG, PNG, WebP, or GIF image.')
      return
    }
    setError('')
    setText('')
    setFile(f)
  }, [])

  const onDrop = useCallback(
    (e) => {
      e.preventDefault()
      setDragOver(false)
      const f = e.dataTransfer.files?.[0] ?? draggedRecentFile
      onFile(f)
      setDraggedRecentFile(null)
    },
    [draggedRecentFile, onFile],
  )

  const extract = async () => {
    if (!file) return
    setLoading(true)
    setError('')
    setText('')
    const form = new FormData()
    form.append('file', file)
    try {
      const res = await fetch('/api/extract', {
        method: 'POST',
        body: form,
      })
      const data = await res.json().catch(() => ({}))
      if (!res.ok) {
        setError(data.detail ?? res.statusText ?? 'Request failed')
        return
      }
      setText(data.text ?? '')
      if (data.model) setModel(data.model)
      setRecentImages((prev) => {
        if (!file) return prev
        const existingIdx = prev.findIndex((item) => item.name === file.name && item.size === file.size)
        const existing = existingIdx >= 0 ? prev[existingIdx] : null
        const entry = {
          id: `${file.name}-${file.size}-${file.lastModified}`,
          name: file.name,
          size: file.size,
          type: file.type,
          preview: existing?.preview ?? URL.createObjectURL(file),
          file,
        }
        const withoutDup = prev.filter((item) => !(item.name === file.name && item.size === file.size))
        const next = [entry, ...withoutDup].slice(0, 4)
        withoutDup.slice(3).forEach((item) => URL.revokeObjectURL(item.preview))
        return next
      })
    } catch (err) {
      setError(err.message ?? 'Network error')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="app">
      <header className="header">
        <div className="header-title">
          <img src="/unnamed.png" alt="GS1 logo" className="header-logo" />
          <h1>GS1 New OCR</h1>
        </div>
        {/* <p className="sub">
          Upload a label or document image. Text is extracted locally via Ollama
          {model ? ` (${model})` : ''}.
        </p> */}
        {healthOk === false && (
          <p className="banner warn" role="status">
            API or Ollama may be offline. Start the backend:{' '}
            <code className="inline">uvicorn main:app --reload</code> from{' '}
            <code className="inline">backend/</code> and ensure Ollama is running.
          </p>
        )}
      </header>

      <div className="layout">
        <section className="panel upload-panel">
          <div
            className={`dropzone ${dragOver ? 'dragover' : ''}`}
            onDragOver={(e) => {
              e.preventDefault()
              setDragOver(true)
            }}
            onDragLeave={() => setDragOver(false)}
            onDrop={onDrop}
            onClick={() => inputRef.current?.click()}
            role="button"
            tabIndex={0}
            onKeyDown={(e) => {
              if (e.key === 'Enter' || e.key === ' ') {
                e.preventDefault()
                inputRef.current?.click()
              }
            }}
          >
            <input
              ref={inputRef}
              type="file"
              accept={ACCEPT}
              className="sr-only"
              onChange={(e) => onFile(e.target.files?.[0])}
            />
            <span className="dropzone-label">
              Drop an image here or click to browse
            </span>
            <span className="hint">JPEG, PNG, WebP, GIF · max 20 MB</span>
          </div>

          {previewUrl && (
            <div className="preview-wrap">
              <img src={previewUrl} alt="Selected upload preview" className="preview" />
            </div>
          )}

          <div className="actions">
            <button
              type="button"
              className="btn primary"
              disabled={!file || loading}
              onClick={extract}
            >
              {loading ? 'Extracting…' : 'Extract text'}
            </button>
          </div>

          {error && (
            <p className="banner error" role="alert">
              {typeof error === 'string' ? error : JSON.stringify(error)}
            </p>
          )}
        </section>

        <section className="panel recent-panel" aria-label="Recent uploads">
          <h2 className="result-title">Recent uploads</h2>
          {/* {recentImages.length === 0 && (
            <p className="placeholder">Your last 4 extracted images appear here.</p>
          )} */}
          <div className="recent-grid">
            {recentImages.map((item) => (
              <button
                key={item.id}
                type="button"
                className="recent-tile"
                draggable
                onClick={() => onFile(item.file)}
                onDragStart={() => setDraggedRecentFile(item.file)}
                onDragEnd={() => setDraggedRecentFile(null)}
                title={`Use ${item.name}`}
              >
                <img src={item.preview} alt={item.name} />
                <span>{item.name}</span>
              </button>
            ))}
          </div>
        </section>

        <section className="panel result-panel" aria-live="polite">
          <h2 className="result-title">Extracted content</h2>
          {!text && !loading && (
            <p className="placeholder">Results appear here after extraction.</p>
          )}
          {loading && <p className="placeholder">Running OCR…</p>}
          {text && (
            <div className="markdown-body">
              <ReactMarkdown remarkPlugins={[remarkGfm]}>{text}</ReactMarkdown>
            </div>
          )}
        </section>
      </div>
    </div>
  )
}

export default App
