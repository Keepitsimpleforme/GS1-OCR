import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import './App.css'

const ACCEPT = 'image/jpeg,image/png,image/webp,image/gif'
const REQUEST_TIMEOUT_MS = Number(import.meta.env.VITE_REQUEST_TIMEOUT_MS ?? 420000)
const HEALTH_TIMEOUT_MS = 7000
const SLOW_NOTICE_MS = 12000
const FIELD_CONFIG = [
  { label: 'MRP', keys: ['mrp'] },
  { label: 'GTIN', keys: ['gtin', 'GTIN'] },
  { label: 'FSSAI', keys: ['fssai'] },
  { label: 'Email', keys: ['email'] },
  { label: 'Phone', keys: ['phone'] },
  { label: 'Net Wt', keys: ['net_wt'] },
  { label: 'Nutritional', keys: ['nutritional', 'nutritable'] },
  { label: 'Ingredients', keys: ['ingredients'] },
]

function extractFirstTableHtml(text) {
  if (!text) return null
  const match = text.match(/<table\b[\s\S]*?<\/table>/i)
  return match ? match[0] : null
}

function parseTableRows(tableHtml) {
  if (!tableHtml || typeof window === 'undefined') return null
  const parser = new window.DOMParser()
  const doc = parser.parseFromString(tableHtml, 'text/html')
  const table = doc.querySelector('table')
  if (!table) return null

  const rows = Array.from(table.querySelectorAll('tr')).map((row) =>
    Array.from(row.querySelectorAll('th, td'))
      .map((cell) => cell.textContent?.trim() ?? '')
      .filter((cell) => cell !== ''),
  )
  const filteredRows = rows.filter((row) => row.length > 0)
  return filteredRows.length > 0 ? filteredRows : null
}

function App() {
  const [file, setFile] = useState(null)
  const [previewUrl, setPreviewUrl] = useState(null)
  const [recentImages, setRecentImages] = useState([])
  const [draggedRecentFile, setDraggedRecentFile] = useState(null)
  const [text, setText] = useState('')
  const [fields, setFields] = useState(null)
  const [model, setModel] = useState('')
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)
  const [slowNotice, setSlowNotice] = useState(false)
  const [dragOver, setDragOver] = useState(false)
  const [healthOk, setHealthOk] = useState(null)
  const inputRef = useRef(null)
  const tableRows = useMemo(() => {
    const fieldTable =
      extractFirstTableHtml(fields?.nutritional) ??
      extractFirstTableHtml(fields?.nutritable)
    const source = fieldTable ?? extractFirstTableHtml(text)
    return parseTableRows(source)
  }, [fields, text])

  useEffect(() => {
    const controller = new AbortController()
    const timeoutId = setTimeout(() => controller.abort(), HEALTH_TIMEOUT_MS)

    fetch('/health', { signal: controller.signal })
      .then((r) => {
        setHealthOk(r.ok)
        return r.json().catch(() => ({}))
      })
      .then((data) => {
        if (data.model) setModel(data.model)
      })
      .catch(() => setHealthOk(false))
      .finally(() => clearTimeout(timeoutId))

    return () => {
      clearTimeout(timeoutId)
      controller.abort()
    }
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
    setFields(null)
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
    setSlowNotice(false)
    setError('')
    setText('')
    setFields(null)
    const form = new FormData()
    form.append('file', file)
    const controller = new AbortController()
    const timeoutId = setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS)
    const slowNoticeId = setTimeout(() => setSlowNotice(true), SLOW_NOTICE_MS)
    try {
      const res = await fetch('/api/extract/product-info', {
        method: 'POST',
        body: form,
        signal: controller.signal,
      })
      const data = await res.json().catch(() => ({}))
      if (!res.ok) {
        setError(data.detail ?? res.statusText ?? 'Request failed')
        return
      }
      setText(data.text ?? '')
      const apiFields =
        data.fields && typeof data.fields === 'object' ? data.fields : null
      setFields(apiFields)
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
      if (err?.name === 'AbortError') {
        setError(
          `Extraction timed out after ${Math.floor(REQUEST_TIMEOUT_MS / 1000)}s. Try a clearer or smaller image.`,
        )
        return
      }
      setError(err.message ?? 'Network error')
    } finally {
      clearTimeout(timeoutId)
      clearTimeout(slowNoticeId)
      setSlowNotice(false)
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
              {loading ? 'Extracting…' : 'Extract fields'}
            </button>
          </div>

          {error && (
            <p className="banner error" role="alert">
              {typeof error === 'string' ? error : JSON.stringify(error)}
            </p>
          )}
          {loading && slowNotice && (
            <p className="banner warn" role="status">
              Extraction is taking longer than usual. We are still processing your image.
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
          <h2 className="result-title">Extracted product info</h2>
          {!fields && !loading && (
            <p className="placeholder">Results appear here after extraction.</p>
          )}
          {loading && <p className="placeholder">Extracting product fields…</p>}
          {fields && (
            <div className="field-list">
              {FIELD_CONFIG.map((field) => {
                const value =
                  field.keys
                    .map((key) => fields[key])
                    .find((entry) => entry !== null && entry !== undefined && entry !== '') ?? null
                const hasNutritionTable = field.label === 'Nutritional' && tableRows
                return (
                  <div
                    className={`field-row ${hasNutritionTable ? 'field-row-stacked' : ''}`}
                    key={field.label}
                  >
                    <p className="field-label">{field.label}</p>
                    {hasNutritionTable ? (
                      <div className="nutrition-table-wrap">
                        <table className="nutrition-table">
                          <tbody>
                            {tableRows.map((row, rowIdx) => (
                              <tr key={`${field.label}-${rowIdx}`}>
                                {row.map((cell, cellIdx) => {
                                  const CellTag = rowIdx === 0 ? 'th' : 'td'
                                  return <CellTag key={`${field.label}-${rowIdx}-${cellIdx}`}>{cell}</CellTag>
                                })}
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </div>
                    ) : value ? (
                      <p className="field-value">{value}</p>
                    ) : (
                      <p className="field-value field-empty">—</p>
                    )}
                  </div>
                )
              })}
            </div>
          )}
          {text && (
            <div className="raw-output">
              <p className="raw-output-title">Raw OCR text</p>
              <div className="markdown-body">
                <ReactMarkdown remarkPlugins={[remarkGfm]}>{text}</ReactMarkdown>
              </div>
            </div>
          )}
        </section>
      </div>
    </div>
  )
}

export default App
