import { useState } from 'react'
import Uploader from './components/Uploader'
import ResultCard from './components/ResultCard'
import { predictImage } from './api/predict'

export default function App() {
  const [result, setResult]     = useState(null)
  const [isLoading, setLoading] = useState(false)
  const [error, setError]       = useState(null)

  const handleImageSelect = async (file) => {
    setLoading(true)
    setError(null)
    setResult(null)

    try {
      const data = await predictImage(file)
      setResult(data)
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 to-blue-50 py-12 px-4">
      <div className="max-w-2xl mx-auto">

        {/* ── Header ───────────────────────────────────── */}
        <div className="text-center mb-10">
          <h1 className="text-4xl font-bold text-slate-800 tracking-tight">
            Brand <span className="text-blue-600">Vision</span>
          </h1>
          <p className="text-gray-500 mt-2 text-lg">
            Upload a product image to identify its brand and category
          </p>
        </div>

        {/* ── Upload area ───────────────────────────────── */}
        <div className="mb-6">
          <Uploader onImageSelect={handleImageSelect} isLoading={isLoading} />
        </div>

        {/* ── Loading spinner ───────────────────────────── */}
        {isLoading && (
          <div className="text-center py-8">
            <div className="inline-block w-8 h-8 border-4 border-blue-500 border-t-transparent rounded-full animate-spin" />
            <p className="text-gray-500 mt-3 font-medium">Analysing image…</p>
            <p className="text-gray-400 text-sm mt-1">
              First request may take up to 30 s (model loading)
            </p>
          </div>
        )}

        {/* ── Error message ─────────────────────────────── */}
        {error && (
          <div className="bg-red-50 border border-red-200 text-red-700 rounded-xl px-5 py-4 text-sm mb-6">
            ⚠️ {error}
          </div>
        )}

        {/* ── Result card ───────────────────────────────── */}
        {result && <ResultCard result={result} />}

      </div>
    </div>
  )
}
