import { useState } from 'react'
import Uploader from './components/Uploader'
import ResultCard from './components/ResultCard'
import Sidebar from './components/Sidebar'
import { predictImage } from './api/predict'
import { User, Search } from 'lucide-react'

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
      // Attach the local URL to display the 'SOURCE VISUAL'
      setResult({ ...data, imageFile: file, localUrl: URL.createObjectURL(file) })
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  const handleReset = () => setResult(null)

  return (
    <div className="flex min-h-screen bg-[var(--color-brand-bg)] w-full text-black font-sans selection:bg-[var(--color-brand-purple)] selection:text-white">
      <Sidebar />
      
      <main className="flex-1 flex flex-col h-screen overflow-y-auto relative">
        {/* Top Header */}
        <header className="h-[76px] border-b-[4px] border-black bg-white flex items-center justify-between px-10 sticky top-0 z-10 shrink-0">
           <div className="flex gap-8 font-display font-black text-lg uppercase">
              <span className="text-[var(--color-brand-purple)] underline decoration-[4px] underline-offset-[8px] cursor-pointer">SCANNER</span>
           </div>
           

        </header>

        {/* Content Body */}
        <div className="p-10 flex-1 w-full max-w-[1400px]">
          {!result && (
            <Uploader onImageSelect={handleImageSelect} isLoading={isLoading} error={error} />
          )}

          {result && !isLoading && (
            <ResultCard result={result} onReset={handleReset} />
          )}
        </div>
      </main>
    </div>
  )
}
