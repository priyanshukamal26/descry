import { useCallback, useState, useEffect } from 'react'
import { useDropzone } from 'react-dropzone'
import { Rocket, Trash2, Zap, CloudUpload } from 'lucide-react'

export default function Uploader({ onImageSelect, isLoading, error }) {
  const [preview, setPreview] = useState(null)
  const [selectedFile, setSelectedFile] = useState(null)
  const [progress, setProgress] = useState(0)

  // Fake progress animation
  useEffect(() => {
    let interval;
    if (isLoading) {
      setProgress(0)
      interval = setInterval(() => {
        setProgress(p => (p < 95 ? p + 5 : p))
      }, 200)
    } else {
      setProgress(100)
    }
    return () => clearInterval(interval)
  }, [isLoading])

  const onDrop = useCallback((acceptedFiles) => {
    const file = acceptedFiles[0]
    if (!file) return
    const url = URL.createObjectURL(file)
    setPreview(url)
    setSelectedFile(file)
  }, [])

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { 'image/*': ['.jpg', '.jpeg', '.png', '.webp'] },
    multiple: false,
    maxSize: 10 * 1024 * 1024, // 10 MB
  })

  // Dimensions mock extraction for UI
  const [dimensions, setDimensions] = useState("AWAITING")
  useEffect(() => {
    if (preview) {
      const img = new Image()
      img.onload = () => setDimensions(`${img.width} X ${img.height}`)
      img.src = preview
    } else {
      setDimensions("AWAITING")
    }
  }, [preview])

  const handleProcess = () => {
    if (selectedFile && !isLoading) {
      onImageSelect(selectedFile)
    }
  }

  const handlePurge = () => {
    if (!isLoading) {
      setPreview(null)
      setSelectedFile(null)
      setDimensions("AWAITING")
    }
  }

  return (
    <div className="flex flex-col h-full animate-fade-in">
      <div className="flex justify-between items-start mb-8 gap-8">
        
        {/* Left Column: Title & Upload Area */}
        <div className="flex-1">
          <div className="mb-6">
            <h1 className="font-display font-black text-[80px] leading-[0.85] tracking-tighter uppercase text-black">
               UPLOAD
            </h1>
            <h1 className="font-display font-black text-[80px] leading-[0.85] tracking-tighter uppercase text-[var(--color-brand-purple)]">
               CORE_01
            </h1>
            <div className="inline-block mt-6 border-[4px] border-black bg-[var(--color-brand-green)] px-4 py-2 shadow-brutal pointer-events-none">
              <span className="font-black text-sm uppercase tracking-widest text-black">
                 STATUS: {isLoading ? 'PROCESSING_DATA' : preview ? 'READY_FOR_SCAN' : 'AWAITING INPUT'}
              </span>
            </div>
            
            {error && (
              <div className="mt-4 border-[4px] border-black bg-red-500 text-white px-4 py-2 shadow-brutal">
                 <span className="font-black text-sm uppercase tracking-widest">ERROR: {error}</span>
              </div>
            )}
          </div>

          <div
            {...getRootProps()}
            className={`relative border-[6px] border-black bg-white h-[320px] shadow-brutal transition-all mb-6 cursor-pointer flex flex-col items-center justify-center overflow-hidden
              ${isDragActive ? 'bg-[var(--color-brand-green)]/10 scale-[1.01]' : 'hover:translate-x-1 hover:translate-y-1 hover:shadow-[4px_4px_0px_#000]'}
              ${isLoading ? 'pointer-events-none opacity-80 backdrop-blur-sm grayscale' : ''}
            `}
          >
             <input {...getInputProps()} />
             
             {/* Viewfinder brackets */}
             <div className="absolute top-6 left-6 w-16 h-16 border-t-[6px] border-l-[6px] border-[var(--color-brand-green)]"></div>
             <div className="absolute top-6 right-6 w-16 h-16 border-t-[6px] border-r-[6px] border-[var(--color-brand-green)]"></div>
             <div className="absolute bottom-6 left-6 w-16 h-16 border-b-[6px] border-l-[6px] border-[var(--color-brand-green)]"></div>
             <div className="absolute bottom-6 right-6 w-16 h-16 border-b-[6px] border-r-[6px] border-[var(--color-brand-green)]"></div>

             {preview ? (
                <>
                  <img src={preview} alt="Upload Preview" className="absolute inset-x-12 inset-y-12 w-[calc(100%-6rem)] h-[calc(100%-6rem)] object-contain z-10" />
                  <div className="absolute top-8 right-8 bg-[var(--color-brand-green)] border-[3px] border-black px-3 py-1 z-20 shadow-[2px_2px_0px_#000]">
                    <span className="text-xs font-black uppercase">MATCH CONF: N/A</span>
                  </div>
                </>
             ) : (
                <div className="flex flex-col items-center text-center z-10 relative pointer-events-none">
                   <CloudUpload className="w-24 h-24 mb-4 text-[#e6d5a1]" strokeWidth={1} fill="#e6d5a1" />
                   <div className="font-black text-xl uppercase tracking-widest text-[#e6d5a1]">
                     UPLOAD PREVIEW
                   </div>
                   <p className="mt-4 font-bold text-gray-400 max-w-xs">{isDragActive ? 'DROP TO UPLOAD' : 'DRAG RAW IMAGE FILE HERE OR CLICK TO BROWSE'}</p>
                </div>
             )}
          </div>

          {/* Action Buttons */}
          <div className="grid grid-cols-2 gap-6">
            <button 
              onClick={handleProcess}
              disabled={!preview || isLoading}
              className={`brutal-btn-purple flex items-center justify-center gap-3 text-lg h-16
                ${!preview ? 'opacity-50 cursor-not-allowed hidden' : ''}`}
            >
              <Rocket className="w-6 h-6" /> PROCESS SCAN
            </button>
            <button 
              onClick={handlePurge}
              disabled={isLoading}
              className={`brutal-btn-white flex items-center justify-center gap-3 text-lg h-16
                ${!preview ? 'opacity-50 cursor-not-allowed hidden' : ''}`}
            >
               <Trash2 className="w-6 h-6" /> PURGE
            </button>
          </div>
        </div>


      </div>

      {/* Progress Bar (Visible during scan) */}
      {isLoading && (
        <div className="mt-auto animate-fade-in pt-8 border-t-[4px] border-black">
           <div className="flex justify-between items-end mb-2">
             <div className="font-display font-black text-xl italic uppercase">
                SCANNING_PROGRESS <span className="text-[var(--color-brand-green)] not-italic ml-2">{progress}%</span>
             </div>
             <div className="text-[10px] font-black uppercase tracking-widest text-gray-500">
                ESTIMATED COMPLETION: 4.2S
             </div>
           </div>
           <div className="h-8 border-[4px] border-black w-full bg-gray-200 flex p-0.5">
              <div 
                 className="h-full bg-[var(--color-brand-green)] transition-all duration-300 ease-out flex items-center overflow-hidden" 
                 style={{ width: `${progress}%` }}
              >
                 <div className="w-[200%] h-full opacity-20 bg-[repeating-linear-gradient(45deg,transparent,transparent_10px,#000_10px,#000_20px)] animate-[pan-bg_1s_linear_infinite]"></div>
              </div>
           </div>
        </div>
      )}
    </div>
  )
}
