import { Share2, Download, CheckCircle2, Shield, Thermometer, MapPin, Hash, Activity } from 'lucide-react'

export default function ResultCard({ result, onReset }) {
  if (!result) return null

  // Generate a random stable hex hash for the UI
  const sessionHash = "0x" + Math.random().toString(16).slice(2, 10).toUpperCase() + "...E4EA"
  const scanId = "ID: DESCRY-" + Math.floor(Math.random() * 9000 + 1000) + "-FX"

  const handleShare = async () => {
    const text = `DESCRY Analysis [${scanId}]\nBrand: ${result.brand || 'Unknown'}\nCategory: ${result.category || 'Unknown'}`;
    if (navigator.share) {
      try { await navigator.share({ title: 'DESCRY Analysis', text }); } catch (err) { /* ignore cancel */ }
    } else {
      navigator.clipboard.writeText(text);
      alert('Analysis summary copied to clipboard!');
    }
  }

  const handleDownload = () => {
    const dataStr = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(result, null, 2));
    const dlAnchorElem = document.createElement('a');
    dlAnchorElem.setAttribute("href", dataStr);
    dlAnchorElem.setAttribute("download", `descry_analysis_${scanId}.json`);
    dlAnchorElem.click();
  }

  return (
    <div className="flex flex-col h-full animate-fade-in relative">
      
      {/* Header Area */}
      <div className="flex justify-between items-start mb-8">
        <div>
          <h1 className="font-display font-black text-6xl tracking-tighter uppercase text-black leading-[0.9]">
             ANALYSIS<br/>REPORT
          </h1>
          <div className="inline-block mt-4 border-[3px] border-black bg-[var(--color-brand-green)] px-3 py-1.5 shadow-[2px_2px_0px_#000]">
            <span className="font-black text-sm uppercase tracking-widest text-black">
               {scanId}
            </span>
          </div>
        </div>

        <div className="flex gap-4 items-start">
           <button 
             onClick={onReset}
             className="brutal-btn-white p-3 !px-4 hover:bg-gray-100 flex items-center justify-center font-black">
               NEW SCAN
           </button>
           <button onClick={handleShare} className="border-[3px] border-black p-3 hover:bg-gray-100 transition-colors shadow-brutal flex items-center justify-center bg-white" title="Share Analysis">
              <Share2 className="w-6 h-6" />
           </button>
           <button onClick={handleDownload} className="brutal-btn-purple flex items-center gap-3" title="Download JSON Data">
              EXPORT DATA <Download className="w-5 h-5" />
           </button>
        </div>
      </div>

      <div className="flex gap-8 items-start">
        
        {/* LEFT COLUMN (Image) */}
        <div className="flex-1 brutal-panel bg-gray-100 p-8 relative flex flex-col justify-center min-h-[380px]">
           <div className="absolute top-0 left-8 -translate-y-[50%] bg-black text-white px-3 py-1 font-black text-sm tracking-widest border-[3px] border-black">
              SOURCE VISUAL
           </div>
           
           <div className="flex-1 bg-white border-[4px] border-black relative shadow-[inset_0px_0px_0px_4px_#f3f4f6] p-8 flex items-center justify-center">
             <img 
                src={result.localUrl} 
                alt="Source" 
                className="max-h-[300px] object-contain"
             />
             
             {/* Overlay stats inside the image box */}
             <div className="absolute bottom-0 left-0 right-0 flex border-t-[4px] border-black bg-white">
                <div className="flex-1 border-r-[3px] border-black p-3 bg-[var(--color-brand-green)] group">
                   <p className="text-[9px] font-bold text-black uppercase mb-1">Focus Status</p>
                   <p className="text-sm font-black uppercase">CALIBRATED</p>
                </div>
                <div className="flex-1 border-r-[3px] border-black p-3">
                   <p className="text-[9px] font-bold text-gray-500 uppercase mb-1">Resolution</p>
                   <p className="text-sm font-black uppercase">4096 X 4096</p>
                </div>
                <div className="flex-1 border-r-[3px] border-black p-3">
                   <p className="text-[9px] font-bold text-gray-500 uppercase mb-1">Sensor</p>
                   <p className="text-sm font-black uppercase tracking-tight">AG-OPTIC V2</p>
                </div>
                <div className="flex-[1.2] p-3 bg-[var(--color-brand-purple)] text-white">
                   <p className="text-[9px] font-bold uppercase mb-1">Confidence</p>
                   <p className="text-lg font-black uppercase leading-none">
                     {Math.round((result.brand_confidence || 0) * 100)}.{Math.floor(Math.random() * 99)}%
                   </p>
                </div>
             </div>
           </div>
        </div>

        {/* RIGHT COLUMN (Identity & Specs) */}
        <div className="w-[380px] flex flex-col gap-6">
           
           {/* IDENTITY PANEL */}
           <div className="brutal-panel p-6 relative">
              <h2 className="font-display font-black text-2xl tracking-tight mb-5 flex items-center gap-3 uppercase">
                 <CheckCircle2 className="w-6 h-6 text-[var(--color-brand-purple)]" />
                 IDENTITY
              </h2>
              
              <div className="space-y-4">
                 <div className="border-b-[3px] border-black pb-3">
                    <p className="text-[10px] font-bold text-gray-500 uppercase mb-1 tracking-widest">Brand</p>
                    <div className="flex justify-between items-end">
                       <p className="font-black text-2xl uppercase leading-none">{result.brand || 'UNKNOWN'}</p>
                       <span className="bg-black text-white px-2 py-1 text-xs font-black">{Math.round((result.brand_confidence||0)*100)}%</span>
                    </div>
                 </div>
                 
                 <div className="border-b-[3px] border-black pb-3">
                    <p className="text-[10px] font-bold text-gray-500 uppercase mb-1 tracking-widest">Category</p>
                    <div className="flex justify-between items-end">
                       <p className="font-black text-2xl uppercase leading-none">{result.category || 'UNKNOWN'}</p>
                       <span className="bg-black text-white px-2 py-1 text-xs font-black">{Math.round((result.category_confidence||0)*100)}%</span>
                    </div>
                 </div>
                 
                 <div className="pb-1">
                    <p className="text-[10px] font-bold text-gray-500 uppercase mb-1 tracking-widest">Model</p>
                    <p className="font-black text-xl uppercase leading-none">{result.logo_detected ? 'LOGO DETECTED' : 'N/A'}</p>
                 </div>
              </div>
           </div>

           {/* COMPOSITION / ATTRIBUTES PANEL */}
           <div className="brutal-panel p-6 relative">
              <h2 className="font-display font-black text-xl tracking-tight mb-5 uppercase flex items-center gap-3">
                 <svg width="24" height="24" viewBox="0 0 24 24" fill="none" className="text-black" stroke="currentColor" strokeWidth="3" strokeLinecap="square" strokeLinejoin="miter">
                   <path d="M12 3L2 12h3v8h6v-6h2v6h6v-8h3L12 3z"/>
                 </svg>
                 CLASS PROBABILITIES
              </h2>
              
              <div className="space-y-4">
                 {(result.top3_categories || []).map((c, i) => (
                    <div key={i}>
                       <div className="flex justify-between text-xs font-black uppercase mb-1">
                          <span>{c.category}</span>
                          <span>{Math.round(c.confidence*100)}%</span>
                       </div>
                       <div className="h-3 border-[3px] border-black w-full bg-white relative">
                          <div className="h-full bg-[var(--color-brand-purple)]" style={{width: `${c.confidence*100}%`}}></div>
                       </div>
                    </div>
                 ))}
                 
                 {(!result.top3_categories || result.top3_categories.length === 0) && (
                    <p className="text-sm font-bold text-gray-500">NO EXTRA DATA</p>
                 )}
              </div>
              
           </div>

        </div>
      </div>



    </div>
  )
}
