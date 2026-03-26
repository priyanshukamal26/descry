import { ScanLine } from 'lucide-react'

export default function Sidebar() {
  return (
    <aside className="w-72 border-r-[4px] border-black flex flex-col bg-[var(--color-brand-bg)] h-screen sticky top-0">
      {/* Brand Header */}
      <div className="h-[76px] border-b-[4px] border-black p-6 font-display font-black text-2xl tracking-tighter italic flex items-center justify-center bg-white shadow-none">
         DESCRY
      </div>
      


      {/* Navigation */}
      <nav className="mt-8 flex flex-col gap-3 px-6 flex-1">
        <a href="#" className="flex items-center gap-4 bg-[var(--color-brand-purple)] text-white brutal-border py-4 px-5 font-black uppercase text-sm shadow-brutal translate-x-1 translate-y-1">
          <ScanLine className="w-5 h-5" /> SCANNER
        </a>

      </nav>


    </aside>
  )
}
