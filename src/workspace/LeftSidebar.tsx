import FileExplorer from './FileExplorer'
import DatasetExplorer from '../datasets/DatasetExplorer'
import ExperimentsExplorer from '../training/ExperimentsExplorer'
import { useSidebarStore } from './sidebarStore'

export default function LeftSidebar() {
  const tab = useSidebarStore((s) => s.tab)
  const setTab = useSidebarStore((s) => s.setTab)
  return (
    <div className="flex h-full min-h-0 flex-col">
      <div className="flex shrink-0 border-b border-[#1f2429] text-[11px]">
        <TabButton active={tab === 'files'} onClick={() => setTab('files')}>Files</TabButton>
        <TabButton active={tab === 'datasets'} onClick={() => setTab('datasets')}>Datasets</TabButton>
        <TabButton active={tab === 'experiments'} onClick={() => setTab('experiments')}>Experiments</TabButton>
      </div>
      <div className="min-h-0 flex-1">
        {tab === 'files' && <FileExplorer />}
        {tab === 'datasets' && <DatasetExplorer />}
        {tab === 'experiments' && <ExperimentsExplorer />}
      </div>
    </div>
  )
}

function TabButton({ active, onClick, children }: { active: boolean; onClick: () => void; children: React.ReactNode }) {
  return (
    <button
      onClick={onClick}
      className={`px-3 py-1.5 ${
        active ? 'border-b border-[var(--accent)] text-[#e6e8eb]' : 'text-[#6f767e] hover:text-[#9aa1a8]'
      }`}
    >
      {children}
    </button>
  )
}
