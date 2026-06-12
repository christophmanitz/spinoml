import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import App from './App'
import ErrorBoundary from './ErrorBoundary'
import './index.css'
import './inference/store'
import './chat/store'
import './history/store'
import './workspace/store'
import { readAutosave, startAutosave } from './persistence/file'
import { useGraphStore } from './canvas/GraphStore'
import { useWorkspaceStore } from './workspace/store'

const ws = useWorkspaceStore.getState()
const activeFile = ws.activeFileId ? ws.entries[ws.activeFileId] : null
if (activeFile && activeFile.kind === 'file') {
  ws.openFile(activeFile.id)
} else {
  const restored = readAutosave()
  if (restored && restored.nodes.length > 1) useGraphStore.getState().loadSnapshot(restored)
}
startAutosave()

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <ErrorBoundary>
      <App />
    </ErrorBoundary>
  </StrictMode>,
)
