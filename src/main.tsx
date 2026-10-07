import './editor/monacoSetup'
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
import { startTrainingAutosaveToFile } from './training/graph/doc'
import { startDataAutosaveToFile } from './data/graph/doc'
import { startArchitectureDocSync } from './canvas/doc'
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

// Training + data canvases are file-bound (canvasdoc/CanvasFileGate): they reopen
// their bound .spinotrain/.spinodata on mount and autosave to it. No localStorage
// restore here — that produced orphan graphs not tied to any file.
startTrainingAutosaveToFile()
startDataAutosaveToFile()
// Mirror the architecture canvas's file binding to the workspace's active .spinoml
// so its header/chooser stay correct (the workspace owns load/save).
startArchitectureDocSync()

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <ErrorBoundary>
      <App />
    </ErrorBoundary>
  </StrictMode>,
)
