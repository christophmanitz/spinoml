import { useMemo } from 'react'

import { useTrainingGraphStore } from './store'
import { compileTrainingGraph } from '../../codegen/trainingGenerator'
import { useTrainingStore } from '../store'

// File open/save/switch live in the CanvasFileGate header and the generated
// script in the bottom Code panel. This top bar carries the run action (like the
// data canvas's "Pipeline ausführen") + clear + an empty-state hint — so the run
// button sits in the SAME place on both canvases.
const BTN = 'rounded border border-[#1f2429] bg-[#13171b] px-2 py-1 text-[11px] text-[#9aa1a8] hover:border-[#3a4148] hover:bg-[#1a1f24] hover:text-[#e6e8eb]'

export default function TrainingGraphBar() {
  const nodes = useTrainingGraphStore((s) => s.nodes)
  const edges = useTrainingGraphStore((s) => s.edges)
  const resetGraph = useTrainingGraphStore((s) => s.resetGraph)
  const openNewRun = useTrainingStore((s) => s.openNewRun)

  const compile = useMemo(
    () => compileTrainingGraph({
      nodes: nodes.map((n) => ({ id: n.id, trainingType: n.data.trainingType, params: n.data.params })),
      edges: edges.map((e) => ({ source: e.source, target: e.target })),
    }),
    [nodes, edges],
  )

  // Compile the graph and hand the plan to the New-Run dialog pre-filled, so the
  // graph-driven launch still gets the dialog's backend/SLURM/sweep/resume knobs.
  function launch() {
    if (!compile.plan) return
    const plan = compile.plan
    openNewRun({
      label: plan.modelRelpath.split('/').pop()!.replace(/\.spinoml$/i, ''),
      modelRelpath: plan.modelRelpath,
      datasetRelpath: plan.datasetRelpath,
      targetColumn: plan.target,
      featureColumns: plan.features,
      training: plan.training,
    })
  }

  return (
    <div className="flex flex-col gap-1 rounded border border-[#1f2429] bg-[#0e1216]/90 p-1.5 backdrop-blur">
      <div className="flex items-center gap-1">
        <button
          onClick={launch}
          disabled={!compile.ok}
          className="rounded bg-[var(--accent-sel)] px-2 py-1 text-[11px] text-[var(--accent)] hover:bg-[var(--accent-sel-hover)] disabled:cursor-not-allowed disabled:opacity-40"
          title="Aus diesem Graph einen Run vorbereiten — öffnet den Dialog mit Backend/SLURM, Sweep und Resume"
        >▶ Run vorbereiten…</button>
        <button onClick={resetGraph} className={BTN} title="Alle Knoten entfernen">Leeren</button>
      </div>
      {nodes.length === 0 && (
        <div className="max-w-xs text-[10px] text-[#6f767e]">
          Leerer Graph — zieh Knoten aus der Palette: DatasetSource, ModelSource, Loss, Optimizer, TrainLoop.
        </div>
      )}
    </div>
  )
}
