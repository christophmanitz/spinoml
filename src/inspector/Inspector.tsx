import { useGraphStore } from '../canvas/GraphStore'
import { LAYERS, type FieldSpec } from '../layers/registry'

export default function Inspector() {
  const selectedNodeId = useGraphStore((s) => s.selectedNodeId)
  const node = useGraphStore((s) => s.nodes.find((n) => n.id === selectedNodeId))
  const updateNodeParams = useGraphStore((s) => s.updateNodeParams)
  const deleteNode = useGraphStore((s) => s.deleteNode)

  if (!node) {
    return (
      <div className="flex h-1/2 min-h-0 flex-col border-b border-[#1f2429] p-3 text-sm">
        <div className="mb-2 text-xs uppercase tracking-wide text-[#7a8088]">Inspector</div>
        <div className="text-xs text-[#7a8088]">Select a node to edit its parameters.</div>
      </div>
    )
  }

  const spec = LAYERS[node.data.layerType]
  const params = node.data.params

  return (
    <div className="flex h-1/2 min-h-0 flex-col border-b border-[#1f2429] p-3 text-sm">
      <div className="mb-2 flex items-center justify-between">
        <div className="text-xs uppercase tracking-wide text-[#7a8088]">Inspector</div>
        {node.id !== 'input' && (
          <button
            className="text-[10px] text-red-400 hover:text-red-300"
            onClick={() => deleteNode(node.id)}
          >
            delete
          </button>
        )}
      </div>

      <div className="mb-2 font-medium">{node.data.layerType}</div>
      <div className="mb-2 text-[10px] text-[#7a8088]">id: {node.id}</div>

      <div className="flex-1 overflow-y-auto pr-1">
        {spec?.fields.length === 0 && (
          <div className="text-xs text-[#7a8088]">No parameters.</div>
        )}
        {spec?.fields.map((field) => (
          <ParamField
            key={field.name}
            field={field}
            value={params[field.name] ?? field.default}
            onChange={(v) => updateNodeParams(node.id, { [field.name]: v })}
          />
        ))}
      </div>
    </div>
  )
}

function ParamField({
  field,
  value,
  onChange,
}: {
  field: FieldSpec
  value: unknown
  onChange: (v: unknown) => void
}) {
  return (
    <label className="mb-2 flex flex-col gap-1">
      <span className="text-[10px] uppercase tracking-wide text-[#7a8088]">{field.name}</span>
      <FieldInput field={field} value={value} onChange={onChange} />
    </label>
  )
}

function FieldInput({
  field,
  value,
  onChange,
}: {
  field: FieldSpec
  value: unknown
  onChange: (v: unknown) => void
}) {
  const baseClass =
    'rounded border border-[#1f2429] bg-[#0e1216] px-2 py-1 text-xs outline-none focus:border-[#3a4148]'

  switch (field.type) {
    case 'int':
    case 'float':
      return (
        <input
          type="number"
          className={baseClass}
          value={value as number}
          min={field.min}
          max={field.max}
          step={field.step ?? (field.type === 'int' ? 1 : 'any')}
          onChange={(e) => {
            const v = field.type === 'int' ? parseInt(e.target.value, 10) : parseFloat(e.target.value)
            if (!Number.isNaN(v)) onChange(v)
          }}
        />
      )
    case 'bool':
      return (
        <input
          type="checkbox"
          className="h-4 w-4 self-start accent-[#6ab7ff]"
          checked={value as boolean}
          onChange={(e) => onChange(e.target.checked)}
        />
      )
    case 'select':
      return (
        <select
          className={baseClass}
          value={value as string}
          onChange={(e) => onChange(e.target.value)}
        >
          {field.options.map((opt) => (
            <option key={opt} value={opt}>
              {opt}
            </option>
          ))}
        </select>
      )
    case 'tuple-int': {
      const tuple = (Array.isArray(value) ? value : [field.default[0], field.default[1]]) as number[]
      return (
        <div className="flex gap-1">
          {Array.from({ length: field.arity }).map((_, i) => (
            <input
              key={i}
              type="number"
              className={`${baseClass} w-full`}
              value={tuple[i] ?? 0}
              step={1}
              onChange={(e) => {
                const v = parseInt(e.target.value, 10)
                if (Number.isNaN(v)) return
                const next = [...tuple]
                next[i] = v
                onChange(next)
              }}
            />
          ))}
        </div>
      )
    }
    case 'shape': {
      const arr = (Array.isArray(value) ? value : field.default) as number[]
      return (
        <input
          type="text"
          className={baseClass}
          defaultValue={arr.join(', ')}
          onBlur={(e) => {
            const parts = e.target.value
              .split(/[,\s]+/)
              .map((s) => s.trim())
              .filter(Boolean)
              .map((s) => parseInt(s, 10))
            if (parts.every((n) => Number.isFinite(n))) onChange(parts)
          }}
        />
      )
    }
  }
}
