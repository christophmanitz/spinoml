const LAYER_GROUPS: { name: string; layers: string[] }[] = [
  { name: 'Conv', layers: ['Conv2d', 'Conv1d', 'ConvTranspose2d'] },
  { name: 'Linear', layers: ['Linear', 'Flatten'] },
  { name: 'Norm', layers: ['BatchNorm2d', 'LayerNorm', 'GroupNorm'] },
  { name: 'Activation', layers: ['ReLU', 'GELU', 'SiLU', 'Sigmoid', 'Tanh'] },
  { name: 'Pool', layers: ['MaxPool2d', 'AvgPool2d', 'AdaptiveAvgPool2d'] },
  { name: 'Regularize', layers: ['Dropout', 'Dropout2d'] },
  { name: 'Attention', layers: ['MultiheadAttention', 'TransformerEncoderLayer'] },
]

export default function Palette() {
  return (
    <aside className="w-[200px] shrink-0 overflow-y-auto border-r border-[#1f2429] p-2 text-sm">
      <div className="mb-2 text-xs uppercase tracking-wide text-[#7a8088]">Layers</div>
      {LAYER_GROUPS.map((group) => (
        <div key={group.name} className="mb-3">
          <div className="mb-1 text-xs font-medium text-[#9aa1a8]">{group.name}</div>
          <div className="flex flex-col gap-1">
            {group.layers.map((layer) => (
              <div
                key={layer}
                className="cursor-grab rounded border border-[#1f2429] bg-[#13171b] px-2 py-1 text-xs hover:border-[#3a4148] hover:bg-[#181d22]"
                draggable
                onDragStart={(e) => {
                  e.dataTransfer.setData('application/mlforge-layer', layer)
                  e.dataTransfer.effectAllowed = 'copy'
                }}
              >
                {layer}
              </div>
            ))}
          </div>
        </div>
      ))}
    </aside>
  )
}
