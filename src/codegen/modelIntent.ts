// The LLM half of "explain this model": a concrete, grounded explanation of WHAT
// the model does AND what happens to the data as it flows through it. Reuses the
// chat sidecar's streaming endpoint but deliberately does NOT touch the chat
// store/history — the result is shown only in the Explain panel. Fails soft:
// callers treat a throw / empty string as "explanation unavailable" and fall
// back to the deterministic flow skeleton.

import { streamChat, type ChatRequest } from '../chat/client'
import { useGraphStore } from '../canvas/GraphStore'
import { getCurrentLlmRequest } from '../chat/providerStore'
import type { ModelExplanation } from './explain'

const PROMPT = [
  'Du bist Teil von SpinoML. Erkläre KONKRET und verständlich, was dieses PyTorch-Modell tut,',
  'sodass jemand den Datenfluss UND den Zweck jeder Stufe wirklich versteht — nicht nur den Typ.',
  'Stütze dich AUSSCHLIESSLICH auf den unten gezeigten Datenfluss (mit echten Formen) und den Code;',
  'erfinde nichts dazu. Beziehe dich auf die tatsächlichen Schichten und Dimensionen.',
  'Antworte als knappes deutsches Markdown mit GENAU diesen drei Abschnitten:',
  '',
  '**Kurz:** Ein Satz — Architektur-Typ + Zweck.',
  '',
  '**Was im Modell passiert:** 3–6 Stichpunkte, die den Datenfluss in sinnvollen BLÖCKEN erklären',
  '(nicht stur Schicht für Schicht). Pro Punkt: was mit den Daten geschieht UND wozu, mit den',
  'konkreten Formen wo sie helfen — z.B. „Drei Conv-Blöcke extrahieren zunehmend abstraktere',
  'Bildmerkmale und halbieren je die Auflösung → [N, 64, 28, 28]". Erkläre auch Verzweigungen,',
  'Fusion (Concat/Add), Pooling/Readouts und Köpfe, falls vorhanden.',
  '',
  '**Ausgabe:** Was der Ausgabetensor bedeutet (z.B. „Logits über 10 Klassen" oder „ein Regressionswert").',
  '',
  'KEINE Tools, KEIN Vorwort, KEINE Wiederholung dieser Anleitung — nur das Markdown.',
].join('\n')

/** Detailed, markdown-formatted model explanation grounded in the deterministic
 *  data-flow skeleton (exp.flowText) + the generated code. */
export async function fetchModelExplanation(exp: ModelExplanation, signal?: AbortSignal): Promise<string> {
  const { nodes, edges } = useGraphStore.getState()
  if (nodes.length === 0) return ''

  const inputs = nodes
    .filter((n) => n.data.layerType === 'Input' || n.data.layerType === 'Graph')
    .map((n) => ({
      id: n.id,
      name: String(n.data.params.name ?? 'x'),
      shape: (n.data.inferredOutputShape ?? (n.data.params.shape as number[]) ?? []) as number[],
    }))

  const user = [
    PROMPT,
    '',
    'DATENFLUSS (deterministisch abgeleitet, mit inferierten Formen):',
    '```',
    exp.flowText,
    '```',
    '',
    'GENERIERTER CODE:',
    '```python',
    exp.code,
    '```',
  ].join('\n')

  const req: ChatRequest = {
    user,
    messages: [],
    graph: {
      input_shape: inputs[0]?.shape ?? [1, 1],
      inputs,
      nodes: nodes.map((n) => ({ id: n.id, layerType: n.data.layerType, params: n.data.params })),
      edges: edges.map((e) => ({ source: e.source, target: e.target })),
    },
    llm: getCurrentLlmRequest(),
    // Pure explanation turn — never document it to the lab notebook.
    docMode: 'off',
  }

  let out = ''
  await streamChat(
    req,
    (ev) => { if (ev.type === 'text') out += ev.value },
    signal,
  )
  return out.trim()
}
