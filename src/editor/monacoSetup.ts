// Bundled Monaco for `@monaco-editor/react` — no CDN at runtime.
//
// R052 / LIMITATIONS §2: the app origin holds full Tauri IPC (file writes, ssh,
// sidecar_token), so loading the editor runtime from cdn.jsdelivr.net was both
// an offline break (HPC login nodes, firewalls) and an XSS-to-IPC bridge. This
// module is imported ONCE, as the first import of `src/main.tsx`, so
// `loader.config({ monaco })` is set before any of the 7 editors can mount.
//
// Everything here is served from 'self' (CSP script-src 'self'); the workers
// are Vite `?worker` chunks (CSP worker-src 'self' blob:).
//
// monaco-editor 0.57 reorganized its ESM entry points (0.56 changelog): the
// old `edcore.main` is now `monaco-editor/editor` (API) + per-feature /
// `features/register.all`. Language *definitions* are separate from the
// CSS/HTML/JSON/TypeScript language *services*; only the definitions the app
// actually uses are imported below.

import * as monaco from 'monaco-editor/editor'
import { loader } from '@monaco-editor/react'
import EditorWorker from 'monaco-editor/editor/editor.worker?worker'
import JsonWorker from 'monaco-editor/languages/features/json/json.worker?worker'

// All editor features (find, folding, bracket matching, tokenization, ...) —
// the 0.57 equivalent of the pre-0.56 `edcore.main`.
import 'monaco-editor/features/register.all'

// Languages: `python` (all 6 code panels + CodeField) plus FileViewerModal's
// LANG map (javascript, typescript, markdown, shell, yaml, ini, rust, html,
// css, sql, xml; `cpp` also registers `.c`/`.h`). These are tokenizers only —
// the TypeScript/CSS/HTML language services and their workers are NOT bundled.
import 'monaco-editor/languages/definitions/python/register'
import 'monaco-editor/languages/definitions/javascript/register'
import 'monaco-editor/languages/definitions/typescript/register'
import 'monaco-editor/languages/definitions/markdown/register'
import 'monaco-editor/languages/definitions/shell/register'
import 'monaco-editor/languages/definitions/yaml/register'
import 'monaco-editor/languages/definitions/ini/register'
import 'monaco-editor/languages/definitions/rust/register'
import 'monaco-editor/languages/definitions/html/register'
import 'monaco-editor/languages/definitions/css/register'
import 'monaco-editor/languages/definitions/sql/register'
import 'monaco-editor/languages/definitions/xml/register'
import 'monaco-editor/languages/definitions/cpp/register'

// JSON has no `languages/definitions` entry in 0.57 (it lives entirely in the
// feature, definition + validation + worker). It is registered with its worker
// rather than degraded to plaintext, so `.json` files still highlight.
import 'monaco-editor/languages/features/json/register'

// The editor worker serves every label except 'json'; the bundled JSON worker
// serves validation/formatting. Both are same-origin Vite chunks.
self.MonacoEnvironment = {
  getWorker(_workerId: string, label: string) {
    if (label === 'json') return new JsonWorker()
    return new EditorWorker()
  },
}

loader.config({ monaco })
