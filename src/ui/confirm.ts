// Async confirm dialog. The Tauri webview (webkitgtk) can't run a synchronous
// window.confirm — it throws "dialog.confirm not allowed", so destructive
// actions silently no-op. Route through the Tauri dialog plugin in Tauri mode,
// fall back to the native dialog in the browser.

import { isTauri } from '../workspace/tauri-fs'

export async function confirmDialog(message: string, title = 'SpinoML'): Promise<boolean> {
  if (isTauri()) {
    const { confirm } = await import('@tauri-apps/plugin-dialog')
    return confirm(message, { title, kind: 'warning' })
  }
  return window.confirm(message)
}
