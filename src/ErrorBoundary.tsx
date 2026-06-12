import { Component, type ErrorInfo, type ReactNode } from 'react'

type Props = { children: ReactNode }
type State = { error: Error | null }

export default class ErrorBoundary extends Component<Props, State> {
  state: State = { error: null }

  static getDerivedStateFromError(error: Error): State {
    return { error }
  }

  componentDidCatch(error: Error, info: ErrorInfo) {
    console.error('UI crash:', error, info)
  }

  reset = () => this.setState({ error: null })

  render() {
    if (!this.state.error) return this.props.children
    const message = this.state.error.message ?? String(this.state.error)
    const stack = this.state.error.stack ?? ''
    return (
      <div className="flex h-screen w-screen items-center justify-center bg-[#0b0d10] p-6 text-[#e6e8eb]">
        <div className="w-full max-w-2xl rounded border border-rose-900/60 bg-rose-950/30 p-4">
          <div className="mb-2 text-sm font-medium text-rose-300">UI crashed — the app stayed alive.</div>
          <pre className="mb-3 max-h-48 overflow-auto rounded bg-[#0e1216] p-2 font-mono text-[11px] leading-snug text-rose-200">
{message}
{stack}
          </pre>
          <div className="flex gap-2 text-xs">
            <button
              className="rounded border border-[#1f2429] bg-[#13171b] px-3 py-1 hover:bg-[#1a1f24]"
              onClick={this.reset}
            >
              Retry
            </button>
            <button
              className="rounded border border-[#1f2429] bg-[#13171b] px-3 py-1 hover:bg-[#1a1f24]"
              onClick={() => location.reload()}
            >
              Reload page
            </button>
          </div>
          <div className="mt-3 text-[10px] text-[#7a8088]">
            Tip: most likely cause is a bad layer-param value. Check the chat for the last
            tool call, undo it, or hit “reset” in the chat panel.
          </div>
        </div>
      </div>
    )
  }
}
