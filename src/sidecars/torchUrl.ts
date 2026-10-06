// Sidecar URL picker. Local workspace: the locally-spawned sidecar on
// 127.0.0.1:7421. Remote (SSH) workspace + Phase 12b sidecar running: an
// ssh-tunnelled local port (REMOTE_LOCAL_PORT = 7424) that forwards to the
// remote sidecar's 7421 on the HPC. We keep the laptop's local sidecar
// running too — it stays available for any "open a quick local file"
// flow without juggling ports.

import { isRemoteActive } from '../connections/store'
import { sidecarFetch, type SidecarEndpoint } from './auth'

const LOCAL = 'http://127.0.0.1:7421'
const REMOTE_TUNNEL = 'http://127.0.0.1:7424'

export function currentTorchUrl(): string {
  return isRemoteActive() ? REMOTE_TUNNEL : LOCAL
}

/** Which `sidecar_token` endpoint the current torch URL belongs to. */
export function currentTorchEndpoint(): SidecarEndpoint {
  return isRemoteActive() ? 'torch-remote' : 'torch-local'
}

/** Authenticated request against the currently-selected torch sidecar. */
export function torchFetch(path: string, init?: RequestInit): Promise<Response> {
  return sidecarFetch(currentTorchEndpoint(), currentTorchUrl() + path, init)
}
