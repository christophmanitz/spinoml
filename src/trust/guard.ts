// Phase 43 — the execution gate. Every place that ships generated model code to
// a sidecar or to the training executor must ask this module first. Only
// human-initiated UI events (a later block) may approve a blob; stores never do.

import { collectCodeBlobs } from './codeBlobs'
import { findUntrusted, UNTRUSTED_MESSAGE, type UntrustedBlob } from './gate'
import { trust } from './trustStore'

/** Thrown when a graph (or frozen snapshot) carries code that has no local
 *  human approval. `blobs` names every offending block so the UI can ask. */
export class UntrustedCodeError extends Error {
  readonly blobs: UntrustedBlob[]

  constructor(blobs: UntrustedBlob[]) {
    super(UNTRUSTED_MESSAGE(blobs.length))
    this.name = 'UntrustedCodeError'
    this.blobs = blobs
  }
}

/** Hash every intentional-arbitrary-code blob in `nodes` and return the ones the
 *  trust store does not know. `nodes` accepts live React-Flow nodes or the flat
 *  nodes of a parsed GraphSnapshot (collectCodeBlobs normalizes both). */
export async function listUntrusted(
  nodes: Parameters<typeof collectCodeBlobs>[0],
  isTrusted: (sha256: string) => boolean = trust.isTrusted,
): Promise<UntrustedBlob[]> {
  const blobs = collectCodeBlobs(nodes)
  return findUntrusted(blobs, isTrusted)
}

/** Reject when ANY blob in the graph is untrusted — fail closed, so a single
 *  unapproved Custom/DataOp block blocks ALL execution from that graph. */
export async function assertTrusted(
  nodes: Parameters<typeof collectCodeBlobs>[0],
  isTrusted: (sha256: string) => boolean = trust.isTrusted,
): Promise<void> {
  const untrusted = await listUntrusted(nodes, isTrusted)
  if (untrusted.length > 0) throw new UntrustedCodeError(untrusted)
}
