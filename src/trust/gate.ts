import { type CodeBlob, type CodeKind, hashBlob } from './codeBlobs'
import { type TrustOrigin, type TrustRecord } from './trustStore'

/** A code blob that did NOT pass the trust gate — the caller must not execute
 *  it and must surface a UI ask-for-approval before retrying. */
export type UntrustedBlob = CodeBlob & { sha256: string }

/** Hash every blob and return those the trust store does NOT know about,
 *  preserving the input order. Pure (no I/O except crypto). */
export async function findUntrusted(
  blobs: CodeBlob[],
  isTrusted: (sha256: string) => boolean,
): Promise<UntrustedBlob[]> {
  const out: UntrustedBlob[] = []
  for (const blob of blobs) {
    const sha256 = await hashBlob(blob.kind, blob.source)
    if (!isTrusted(sha256)) out.push({ ...blob, sha256 })
  }
  return out
}

/** Provenance row for run.json: every intentional-arbitrary-code blob that
 *  ran in this experiment, with the origin & timestamp of the human approval
 *  (or 'unrecorded' / null for anything that slipped through — the caller
 *  is supposed to refuse to run those). Snake_case on purpose: this object is
 *  written verbatim to JSON. */
export type CodeTrustEntry = {
  node: string
  path: string
  kind: CodeKind
  sha256: string
  origin: TrustOrigin | 'unrecorded'
  approved_at: string | null
}

export async function buildCodeTrustManifest(
  blobs: CodeBlob[],
  get: (sha256: string) => TrustRecord | undefined,
): Promise<CodeTrustEntry[]> {
  const out: CodeTrustEntry[] = []
  for (const blob of blobs) {
    const sha256 = await hashBlob(blob.kind, blob.source)
    const rec = get(sha256)
    out.push({
      node: blob.nodeId,
      path: blob.path,
      kind: blob.kind,
      sha256,
      origin: rec ? rec.origin : 'unrecorded',
      approved_at: rec ? rec.approvedAt : null,
    })
  }
  return out
}

export const UNTRUSTED_MESSAGE = (n: number) =>
  `${n} code block(s) not approved by you — nothing was executed.`
