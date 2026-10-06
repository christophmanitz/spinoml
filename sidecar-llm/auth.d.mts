// Hand-written type declarations for sidecar-llm/auth.mjs.
// Mirrors the module's runtime exports exactly (no `any`):
//   * the env argument is a read-only view of `process.env`;
//   * `decide` returns a discriminated union that the HTTP layer matches on;
//   * the duplicate-header sentinel is a unique symbol so it cannot collide
//     with any string a header could plausibly contain.

export interface AuthConfig {
  token: string | null
  requireToken: boolean
  origins: readonly string[]
}

export const DEFAULT_ORIGINS: readonly string[]
export const CODE_UNAUTHORIZED: 'unauthorized'
export const CODE_BAD_ORIGIN: 'bad_origin'
export const CODE_BAD_HOST: 'bad_host'
export const DUPLICATE_HEADER: unique symbol
/** `/internal/mcp/<id>/list|call`: authenticated by the per-turn session secret, not the master token. */
export const MCP_ROUTE_RE: RegExp

export class AuthConfigError extends Error {
  constructor(message: string)
  override name: 'AuthConfigError'
}

export function loadConfig(env: Readonly<Record<string, string | undefined>>): AuthConfig

export function scrubEnv(env: NodeJS.ProcessEnv | Record<string, unknown>): void

export function checkHost(hostHeader: string | null | undefined): boolean

export function checkOrigin(
  originHeader: string | string[] | null | undefined,
  cfg: AuthConfig,
): boolean

export function tokenMatches(
  supplied: string | string[] | null | undefined,
  cfg: AuthConfig,
): boolean

export function sessionSecretMatches(
  supplied: string | string[] | null | undefined,
  secret: string | null | undefined,
): boolean

export function generateMcpSecret(): string

export type HeaderGetResult = string | null | typeof DUPLICATE_HEADER
export type HeaderGet = (name: string) => HeaderGetResult

export interface DecisionAllow {
  kind: 'ok'
  status: 0
  code: ''
  reason: ''
  origin: string | null
  tokenOk: boolean
}

export interface DecisionHealthLimited {
  kind: 'ok_health_limited'
  status: 0
  code: ''
  reason: ''
  origin: string | null
  tokenOk: false
}

export interface DecisionReject {
  kind: 'reject'
  status: number
  code: 'unauthorized' | 'bad_origin' | 'bad_host'
  reason: string
  origin: string | null
  tokenOk: false
}

export type Decision = DecisionAllow | DecisionHealthLimited | DecisionReject

export function decide(
  method: string,
  url: string,
  headersGet: HeaderGet,
  cfg: AuthConfig,
): Decision

export interface RequestLike {
  headers: Record<string, string | string[] | undefined>
  rawHeaders: string[]
}

export function decideFromHeaders(
  method: string,
  url: string,
  req: RequestLike,
  cfg: AuthConfig,
): Decision

export function healthBody(
  cfg: AuthConfig,
  fullBody: Record<string, unknown> | null,
  tokenOk: boolean,
): Record<string, unknown>

export function limitedHealthBody(): {
  ok: true
  auth: 'token'
  requiresAuth: true
  tokenOk: false
}

export function applyCorsHeaders(
  res: { setHeader(name: string, value: string): unknown },
  origin: string | null,
): void
