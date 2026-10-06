// Pure command/URL safety helpers for the LLM sidecar. No deps beyond Node.
//
// Three concerns, one module:
//   - posixQuote / splitArgs / quoteArgv: turn LLM-supplied argument text into
//     a list of argv tokens and re-quote them so the shell can never interpret
//     a token as an operator, substitution, glob or variable.
//   - checkDownloadUrl / isBlockedAddress / safeFetch: reject SSRF targets
//     (loopback, link-local, private ranges, metadata hosts, obfuscated IP
//     forms, IPv4-in-IPv6 embeddings, and hostnames that resolve to a blocked
//     address) before curl/fetch sees them.
//   - checkSshTarget: keep a target from being parsed as an ssh option.

import dns from 'node:dns'

// ── POSIX single-quote quoting ──────────────────────────────────────────────
// Every `'` becomes `'\''` so the result is always exactly one shell word.
export function posixQuote(s) {
  return "'" + String(s).replace(/'/g, "'\\''") + "'"
}

// ── argv tokenizer ──────────────────────────────────────────────────────────
// POSIX-like but deliberately inert: NO variable expansion, NO command
// substitution, NO globbing, NO operators. `;`, `&&`, `|`, `>`, `$(`, backticks
// are ordinary characters inside a token. Quotes only group/split; they never
// introduce interpretation.
const MAX_ARGS = 64
const MAX_INPUT = 4096

export function splitArgs(str) {
  if (str == null) return { ok: true, argv: [] }
  if (typeof str !== 'string') return { ok: false, error: 'args must be a string' }
  if (str === '') return { ok: true, argv: [] }
  if (str.length > MAX_INPUT) return { ok: false, error: `args too long (max ${MAX_INPUT} chars)` }
  if (str.indexOf('\u0000') !== -1) return { ok: false, error: 'args may not contain NUL' }
  if (str.indexOf('\n') !== -1 || str.indexOf('\r') !== -1) {
    return { ok: false, error: 'args may not contain newlines' }
  }

  const argv = []
  let token = ''
  let started = false
  let i = 0
  while (i < str.length) {
    const c = str[i]
    if (c === "'") {
      // Single quotes: fully literal until the closing quote.
      started = true
      const end = str.indexOf("'", i + 1)
      if (end === -1) return { ok: false, error: 'unbalanced single quote' }
      token += str.slice(i + 1, end)
      i = end + 1
      continue
    }
    if (c === '"') {
      // Double quotes: only \" and \\ are escapes; everything else literal.
      started = true
      i++
      let closed = false
      while (i < str.length) {
        const d = str[i]
        if (d === '\\') {
          const n = str[i + 1]
          if (n === '"' || n === '\\') { token += n; i += 2; continue }
          token += '\\'
          i++
          continue
        }
        if (d === '"') { closed = true; i++; break }
        token += d
        i++
      }
      if (!closed) return { ok: false, error: 'unbalanced double quote' }
      continue
    }
    if (c === '\\') {
      // Outside quotes a backslash escapes the next char; a trailing backslash
      // has nothing to escape, so it is kept literally.
      started = true
      if (i + 1 < str.length) { token += str[i + 1]; i += 2; continue }
      token += '\\'
      i++
      continue
    }
    if (c === ' ' || c === '\t') {
      if (started) { argv.push(token); token = ''; started = false }
      i++
      continue
    }
    token += c
    started = true
    i++
  }
  if (started) argv.push(token)
  if (argv.length > MAX_ARGS) return { ok: false, error: `too many args (max ${MAX_ARGS})` }
  return { ok: true, argv }
}

export function quoteArgv(argv) {
  return argv.map(posixQuote).join(' ')
}

// ── SSRF policy for download URLs ───────────────────────────────────────────
const ALLOWED_PORTS = new Set(['80', '443', '8080', '8443'])
const NAME_RE = /^(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?$/i

function ipv4Octets(host) {
  const m = /^(\d{1,3})\.(\d{1,3})\.(\d{1,3})\.(\d{1,3})$/.exec(host)
  if (!m) return null
  const oct = [Number(m[1]), Number(m[2]), Number(m[3]), Number(m[4])]
  if (oct.some((o) => o > 255)) return null
  return oct
}

function blockedV4(oct) {
  const [a, b, c] = oct
  if (a === 0) return true            // 0.0.0.0/8
  if (a === 10) return true           // 10/8
  if (a === 100 && b >= 64 && b <= 127) return true // 100.64/10
  if (a === 127) return true          // 127/8
  if (a === 169 && b === 254) return true // 169.254/16
  if (a === 172 && b >= 16 && b <= 31) return true // 172.16/12
  if (a === 192 && b === 0 && c === 0) return true // 192.0.0/24
  if (a === 192 && b === 168) return true // 192.168/16
  if (a === 198 && (b === 18 || b === 19)) return true // 198.18/15
  if (a >= 224) return true           // 224/4 + 240/4 (multicast + reserved)
  return false
}

function ipv6Groups(host) {
  const h = host.toLowerCase()
  if (h.indexOf(':') === -1) return null
  const parts = h.split('::')
  if (parts.length > 2) return null
  const toGroups = (s) => (s === '' ? [] : s.split(':'))
  const head = toGroups(parts[0])
  const tail = parts.length === 2 ? toGroups(parts[1]) : []
  let groups
  if (parts.length === 1) {
    if (head.length !== 8) return null
    groups = head
  } else {
    const missing = 8 - head.length - tail.length
    if (missing < 1) return null
    groups = [...head, ...Array(missing).fill('0'), ...tail]
  }
  const nums = groups.map((g) => (/^[0-9a-f]{1,4}$/.test(g) ? parseInt(g, 16) : NaN))
  if (nums.some((n) => Number.isNaN(n))) return null
  return nums
}

function blockedV6(groups) {
  if (groups.every((g) => g === 0)) return true            // ::
  // IPv4-mapped (::ffff:a.b.c.d) / IPv4-compatible (::a.b.c.d) with blocked v4.
  const mapped = groups.slice(0, 5).every((g) => g === 0) && groups[5] === 0xffff
  const compatible = groups.slice(0, 6).every((g) => g === 0)
  if (mapped || compatible) {
    const v4 = (groups[6] << 16) | groups[7]
    const oct = [(v4 >>> 24) & 0xff, (v4 >>> 16) & 0xff, (v4 >>> 8) & 0xff, v4 & 0xff]
    if (blockedV4(oct)) return true
  }
  if ((groups[0] & 0xffc0) === 0xfe80) return true         // fe80::/10
  if ((groups[0] & 0xfe00) === 0xfc00) return true         // fc00::/7
  if ((groups[0] & 0xff00) === 0xff00) return true         // ff00::/8
  return false
}

// Single blocklist for IP literals: IPv4 dotted-quad OR plain IPv6 (no
// brackets), as returned by dns.lookup / accepted by Node's URL class.
export function isBlockedAddress(ip) {
  // Anything not a strict IPv4 dotted-quad or a valid IPv6 literal (no brackets) is blocked.
  if (typeof ip !== 'string' || ip.length === 0) return true
  // Strict IPv4 dotted-quad: four decimal octets, no leading zeros (except single 0), each 0-255.
  const ipv4Strict = /^(?:0|[1-9]\d{0,2})(?:\.(?:0|[1-9]\d{0,2})){3}$/
  if (!ip.includes(':')) {
    if (!ipv4Strict.test(ip)) return true
    const oct = ipv4Octets(ip)
    if (!oct) return true
    return blockedV4(oct)
  }
  // Handle IPv6 literals possibly with a dotted-quad tail.
  let ipv6Str = ip
  if (ip.includes('.')) {
    const lastColon = ip.lastIndexOf(':')
    if (lastColon === -1) return true
    const prefix = ip.slice(0, lastColon)
    const ipv4Part = ip.slice(lastColon + 1)
    if (!ipv4Strict.test(ipv4Part)) return true
    const oct = ipv4Octets(ipv4Part)
    if (!oct) return true
    const high = ((oct[0] << 8) | oct[1]).toString(16)
    const low = ((oct[2] << 8) | oct[3]).toString(16)
    ipv6Str = `${prefix}:${high}:${low}`
  }
  const groups = ipv6Groups(ipv6Str)
  if (!groups) return true
  if (groups.every((g) => g === 0)) return true // ::
  // IPv4-mapped (::ffff:a.b.c.d) / IPv4-compatible (::a.b.c.d)
  const mapped = groups.slice(0, 5).every((g) => g === 0) && groups[5] === 0xffff
  const compatible = groups.slice(0, 6).every((g) => g === 0)
  if (mapped || compatible) {
    const v4 = (groups[6] << 16) | groups[7]
    const oct = [(v4 >>> 24) & 0xff, (v4 >>> 16) & 0xff, (v4 >>> 8) & 0xff, v4 & 0xff]
    if (blockedV4(oct)) return true
  }
  // NAT64 well-known prefix 64:ff9b::/96 — embedded IPv4 occupies groups 6+7.
  if (groups[0] === 0x0064 && groups[1] === 0xff9b && groups[2] === 0 && groups[3] === 0 && groups[4] === 0 && groups[5] === 0) {
    const v4 = (groups[6] << 16) | groups[7]
    const oct = [(v4 >>> 24) & 0xff, (v4 >>> 16) & 0xff, (v4 >>> 8) & 0xff, v4 & 0xff]
    if (blockedV4(oct)) return true
  }
  // NAT64 local-use prefix 64:ff9b:1::/48 — reject the whole /48.
  if (groups[0] === 0x0064 && groups[1] === 0xff9b && groups[2] === 0x0001) return true
  // 6to4 prefix 2002::/16 — embedded IPv4 occupies groups 1+2 (bits 16..47).
  if (groups[0] === 0x2002) {
    const v4 = (groups[1] << 16) | groups[2]
    const oct = [(v4 >>> 24) & 0xff, (v4 >>> 16) & 0xff, (v4 >>> 8) & 0xff, v4 & 0xff]
    if (blockedV4(oct)) return true
  }
  // Teredo prefix 2001:0000::/32 — reject the whole /32.
  if (groups[0] === 0x2001 && groups[1] === 0x0000) return true
  // Documentation prefix 2001:db8::/32.
  if (groups[0] === 0x2001 && groups[1] === 0xdb8) return true
  // Link-local, unique-local, multicast.
  if ((groups[0] & 0xffc0) === 0xfe80) return true // fe80::/10
  if ((groups[0] & 0xfe00) === 0xfc00) return true // fc00::/7
  if ((groups[0] & 0xff00) === 0xff00) return true // ff00::/8
  return false
}

export function checkDownloadUrl(url) {
  let u
  try { u = new URL(String(url)) } catch { return { ok: false, error: 'not a valid URL' } }

  if (u.protocol !== 'http:' && u.protocol !== 'https:') {
    return { ok: false, error: 'url must use http:// or https://' }
  }
  if (u.username || u.password) return { ok: false, error: 'url must not contain credentials' }
  if (!u.hostname) return { ok: false, error: 'url has no hostname' }
  if (u.port && !ALLOWED_PORTS.has(u.port)) {
    return { ok: false, error: `port ${u.port} is not allowed` }
  }

  let host = u.hostname.toLowerCase()
  let isIpv6 = false
  if (host.startsWith('[') && host.endsWith(']')) { isIpv6 = true; host = host.slice(1, -1) }

  if (isIpv6) {
    const groups = ipv6Groups(host)
    if (!groups) return { ok: false, error: 'invalid IPv6 host' }
    if (isBlockedAddress(host)) return { ok: false, error: 'blocked address range' }
    return { ok: true, url: u.href }
  }

  // IPv4 or numeric host
  if (/^[0-9.]+$/.test(host)) {
    // plain IPv4 literal
    if (isBlockedAddress(host)) return { ok: false, error: 'blocked address range' }
    return { ok: true, url: u.href }
  }

  // Check for obfuscated numeric host
  if (/^0x[0-9a-f]+$/i.test(host) || /^[0-9]+$/.test(host)) {
    return { ok: false, error: 'obfuscated numeric host' }
  }

  if (host === 'localhost' || host.endsWith('.localhost')) {
    return { ok: false, error: 'localhost is not allowed' }
  }
  if (host.endsWith('.local') || host.endsWith('.internal') || host.endsWith('.localdomain')) {
    return { ok: false, error: 'internal hostname is not allowed' }
  }
  if (!NAME_RE.test(host)) return { ok: false, error: 'hostname is not a registered name' }

  return { ok: true, url: u.href }
}

// ── ssh target policy ───────────────────────────────────────────────────────
// Mirrors validate_alias in src-tauri/src/ssh.rs (1..128, [A-Za-z0-9._@:-])
// plus the leading-dash rule so a target can never be read as an option.
export function checkSshTarget(s) {
  if (typeof s !== 'string') return { ok: false, error: 'target must be a string' }
  if (s.length < 1 || s.length > 128) return { ok: false, error: 'target must be 1..128 chars' }
  if (s.startsWith('-')) return { ok: false, error: 'target may not start with "-"' }
  if (!/^[A-Za-z0-9._@:-]+$/.test(s)) return { ok: false, error: 'target has invalid characters' }
  return { ok: true }
}

// ── Safe HTTP fetch with SSRF policy + manual redirect handling ─────────────
// For every hop (including the first):
//   1. Run checkDownloadUrl (scheme / port / literal IP block).
//   2. If the host is a name (not an IP literal), resolve it via `lookup` and
//      reject the URL if ANY returned address is blocked by isBlockedAddress.
//   3. Fetch with redirect:'manual' so redirects are visible to us.
//   4. On 301/302/303/307/308 take the Location header, resolve it relative to
//      the current URL, and loop. On the (maxRedirects+1)-th redirect we throw.
//   5. Any other status is returned to the caller.
//
// KNOWN LIMITATION: a DNS-rebinding race between our lookup and the connection
// is still possible (lookup says public, the connection lands on a different
// answer). The remote curl branch in main.mjs cannot be resolved locally and
// is intentionally out of scope.
export async function safeFetch(url, opts = {}) {
  const {
    lookup = dns.promises.lookup,
    fetchImpl = globalThis.fetch,
    maxRedirects = 5,
    timeoutMs = 300_000,
  } = opts
  let current
  try { current = new URL(String(url)) }
  catch { throw new Error('url rejected: not a valid URL') }

  let hops = 0
  while (true) {
    const chk = checkDownloadUrl(current.toString())
    if (!chk.ok) throw new Error(`url rejected: ${chk.error}`)

    let host = current.hostname
    if (host.startsWith('[') && host.endsWith(']')) host = host.slice(1, -1)

    if (host.includes(':') || /^[0-9.]+$/.test(host)) {
      if (isBlockedAddress(host)) throw new Error(`url rejected: ${host} is a blocked address`)
    } else {
      let addrs
      try { addrs = await lookup(host, { all: true, verbatim: true }) }
      catch { throw new Error(`url rejected: ${host} DNS lookup failed`) }
      if (!addrs || addrs.length === 0) {
        throw new Error(`url rejected: ${host} resolves to no address`)
      }
      for (const a of addrs) {
        if (isBlockedAddress(a.address)) {
          throw new Error(`url rejected: ${host} resolves to a blocked address ${a.address}`)
        }
      }
    }

    const resp = await fetchImpl(current.toString(), {
      redirect: 'manual',
      signal: AbortSignal.timeout(timeoutMs),
    })

    if ([301, 302, 303, 307, 308].includes(resp.status)) {
      const loc = resp.headers.get('location')
      if (!loc) throw new Error('url rejected: redirect without location')
      if (hops + 1 > maxRedirects) throw new Error('too many redirects')
      current = new URL(loc, current)
      hops++
      continue
    }
    return resp
  }
}
