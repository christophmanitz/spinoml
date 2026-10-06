// Phase 77: per-launch sidecar authentication token (see docs/engineering/SIDECAR_AUTH.md).
//
// One 256-bit random token is generated at startup from the OS RNG and shared with
// the two managed (local) sidecars through the env of their spawn (NEVER argv, NEVER
// `std::env::set_var` — that would leak it to every other child of the GUI process).
// The webview reads it via the `sidecar_token` Tauri command so it can attach it to
// every request as `X-SpinoML-Token`.
//
// A second token exists for the remote (ssh-tunneled) torch sidecar: it's freshly
// generated per `ensure_remote_sidecar` run, delivered to the remote through ssh
// stdin (not argv, not the remote command string), and cleared the moment the
// session ends. The `SidecarTokens` struct deliberately has no `Debug`/`Display`/
// `Clone` derives — tokens must never be formatted, printed or duplicated.

use std::sync::Mutex;

use tauri::State;

/// 32 random bytes from the OS, encoded as 64 lowercase hex chars.
///
/// We pin getrandom to a `0.3.x` line because that's already in the dependency
/// closure (transitive) and its API (`getrandom::fill`) is what the brief calls
/// for. If the lock tree changes and 0.3 no longer resolves, fall back to the
/// `0.2.x` line that's also already in the lock file (`getrandom::getrandom`).
pub fn generate_token() -> Result<String, String> {
    let mut buf = [0u8; 32];
    getrandom::fill(&mut buf).map_err(|e| format!("OS RNG failed: {e}"))?;
    Ok(hex_lower(&buf))
}

fn hex_lower(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(bytes.len() * 2);
    for &b in bytes {
        out.push(HEX[(b >> 4) as usize] as char);
        out.push(HEX[(b & 0x0f) as usize] as char);
    }
    out
}

/// Holds the two kinds of token the app may need to hand out.
///
/// IMPORTANT: this struct intentionally does NOT derive `Debug`, `Display` or
/// `Clone`. A `Debug` derive would let any `format!("{:?}", tokens)` or
/// `dbg!(&tokens)` call leak both tokens into logs; `Clone` would let a
/// handler keep using a token after we've cleared the remote one. Every new
/// way to obtain a token value must go through `endpoint_token` (which
/// returns `Option<String>`) and never the struct itself.
pub struct SidecarTokens {
    local: String,
    remote: Mutex<Option<String>>,
}

impl SidecarTokens {
    /// Generate the local token at app launch. If the OS RNG is unavailable
    /// the call returns Err — the caller (Tauri `.manage(...)`) MUST surface
    /// that as a startup failure rather than falling back to "no token", which
    /// would silently downgrade the managed sidecars to `unauthenticated-dev`
    /// mode and defeat the whole point of this module.
    pub fn new() -> Result<Self, String> {
        Ok(Self { local: generate_token()?, remote: Mutex::new(None) })
    }

    /// Pure mapping: which endpoint needs which token (if any)?
    /// - "torch-local" / "llm" → the local token
    /// - "torch-remote"        → the current remote token, only if set
    /// - anything else         → Err (the webview never asks for anything
    ///   else; refusing loudly beats silently returning the wrong token)
    pub fn endpoint_token(&self, endpoint: &str) -> Result<Option<String>, String> {
        match endpoint {
            "torch-local" | "llm" => Ok(Some(self.local.clone())),
            "torch-remote" => {
                let g = self.remote.lock().map_err(|e| e.to_string())?;
                Ok(g.clone())
            }
            other => Err(format!("unknown sidecar endpoint: {other}")),
        }
    }

    /// Set the remote token. Called from `ensure_remote_sidecar` ONLY after
    /// the sidecar process has been spawned successfully — otherwise an
    /// `Option<String>` would briefly exist for a tunnel that never came up.
    pub fn set_remote(&self, token: String) -> Result<(), String> {
        let mut g = self.remote.lock().map_err(|e| e.to_string())?;
        *g = Some(token);
        Ok(())
    }

    /// Clear the remote token. Called whenever `current` is reset to None:
    /// a fresh `ensure_remote_sidecar` (before spawn), a `stop_remote_sidecar`,
    /// a reaped exited child in `remote_sidecar_status`, the
    /// `on_window_event(CloseRequested)` shutdown, or an error path. The
    /// `SidecarTokens` struct outlives the `RemoteSidecarState` (Tauri keeps
    /// both `.manage`d), so the token must be cleared explicitly here.
    pub fn clear_remote(&self) -> Result<(), String> {
        let mut g = self.remote.lock().map_err(|e| e.to_string())?;
        *g = None;
        Ok(())
    }
}

/// Tauri command: hand the webview the token for one sidecar endpoint, if any.
/// The webview uses it to add `X-SpinoML-Token` to every request. We return
/// `null` (None → JSON null) instead of `Err` for "no remote sidecar alive
/// right now" — that's a normal state, not an error.
#[tauri::command]
pub fn sidecar_token(state: State<SidecarTokens>, endpoint: String) -> Result<Option<String>, String> {
    state.endpoint_token(&endpoint)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn token_format_64_lowercase_hex() {
        for _ in 0..16 {
            let t = generate_token().expect("OS RNG must work in tests");
            assert_eq!(t.len(), 64, "token length must be 64 hex chars");
            assert!(
                t.chars().all(|c| matches!(c, '0'..='9' | 'a'..='f')),
                "token must be lowercase hex: {t:?}"
            );
        }
    }

    #[test]
    fn tokens_pairwise_distinct() {
        // 2000 calls — collision probability for 256-bit tokens is astronomically
        // small (birthday at ~2^128), so a collision here means our RNG is broken.
        let mut seen = std::collections::HashSet::new();
        for _ in 0..2000 {
            let t = generate_token().expect("OS RNG must work");
            assert!(seen.insert(t), "two tokens collided — RNG is broken");
        }
        assert_eq!(seen.len(), 2000);
    }

    #[test]
    fn endpoint_token_local_endpoints_share_local() {
        let s = SidecarTokens::new().expect("must construct");
        assert_eq!(
            s.endpoint_token("torch-local").unwrap().as_deref(),
            Some(s.local.as_str())
        );
        assert_eq!(
            s.endpoint_token("llm").unwrap().as_deref(),
            Some(s.local.as_str())
        );
    }

    #[test]
    fn endpoint_token_remote_none_before_set_errs_on_unknown() {
        let s = SidecarTokens::new().expect("must construct");
        assert!(matches!(s.endpoint_token("torch-remote"), Ok(None)));
        assert!(s.endpoint_token("bogus").is_err());
        assert!(s.endpoint_token("").is_err());
        assert!(s.endpoint_token("Torch-Local").is_err()); // case-sensitive on purpose
    }

    #[test]
    fn endpoint_token_remote_set_then_clear() {
        let s = SidecarTokens::new().expect("must construct");
        s.set_remote("abcdef".into()).unwrap();
        assert_eq!(s.endpoint_token("torch-remote").unwrap().as_deref(), Some("abcdef"));
        s.clear_remote().unwrap();
        assert!(matches!(s.endpoint_token("torch-remote"), Ok(None)));
    }

    #[test]
    fn generate_token_keeps_working_when_remote_is_set() {
        // Sanity: clearing/setting the remote field never touches the local field.
        let s = SidecarTokens::new().expect("must construct");
        let local_before = s.endpoint_token("llm").unwrap().unwrap();
        s.set_remote("x".into()).unwrap();
        s.clear_remote().unwrap();
        let local_after = s.endpoint_token("llm").unwrap().unwrap();
        assert_eq!(local_before, local_after);
    }
}
