use crate::{app::AppState, error::ApiError};
use axum::{
    extract::{Request, State},
    middleware::Next,
    response::{IntoResponse, Response},
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Role {
    /// Feeds transactions in (core banking / payment switch).
    Ingest,
    /// Triages alerts and works cases.
    Analyst,
    /// Rules, watchlists, SAR filing (MLRO / compliance officer).
    Admin,
}

#[derive(Clone, Debug)]
pub struct Actor {
    pub name: String,
    pub role: Role,
}

impl Actor {
    pub fn require_ingest(&self) -> Result<(), ApiError> {
        match self.role {
            Role::Ingest | Role::Admin => Ok(()),
            _ => Err(ApiError::Forbidden("role may not submit transactions".into())),
        }
    }
    pub fn require_triage(&self) -> Result<(), ApiError> {
        match self.role {
            Role::Analyst | Role::Admin => Ok(()),
            _ => Err(ApiError::Forbidden("analyst role required".into())),
        }
    }
    pub fn require_admin(&self) -> Result<(), ApiError> {
        match self.role {
            Role::Admin => Ok(()),
            _ => Err(ApiError::Forbidden("admin role required".into())),
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct Keys(pub Vec<(String, Role, String)>);

impl Keys {
    /// `AML_API_KEYS="key1:admin:alice,key2:analyst:bob,key3:ingest:core-banking"`
    pub fn parse(s: &str) -> anyhow::Result<Keys> {
        let mut out = vec![];
        for item in s.split(',').map(str::trim).filter(|x| !x.is_empty()) {
            let parts: Vec<&str> = item.splitn(3, ':').collect();
            if parts.len() != 3 || parts[0].len() < 16 {
                anyhow::bail!("AML_API_KEYS entries must be key:role:name with a key of at least 16 characters");
            }
            let role = match parts[1] {
                "admin" => Role::Admin,
                "analyst" => Role::Analyst,
                "ingest" => Role::Ingest,
                other => anyhow::bail!("unknown role '{other}' (admin|analyst|ingest)"),
            };
            out.push((parts[0].to_string(), role, parts[2].to_string()));
        }
        Ok(Keys(out))
    }
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
    pub fn lookup(&self, presented: &str) -> Option<Actor> {
        let mut found = None;
        for (k, role, name) in &self.0 {
            // compare every key so timing does not reveal which (or how much of one) matched
            if ct_eq(k.as_bytes(), presented.as_bytes()) {
                found = Some(Actor { name: name.clone(), role: *role });
            }
        }
        found
    }
}

fn ct_eq(a: &[u8], b: &[u8]) -> bool {
    let mut diff = (a.len() ^ b.len()) as u8;
    for i in 0..a.len().max(b.len()) {
        diff |= a.get(i).copied().unwrap_or(0) ^ b.get(i).copied().unwrap_or(0);
    }
    diff == 0
}

pub async fn middleware(State(st): State<AppState>, mut req: Request, next: Next) -> Response {
    if req.uri().path() == "/health" {
        return next.run(req).await;
    }
    let actor = if st.keys.is_empty() {
        // Development mode: no keys configured. A warning is logged once at startup.
        Actor { name: "dev".into(), role: Role::Admin }
    } else {
        let h = req.headers();
        let presented = h
            .get("x-api-key")
            .and_then(|v| v.to_str().ok())
            .map(str::to_owned)
            .or_else(|| {
                h.get("authorization")
                    .and_then(|v| v.to_str().ok())
                    .and_then(|v| v.strip_prefix("Bearer "))
                    .map(str::to_owned)
            });
        match presented.and_then(|p| st.keys.lookup(&p)) {
            Some(a) => a,
            None => return ApiError::Unauthorized.into_response(),
        }
    };
    req.extensions_mut().insert(actor);
    next.run(req).await
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_and_looks_up_keys() {
        let k = Keys::parse("aaaaaaaaaaaaaaaa:admin:alice, bbbbbbbbbbbbbbbb:analyst:bob").unwrap();
        assert_eq!(k.lookup("aaaaaaaaaaaaaaaa").unwrap().role, Role::Admin);
        assert_eq!(k.lookup("bbbbbbbbbbbbbbbb").unwrap().name, "bob");
        assert!(k.lookup("bbbbbbbbbbbbbbb").is_none());
        assert!(k.lookup("").is_none());
    }

    #[test]
    fn rejects_weak_or_malformed_keys() {
        assert!(Keys::parse("short:admin:x").is_err());
        assert!(Keys::parse("aaaaaaaaaaaaaaaa:root:x").is_err());
        assert!(Keys::parse("aaaaaaaaaaaaaaaa").is_err());
        assert!(Keys::parse("").unwrap().is_empty());
    }

    #[test]
    fn role_capabilities_are_separated() {
        let a = Actor { name: "a".into(), role: Role::Analyst };
        assert!(a.require_triage().is_ok() && a.require_ingest().is_err() && a.require_admin().is_err());
        let i = Actor { name: "i".into(), role: Role::Ingest };
        assert!(i.require_ingest().is_ok() && i.require_triage().is_err());
    }
}
