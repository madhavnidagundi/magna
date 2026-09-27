// =============================================================================
// Magna Middleware — Lifecycle / State Manager
// =============================================================================
//! Implements a strict state machine that governs the middleware lifecycle.
//!
//! Valid states and transitions:
//!
//! ```text
//!   Uninitialized ──► Initialized ──► EngineLoaded ──► Ready
//!         │                │                │             │
//!         └────────────────┴────────────────┴─────────────┘
//!                                   │
//!                                   ▼
//!                                 Error
//! ```
//!
//! Any state can transition to `Error`.
//! `Error` can transition back to `Uninitialized` (reset).

use std::fmt;

use tracing::{debug, info, warn};

use crate::utils::errors::{MiddlewareError, MiddlewareResult};

// ---------------------------------------------------------------------------
// States
// ---------------------------------------------------------------------------

/// Middleware lifecycle states.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum State {
    Uninitialized,
    Initialized,
    EngineLoaded,
    Ready,
    Error,
}

impl fmt::Display for State {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            State::Uninitialized => write!(f, "Uninitialized"),
            State::Initialized => write!(f, "Initialized"),
            State::EngineLoaded => write!(f, "EngineLoaded"),
            State::Ready => write!(f, "Ready"),
            State::Error => write!(f, "Error"),
        }
    }
}

// ---------------------------------------------------------------------------
// StateManager
// ---------------------------------------------------------------------------

/// Enforces valid lifecycle state transitions.
#[derive(Debug)]
pub struct StateManager {
    current: State,
}

impl StateManager {
    /// Create a new state manager starting at `Uninitialized`.
    pub fn new() -> Self {
        Self {
            current: State::Uninitialized,
        }
    }

    /// Current state.
    pub fn current(&self) -> State {
        self.current
    }

    /// Attempt to transition to `target`.
    ///
    /// Returns `Ok(())` if the transition is valid, or
    /// [`MiddlewareError::InvalidStateTransition`] otherwise.
    pub fn transition_to(&mut self, target: State) -> MiddlewareResult<()> {
        if self.is_valid_transition(target) {
            info!(from = %self.current, to = %target, "State transition");
            self.current = target;
            Ok(())
        } else {
            warn!(from = %self.current, to = %target, "Invalid state transition attempted");
            Err(MiddlewareError::InvalidStateTransition {
                from: self.current.to_string(),
                to: target.to_string(),
            })
        }
    }

    /// Assert that the current state is `expected`, or return an error.
    pub fn require(&self, expected: State) -> MiddlewareResult<()> {
        if self.current == expected {
            Ok(())
        } else {
            Err(MiddlewareError::InvalidState {
                required: expected.to_string(),
                current: self.current.to_string(),
            })
        }
    }

    /// Assert that the current state is at least `min` (in lifecycle order).
    pub fn require_at_least(&self, min: State) -> MiddlewareResult<()> {
        if self.current != State::Error && self.ordinal(self.current) >= self.ordinal(min) {
            Ok(())
        } else {
            Err(MiddlewareError::InvalidState {
                required: format!("at least {}", min),
                current: self.current.to_string(),
            })
        }
    }

    /// Force the state to `Error`.
    pub fn set_error(&mut self) {
        warn!(from = %self.current, "Forcing state to Error");
        self.current = State::Error;
    }

    /// Reset from `Error` back to `Uninitialized`.
    pub fn reset(&mut self) -> MiddlewareResult<()> {
        if self.current == State::Error || self.current == State::Uninitialized {
            debug!("Resetting state to Uninitialized");
            self.current = State::Uninitialized;
            Ok(())
        } else {
            Err(MiddlewareError::InvalidStateTransition {
                from: self.current.to_string(),
                to: State::Uninitialized.to_string(),
            })
        }
    }

    // -- helpers ------------------------------------------------------------

    fn is_valid_transition(&self, target: State) -> bool {
        // Any → Error is always valid.
        if target == State::Error {
            return true;
        }

        matches!(
            (self.current, target),
            (State::Uninitialized, State::Initialized)
                | (State::Initialized, State::EngineLoaded)
                | (State::EngineLoaded, State::Ready)
                // Allow re-loading a new engine from Ready.
                | (State::Ready, State::EngineLoaded)
                // Allow shutdown (any → Uninitialized only from specific states).
                | (State::Initialized, State::Uninitialized)
                | (State::EngineLoaded, State::Uninitialized)
                | (State::Ready, State::Uninitialized)
                | (State::Error, State::Uninitialized)
        )
    }

    fn ordinal(&self, s: State) -> u8 {
        match s {
            State::Uninitialized => 0,
            State::Initialized => 1,
            State::EngineLoaded => 2,
            State::Ready => 3,
            State::Error => 255,
        }
    }
}

impl Default for StateManager {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn happy_path() {
        let mut sm = StateManager::new();
        assert_eq!(sm.current(), State::Uninitialized);

        sm.transition_to(State::Initialized).unwrap();
        sm.transition_to(State::EngineLoaded).unwrap();
        sm.transition_to(State::Ready).unwrap();

        assert_eq!(sm.current(), State::Ready);
    }

    #[test]
    fn skip_state_fails() {
        let mut sm = StateManager::new();
        // Cannot jump from Uninitialized to Ready.
        assert!(sm.transition_to(State::Ready).is_err());
    }

    #[test]
    fn any_to_error() {
        let mut sm = StateManager::new();
        sm.transition_to(State::Initialized).unwrap();
        sm.transition_to(State::Error).unwrap();
        assert_eq!(sm.current(), State::Error);
    }

    #[test]
    fn error_to_uninitialized() {
        let mut sm = StateManager::new();
        sm.set_error();
        sm.reset().unwrap();
        assert_eq!(sm.current(), State::Uninitialized);
    }

    #[test]
    fn require_state() {
        let sm = StateManager::new();
        assert!(sm.require(State::Uninitialized).is_ok());
        assert!(sm.require(State::Initialized).is_err());
    }

    #[test]
    fn require_at_least() {
        let mut sm = StateManager::new();
        sm.transition_to(State::Initialized).unwrap();
        sm.transition_to(State::EngineLoaded).unwrap();

        assert!(sm.require_at_least(State::Initialized).is_ok());
        assert!(sm.require_at_least(State::EngineLoaded).is_ok());
        assert!(sm.require_at_least(State::Ready).is_err());
    }

    #[test]
    fn error_does_not_satisfy_require_at_least() {
        let mut sm = StateManager::new();
        sm.set_error();

        assert!(sm.require_at_least(State::Initialized).is_err());
        assert!(sm.require_at_least(State::EngineLoaded).is_err());
        assert!(sm.require_at_least(State::Ready).is_err());
    }

    #[test]
    fn reload_engine_from_ready() {
        let mut sm = StateManager::new();
        sm.transition_to(State::Initialized).unwrap();
        sm.transition_to(State::EngineLoaded).unwrap();
        sm.transition_to(State::Ready).unwrap();

        // Should be able to load a new engine.
        sm.transition_to(State::EngineLoaded).unwrap();
        sm.transition_to(State::Ready).unwrap();
    }

    #[test]
    fn shutdown_from_ready() {
        let mut sm = StateManager::new();
        sm.transition_to(State::Initialized).unwrap();
        sm.transition_to(State::EngineLoaded).unwrap();
        sm.transition_to(State::Ready).unwrap();

        sm.transition_to(State::Uninitialized).unwrap();
        assert_eq!(sm.current(), State::Uninitialized);
    }
}
