# Changelog

## v0.3.0 — 2026-05-30

Add bounded in-memory turn-history ring so the brain chains conversation turns.

`wmd` was provably stateless — each turn built a single-message request with no
prior context. This extends `DaemonState` with a `History` ring (capacity from
`BrainConfig.history_turns`, default 6) fed into `compose_request` as a flat
`[user, assistant, …, current_user]` prefix. Failed / empty-reply turns are not
pushed (AC3). Destructive-intent turns store the spoken prefix, not the JSON fence
(AC5). A token-budget guard trims the oldest pairs first (AC6). `history_turns = 0`
restores the single-message invariant (AC4). Verified: 176 lib tests pass, AC1–AC4
covered by integration tests in `daemon::tests`.

## v0.1.1 — 2026-05-28

Fix post-announce bus-startup defect (PRD-wintermute-fleet-bus-startup-defect).

The announce-before-subscribe fix that shipped overnight was install-stale, not
source-buggy: the binaries under ~/.local/bin/ predated the fix, while the source
already had the dual-Client + announce-first pattern. Tightened the agorabus
path-dependency pin from a wildcard/^0.1 to ^0.3 (agorabus 0.3.0's let_chains
need system cargo 1.95), rebuilt, and reinstalled so the systemd-launched daemons
run post-fix bytes. Daemons now survive a 60s soak (NRestarts=0) and round-trip
their subscribed topics. Note: AC3-strict (peer presence after the 60s window)
is deferred to PRD-wintermute-fleet-bus-heartbeat-keepalive — these daemons still
lack a post-announce heartbeat, so the bus prunes them from the peer snapshot.
