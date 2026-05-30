# Changelog

## v0.5.0 — 2026-05-30

`wm.almanac.due` is published but nothing spoke it. This version teaches
`wmd` to subscribe to `wm.almanac.due` and speak the prompt by reusing the
exact proactive publish path `recap_opener` already uses — so the reminder
comes out in hearth's persona, at earshot's pace, through wm-tts, with no
new speech mechanism invented.

New in v0.5.0:
- `AlmanacDueEvent` wire type + `ALMANAC_DUE_TOPIC` / `ALMANAC_TOPIC_PREFIX` constants
- `BrainConfig.almanac_speak` gate (default `true`; env `WM_BRAIN_ALMANAC_SPEAK`)
- `handle_speak_almanac_due`: speaks `ev.say` verbatim via `wm.brain.reply`;
  arms the PendingAck window; gracefully degrades on empty/malformed say
- Subscribe loop wired to `wm.almanac.` prefix; routes due events to handler

## v0.4.0 — 2026-05-29

Add almanac acknowledgment FSM — close the reminder loop with spoken reply classification.

After `wm.almanac.due` fires a spoken prompt, the brain now sets a `PendingAck` on
`DaemonState` and classifies the next `wm.stt.final` transcript within earshot's patience
window: "took it" / "okay" / "done" → `wm.almanac.ack {state:"done"}`; "later" / "in a
minute" → `{state:"snoozed"}` + `wm.almanac.snooze {resume_ts}`; unrelated transcript →
pending left open. On timeout the daemon emits `{state:"missed"}` and speaks one gentle
re-ask; a second timeout finalises missed with no further re-ask. Classification is
deterministic keyword tiers (no network/LLM call). 204 lib tests pass.

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
