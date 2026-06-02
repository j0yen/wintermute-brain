# Changelog

## v0.15.0 — 2026-06-02

brain-prompt-cache: make every brain turn pay for its prefix once.

`wintermute-brain` previously re-sent and re-billed its entire stable prefix (persona, tool definitions, recall context, conversation history) at full input-token rate on every turn — the request carried no `cache_control` breakpoints. This version adds ephemeral `cache_control` breakpoints to the Anthropic request and restructures context composition so the stable persona prefix is cacheable and volatile per-turn recall context no longer busts the cache.

- `MessageRequest`/`SystemField` now serialize `system` as a typed content-block array (`SystemBlock`) carrying optional `cache_control: {"type":"ephemeral"}`, with byte-identical plain-string fallback when no breakpoint is set.
- Composition splits the cacheable persona prefix (base persona + child-lock + tool preamble, breakpoint on the last stable block) from the volatile recall/recap tail, which is positioned after the breakpoint so it never busts the cached prefix.
- `cache_read_input_tokens` / `cache_creation_input_tokens` parsed from `message_start.usage` and emitted in a structured per-turn log line.
- child-lock semantics unchanged (clause stays inside the cached prefix). 351 lib tests pass.

## v0.14.0 — 2026-06-02

Add routing classifier and wm.brain.route observability to wintermute-brain.

New src/router.rs implements command-vs-conversational turn classification (deterministic, no model call on the hot path), a six-row routing policy table (PRD §2.2), RoutingConfig ([routing] brain.toml section), and a canned degrade phrase bank. The daemon now publishes wm.brain.route after every ladder turn with tier, reason, latency_ms, and model for operator observability. wmd gains `route status` and `route prefer` subcommands. 328 tests pass.

## v0.13.0 — 2026-06-02

Adds first-contact greeting module: GreetingMode (off/auto/first-ever-always), GreetingKind
(FirstEver/Returning/Silent), RecallPresence probe struct, select_greeting_kind, compose_greeting,
and GreetingGuard (greet-once). Extends PersonaConfig with `greeting` and `wake_word` fields.
14 new lib tests cover all 6 automated ACs.

## v0.12.0 — 2026-06-02

hearth-persona-config: lift the companion's persona out of a hardcoded `const` into a configurable `[persona]` table in `brain.toml`.

- New `PersonaConfig` + `Register` enum (WarmElder default, Plain, Brisk) in `lib.rs`; all fields `#[serde(default)]` so existing configs load unchanged.
- `Register::compose_base` builds the persona prose with `{self_name}`/`{user_name}` substitution; WarmElder is calibrated for a non-technical elder (short sentences, no jargon), Plain reproduces the retained `DEFAULT_PERSONA` byte-for-byte.
- Persona composed once at config load (`DaemonState::new`) so it stays a byte-stable prompt-cache prefix; per-turn recall/recap still layer after via `compose_persona`.
- CLI: `wmd persona show` and `wmd persona set-register <warm-elder|plain|brisk>` (atomic-write, mirrors swap-model).
- 9 new lib tests (deserialization defaults, per-field override, register→prose, name substitution, user-clause omission, cache-prefix stability, extra append, serde round-trip). 311 tests pass.

## v0.11.0 — 2026-06-02

brain-backend-ladder: local-first tier ladder for wintermute-brain.

Implements PRD-brain-backend-ladder: a 5-rung tier ladder (local-3b →
local-8b → haiku → sonnet → opus) with LadderClient orchestrator,
dual-signal escalation (hard LocalOutcome::Escalate + soft wm-verify
reject), safety-override pre-route for high-stakes turns, conversational
stickiness (SessionFloor), filler-while-escalating (ESCALATION_FILLER),
key-gate relaxation (brain works with no Anthropic API key), and full
config integration (default_tier, pending_tier, local_endpoint in
brain.toml). Also fixes bus_smoke compile error: agorabus DaemonConfig
gained drain_grace_ms/drain_resume_hint_ms fields.

All 302+ tests green; no new clippy warnings beyond baseline.

## v0.9.0 — 2026-05-30

Adds cross-session thread memory recall (wmd-session-recap): on session.start
wmd queries recall for recent committed wintermute-thread-* memories and holds
them session-scoped; spliced into every turn's system prompt under "Recent
conversations:" distinct from per-turn profile recall. Optional recap_opener
(default off) publishes a proactive continuity greeting before first turn.
Recall outages tolerated. 15 new tests (293 total). AC1-AC8 covered.

## v0.8.0 — 2026-05-30

PRD-wmd-turn-history gave the brain a rolling last-N buffer, but without session
boundaries context bleeds across unrelated conversations. This increment adds
conversation session tracking: wmd infers sessions from idle gaps (default 5 min)
and explicit close phrases ("goodbye", "go to sleep", etc.), emits
wm.brain.session.start/end events, and clears the history ring on each boundary.

## v0.7.0 — 2026-05-30

wmd-memory-writeback: end-of-session recall writeback pipeline.
New: src/writeback.rs (ExtractorClient, WritebackGuard, FACT parsing, trigger_writeback).
BrainConfig: writeback_auto_commit/model/confidence_floor/idle_gap_ms.
DaemonState: session tracking + extractor; fires on idle-gap expiry.
AnthropicExtractor wired in run(). 249 tests (+26). cargo deny clean.

## v0.6.0 — 2026-05-30

feat(degrade): graceful-degradation phrase bank, aggregator, and health snapshot

Adds src/degrade.rs with: phrase bank (9 error kinds + fallback), RateLimitState
(30-s per-kind gate), HealthState (4 components), process_error_envelope aggregator.
Wires error-topic subscriptions and health snapshot ticker (60 s) into daemon.rs.
+16 lib tests; 239 total pass.

## v0.5.1 — 2026-05-29

PRD-wmd-turn-history: complete test coverage pass — add 10 new integration
tests closing the missing acceptance criteria.

New tests in v0.5.1:
- AC5: destructive-intent turns store the spoken prefix (not the JSON fence)
  as the assistant history entry, verified end-to-end through `dispatch()`
- AC7: `history_turns` persists through `brain.toml` save/reload (both
  enabled and disabled=0 variants)
- AC3 variant: empty-text LLM reply is not stored in history
- Additional integration tests: monotonic growth, disabled accumulation,
  single-pair `compose_request`, `DaemonState` capacity initialised from
  `config.history_turns`

223 lib tests pass (up from 204 at the last checkpoint). `cargo deny check
bans licenses sources` clean.

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
