# Changelog

## v0.24.0 — 2026-06-15

inoculate-immune: Persona floor enforcement — base strain Boundaries are a
non-lowerable floor. Effective policy = base ∪ persona_additions; persona
rules that weaken the base floor are rejected with `floor-violation: <rule>`.
`persona lint <profile>` exits 0=compliant, 1=would-weaken. `PersonaConfig`
carries `strain_hash` from the resolved base. Base-forbid always beats
persona-allow; persona-forbid adds to base. Fail-open when `inoculate` is
absent. 480+ tests pass.

## v0.23.0 — 2026-06-13

persona-redline-regenerate: Added Regenerate variant to RedlineAction for graceful redline recovery. Re-issues LLM request with hardened system prompt, falls back to safe phrase only on exhaustion. 451 tests pass.

## v0.22.0 — 2026-06-13

persona-profile: named profile registry + wm-brain persona subcommand (list/show/diff/apply); jocelyn and default presets; apply --write patches only [persona] section preserving other tables

## v0.21.0 — 2026-06-13

persona-redline: output-side enforcement — scan reply for forbidden terms before TTS publish; SafePhrase (replace with configured phrase or built-in default) when RedlineAction::SafePhrase active; RedlineAction::Off is default (existing configs unaffected); all 427 tests pass including 23 new redline tests

## v0.20.0 — 2026-06-12

Add `forbidden_terms` to `PersonaConfig`: firm LLM avoidance instruction for tech jargon (Jocelyn elder-companion preset). When non-empty, `compose_base` appends an explicit "never use" clause listing the terms. Includes `brain.toml.example` with annotated Jocelyn preset.

## v0.19.0 — 2026-06-12

Add SemanticCache to intercept semantically identical turns before the LLM tier ladder, reducing redundant API calls. SQLite-backed, 256-entry LRU, 0.92 cosine threshold. Time/date/weather turns bypass the cache safely.

## v0.18.0 — 2026-06-05

### lucid-turn-id: adopt inbound turn_id for brain.route/reply (AC4 + AC3-brain)

The brain now adopts the cross-daemon `turn_id` minted at wake by `wm-audio` and
threaded through stt → dialog, instead of minting `now_ms` locally. The inbound
`wm.dialog.turn.user.turn_id` (optional `String`) is propagated onto
`wm.brain.reply` (`turn_id`), `wm.brain.route` (`turn_corr`), and the
tool/destructive envelopes, so a whole spoken turn shares one correlation id
end-to-end (`wm-tts` will copy `reply.turn_id` in a later tick). When no inbound
id is present (system-injected / pre-PRD turns), a freshly-minted, `gen-`-flagged
id is used so consumers can tell it was synthesized. Fully additive and
backward-compatible (AC5): the legacy numeric `route.turn_id` (u64) is retained
for lucid-mind's route⇄context join, and pre-PRD envelopes with no `turn_id`
still deserialize. One repo of the multi-repo lucid-turn-id PRD.

### constellation-brain-local: local-llm tier for 5700U APU node

Added `local-llm` tier to the brain ladder — a dedicated qwen2.5-8B Q4_K_M
rung served by the Ryzen 7 5700U node running llama-server. Follows the same
optional/graceful-absence pattern as `local-gpu`: enabled only when
`WM_BRAIN_LOCAL_LLM_ENDPOINT` is set, absent otherwise (falls through to cloud).
Composes with `local-gpu` when both endpoints are set (gpu first, then local-llm).

New constellation config in `constellation/brain-local/`:
- `detect-backend.sh`: benchmarks Vulkan RADV vs CPU generation on the Vega 8
  iGPU, picks the faster, explicitly never ROCm (AC1).
- `llama-server.service`: resource-isolated unit (brain-local.slice, api-key
  from env file, 0.0.0.0 bind for Tailscale mesh, no-build-worker marker) (AC2/AC4/AC7).
- `brain-local.slice`: 20 GB memory reservation, swap-off, 6+ of 16 threads (AC4).
- `brain-local.env.example`: documents model path + api-key config (AC7).

Rust: `TIER_LOCAL_LLM` const, `DEFAULT_LOCAL_LLM_MODEL`, `ladder_with_local_llm()`
function, `local_llm_endpoint`/`local_llm_model` BrainConfig fields, env vars
`WM_BRAIN_LOCAL_LLM_ENDPOINT`/`WM_BRAIN_LOCAL_LLM_MODEL`. 8 new tests. 369 total pass.

## v0.17.0 — 2026-06-04

constellation-brain-gpu: add `local-gpu` tier to the brain ladder. A new rung
(`TIER_LOCAL_GPU`) sits between `local-8b` and the cloud tiers, pointing at the
desktop Radeon's `llama-server` via `WM_BRAIN_GPU_ENDPOINT`. When the endpoint
is absent or unreachable the tier is either not inserted or naturally escalates
to the next rung — cloud fallback unchanged. `Tier` gains `endpoint_override`
so per-rung local endpoints work without a shared ollama. `BrainConfig` gains
`gpu_endpoint` + `gpu_model` (env `WM_BRAIN_GPU_ENDPOINT` / `WM_BRAIN_GPU_MODEL`).
`capped_ladder()` reads these and inserts the rung; `wmd swap-model local-gpu`
is now accepted. Four new unit tests cover presence, absence, empty-string, and
allow-list validation (AC5, AC6, AC7).

## v0.16.0 — 2026-06-04

lucid-mind-brain-context: publish `wm.brain.context` digest per turn so lucid-mind can show injected recall context.

Adds `BrainContextEvent` (`turn_id`, `recall_hits: [{id, subject}]`, `persona_tier`, `history_turns`, `ts`) published once per turn right before the LLM call on both the ladder path and the single-client path. `turn_id` matches `wm.brain.route` so lucid-mind joins the two events without extra wiring (AC2). `recall_hits` carries only id+subject — no body text — keeping the payload privacy-light (AC3). New unit tests verify serde round-trip (AC4) and no body-field leakage (AC3). Two daemon integration tests verify exactly-once-per-turn (AC1) and turn_id alignment (AC2). AC6 (runtime lucid-mind render) deferred per PRD.

## v0.15.1 — 2026-06-02

brain-prompt-cache AC5: deterministic prompt-cache ratio test.

Adds an in-process fake Anthropic client (CacheSimClient) that bills a request per the prompt-caching contract — the cache_control-terminated stable prefix is a write on turn 1 and a read on subsequent turns; volatile tail + messages are full-rate every turn. A new 50-turn fixture drives the real compose_request path and asserts sum(cache_read)/sum(input) >= 0.60, closing the PRD's AC5, which previously existed only as an #[ignore]-gated live-fleet test requiring a (currently exhausted) Anthropic API key. 352 lib tests pass.

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

## v0.10.0 — 2026-05-30

wmd-repair-affordances: local-replay repair path — "say that again, louder."

Once the history ring (v0.3.0) is in place, verbatim repair requests become
near-free: no model round-trip, no token cost, no latency. Matches "say that
again", "what did you say", "louder", "speak up" (and related) brain-side
and replays the last assistant turn from the ring.

- New `src/repair.rs`: `Repair` enum + `classify()` + phrase normalisation;
  config-driven phrase sets with built-in defaults (12 repeat phrases, 10
  louder phrases); word-count ambiguity guard (≤5 words) prevents long
  sentences containing a keyword from being mis-classified (AC5).
- `src/bus.rs`: `ReplyEvent` gains optional `loudness: Option<String>` field
  (`#[serde(skip_serializing_if = "Option::is_none")]` — backward-compatible;
  AC6).
- `src/lib.rs`: `BrainConfig` gains `repair_repeat_phrases` /
  `repair_louder_phrases` `Vec<String>` (empty = use defaults; AC7).
- `src/history.rs`: `History::last()` accessor for repair replay path.
- `src/daemon.rs`: repair check runs before LLM dispatch in
  `handle_turn_user()`; `handle_repair()` publishes replay with optional
  loudness hint; empty history degrades to "I haven't said anything yet"
  (AC3). Replayed turns are not pushed back into history (AC4).

302 lib tests (was 293), +9 new. `cargo deny check bans licenses sources` clean.

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
