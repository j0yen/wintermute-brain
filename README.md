# wintermute-brain

> `wmd` is the brain — the Claude API conversation loop with
> recall-backed persistent memory. Sonnet 4.6 by default (Opus 4.7
> opt-in), prompt-cached on the persistent profile + the day's history,
> with sub-10 ms recall retrieval for context. Routes tool calls to the
> action layer (Fleet 1 ships a minimal tool stub; Fleet 2 plugs in
> browser/mail/etc.). Returns destructive intents to wm-dialog for
> verbal confirmation — the brain never acts destructively without
> dialog gating.

Part of the wintermute fleet
([`wintermute-platform`](https://github.com/j0yen/wintermute-platform),
[`wintermute-audio`](https://github.com/j0yen/wintermute-audio),
[`wintermute-stt`](https://github.com/j0yen/wintermute-stt),
[`wintermute-tts`](https://github.com/j0yen/wintermute-tts),
[`wintermute-dialog`](https://github.com/j0yen/wintermute-dialog)).
Subscribes to `wm.dialog.{turn.user,confirm.granted,confirm.denied}`
on agorabus; publishes `wm.brain.{reply,reply.destructive,tool.call,tool.result,error}`.

Built with Rust 2024 / `rustc 1.85`. Configuration lives at
`$XDG_CONFIG_HOME/wintermute/brain.toml` (atomic write on swap-model /
default-model mutations).

## Install

```sh
git clone --depth 1 https://github.com/j0yen/wintermute-brain.git
cd wintermute-brain
cargo install --path . --root ~/.local
```

`cargo install` puts `wmd` into `~/.local/bin/`. The wintermute systemd
user target (from
[`wintermute-platform`](https://github.com/j0yen/wintermute-platform))
supervises it once the unit file is in place.

### Prerequisites

- `cargo` / `rustc 1.85+`
- An Anthropic API key in `WM_ANTHROPIC_API_KEY` (env var name is
  configurable via `brain.toml` or `--api-key-env`).
- [`recall`](https://github.com/j0yen/recall) daemon listening on
  `$XDG_RUNTIME_DIR/recall.sock` (override with `--recall-sock` or
  `WM_BRAIN_RECALL_SOCK`).
- [`wintermute-platform`](https://github.com/j0yen/wintermute-platform)
  for the wintermute.target ordering and the agorabus socket.

## Quick start

```sh
wmd status                       # dump effective config as JSON
wmd swap-model opus              # next turn only, then revert
wmd default-model sonnet         # persistent default in brain.toml
wmd start                        # run the daemon
```

## Acceptance criteria (PRD §4)

Eight criteria. AC5/6/7 are unit-paired and green; AC1/2/3/4/8 require
the full live Fleet 1 harness (Anthropic key, recall daemon, agorabus,
wm-dialog upstream) and are validated post-boot.

| # | Criterion | Tests |
|---|---|---|
| 1 | End-to-end "wake → first TTS audio of brain reply" ≤2 s warm + cache-hot | live-only (Fleet 1 harness) |
| 2 | Conversation context survives reboot — recall surfaces today's earlier turns | live-only |
| 3 | Prompt cache hit rate ≥60% across a typical day | live-only (API `cache_read_input_tokens`) |
| 4 | Network drop → spoken apology within 3 s; pending utterance replays on restore | live-only |
| 5 | 10 scripted destructive prompts each emit `wm.brain.reply.destructive` with valid `intent_id` + `confirm_keyword`; none execute without confirmation | `tests/ac5_ten_scripted_destructive_prompts` |
| 6 | `wmd swap-model opus` uses Opus exactly once, then reverts | `daemon::tests::pending_model_consumed_*` |
| 7 | `wm.recall.search` + `wm.recall.save_fact` end-to-end | `daemon::tests::recall_tools_router_*` + `recall_client` integration |
| 8 | 8-hour steady-state, 100 simulated turns: RSS growth <100 MB, no zombie pending utterances | live-only soak |

## Conversation loop

```
wm.dialog.turn.user
   │
   ▼
┌──────────┐  recall.query top-K     ┌─────────────────┐
│ retrieve ├────────────────────────▶│ compose request │
└──────────┘  thread-<day>, profile  └────────┬────────┘
                                              │  Anthropic /v1/messages (stream=true)
                                              ▼
                                      ┌───────────────┐
                                      │   route       │
                                      └─┬───┬─────┬───┘
                                        │   │     │
                  destructive intent?   │   │     │  tool call?
                  ┌─────────────────────┘   │     └───────────────┐
                  ▼                         ▼                     ▼
        wm.brain.reply.destructive   wm.brain.reply       wm.brain.tool.call
                  │                                              │
       confirm.granted / denied                          wm.brain.tool.result
                  │
                  ▼
          tool.call (granted) / cancellation-reply (denied)
                                              │
                                              ▼  recall.touch on retrieved hits
                                       memorize turn
```

## Tool surface (Fleet 1 minimum)

| Tool | Notes |
|---|---|
| `wm.recall.search` | Recall daemon query — returns top-K hits |
| `wm.recall.save_fact` | Writes a `wintermute-profile`-subject memory |
| `wm.fleet2.*` | Stubs — return "no tools registered" until Fleet 2 plugs in |

## Topics

Subscribed: `wm.dialog.{turn.user, confirm.granted, confirm.denied}`.

Published: `wm.brain.{reply, reply.destructive, tool.call, tool.result, error}`.

## License

Dual-licensed MIT or Apache-2.0 at your option.
