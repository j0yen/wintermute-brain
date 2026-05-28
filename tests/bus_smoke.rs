//! Bus-smoke regression test for the announce-before-subscribe protocol
//! bug class (PRD-wintermute-fleet-bus-smoke-convention.md).
//!
//! Spawns an in-process `agorabus` daemon on a temp socket, points the
//! `wm-brain` daemon at it via the `WM_BRAIN_BUS_SOCKET` env override,
//! waits for the daemon to connect + announce + subscribe on both
//! connections, then queries the bus's peer snapshot and asserts the
//! daemon's two announced session-ids (`wm-brain-<pid>-sub` and
//! `wm-brain-<pid>`) are both present. A daemon that connected without
//! announcing would have been torn down by agorabus with
//! `announce_required` before either session could land in the peers
//! map — appearance in `peers()` is positive evidence that the
//! `connect()` → `announce()` → `subscribe()` ordering is correct on
//! both connections.
//!
//! Why a peer-snapshot probe and not a publish-through driver: brain
//! subscribes to a single prefix (`wm.dialog.`), so the agorabus
//! single-`Option<String>` `subscribed_prefix` limitation (see
//! `agorabus/src/daemon.rs:367` — `*subscribed_prefix = Some(prefix)`
//! overwrites on every Subscribe) is NOT what blocks publish-through
//! here. Brain blocks publish-through because every incoming
//! `wm.dialog.*` event the daemon would echo back needs either an
//! `ANTHROPIC_API_KEY` (`wm.dialog.turn.user` → silent drop without
//! one — see `daemon.rs:1252`) or a pending intent in `DaemonState`
//! (`wm.dialog.confirm.granted` / `confirm.denied` → silent drop for
//! unknown intent ids — `daemon.rs:1151,1200`). With no observable
//! echo path from a cold-start daemon, the peer-snapshot probe is the
//! cleanest positive evidence of announce-before-subscribe correctness.
//! It still catches the bug class this PRD targets: any announce
//! regression would prevent the session-ids from landing in `peers()`.

#![allow(
    unsafe_code,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::too_many_lines,
    clippy::missing_panics_doc,
    clippy::missing_assert_message,
    clippy::missing_errors_doc
)]

use std::path::PathBuf;
use std::time::Duration;

use agorabus::{Client, DaemonConfig, run_daemon};
use tokio::time::timeout;
use wintermute_brain::BrainConfig;

fn tmp_path(tag: &str, ext: &str) -> PathBuf {
    let pid = std::process::id();
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.subsec_nanos())
        .unwrap_or(0);
    // agorabus chmods the socket parent to 0700 on bind; pointing at
    // /tmp directly silently goes wrong. Use a fresh pid+nanos subdir.
    let dir = std::env::temp_dir().join(format!("wm-brain-test-{pid}-{nanos}"));
    let _ = std::fs::create_dir_all(&dir);
    dir.join(format!("{tag}.{ext}"))
}

async fn run_bus_smoke() -> Result<(), String> {
    // 1. Spawn an in-process agorabus on a unique temp socket.
    let bus_sock = tmp_path("bus", "sock");
    let _ = std::fs::remove_file(&bus_sock);
    let bus_cfg = DaemonConfig {
        socket_path: bus_sock.clone(),
        heartbeat_timeout: Duration::from_secs(60),
        broadcast_capacity: 1024,
    };
    let (ready_tx, ready_rx) = tokio::sync::oneshot::channel::<()>();
    let (bus_shutdown_tx, bus_shutdown_rx) = tokio::sync::oneshot::channel::<()>();
    let bus_task = tokio::spawn(async move {
        let _ = run_daemon(bus_cfg, Some(ready_tx), bus_shutdown_rx).await;
    });
    timeout(Duration::from_secs(2), ready_rx)
        .await
        .map_err(|_| "bus never signalled ready".to_string())?
        .map_err(|e| format!("bus ready_tx dropped: {e}"))?;

    // 2. Open a query client BEFORE the wm-brain daemon starts.
    //    Announce first — positive evidence the test author understood
    //    the ordering (AC7 anti-cargo-cult gate). We don't subscribe
    //    here because the probe is a `peers()` query, not a
    //    broadcast-listener.
    let mut query = Client::connect(&bus_sock)
        .await
        .map_err(|e| format!("query connect: {e:#}"))?;
    query
        .announce(
            "wm-brain-bus-smoke-query",
            std::process::id(),
            "",
            "test-query",
        )
        .await
        .map_err(|e| format!("query announce: {e:#}"))?;

    // 3. Point the wm-brain daemon at our temp bus socket.
    //    SAFETY: tests in this file are the only consumer of this
    //    var; cargo runs separate test binaries in separate processes
    //    so cross-file env races are impossible. Intra-file there's
    //    only this one test fn.
    let bus_sock_for_env = bus_sock.clone();
    // SAFETY: see comment above.
    unsafe {
        std::env::set_var("WM_BRAIN_BUS_SOCKET", &bus_sock_for_env);
    }

    // 4. Spawn the wm-brain daemon. It will announce on TWO
    //    connections — sub_client (session_id `wm-brain-<pid>-sub`)
    //    and pub_client (session_id `wm-brain-<pid>`) — and subscribe
    //    to `wm.dialog.` on the sub connection. With no
    //    ANTHROPIC_API_KEY set, the daemon logs the key-absent warning
    //    and continues looping on next_event() — exactly the state
    //    this peer-snapshot probe needs.
    let daemon_task =
        tokio::spawn(
            async move { wintermute_brain::daemon::run(BrainConfig::default(), None).await },
        );

    // 5. Give the daemon time to connect + announce + subscribe on
    //    both connections. agorabus's announce path adds the peer to
    //    the state map BEFORE replying ok, so a peers() query after a
    //    short wait will see both sessions if the wire-up succeeded.
    tokio::time::sleep(Duration::from_millis(1_500)).await;

    // 6. Probe the peer snapshot. The two expected session_ids come
    //    from daemon.rs:1315-1320 (sub) and daemon.rs:1330-1336 (pub)
    //    in wintermute-brain. Both must appear; either missing would
    //    indicate an announce failure (agorabus drops the connection
    //    before recording the peer).
    let peers = query
        .peers()
        .await
        .map_err(|e| format!("peers query: {e:#}"))?;
    let pid = std::process::id();
    let want_sub = format!("wm-brain-{pid}-sub");
    let want_pub = format!("wm-brain-{pid}");
    let session_ids: Vec<String> = peers.iter().map(|p| p.session_id.clone()).collect();
    let saw_sub = session_ids.iter().any(|s| s == &want_sub);
    let saw_pub = session_ids.iter().any(|s| s == &want_pub);

    // 7. Tear down regardless of outcome — never leak the daemon task
    //    or the bus task. Order: drop the query (closes its UDS), shut
    //    down the bus (daemon's next_event returns None, daemon exits
    //    cleanly per daemon.rs:1357), await both tasks with a deadline.
    drop(query);
    let _ = bus_shutdown_tx.send(());
    let _ = timeout(Duration::from_secs(3), bus_task).await;
    let daemon_outcome = timeout(Duration::from_secs(3), daemon_task).await;
    let _ = std::fs::remove_file(&bus_sock);
    // SAFETY: same single-test-consumer reasoning as the set_var
    // above. Removing the var so any later test in the same binary
    // sees a clean env.
    unsafe {
        std::env::remove_var("WM_BRAIN_BUS_SOCKET");
    }

    // 8. The implicit anti-announce_required check: if the daemon had
    //    failed at announce, it would have exited within ~1 s of
    //    contacting the bus and its anyhow chain would surface
    //    `announce_required`. We log the outcome and bail loudly if
    //    that string appears.
    match &daemon_outcome {
        Err(_) => eprintln!("daemon_outcome: still running at 3s (expected — bus drove its exit)"),
        Ok(Err(join_err)) => eprintln!("daemon_outcome: JoinError: {join_err}"),
        Ok(Ok(Ok(()))) => eprintln!("daemon_outcome: clean exit (expected once bus closed)"),
        Ok(Ok(Err(e))) => eprintln!("daemon_outcome: Err: {e:#}"),
    }
    if let Ok(Ok(Err(daemon_err))) = daemon_outcome {
        let chain = format!("{daemon_err:#}");
        if chain.contains("announce_required") {
            return Err(format!(
                "daemon hit announce_required — bus wire-up regression: {chain}"
            ));
        }
        return Err(format!("daemon exited with error: {chain}"));
    }

    if !saw_sub {
        return Err(format!(
            "wm-brain sub-client session-id {want_sub} not in peers: {session_ids:?}"
        ));
    }
    if !saw_pub {
        return Err(format!(
            "wm-brain pub-client session-id {want_pub} not in peers: {session_ids:?}"
        ));
    }
    Ok(())
}

#[test]
fn wm_brain_bus_smoke_announces_before_subscribe() {
    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .worker_threads(2)
        .build()
        .expect("build tokio runtime");
    rt.block_on(async {
        run_bus_smoke().await.expect("wm-brain bus smoke lifecycle");
    });
}
