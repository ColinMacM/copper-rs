//! The cycle thread allocates nothing with the policy link live and chunks flowing. A counting
//! global allocator counts per thread, so the link's worker thread (which does allocate, by
//! design) is outside the measurement.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::process::{Command, Stdio};

use cu_vla_loop::{listen_config, run_hooked};

thread_local! {
    static ON: Cell<bool> = const { Cell::new(false) };
    static ALLOCS: Cell<usize> = const { Cell::new(0) };
    static BYTES: Cell<usize> = const { Cell::new(0) };
}

struct Counting;
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        note(l.size());
        unsafe { System.alloc(l) }
    }
    unsafe fn alloc_zeroed(&self, l: Layout) -> *mut u8 {
        note(l.size());
        unsafe { System.alloc_zeroed(l) }
    }
    unsafe fn realloc(&self, p: *mut u8, l: Layout, n: usize) -> *mut u8 {
        note(n);
        unsafe { System.realloc(p, l, n) }
    }
    unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
        unsafe { System.dealloc(p, l) }
    }
}

fn note(n: usize) {
    let _ = ON.try_with(|on| {
        if on.get() {
            let _ = ALLOCS.try_with(|a| a.set(a.get() + 1));
            let _ = BYTES.try_with(|b| b.set(b.get() + n));
        }
    });
}

#[global_allocator]
static A: Counting = Counting;

const CYCLES: usize = 330;
const WARMUP: usize = 120;

#[test]
fn the_cycle_thread_allocates_nothing_with_the_link_live() {
    let ok = Command::new("python3")
        .args(["-c", "import zenoh"])
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .is_ok_and(|s| s.success());
    if !ok {
        eprintln!("skipped: python3 cannot import zenoh");
        return;
    }
    // Control: the counter must see a deliberate allocation, or a zero below proves nothing.
    ALLOCS.with(|a| a.set(0));
    ON.with(|o| o.set(true));
    let probe = std::hint::black_box(vec![1u8; 10]);
    ON.with(|o| o.set(false));
    assert!(
        ALLOCS.with(Cell::get) >= 1,
        "the counting allocator is not counting"
    );
    drop(probe);
    // Copper's pool allocates the handle's Arc on every `acquire`; that is the camera's cost,
    // not the link's, so counting is paused across exactly that call.
    let _ = cu_vla_loop::POOL_ACQUIRE_PROBE.set(|enter| {
        thread_local!(static WAS_ON: Cell<bool> = const { Cell::new(false) });
        if enter {
            WAS_ON.with(|w| w.set(ON.with(Cell::get)));
            ON.with(|o| o.set(false));
        } else {
            ON.with(|o| o.set(WAS_ON.with(Cell::get)));
        }
    });
    let port = {
        let l = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        l.local_addr().unwrap().port()
    };
    let dir = tempfile::tempdir().unwrap();
    let mut child = None;
    let mut per_cycle = [0usize; CYCLES];
    let mut bytes = 0usize;
    let started = Cell::new(std::time::Instant::now());
    let mut busy_ns = [0u64; CYCLES];
    run_hooked(
        CYCLES,
        30.0,
        &dir.path().join("a.copper"),
        &listen_config(port),
        |i| {
            if i == 0 {
                child = Some(
                    Command::new("python3")
                        .args([
                            "-m",
                            "vla_runner",
                            "--connect-port",
                            &port.to_string(),
                            "--seconds",
                            "14",
                        ])
                        .current_dir(concat!(env!("CARGO_MANIFEST_DIR"), "/python"))
                        .stdout(Stdio::null())
                        .stderr(Stdio::null())
                        .spawn()
                        .unwrap(),
                );
            }
            started.set(std::time::Instant::now());
            if i >= WARMUP {
                ALLOCS.with(|a| a.set(0));
                BYTES.with(|b| b.set(0));
                ON.with(|o| o.set(true));
            }
        },
        |i| {
            busy_ns[i] = started.get().elapsed().as_nanos() as u64;
            if i >= WARMUP {
                ON.with(|o| o.set(false));
                per_cycle[i] = ALLOCS.with(Cell::get);
                bytes += BYTES.with(Cell::get);
            }
        },
    )
    .unwrap();
    if let Some(mut c) = child {
        let _ = c.kill();
        let _ = c.wait();
    }
    let mut busy: Vec<u64> = busy_ns[WARMUP..].to_vec();
    busy.sort_unstable();
    let pct = |q: f64| busy[((busy.len() - 1) as f64 * q) as usize];
    println!(
        "busy ns per cycle (link live): p50={} p99={} max={}",
        pct(0.5),
        pct(0.99),
        busy[busy.len() - 1]
    );
    let window = &per_cycle[WARMUP..];
    let total: usize = window.iter().sum();
    let worst = window.iter().max().copied().unwrap_or(0);
    println!(
        "cycles measured: {}, allocations: {total}, bytes: {bytes}, worst cycle: {worst}",
        window.len()
    );
    assert_eq!(
        total, 0,
        "{total} allocations ({bytes} bytes) on the cycle thread; worst cycle {worst}"
    );
}
