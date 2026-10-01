//! Zero-allocation proof: a counting global allocator (per-thread counters, enabled only
//! around the measured region) around (1) the governor task's `process()` called directly
//! with adversarial inputs and (2) whole `run_one_iteration`s of the graph with and
//! without the governor.
use cu_policy::governor::{ActionGovernor, GovernorParams, JointPositions, SchedParams};
use cu_policy::{ActionChunk, ExecState, InferenceRequest, JOINTS, MAX_STEPS, ObsStamp};
use cu29::prelude::*;
use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

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
            ALLOCS.with(|a| a.set(a.get() + 1));
            BYTES.with(|b| b.set(b.get() + n));
        }
    });
}
#[global_allocator]
static A: Counting = Counting;

fn measure<R>(f: impl FnOnce() -> R) -> (R, usize, usize) {
    ALLOCS.with(|a| a.set(0));
    BYTES.with(|b| b.set(0));
    ON.with(|o| o.set(true));
    let r = f();
    ON.with(|o| o.set(false));
    (r, ALLOCS.with(|a| a.get()), BYTES.with(|b| b.get()))
}

fn params() -> GovernorParams {
    GovernorParams {
        min: [-1.0; JOINTS],
        max: [1.0; JOINTS],
        max_step: 0.02,
        max_lead: 0.3,
        max_age_ns: 150_000_000,
        hold_deadline_ns: 300_000_000,
        cycle_ns: 10_000_000,
        time_from_feedback: true,
        // The scheduler is on, with an event threshold the test's feedback crosses, so the
        // request path (including copying the unplayed remainder) is part of what is counted.
        sched: SchedParams {
            s_min: 4,
            margin: 2,
            d_init: 2,
            horizon: 50,
            replan_threshold: 0.05,
            pending_timeout: 25,
            blend_steps: 3,
        },
    }
}

#[test]
fn process_allocates_nothing_on_every_path() {
    let mut gov = ActionGovernor::from_params(params());
    let (ctx, _clock) = CuContext::new_mock_clock();
    let mut out = (
        CuMsg::<JointPositions>::default(),
        CuMsg::<ExecState>::default(),
        CuMsg::<InferenceRequest>::default(),
    );
    let mut fb = CuMsg::<JointPositions>::default();
    let mut chunk = CuMsg::<ActionChunk>::default();
    let mut stamp = CuMsg::<ObsStamp>::default();
    let mut good = ActionChunk::default();
    good.values
        .fill_from_iter((0..MAX_STEPS * JOINTS).map(|i| 0.001 * i as f32));

    let requests = Cell::new(0u32);
    let events = Cell::new(0u32);
    let mut run = |cycle: u64| {
        let now = 1_000_000_000 + cycle * 10_000_000;
        let mut a = JointPositions::new();
        a.fill_from_iter((0..JOINTS).map(|j| {
            if cycle % 53 == 52 && j == 1 {
                f32::NAN
            } else {
                0.0
            }
        }));
        fb.set_payload(a);
        fb.tov = Tov::Time(CuTime(now));
        if cycle.is_multiple_of(5) {
            stamp.set_payload(ObsStamp { seq: cycle / 5 });
        } else {
            stamp.clear_payload();
        }
        // accepted, nan, wild, stale, out-of-order, wrong shape, absent: rotate through them
        match cycle % 14 {
            0 => {
                good.obs_seq = cycle / 5;
                chunk.set_payload(good.clone());
            }
            2 => {
                let mut c = good.clone();
                c.obs_seq = cycle / 5;
                c.values.as_slice(); // clone is stack copy
                let mut v = good.values.clone();
                v.fill_from_iter(
                    good.values
                        .as_slice()
                        .iter()
                        .map(|x| if *x > 0.1 { f32::NAN } else { *x }),
                );
                c.values = v;
                chunk.set_payload(c);
            }
            4 => {
                let mut c = good.clone();
                c.obs_seq = cycle / 5;
                c.values.fill_from_iter((0..60).map(|_| 1e9));
                chunk.set_payload(c);
            }
            6 => {
                let mut c = good.clone();
                c.obs_seq = 0; // long gone from the ring, or older than the last accepted
                chunk.set_payload(c);
            }
            8 => {
                let mut c = good.clone();
                c.obs_seq = cycle / 5;
                c.values.fill_from_iter((0..7).map(|_| 0.0));
                chunk.set_payload(c);
            }
            _ => chunk.clear_payload(),
        }
        gov.process(&ctx, &(&fb, &stamp, &chunk), &mut out).unwrap();
        if let Some(r) = out.2.payload() {
            requests.set(requests.get() + 1);
            if r.reason & InferenceRequest::REASON_EVENT != 0 {
                events.set(events.get() + 1);
            }
        }
    };
    for c in 0..2000 {
        run(c);
    }
    let (_, allocs, bytes) = measure(|| {
        for c in 2000..6000 {
            run(c);
        }
    });
    let core = gov.core();
    println!(
        "process(): 4000 cycles, allocs={allocs} bytes={bytes}; accepted={} shape={} nonfinite={} order={} unknown_obs={} stale={} held={} bad_fb={}",
        core.accepted,
        core.rej_shape,
        core.rej_nonfinite,
        core.rej_order,
        core.rej_unknown_obs,
        core.rej_stale,
        core.held_cycles,
        core.bad_feedback
    );
    println!(
        "requests={} of which event-triggered={}",
        requests.get(),
        events.get()
    );
    assert_eq!(allocs, 0);
    assert!(
        requests.get() > 100 && events.get() > 0,
        "the scheduler never fired, so nothing was proven"
    );
    assert!(core.accepted > 0 && core.rej_shape > 0 && core.rej_nonfinite > 0);
    assert!(core.rej_order + core.rej_unknown_obs > 0);
}

macro_rules! graph {
    ($m:ident, $cfg:literal) => {
        mod $m {
            use super::*;
            #[copper_runtime(config = $cfg)]
            struct App {}
            pub fn allocs_per_window() -> (usize, usize) {
                let dir = tempfile::tempdir().unwrap();
                let app = App::builder()
                    .with_log_path(&dir.path().join("a.copper"), Some(64 * 1024 * 1024))
                    .unwrap()
                    .build()
                    .unwrap();
                let mut running = app.start().unwrap();
                for _ in 0..1500 {
                    running.run_one_iteration().unwrap();
                }
                let (_, a, b) = measure(|| {
                    for _ in 0..3000 {
                        running.run_one_iteration().unwrap();
                    }
                });
                running.stop().unwrap();
                (a, b)
            }
        }
    };
}
graph!(with_gov, "tests/configs/gov.ron");
graph!(without_gov, "tests/configs/base.ron");

#[test]
fn graph_with_governor_allocates_no_more_than_without() {
    // Logger/runtime singletons: run sequentially inside one test.
    let base = without_gov::allocs_per_window();
    let gov = with_gov::allocs_per_window();
    println!(
        "graph 3000 iterations: baseline allocs={} bytes={}; with governor allocs={} bytes={}",
        base.0, base.1, gov.0, gov.1
    );
    assert!(gov.0 <= base.0);
}
