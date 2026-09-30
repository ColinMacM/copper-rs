//! Safety behavior of the bridge, against a fake servo bus.

use crate::FeetechBridge;
use crate::messages::JointPositions;
use crate::testbus::{BusPort, Event, Shared, bus};

const IDS: [u8; 3] = [1, 2, 3];
const START: [(u8, u16); 3] = [(1, 2000), (2, 2100), (3, 2200)];

fn joints(v: &[f32]) -> JointPositions {
    let mut p = JointPositions::new();
    p.fill_from_iter(v.iter().copied());
    p
}

fn commander(limits: (u16, u16), timeout_ms: Option<u64>) -> (FeetechBridge, Shared) {
    let b = bus(&START);
    let bridge =
        FeetechBridge::for_test(Box::new(BusPort(b.clone())), &IDS, true, limits, timeout_ms);
    (bridge, b)
}

fn rx(bridge: &mut FeetechBridge, now_ns: u64) -> Option<JointPositions> {
    bridge.cycle_receive(now_ns).0
}

fn goals(b: &Shared) -> Vec<Vec<(u8, u16)>> {
    b.lock()
        .unwrap()
        .events
        .iter()
        .filter_map(|e| match e {
            Event::Goals(g) => Some(g.clone()),
            _ => None,
        })
        .collect()
}

#[test]
fn start_sets_the_goal_to_the_present_position_before_torque_comes_on() {
    let (mut bridge, b) = commander((0, 65535), None);
    bridge.begin().unwrap();
    let events = b.lock().unwrap().events.clone();
    let first_torque = events
        .iter()
        .position(|e| matches!(e, Event::Torque(_, true)))
        .unwrap();
    let hold = events
        .iter()
        .position(|e| matches!(e, Event::Goals(_)))
        .unwrap();
    assert!(
        hold < first_torque,
        "goal was written after torque came on: {events:?}"
    );
    assert_eq!(
        goals(&b)[0],
        vec![(1, 2000), (2, 2100), (3, 2200)],
        "goal is where the arm is"
    );
    let servos = b.lock().unwrap().servos.clone();
    assert!(servos.values().all(|s| s.torque));
    assert_eq!(
        servos[&1].goal, 2000,
        "the stale goal register (2700) was replaced"
    );
}

#[test]
fn start_refuses_to_enable_torque_if_a_servo_cannot_be_read() {
    let (mut bridge, b) = commander((0, 65535), None);
    b.lock().unwrap().servos.get_mut(&2).unwrap().mute = true;
    assert!(bridge.begin().is_err());
    assert!(
        b.lock().unwrap().events.is_empty(),
        "nothing may be written when a servo is silent"
    );
    assert!(b.lock().unwrap().servos.values().all(|s| !s.torque));
}

#[test]
fn a_nan_goal_writes_nothing_instead_of_driving_a_joint_to_raw_zero() {
    let (mut bridge, b) = commander((0, 65535), None);
    bridge.begin().unwrap();
    let before = goals(&b).len();
    for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let e = bridge
            .cycle_send(1, &joints(&[2000.0, bad, 2200.0]))
            .unwrap_err()
            .to_string();
        assert!(
            e.contains("servo 2") && e.contains("nothing was written"),
            "{e}"
        );
    }
    assert_eq!(
        goals(&b).len(),
        before,
        "a refused message must not reach the bus"
    );
    assert_eq!(b.lock().unwrap().servos[&1].goal, 2000);
    assert_eq!(bridge.stats().goals_rejected, 3);
}

#[test]
fn goals_outside_the_calibrated_range_are_clamped() {
    let (mut bridge, b) = commander((1000, 3000), None);
    bridge.begin().unwrap();
    bridge
        .cycle_send(1, &joints(&[0.0, 2500.0, 60000.0]))
        .unwrap();
    assert_eq!(
        goals(&b).last().unwrap(),
        &vec![(1, 1000), (2, 2500), (3, 3000)]
    );
    assert_eq!(bridge.stats().goals_clamped, 2);
}

#[test]
fn a_failed_read_yields_no_measurement_and_never_a_stale_position() {
    let (mut bridge, b) = commander((0, 65535), None);
    bridge.begin().unwrap();
    let ok = rx(&mut bridge, 1).expect("all servos answer");
    assert_eq!(ok.as_slice()[..3], [2000.0, 2100.0, 2200.0]);
    {
        let mut bus = b.lock().unwrap();
        bus.servos.get_mut(&2).unwrap().mute = true;
        bus.servos.get_mut(&1).unwrap().position = 2500; // the arm moved
    }
    assert!(
        rx(&mut bridge, 2).is_none(),
        "the old positions must not be republished"
    );
    assert_eq!(bridge.stats().read_failures, 1);
    b.lock().unwrap().servos.get_mut(&2).unwrap().mute = false;
    let back = rx(&mut bridge, 3).expect("recovers as soon as the servo answers");
    assert_eq!(
        back.as_slice()[0],
        2500.0,
        "the fresh reading, not the cached one"
    );
}

#[test]
fn a_protective_error_cuts_torque_and_latches_a_fault() {
    let (mut bridge, b) = commander((0, 65535), None);
    bridge.begin().unwrap();
    b.lock().unwrap().servos.get_mut(&3).unwrap().error = 0x20; // overload
    assert!(rx(&mut bridge, 1).is_none());
    assert!(
        bridge.fault().unwrap().contains("servo 3") && bridge.fault().unwrap().contains("0x20")
    );
    assert!(
        b.lock().unwrap().servos.values().all(|s| !s.torque),
        "torque must be off on the whole bus"
    );
    let writes = goals(&b).len();
    assert!(
        bridge
            .cycle_send(2, &joints(&[2000.0, 2100.0, 2200.0]))
            .is_err(),
        "goals are refused while faulted"
    );
    assert_eq!(goals(&b).len(), writes);
    // The fault stays latched even if the servo recovers: a person has to look at the arm.
    b.lock().unwrap().servos.get_mut(&3).unwrap().error = 0;
    assert!(rx(&mut bridge, 3).is_none());
}

#[test]
fn a_nonprotective_error_byte_is_counted_but_does_not_stop_the_arm() {
    let (mut bridge, b) = commander((0, 65535), None);
    bridge.begin().unwrap();
    b.lock().unwrap().servos.get_mut(&1).unwrap().error = 0x02; // angle sensor
    assert!(rx(&mut bridge, 1).is_some());
    assert_eq!(bridge.stats().servo_errors, 1);
    assert!(bridge.fault().is_none());
}

#[test]
fn an_overdue_goal_commands_a_hold_at_the_present_position_once() {
    let (mut bridge, b) = commander((0, 65535), Some(100));
    bridge.begin().unwrap();
    bridge
        .cycle_send(1_000_000, &joints(&[2010.0, 2110.0, 2210.0]))
        .unwrap();
    b.lock().unwrap().servos.get_mut(&1).unwrap().position = 2300; // pushed away by a person
    let n = goals(&b).len();
    let _ = rx(&mut bridge, 50_000_000); // 49 ms: not overdue
    assert_eq!(goals(&b).len(), n);
    let _ = rx(&mut bridge, 150_000_000); // 149 ms: overdue
    assert_eq!(goals(&b).len(), n + 1);
    assert_eq!(
        goals(&b).last().unwrap()[0],
        (1, 2300),
        "hold where the arm is now"
    );
    let _ = rx(&mut bridge, 200_000_000);
    assert_eq!(goals(&b).len(), n + 1, "one hold, not one per cycle");
    assert_eq!(bridge.stats().holds, 1);
    // A new goal re-arms the watchdog.
    bridge
        .cycle_send(300_000_000, &joints(&[2300.0, 2110.0, 2210.0]))
        .unwrap();
    let _ = rx(&mut bridge, 450_000_000);
    assert_eq!(bridge.stats().holds, 2);
}

#[test]
fn without_a_goal_timeout_nothing_is_held() {
    let (mut bridge, b) = commander((0, 65535), None);
    bridge.begin().unwrap();
    bridge
        .cycle_send(1, &joints(&[2010.0, 2110.0, 2210.0]))
        .unwrap();
    let n = goals(&b).len();
    let _ = rx(&mut bridge, 10_000_000_000);
    assert_eq!(goals(&b).len(), n);
}

#[test]
fn dropping_the_bridge_turns_torque_off() {
    let (mut bridge, b) = commander((0, 65535), None);
    bridge.begin().unwrap();
    assert!(b.lock().unwrap().servos.values().all(|s| s.torque));
    drop(bridge);
    assert!(b.lock().unwrap().servos.values().all(|s| !s.torque));
}

#[test]
fn a_panic_unwinding_through_the_bridge_turns_torque_off() {
    let (mut bridge, b) = commander((0, 65535), None);
    bridge.begin().unwrap();
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(move || {
        let _own = bridge; // dropped while unwinding
        panic!("a task panicked");
    }));
    assert!(result.is_err());
    assert!(b.lock().unwrap().servos.values().all(|s| !s.torque));
}

#[test]
fn a_bridge_that_never_enabled_torque_does_not_touch_the_bus_on_drop() {
    let b = bus(&START);
    let bridge =
        FeetechBridge::for_test(Box::new(BusPort(b.clone())), &IDS, false, (0, 65535), None);
    drop(bridge);
    assert!(
        b.lock().unwrap().events.is_empty(),
        "read-only mode must leave the servos alone"
    );
}
