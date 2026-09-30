//! The byte layout the Python runner relies on, checked from the Rust side.
//! ObsPacket: seq u64 | tov_ns u64 | len u32 | len x f32. ActionChunk: obs_seq u64 | len u32 | len x f32, little-endian, fixed-width integers.

use cu_action_governor::{ActionChunk, CHUNK_LEN, OBS_JOINTS, ObsPacket};
use cu29::bincode::{config, decode_from_slice, encode_into_slice};
use cu29::prelude::CuArray;

fn cfg() -> impl cu29::bincode::config::Config {
    config::standard().with_fixed_int_encoding()
}

fn chunk(seq: u64, values: &[f32]) -> ActionChunk {
    let mut v = CuArray::new();
    v.fill_from_iter(values.iter().copied());
    ActionChunk {
        obs_seq: seq,
        values: v,
    }
}

#[test]
fn an_observation_encodes_to_the_documented_bytes() {
    let mut state = CuArray::new();
    state.fill_from_iter([1.5f32, -2.0, 3.25]);
    let mut buf = [0u8; 64];
    let n = encode_into_slice(
        ObsPacket {
            seq: (1 << 40) + 7,
            tov_ns: 123_456_789_012,
            state,
        },
        &mut buf,
        cfg(),
    )
    .unwrap();
    let mut expected = Vec::new();
    expected.extend_from_slice(&((1u64 << 40) + 7).to_le_bytes());
    expected.extend_from_slice(&123_456_789_012u64.to_le_bytes());
    expected.extend_from_slice(&3u32.to_le_bytes());
    for v in [1.5f32, -2.0, 3.25] {
        expected.extend_from_slice(&v.to_le_bytes());
    }
    assert_eq!(&buf[..n], expected.as_slice());
}

#[test]
fn a_chunk_from_python_decodes_and_roundtrips_exactly() {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&9u64.to_le_bytes());
    bytes.extend_from_slice(&12u32.to_le_bytes());
    for i in 0..12 {
        bytes.extend_from_slice(&(i as f32 * 0.5).to_le_bytes());
    }
    let (c, used): (ActionChunk, usize) = decode_from_slice(&bytes, cfg()).unwrap();
    assert_eq!(used, bytes.len());
    assert_eq!(c.obs_seq, 9);
    assert_eq!(c.values.as_slice().len(), 12);
    assert_eq!(c.values.as_slice()[11], 5.5);
    let mut again = [0u8; 256];
    let n = encode_into_slice(&c, &mut again, cfg()).unwrap();
    assert_eq!(&again[..n], bytes.as_slice());
}

#[test]
fn a_full_chunk_fits_and_an_oversized_or_truncated_one_is_refused() {
    let full = chunk(1, &vec![0.25; CHUNK_LEN]);
    let mut buf = vec![0u8; 12 + 4 * CHUNK_LEN];
    let n = encode_into_slice(&full, &mut buf, cfg()).unwrap();
    assert_eq!(n, buf.len());
    assert!(decode_from_slice::<ActionChunk, _>(&buf, cfg()).is_ok());

    let mut over = Vec::new();
    over.extend_from_slice(&1u64.to_le_bytes());
    over.extend_from_slice(&((CHUNK_LEN + 1) as u32).to_le_bytes());
    over.extend(std::iter::repeat_n(0u8, 4 * (CHUNK_LEN + 1)));
    assert!(
        decode_from_slice::<ActionChunk, _>(&over, cfg()).is_err(),
        "longer than the capacity"
    );

    assert!(
        decode_from_slice::<ActionChunk, _>(&buf[..buf.len() - 1], cfg()).is_err(),
        "truncated"
    );
    let mut obs_over = Vec::new();
    obs_over.extend_from_slice(&1u64.to_le_bytes());
    obs_over.extend_from_slice(&0u64.to_le_bytes()); // tov_ns
    obs_over.extend_from_slice(&((OBS_JOINTS + 1) as u32).to_le_bytes());
    obs_over.extend(std::iter::repeat_n(0u8, 4 * (OBS_JOINTS + 1)));
    assert!(decode_from_slice::<ObsPacket, _>(&obs_over, cfg()).is_err());
}

#[test]
fn non_finite_values_decode_unchanged_so_the_governor_can_see_and_reject_them() {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&1u64.to_le_bytes());
    bytes.extend_from_slice(&6u32.to_le_bytes());
    for v in [f32::NAN, f32::INFINITY, 0.0, 0.0, 0.0, 0.0] {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    let (c, _): (ActionChunk, usize) = decode_from_slice(&bytes, cfg()).unwrap();
    assert!(c.values.as_slice()[0].is_nan() && c.values.as_slice()[1].is_infinite());
}
