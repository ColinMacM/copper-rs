//! The wire vectors in `tests/golden/vectors.json` were written by an independent encoder; the
//! Python tests read the same file. Every vector must encode to its bytes and decode from them,
//! and every rejected input must fail with its error.

use cu_policy::wire::{
    self, CHUNK_LEN, Exec, IMAGE_HEADER_BYTES, ImageHeader, OBS_JOINTS, PolicyOptions, SliceWriter,
    WireError,
};
use serde_json::Value;

fn vectors() -> Value {
    serde_json::from_str(include_str!("golden/vectors.json")).unwrap()
}

fn unhex(s: &str) -> Vec<u8> {
    (0..s.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&s[i..i + 2], 16).unwrap())
        .collect()
}

fn floats(v: &Value) -> Vec<f32> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_f64().unwrap() as f32)
        .collect()
}

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

fn u64_of(v: &Value, key: &str) -> u64 {
    v[key].as_u64().unwrap()
}

fn u32_of(v: &Value, key: &str) -> u32 {
    u32::try_from(u64_of(v, key)).unwrap()
}

#[test]
fn every_vector_encodes_to_its_bytes_and_decodes_from_them() {
    let doc = vectors();
    let mut seen = std::collections::BTreeSet::new();
    for v in doc["vectors"].as_array().unwrap() {
        let name = v["name"].as_str().unwrap();
        let kind = v["kind"].as_str().unwrap();
        let want = unhex(v["hex"].as_str().unwrap());
        let f = &v["fields"];
        seen.insert(kind.to_owned());
        let mut buf = vec![0u8; 4096];
        match kind {
            "obs" => {
                let state = floats(&f["state"]);
                let n = wire::encode_obs(&mut buf, u64_of(f, "seq"), u64_of(f, "tov_ns"), &state)
                    .unwrap();
                assert_eq!(&buf[..n], want.as_slice(), "{name}");
                let (obs, used) = wire::decode_obs(&want).unwrap();
                assert_eq!(used, want.len(), "{name}");
                assert_eq!(
                    (obs.seq, obs.tov_ns),
                    (u64_of(f, "seq"), u64_of(f, "tov_ns"))
                );
                assert_eq!(bits(obs.state()), bits(&state), "{name}");
            }
            "chunk" => {
                let values = floats(&f["values"]);
                let n = wire::encode_chunk(&mut buf, u64_of(f, "obs_seq"), &values).unwrap();
                assert_eq!(&buf[..n], want.as_slice(), "{name}");
                let (chunk, used) = wire::decode_chunk(&want).unwrap();
                assert_eq!(used, want.len(), "{name}");
                assert_eq!(chunk.obs_seq, u64_of(f, "obs_seq"));
                assert_eq!(bits(chunk.values()), bits(&values), "{name}");
            }
            "exec" => {
                let e = Exec {
                    stamp_seq: u64_of(f, "stamp_seq"),
                    chunk_seq: u64_of(f, "chunk_seq"),
                    next_index: u32_of(f, "next_index"),
                    flags: u32_of(f, "flags"),
                    accept_skip: u32_of(f, "accept_skip"),
                    reject: u32_of(f, "reject"),
                    tracking_err: f["tracking_err"].as_f64().unwrap() as f32,
                };
                let n = wire::encode_exec(&mut buf, &e).unwrap();
                assert_eq!(&buf[..n], want.as_slice(), "{name}");
                assert_eq!(wire::decode_exec(&want).unwrap(), (e, want.len()), "{name}");
            }
            "request" => {
                let state = floats(&f["state"]);
                let previous = floats(&f["previous"]);
                let options = PolicyOptions {
                    horizon: u32_of(f, "horizon"),
                    mode: u32_of(f, "mode"),
                    denoise_steps: u32_of(f, "denoise_steps"),
                    best_of: u32_of(f, "best_of"),
                    flags: u32_of(f, "flags"),
                    beta: f["beta"].as_f64().unwrap() as f32,
                };
                let n = wire::encode_request(
                    &mut buf,
                    u64_of(f, "obs_seq"),
                    u32_of(f, "delay"),
                    u32_of(f, "executed"),
                    u32_of(f, "reason"),
                    &options,
                    &state,
                    &previous,
                )
                .unwrap();
                assert_eq!(&buf[..n], want.as_slice(), "{name}");
                let (r, used) = wire::decode_request(&want).unwrap();
                assert_eq!(used, want.len(), "{name}");
                assert_eq!(
                    (r.obs_seq, r.delay, r.executed, r.reason),
                    (
                        u64_of(f, "obs_seq"),
                        u32_of(f, "delay"),
                        u32_of(f, "executed"),
                        u32_of(f, "reason")
                    )
                );
                assert_eq!(r.options, options, "{name}");
                assert_eq!(bits(r.state()), bits(&state), "{name}");
                assert_eq!(bits(r.previous()), bits(&previous), "{name}");
            }
            "image_header" => {
                let fmt = f["pixel_format"].as_str().unwrap().as_bytes();
                let h = ImageHeader {
                    seq: u64_of(f, "seq"),
                    tov_ns: u64_of(f, "tov_ns"),
                    width: u32_of(f, "width"),
                    height: u32_of(f, "height"),
                    stride: u32_of(f, "stride"),
                    pixel_format: [fmt[0], fmt[1], fmt[2], fmt[3]],
                    len: u32_of(f, "len"),
                };
                assert_eq!(h.to_bytes().as_slice(), want.as_slice(), "{name}");
                assert_eq!(want.len(), IMAGE_HEADER_BYTES);
                assert_eq!(ImageHeader::from_bytes(&want).unwrap(), h, "{name}");
            }
            other => panic!("unknown kind {other} in {name}"),
        }
    }
    assert_eq!(seen.len(), 5, "a message kind has no vector: {seen:?}");
}

#[test]
fn every_rejected_input_fails_with_its_error() {
    let doc = vectors();
    let rejects = doc["reject"].as_array().unwrap();
    assert!(rejects.len() >= 10);
    for r in rejects {
        let name = r["name"].as_str().unwrap();
        let bytes = unhex(r["hex"].as_str().unwrap());
        let err = match r["kind"].as_str().unwrap() {
            "obs" => wire::decode_obs(&bytes).map(|_| ()),
            "chunk" => wire::decode_chunk(&bytes).map(|_| ()),
            "exec" => wire::decode_exec(&bytes).map(|_| ()),
            "request" => wire::decode_request(&bytes).map(|_| ()),
            "image_header" => ImageHeader::from_bytes(&bytes).map(|_| ()),
            other => panic!("unknown kind {other}"),
        }
        .expect_err(name);
        match r["error"].as_str().unwrap() {
            "truncated" => assert_eq!(err, WireError::Truncated, "{name}"),
            "too_long" => assert!(matches!(err, WireError::TooLong { .. }), "{name}: {err:?}"),
            other => panic!("unknown error {other}"),
        }
    }
}

#[test]
fn a_length_above_the_capacity_is_refused_before_any_value_is_read() {
    // 9 values announced, none present: the capacity check fires, not a truncation.
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&1u64.to_le_bytes());
    bytes.extend_from_slice(&2u64.to_le_bytes());
    bytes.extend_from_slice(&((OBS_JOINTS + 1) as u32).to_le_bytes());
    assert_eq!(
        wire::decode_obs(&bytes).unwrap_err(),
        WireError::TooLong {
            max: OBS_JOINTS,
            found: OBS_JOINTS + 1
        }
    );
}

#[test]
fn encoding_refuses_what_the_decoder_would_refuse_and_what_does_not_fit() {
    let mut buf = [0u8; 4096];
    let too_many = vec![0.0f32; CHUNK_LEN + 1];
    assert_eq!(
        wire::encode_chunk(&mut buf, 1, &too_many).unwrap_err(),
        WireError::TooLong {
            max: CHUNK_LEN,
            found: CHUNK_LEN + 1
        }
    );
    let mut small = [0u8; 10];
    assert_eq!(
        wire::encode_obs(&mut small, 1, 2, &[1.0]).unwrap_err(),
        WireError::BufferTooSmall
    );
    let mut exact = [0u8; 12];
    let mut w = SliceWriter::new(&mut exact);
    assert!(w.is_empty());
    wire::write_chunk(&mut w, 1, &[]).unwrap();
    assert_eq!(
        w.len(),
        12,
        "an empty chunk is its sequence number and a zero length"
    );
}

#[test]
fn trailing_bytes_are_left_for_the_caller() {
    let mut buf = [0u8; 64];
    let n = wire::encode_chunk(&mut buf, 3, &[1.0, 2.0]).unwrap();
    buf[n] = 0xAA;
    let (chunk, used) = wire::decode_chunk(&buf[..n + 1]).unwrap();
    assert_eq!((chunk.len, used), (2, n));
}
