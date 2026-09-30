//! A fake Feetech servo bus: a `SerialPort` that speaks the packet protocol, for tests.

use std::collections::{BTreeMap, VecDeque};
use std::io::{self, Read, Write};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use serialport::{ClearBuffer, DataBits, FlowControl, Parity, SerialPort, StopBits};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Event {
    Torque(u8, bool),
    /// `(id, goal)` pairs of one sync-write packet.
    Goals(Vec<(u8, u16)>),
}

#[derive(Debug, Clone, Default)]
pub struct Servo {
    pub position: u16,
    pub goal: u16,
    pub torque: bool,
    /// Error byte returned with every status packet.
    pub error: u8,
    /// Reads of this servo time out.
    pub mute: bool,
}

#[derive(Debug, Default)]
pub struct Bus {
    pub servos: BTreeMap<u8, Servo>,
    pub events: Vec<Event>,
    rx: VecDeque<u8>,
}

pub type Shared = Arc<Mutex<Bus>>;

pub struct BusPort(pub Shared);

pub fn bus(ids: &[(u8, u16)]) -> Shared {
    let mut b = Bus::default();
    for (id, pos) in ids {
        // A goal left over from an earlier session, different from where the arm is.
        b.servos.insert(
            *id,
            Servo {
                position: *pos,
                goal: pos.wrapping_add(700),
                ..Servo::default()
            },
        );
    }
    Arc::new(Mutex::new(b))
}

fn status(id: u8, error: u8, data: &[u8]) -> Vec<u8> {
    let length = (data.len() + 2) as u8;
    let mut p = vec![0xFF, 0xFF, id, length, error];
    p.extend_from_slice(data);
    let sum = p[2..].iter().fold(0u8, |a, b| a.wrapping_add(*b));
    p.push(!sum);
    p
}

impl Bus {
    fn handle(&mut self, packet: &[u8]) {
        let (id, instr, params) = (packet[2], packet[4], &packet[5..packet.len() - 1]);
        match instr {
            0x02 => {
                let (addr, count) = (params[0], params[1] as usize);
                let Some(s) = self.servos.get(&id) else {
                    return;
                };
                if s.mute {
                    return;
                }
                let data: Vec<u8> = match addr {
                    56 => s.position.to_le_bytes()[..count.min(2)].to_vec(),
                    _ => vec![0; count],
                };
                let reply = status(id, s.error, &data);
                self.rx.extend(reply);
            }
            0x03 => {
                let (addr, value) = (params[0], &params[1..]);
                let Some(s) = self.servos.get_mut(&id) else {
                    return;
                };
                if addr == 40 {
                    s.torque = value[0] != 0;
                    self.events.push(Event::Torque(id, s.torque));
                }
                let reply = status(id, s.error, &[]);
                self.rx.extend(reply);
            }
            0x83 => {
                let per = params[1] as usize;
                let mut goals = Vec::new();
                for chunk in params[2..].chunks(1 + per) {
                    let goal = u16::from_le_bytes([chunk[1], chunk[2]]);
                    if let Some(s) = self.servos.get_mut(&chunk[0]) {
                        s.goal = goal;
                    }
                    goals.push((chunk[0], goal));
                }
                self.events.push(Event::Goals(goals));
            }
            _ => {}
        }
    }
}

impl Read for BusPort {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        let mut b = self.0.lock().unwrap();
        if b.rx.is_empty() {
            return Err(io::Error::new(io::ErrorKind::TimedOut, "no reply"));
        }
        let n = buf.len().min(b.rx.len());
        for slot in buf.iter_mut().take(n) {
            *slot = b.rx.pop_front().unwrap();
        }
        Ok(n)
    }
}

impl Write for BusPort {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        // One instruction packet per write; the bridge builds and sends each in one call.
        assert!(
            buf.len() >= 6 && buf[0] == 0xFF && buf[1] == 0xFF,
            "malformed packet {buf:02X?}"
        );
        self.0.lock().unwrap().handle(buf);
        Ok(buf.len())
    }
    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

impl SerialPort for BusPort {
    fn name(&self) -> Option<String> {
        Some("fake-bus".into())
    }
    fn baud_rate(&self) -> serialport::Result<u32> {
        Ok(1_000_000)
    }
    fn data_bits(&self) -> serialport::Result<DataBits> {
        Ok(DataBits::Eight)
    }
    fn flow_control(&self) -> serialport::Result<FlowControl> {
        Ok(FlowControl::None)
    }
    fn parity(&self) -> serialport::Result<Parity> {
        Ok(Parity::None)
    }
    fn stop_bits(&self) -> serialport::Result<StopBits> {
        Ok(StopBits::One)
    }
    fn timeout(&self) -> Duration {
        Duration::from_millis(1)
    }
    fn set_baud_rate(&mut self, _: u32) -> serialport::Result<()> {
        Ok(())
    }
    fn set_data_bits(&mut self, _: DataBits) -> serialport::Result<()> {
        Ok(())
    }
    fn set_flow_control(&mut self, _: FlowControl) -> serialport::Result<()> {
        Ok(())
    }
    fn set_parity(&mut self, _: Parity) -> serialport::Result<()> {
        Ok(())
    }
    fn set_stop_bits(&mut self, _: StopBits) -> serialport::Result<()> {
        Ok(())
    }
    fn set_timeout(&mut self, _: Duration) -> serialport::Result<()> {
        Ok(())
    }
    fn write_request_to_send(&mut self, _: bool) -> serialport::Result<()> {
        Ok(())
    }
    fn write_data_terminal_ready(&mut self, _: bool) -> serialport::Result<()> {
        Ok(())
    }
    fn read_clear_to_send(&mut self) -> serialport::Result<bool> {
        Ok(false)
    }
    fn read_data_set_ready(&mut self) -> serialport::Result<bool> {
        Ok(false)
    }
    fn read_ring_indicator(&mut self) -> serialport::Result<bool> {
        Ok(false)
    }
    fn read_carrier_detect(&mut self) -> serialport::Result<bool> {
        Ok(false)
    }
    fn bytes_to_read(&self) -> serialport::Result<u32> {
        Ok(self.0.lock().unwrap().rx.len() as u32)
    }
    fn bytes_to_write(&self) -> serialport::Result<u32> {
        Ok(0)
    }
    fn clear(&self, _: ClearBuffer) -> serialport::Result<()> {
        Ok(())
    }
    fn try_clone(&self) -> serialport::Result<Box<dyn SerialPort>> {
        Ok(Box::new(BusPort(Arc::clone(&self.0))))
    }
    fn set_break(&self) -> serialport::Result<()> {
        Ok(())
    }
    fn clear_break(&self) -> serialport::Result<()> {
        Ok(())
    }
}
