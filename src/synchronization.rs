use mio::net::UdpSocket;
use mio::{Events, Interest, Poll, Token};
use pyo3::prelude::*;
use std::io;
use std::net::SocketAddr;

pub fn get_handshake_poll(socket: &mut UdpSocket) -> PyResult<(Poll, Events)> {
    let poll = Poll::new()?;
    let events = Events::with_capacity(1);
    poll.registry()
        .register(socket, Token(0), Interest::READABLE)?;
    Ok((poll, events))
}

pub fn drain_socket(socket: &UdpSocket) -> PyResult<bool> {
    let mut received_any = false;
    loop {
        match socket.recv_from(&mut [0]) {
            Ok(_) => received_any = true,
            Err(e) if e.kind() == io::ErrorKind::WouldBlock => return Ok(received_any),
            Err(e) => Err(e)?,
        }
    }
}

pub fn recvfrom_byte(
    socket: &mut UdpSocket,
    poll: &mut Poll,
    events: &mut Events,
) -> PyResult<SocketAddr> {
    loop {
        poll.poll(events, None)?;
        if !events.iter().any(|event| event.is_readable()) {
            continue;
        }
        let v = socket.recv_from(&mut [0]);
        match v {
            Ok((_, send_addr)) => return Ok(send_addr),
            Err(e) if e.kind() == io::ErrorKind::WouldBlock => continue,
            Err(e) => Err(e)?,
        }
    }
}

pub fn sendto_byte(socket: &UdpSocket, address: SocketAddr) -> PyResult<()> {
    socket.send_to(&[0], address)?;
    Ok(())
}

pub fn get_flink(flinks_folder: &str, proc_id: u128) -> String {
    format!("{}/{}", flinks_folder, proc_id)
}
