use mio::net::UdpSocket;
use pyo3::prelude::*;
use std::io;
use std::net::{SocketAddr, UdpSocket as StdUdpSocket};

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

pub fn recvfrom_byte(socket: &StdUdpSocket) -> PyResult<SocketAddr> {
    let (_, child_addr) = socket.recv_from(&mut [0])?;
    Ok(child_addr)
}
pub fn sendto_byte(socket: &StdUdpSocket, address: SocketAddr) -> PyResult<()> {
    socket.send_to(&[0], address)?;
    Ok(())
}
pub fn mio_sendto_byte(socket: &UdpSocket, address: SocketAddr) -> PyResult<()> {
    socket.send_to(&[0], address)?;
    Ok(())
}

pub fn get_flink(flinks_folder: &str, proc_id: u128) -> String {
    format!("{}/{}", flinks_folder, proc_id)
}
