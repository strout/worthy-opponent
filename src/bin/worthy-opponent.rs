extern crate tiny_http;

use tiny_http::ServerBuilder;

fn main() {
    let srv = ServerBuilder::new().with_port(8080).build().unwrap();
    for mut req in srv.incoming_requests() {
        let mut body = String::new();
        req.as_reader().read_to_string(&mut body).unwrap();
        println!("Got {}", body);
    }
}
