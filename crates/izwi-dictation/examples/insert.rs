//! Insert text into the focused app after a countdown:
//! `cargo run -p izwi-dictation --example insert -- [paste|type|clipboard] [text]`.

use izwi_dictation::insert;
use izwi_dictation::InsertMethod;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let method = match args.first().map(String::as_str) {
        Some("type") => InsertMethod::Type,
        Some("clipboard") => InsertMethod::Clipboard,
        _ => InsertMethod::Paste,
    };
    let text = args
        .get(1)
        .cloned()
        .unwrap_or_else(|| "Hello from Izwi dictation. Ünïcödé 你好 🎙️ ".to_string());
    let seconds: u64 = args.get(2).and_then(|arg| arg.parse().ok()).unwrap_or(3);
    println!(
        "keyboard control allowed: {}",
        insert::can_control_keyboard()
    );
    println!("focus the target app; inserting in {seconds} seconds");
    std::thread::sleep(std::time::Duration::from_secs(seconds));
    println!("focused app: {:?}", insert::focused_app());
    println!("{:?}", insert::insert(&text, method));
    // Let the clipboard restore run before exiting.
    std::thread::sleep(std::time::Duration::from_secs(1));
}
