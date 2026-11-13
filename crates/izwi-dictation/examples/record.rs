//! Record from the microphone for a few seconds and write `dictation.wav`:
//! `cargo run -p izwi-dictation --example record -- [seconds] [--warm]`.
//! Prints how long the microphone took to deliver its first audio.

use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use izwi_dictation::audio;
use izwi_dictation::capture::{microphones, CaptureConfig, CaptureEvent, Recorder};

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let seconds: u64 = args.iter().find_map(|arg| arg.parse().ok()).unwrap_or(3);
    let keep_ready = args.iter().any(|arg| arg == "--warm");
    for microphone in microphones() {
        let default = if microphone.is_default {
            " (default)"
        } else {
            ""
        };
        println!("microphone: {}{default}", microphone.name);
    }

    let first_audio = Arc::new(Mutex::new(None::<Instant>));
    let seen = Arc::clone(&first_audio);
    let recorder = Recorder::new(
        CaptureConfig {
            microphone: None,
            keep_ready,
        },
        Arc::new(move |event| {
            if event == CaptureEvent::Listening {
                *seen.lock().unwrap() = Some(Instant::now());
            }
        }),
    );
    if keep_ready {
        std::thread::sleep(Duration::from_secs(1));
    }
    let pressed = Instant::now();
    if let Err(error) = recorder.start() {
        eprintln!("start failed: {error}");
        std::process::exit(1);
    }
    println!("opened in {} ms", pressed.elapsed().as_millis());
    std::thread::sleep(Duration::from_secs(seconds));
    match recorder.stop() {
        Ok(samples) => {
            if let Some(at) = *first_audio.lock().unwrap() {
                println!("first audio after {} ms", (at - pressed).as_millis());
            }
            println!(
                "{:.2} s recorded, level {:.1} dBFS",
                samples.len() as f32 / audio::SAMPLE_RATE as f32,
                audio::level_dbfs(&samples)
            );
            match audio::speech_clip(&samples) {
                Some(clip) => println!(
                    "speech: {:.2} s",
                    clip.len() as f32 / audio::SAMPLE_RATE as f32
                ),
                None => println!("no speech detected"),
            }
            std::fs::write("dictation.wav", audio::wav_bytes(&samples)).unwrap();
        }
        Err(error) => eprintln!("stop failed: {error}"),
    }
}
