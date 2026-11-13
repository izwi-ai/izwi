//! Microphone capture. One thread owns the input stream (cpal streams must
//! stay on the thread that built them) and turns whatever the device
//! delivers into 16 kHz mono.
//!
//! The microphone is opened when a recording starts and closed when it ends,
//! unless `keep_ready` is set: then it stays open and the last moments
//! before the hotkey are kept, so the first word is never clipped.

use std::collections::VecDeque;
use std::sync::mpsc::{self, Receiver, RecvTimeoutError, Sender};
use std::sync::Arc;
use std::time::{Duration, Instant};

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{FromSample, Sample, SampleFormat, SizedSample};
use izwi_audio_codec::StreamResampler;

use crate::audio::{self, SAMPLE_RATE};

/// Audio from before the hotkey that starts a recording when the microphone
/// was already open.
const PRE_ROLL_MS: usize = 450;
/// Recording goes on this long after the hotkey is released: people let go
/// while still finishing the last word.
const RELEASE_TAIL_MS: u64 = 200;
/// A cold microphone that delivers nothing for this long has failed.
const FIRST_AUDIO_TIMEOUT: Duration = Duration::from_secs(4);
const METER_INTERVAL: Duration = Duration::from_millis(50);

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum CaptureError {
    #[error("No microphone was found.")]
    NoMicrophone,
    #[error("The microphone \"{0}\" is not connected.")]
    MicrophoneMissing(String),
    #[error("The microphone could not be opened: {0}")]
    Open(String),
    #[error("The microphone delivered no audio. Check that Izwi is allowed to use it.")]
    Silent,
    #[error("No recording is in progress.")]
    NotRecording,
    #[error("The audio thread stopped.")]
    Stopped,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Microphone {
    pub name: String,
    pub is_default: bool,
}

/// The input devices the system offers, default first.
pub fn microphones() -> Vec<Microphone> {
    let host = cpal::default_host();
    let default_name = host
        .default_input_device()
        .and_then(|device| name_of(&device));
    let mut found: Vec<Microphone> = Vec::new();
    if let Ok(devices) = host.input_devices() {
        for name in devices.filter_map(|device| name_of(&device)) {
            if !found.iter().any(|known| known.name == name) {
                found.push(Microphone {
                    is_default: default_name.as_deref() == Some(name.as_str()),
                    name,
                });
            }
        }
    }
    found.sort_by_key(|microphone| !microphone.is_default);
    found
}

fn name_of(device: &cpal::Device) -> Option<String> {
    device
        .description()
        .ok()
        .map(|description| description.name().to_string())
}

/// What the recorder reports while a recording is in progress.
#[derive(Debug, Clone, PartialEq)]
pub enum CaptureEvent {
    /// The first audio arrived: the user can speak now.
    Listening,
    /// Input level, 0..1, for the indicator.
    Level(f32),
    /// New 16 kHz mono audio, for live transcription.
    Audio(Vec<f32>),
}

pub type CaptureSink = Arc<dyn Fn(CaptureEvent) + Send + Sync>;

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct CaptureConfig {
    /// Device name; the system default when unset.
    pub microphone: Option<String>,
    pub keep_ready: bool,
}

enum Command {
    Configure(CaptureConfig),
    Start(Sender<Result<(), CaptureError>>),
    Stop(Sender<Result<Vec<f32>, CaptureError>>),
    Discard,
}

/// Handle to the capture thread. Dropping it closes the microphone.
pub struct Recorder {
    commands: Sender<Command>,
}

impl Recorder {
    pub fn new(config: CaptureConfig, sink: CaptureSink) -> Self {
        let (commands, inbox) = mpsc::channel();
        let _ = std::thread::Builder::new()
            .name("izwi-dictation-audio".to_string())
            .spawn(move || Worker::new(config, sink).run(inbox));
        Self { commands }
    }

    /// Apply new settings. A recording in progress keeps its microphone.
    pub fn configure(&self, config: CaptureConfig) {
        let _ = self.commands.send(Command::Configure(config));
    }

    /// Begin a recording. Returns once the microphone is open (at once when
    /// it was kept ready).
    pub fn start(&self) -> Result<(), CaptureError> {
        let (reply, result) = mpsc::channel();
        self.commands
            .send(Command::Start(reply))
            .map_err(|_| CaptureError::Stopped)?;
        result.recv().map_err(|_| CaptureError::Stopped)?
    }

    /// End the recording and return it as 16 kHz mono samples.
    pub fn stop(&self) -> Result<Vec<f32>, CaptureError> {
        let (reply, result) = mpsc::channel();
        self.commands
            .send(Command::Stop(reply))
            .map_err(|_| CaptureError::Stopped)?;
        result.recv().map_err(|_| CaptureError::Stopped)?
    }

    /// Throw the recording away.
    pub fn discard(&self) {
        let _ = self.commands.send(Command::Discard);
    }
}

struct OpenMicrophone {
    /// Dropping the stream closes the device.
    _stream: cpal::Stream,
    audio: Receiver<Vec<f32>>,
    resampler: StreamResampler,
    /// Which device the user asked for when it was opened.
    requested: Option<String>,
}

struct Take {
    samples: Vec<f32>,
    listening: bool,
    opened_at: Instant,
    last_meter: Instant,
    meter_window: Vec<f32>,
}

struct Worker {
    config: CaptureConfig,
    sink: CaptureSink,
    microphone: Option<OpenMicrophone>,
    pre_roll: VecDeque<f32>,
    take: Option<Take>,
}

impl Worker {
    fn new(config: CaptureConfig, sink: CaptureSink) -> Self {
        Self {
            config,
            sink,
            microphone: None,
            pre_roll: VecDeque::new(),
            take: None,
        }
    }

    fn run(mut self, inbox: Receiver<Command>) {
        self.apply_keep_ready();
        loop {
            // With the microphone open, wait on its audio and look at the
            // commands between chunks; otherwise sleep until a command.
            let command = if self.microphone.is_some() {
                self.pump(Duration::from_millis(10));
                match inbox.try_recv() {
                    Ok(command) => Some(command),
                    Err(mpsc::TryRecvError::Empty) => None,
                    Err(mpsc::TryRecvError::Disconnected) => return,
                }
            } else {
                match inbox.recv() {
                    Ok(command) => Some(command),
                    Err(_) => return,
                }
            };
            match command {
                Some(Command::Configure(config)) => {
                    self.config = config;
                    if self.take.is_none() {
                        self.apply_keep_ready();
                    }
                }
                Some(Command::Start(reply)) => {
                    let _ = reply.send(self.start());
                }
                Some(Command::Stop(reply)) => {
                    let _ = reply.send(self.stop());
                }
                Some(Command::Discard) => {
                    self.take = None;
                    self.apply_keep_ready();
                }
                None => {}
            }
        }
    }

    /// Open or close the idle microphone to match `keep_ready`, reopening it
    /// when the chosen device changed.
    fn apply_keep_ready(&mut self) {
        let stale = self
            .microphone
            .as_ref()
            .is_some_and(|open| open.requested != self.config.microphone);
        if !self.config.keep_ready || stale {
            self.microphone = None;
            self.pre_roll.clear();
        }
        if self.config.keep_ready && self.microphone.is_none() {
            // Failure is reported when a recording starts.
            self.microphone = open_microphone(self.config.microphone.as_deref()).ok();
        }
    }

    fn start(&mut self) -> Result<(), CaptureError> {
        if self.microphone.is_none() {
            self.microphone = Some(open_microphone(self.config.microphone.as_deref())?);
        }
        let now = Instant::now();
        let warm = !self.pre_roll.is_empty();
        self.take = Some(Take {
            samples: self.pre_roll.drain(..).collect(),
            listening: false,
            opened_at: now,
            last_meter: now,
            meter_window: Vec::new(),
        });
        if warm {
            self.mark_listening();
            // Live transcription hears the pre-roll too.
            if let Some(take) = &self.take {
                (self.sink)(CaptureEvent::Audio(take.samples.clone()));
            }
        }
        Ok(())
    }

    fn stop(&mut self) -> Result<Vec<f32>, CaptureError> {
        if self.take.is_none() {
            return Err(CaptureError::NotRecording);
        }
        let until = Instant::now() + Duration::from_millis(RELEASE_TAIL_MS);
        while Instant::now() < until && self.microphone.is_some() {
            self.pump(Duration::from_millis(10));
        }
        let take = self.take.take().ok_or(CaptureError::NotRecording)?;
        self.apply_keep_ready();
        if !take.listening || audio::is_digital_silence(&take.samples) {
            return Err(CaptureError::Silent);
        }
        Ok(take.samples)
    }

    fn mark_listening(&mut self) {
        if let Some(take) = self.take.as_mut() {
            if !take.listening {
                take.listening = true;
                (self.sink)(CaptureEvent::Listening);
            }
        }
    }

    /// Move audio that has arrived from the device into the recording, or
    /// into the pre-roll when idle. Waits up to `wait` for the first chunk.
    fn pump(&mut self, wait: Duration) {
        let Some(microphone) = self.microphone.as_mut() else {
            return;
        };
        let mut fresh: Vec<f32> = Vec::new();
        let mut disconnected = false;
        match microphone.audio.recv_timeout(wait) {
            Ok(chunk) => {
                fresh.extend(microphone.resampler.push(&chunk).unwrap_or_default());
                while let Ok(chunk) = microphone.audio.try_recv() {
                    fresh.extend(microphone.resampler.push(&chunk).unwrap_or_default());
                }
            }
            Err(RecvTimeoutError::Timeout) => {}
            Err(RecvTimeoutError::Disconnected) => disconnected = true,
        }
        if disconnected {
            // The device went away; a recording keeps what it has.
            self.microphone = None;
            return;
        }
        if fresh.is_empty() {
            if let Some(take) = &self.take {
                if !take.listening && take.opened_at.elapsed() > FIRST_AUDIO_TIMEOUT {
                    self.microphone = None;
                }
            }
            return;
        }
        if self.take.is_none() {
            self.pre_roll.extend(fresh);
            let keep = PRE_ROLL_MS * SAMPLE_RATE as usize / 1000;
            while self.pre_roll.len() > keep {
                self.pre_roll.pop_front();
            }
            return;
        }
        self.mark_listening();
        let Some(take) = self.take.as_mut() else {
            return;
        };
        take.samples.extend_from_slice(&fresh);
        take.meter_window.extend_from_slice(&fresh);
        if take.last_meter.elapsed() >= METER_INTERVAL {
            let level = audio::meter(&take.meter_window);
            take.meter_window.clear();
            take.last_meter = Instant::now();
            (self.sink)(CaptureEvent::Level(level));
        }
        (self.sink)(CaptureEvent::Audio(fresh));
    }
}

fn open_microphone(requested: Option<&str>) -> Result<OpenMicrophone, CaptureError> {
    let host = cpal::default_host();
    let device = match requested {
        Some(name) => host
            .input_devices()
            .map_err(|error| CaptureError::Open(error.to_string()))?
            .find(|device| name_of(device).as_deref() == Some(name))
            .ok_or_else(|| CaptureError::MicrophoneMissing(name.to_string()))?,
        None => host
            .default_input_device()
            .ok_or(CaptureError::NoMicrophone)?,
    };
    let supported = device
        .default_input_config()
        .map_err(|error| CaptureError::Open(error.to_string()))?;
    let channels = usize::from(supported.channels()).max(1);
    let rate = supported.sample_rate();
    let format = supported.sample_format();
    let (sender, audio) = mpsc::channel();
    let stream = match format {
        SampleFormat::F32 => build::<f32>(&device, supported.into(), channels, sender),
        SampleFormat::I16 => build::<i16>(&device, supported.into(), channels, sender),
        SampleFormat::U16 => build::<u16>(&device, supported.into(), channels, sender),
        SampleFormat::I32 => build::<i32>(&device, supported.into(), channels, sender),
        SampleFormat::I8 => build::<i8>(&device, supported.into(), channels, sender),
        other => Err(CaptureError::Open(format!(
            "unsupported sample format {other}"
        ))),
    }?;
    stream
        .play()
        .map_err(|error| CaptureError::Open(error.to_string()))?;
    let resampler = StreamResampler::new(rate, SAMPLE_RATE)
        .map_err(|error| CaptureError::Open(error.to_string()))?;
    Ok(OpenMicrophone {
        _stream: stream,
        audio,
        resampler,
        requested: requested.map(str::to_string),
    })
}

/// Build an input stream that sends each buffer, mixed down to mono.
fn build<T>(
    device: &cpal::Device,
    config: cpal::StreamConfig,
    channels: usize,
    sender: Sender<Vec<f32>>,
) -> Result<cpal::Stream, CaptureError>
where
    T: SizedSample,
    f32: FromSample<T>,
{
    device
        .build_input_stream(
            config,
            move |data: &[T], _: &cpal::InputCallbackInfo| {
                let mono: Vec<f32> = data
                    .chunks(channels)
                    .map(|frame| {
                        frame
                            .iter()
                            .map(|sample| f32::from_sample(*sample))
                            .sum::<f32>()
                            / frame.len() as f32
                    })
                    .collect();
                let _ = sender.send(mono);
            },
            // Stream errors end in a closed channel or no audio, which the
            // worker reports when the recording stops.
            |_error| {},
            None,
        )
        .map_err(|error| CaptureError::Open(error.to_string()))
}
