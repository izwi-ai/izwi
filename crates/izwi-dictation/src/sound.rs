//! The short tones that mark the start and end of a recording, so the user
//! knows when to speak without looking at the screen.

use std::time::Duration;

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::SampleFormat;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Tone {
    /// Recording started: rising.
    Start,
    /// Recording ended: falling.
    Stop,
    /// Something went wrong: low and flat.
    Error,
}

impl Tone {
    /// Frequency at the start and at the end, in Hz.
    fn sweep(self) -> (f32, f32) {
        match self {
            Tone::Start => (587.0, 880.0),
            Tone::Stop => (880.0, 587.0),
            Tone::Error => (247.0, 233.0),
        }
    }

    fn duration(self) -> Duration {
        match self {
            Tone::Start | Tone::Stop => Duration::from_millis(110),
            Tone::Error => Duration::from_millis(220),
        }
    }
}

const VOLUME: f32 = 0.12;

/// One mono sample of `tone` at `position` (0..1 through the tone).
fn sample(tone: Tone, position: f32, phase: &mut f32, rate: f32) -> f32 {
    let (from, to) = tone.sweep();
    let frequency = from + (to - from) * position;
    *phase = (*phase + frequency / rate).fract();
    // Fade in and out so the tone does not click.
    let envelope = (position * 8.0).min(1.0) * ((1.0 - position) * 5.0).min(1.0);
    (*phase * std::f32::consts::TAU).sin() * envelope * VOLUME
}

/// Play `tone` on the default output without blocking the caller. Fails
/// silently: a missing output device must not stop a dictation.
pub fn play(tone: Tone) {
    let _ = std::thread::Builder::new()
        .name("izwi-dictation-tone".to_string())
        .spawn(move || {
            let _ = play_blocking(tone);
        });
}

fn play_blocking(tone: Tone) -> Option<()> {
    let device = cpal::default_host().default_output_device()?;
    let supported = device.default_output_config().ok()?;
    let format = supported.sample_format();
    let config: cpal::StreamConfig = supported.into();
    let channels = usize::from(config.channels).max(1);
    let rate = config.sample_rate as f32;
    let total = (rate * tone.duration().as_secs_f32()) as usize;
    let mut frame = 0usize;
    let mut phase = 0.0f32;
    let mut next = move || {
        let value = if frame < total {
            sample(tone, frame as f32 / total as f32, &mut phase, rate)
        } else {
            0.0
        };
        frame += 1;
        value
    };
    let stream = match format {
        SampleFormat::F32 => device.build_output_stream(
            config,
            move |data: &mut [f32], _: &cpal::OutputCallbackInfo| {
                for out in data.chunks_mut(channels) {
                    out.fill(next());
                }
            },
            |_| {},
            None,
        ),
        SampleFormat::I16 => device.build_output_stream(
            config,
            move |data: &mut [i16], _: &cpal::OutputCallbackInfo| {
                for out in data.chunks_mut(channels) {
                    out.fill((next() * f32::from(i16::MAX)) as i16);
                }
            },
            |_| {},
            None,
        ),
        _ => return None,
    }
    .ok()?;
    stream.play().ok()?;
    std::thread::sleep(tone.duration() + Duration::from_millis(80));
    Some(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tones_are_quiet_and_fade_at_both_ends() {
        for tone in [Tone::Start, Tone::Stop, Tone::Error] {
            let rate = 48_000.0;
            let total = (rate * tone.duration().as_secs_f32()) as usize;
            let mut phase = 0.0;
            let samples: Vec<f32> = (0..total)
                .map(|frame| sample(tone, frame as f32 / total as f32, &mut phase, rate))
                .collect();
            assert!(samples.iter().all(|value| value.abs() <= VOLUME));
            assert!(samples[0].abs() < 0.001);
            assert!(samples[total - 1].abs() < 0.01);
            assert!(samples.iter().any(|value| value.abs() > VOLUME * 0.8));
        }
    }
}
