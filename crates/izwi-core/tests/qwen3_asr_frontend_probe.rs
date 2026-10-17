//! Real-checkpoint probe for the Qwen3-ASR front end (ignored, env-gated).
//! Qwen3-ASR shares the mel front end with the forced aligner, so a change
//! there must keep its transcripts of the repository fixtures. Runs on CPU,
//! or on the default accelerator with `IZWI_PROBE_DEVICE=auto`.

use izwi_core::RuntimeService;

const QWEN3_ASR: &str = "Qwen3-ASR-0.6B-GGUF";

fn models_root() -> std::path::PathBuf {
    std::env::var_os("IZWI_MODELS_DIR")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| {
            let home = std::env::var_os("HOME").expect("HOME is set");
            std::path::PathBuf::from(home).join("Library/Application Support/izwi/models")
        })
}

fn words(text: &str) -> Vec<String> {
    text.split_whitespace()
        .map(|word| {
            word.chars()
                .filter(|ch| ch.is_alphanumeric())
                .collect::<String>()
                .to_lowercase()
        })
        .filter(|word| !word.is_empty())
        .collect()
}

/// Word-level edit distance.
fn word_errors(reference: &[String], hypothesis: &[String]) -> usize {
    let mut previous = (0..=hypothesis.len()).collect::<Vec<_>>();
    for (i, expected) in reference.iter().enumerate() {
        let mut current = vec![i + 1; hypothesis.len() + 1];
        for (j, actual) in hypothesis.iter().enumerate() {
            let substitution = previous[j] + usize::from(expected != actual);
            current[j + 1] = substitution.min(previous[j + 1] + 1).min(current[j] + 1);
        }
        previous = current;
    }
    previous[hypothesis.len()]
}

#[tokio::test]
#[ignore = "requires the local Qwen3-ASR-0.6B-GGUF checkpoint (IZWI_MODELS_DIR)"]
async fn qwen3_asr_transcribes_the_fixtures() {
    let engine = izwi_core::EngineConfig {
        models_dir: models_root(),
        ..Default::default()
    };
    let runtime = match std::env::var("IZWI_PROBE_DEVICE").as_deref() {
        Ok("auto") => RuntimeService::new(engine),
        _ => {
            RuntimeService::new_assigned(engine, izwi_core::backends::RuntimeDeviceAssignment::Cpu)
        }
    }
    .expect("runtime constructs");
    let data = concat!(env!("CARGO_MANIFEST_DIR"), "/../../data");
    for name in ["fox", "test"] {
        let audio = std::fs::read(format!("{data}/{name}.wav")).expect("audio fixture");
        let reference = std::fs::read_to_string(format!("{data}/{name}.md")).expect("transcript");
        let transcription = runtime
            .asr_transcribe_bytes(&audio, Some(QWEN3_ASR), None)
            .await
            .expect("Qwen3-ASR transcription");
        let errors = word_errors(&words(&reference), &words(&transcription.text));
        let reference_words = words(&reference).len();
        eprintln!(
            "{name}: {errors}/{reference_words} word errors: {:?}",
            transcription.text
        );
        // test.md spells "2026" and "AI—autonomous", which are read out as
        // five words: up to 15 % word errors.
        assert!(errors * 100 <= reference_words * 15, "{name}");
    }
}
