//! Inkling 2: Thinking Machines' text + image chat models, as their own
//! family.
//!
//! The published model (`thinkingmachines/Inkling-2-Mini`) is a looped
//! sparse-expert decoder: 24 layers of weights, of which the middle 12 run
//! three times, so a token passes through 48 layer applications. Its expert
//! weights are published quantized (NVFP4 and MXFP8) and stay packed in
//! memory; nothing here expands them.
//!
//! * `config`: the published `config.json` and the derived geometry;
//! * `checkpoint`: the memory-mapped weight file and its census;
//! * `quant`: the two expert formats and products straight from packed bytes;
//! * `experts`: routing and the expert feed-forward on the host;
//! * `trunk`: the text trunk composed from plain tensor operations, the
//!   reference every faster path is checked against;
//! * `prompt`: the chat prompt, matching the published template;
//! * `reply`: the model's reply messages as the text the server reads;
//! * `memory`: what a loaded model holds, from the checkpoint header;
//! * `state`: the engine-owned state contract (paged keys and values, a ring
//!   of convolution inputs);
//! * `chat`: the served chat model.
//!
//! Nothing here imports another family: the family may use `kernels`,
//! `models::shared`, `kv`, `backends` and `engine`, and builds its own state
//! contract. The isolation test at the bottom of this file holds that line.

pub mod chat;
pub mod checkpoint;
pub mod config;
pub mod experts;
pub mod memory;
pub mod prompt;
pub mod quant;
pub mod reply;
pub(crate) mod state;
pub mod trunk;
pub mod vision;

#[cfg(test)]
mod chat_tests;
#[cfg(test)]
pub(crate) mod synthetic;
#[cfg(test)]
mod tests;
#[cfg(test)]
mod vision_tests;

pub use chat::{
    ChatDecodeState as Inkling2ChatDecodeState, ChatDecodeStep as Inkling2ChatDecodeStep,
    Inkling2ChatModel,
};
pub use checkpoint::Inkling2Checkpoint;
pub use config::{Inkling2Config, Inkling2LayerKind};
pub use trunk::{Inkling2SequenceState, Inkling2Trunk};

#[cfg(test)]
mod isolation_tests {
    use std::path::Path;

    fn visit(dir: &Path, files: &mut Vec<std::path::PathBuf>) {
        for entry in std::fs::read_dir(dir).unwrap() {
            let path = entry.unwrap().path();
            if path.is_dir() {
                visit(&path, files);
            } else if path.extension().is_some_and(|ext| ext == "rs") {
                files.push(path);
            }
        }
    }

    /// The family imports no other architecture: every `architectures::`
    /// path and every `super::super::<family>` reach stays inside
    /// `inkling2`.
    #[test]
    fn the_family_imports_no_other_architecture() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("src/models/architectures");
        let families = std::fs::read_dir(&root)
            .unwrap()
            .map(|entry| entry.unwrap().path())
            .filter(|path| path.is_dir())
            .map(|path| path.file_name().unwrap().to_string_lossy().into_owned())
            .filter(|name| name != "inkling2")
            .collect::<Vec<_>>();
        assert!(families.contains(&"qwen36moe".to_string()));
        let mut files = Vec::new();
        visit(&root.join("inkling2"), &mut files);
        assert!(files.len() >= 6);
        for file in files {
            let source = std::fs::read_to_string(&file).unwrap();
            for (number, line) in source.lines().enumerate() {
                let code = line.split("//").next().unwrap_or("");
                for family in &families {
                    for pattern in [
                        format!("architectures::{family}::"),
                        format!("architectures::{family};"),
                        format!("architectures::{{{family}"),
                        format!("super::super::{family}::"),
                        format!("super::super::super::{family}::"),
                    ] {
                        assert!(
                            !code.contains(&pattern),
                            "{}:{} imports the {family} family: {line}",
                            file.display(),
                            number + 1
                        );
                    }
                }
            }
        }
    }
}
