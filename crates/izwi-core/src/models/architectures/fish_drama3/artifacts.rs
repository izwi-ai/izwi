//! The pinned Fish Audio Drama 3 bundle and its two identity checks.
//!
//! The weights were re-uploaded on 2026-10-29 (8 shards became 15), so the
//! family follows one revision, never `main`. The manifest lists every file
//! at that revision except `.gitattributes`: 24 files, 41,502,569,692 bytes.
//! LFS files carry their LFS SHA-256 from the Hub API; the seven small Git
//! files were hashed from copies whose Git blob ids match the API's.
//!
//! The two checks cost very different amounts:
//! * [`verify_identity`]: presence, exact sizes and the `config.json` hash.
//!   A few `stat` calls and 5.5 kB of hashing, so every load runs it.
//! * [`verify_full`]: the SHA-256 of every byte. Run once after a download,
//!   never per load: hashing 41.5 GB at each load, as S2's content
//!   fingerprint does, costs minutes of I/O on every start.

use std::io::Read;
use std::path::Path;

use sha2::{Digest, Sha256};

use crate::error::{Error, Result};

pub const FISH_DRAMA3_REPO: &str = "fishaudio/drama-3";
pub const FISH_DRAMA3_REVISION: &str = "78e5ab79e49aa9a25abe8799eff2c4d73a1c3216";
/// Every manifest file's bytes.
pub const FISH_DRAMA3_BUNDLE_BYTES: u64 = 41_502_569_692;
/// The file [`verify_identity`] hashes.
pub const FISH_DRAMA3_CONFIG_FILE: &str = "config.json";

/// One file of the pinned revision.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FishDrama3PinnedFile {
    pub name: &'static str,
    pub bytes: u64,
    /// Lowercase hex SHA-256 of the whole file.
    pub sha256: &'static str,
}

const fn pinned(name: &'static str, bytes: u64, sha256: &'static str) -> FishDrama3PinnedFile {
    FishDrama3PinnedFile {
        name,
        bytes,
        sha256,
    }
}

/// `fishaudio/drama-3@78e5ab79`, without `.gitattributes`.
pub const FISH_DRAMA3_FILES: &[FishDrama3PinnedFile] = &[
    pinned(
        "LICENSE",
        19_685,
        "733a32442087acbda12ecafe8e71eb8ee5890c89b50ce7ac4ea3777faa3c823e",
    ),
    pinned(
        "README.md",
        7_996,
        "6d147e079b87e0b9bb4e1c899f75da3e658754933457a3ca753ca9912d1f5e10",
    ),
    // Stock Qwen3.5 template: shipped, never rendered.
    pinned(
        "chat_template.jinja",
        7_756,
        "a4aee8afcf2e0711942cf848899be66016f8d14a889ff9ede07bca099c28f715",
    ),
    // Byte-identical to fishaudio/s2-pro's codec.
    pinned(
        "codec.pth",
        1_871_099_728,
        "74fc41c5a7151c6f350af8bd7e5d6e3accfcc7f3dfbfac23afd35af07052bb2f",
    ),
    pinned(
        FISH_DRAMA3_CONFIG_FILE,
        5_534,
        "aea53f0f2b9a15023fc0f8eb0f42360663671c99814037bada6867d508010111",
    ),
    pinned(
        "model-00001-of-00015.safetensors",
        2_862_881_576,
        "a1b7af15210d2eda2d75a2cf976295f80e345855b745b109aa3ce49d26a44497",
    ),
    pinned(
        "model-00002-of-00015.safetensors",
        2_628_610_248,
        "a95e113c6f32f8fcb23ecb618cd4384fad6358f3c0ac31cfbf352daf5370db94",
    ),
    pinned(
        "model-00003-of-00015.safetensors",
        2_628_610_224,
        "a34d0952d6369f5c73e6ae06138ef96bc6a3af9bb259c4e41425a33706cf5b68",
    ),
    pinned(
        "model-00004-of-00015.safetensors",
        2_628_611_792,
        "13ac368d87441a18752e424e485b711a115ea1084e3a18ab13276e73eedef8b4",
    ),
    pinned(
        "model-00005-of-00015.safetensors",
        2_641_525_264,
        "c432fa0dac97471f0d625bc67eb94fe084ca2e5ecd8865f457b751479b50e363",
    ),
    pinned(
        "model-00006-of-00015.safetensors",
        2_628_614_896,
        "661a31433747fa9d029632a61350ea9eb105bf7276d013ff0901b6a939fb2dcb",
    ),
    pinned(
        "model-00007-of-00015.safetensors",
        2_628_614_880,
        "4a7574b5c238f424b37aa945a736da1994a26bd38b86bf3ec4abf524c0969cc8",
    ),
    pinned(
        "model-00008-of-00015.safetensors",
        2_628_614_888,
        "7fac6d1e899939eb03c9cf28a33b0bfd9c7a599298662f26d9a59d64c96d007f",
    ),
    pinned(
        "model-00009-of-00015.safetensors",
        2_641_525_264,
        "fcf1313744ea5978e363b2bb49c3b66748d886d79f3ec102f4346a462385de4b",
    ),
    pinned(
        "model-00010-of-00015.safetensors",
        2_628_614_896,
        "6db3f7cd9d49806ae4f17b6f6d17a0c8dd9686aee57728da1b50ec0c6139a629",
    ),
    pinned(
        "model-00011-of-00015.safetensors",
        2_628_614_880,
        "a9027ef4db30bb1875bd2bc93b80bae785c5943807efe63bf09834b66e49fc16",
    ),
    pinned(
        "model-00012-of-00015.safetensors",
        2_628_614_888,
        "c16fea955e973010064ff7bc907a6fa67821956e0de23f74807c31fb48339f3a",
    ),
    pinned(
        "model-00013-of-00015.safetensors",
        2_641_525_264,
        "64861c8e1ee13a0cbb5ec9431f402cee33edcbcbc744ff63091fe3a997f0da19",
    ),
    pinned(
        "model-00014-of-00015.safetensors",
        3_749_111_392,
        "ba5319475a6258984143b69af8380f166df8179af90378f2f4e02694cad87baf",
    ),
    pinned(
        "model-00015-of-00015.safetensors",
        1_409_461_016,
        "1e2ddf01c1be73fb19f3022c28e90d24e730c1bc759e5de78083afb39aab3fb7",
    ),
    pinned(
        "model.safetensors.index.json",
        6_880_196,
        "a8e9aca4e9fddca51f03fad63f52c82c470e98e28b6b0b1b1562dbe6456b50bb",
    ),
    pinned(
        "special_tokens_map.json",
        102_779,
        "08aed9e321dc05275ac68cf883bfa96006b3fc09ff5cccadf262b6bce0873adb",
    ),
    pinned(
        "tokenizer.json",
        20_791_070,
        "0e9b9db76b2f0389b04b6ef057a2ac3b89100b7e96f2dd7034d19580a09654d6",
    ),
    pinned(
        "tokenizer_config.json",
        103_580,
        "00e800279d8436c324b924e97872c7eacd2b19406aa0fe71eeb557b4cd53aa3e",
    ),
];

/// The manifest entry for `name`.
pub fn pinned_file(name: &str) -> Option<&'static FishDrama3PinnedFile> {
    FISH_DRAMA3_FILES.iter().find(|file| file.name == name)
}

/// The cheap identity check every load runs: each pinned file is present
/// at its exact size, and `config.json` hashes to the pinned value.
pub fn verify_identity(model_dir: &Path) -> Result<()> {
    let problems = identity_problems(model_dir, FISH_DRAMA3_FILES, &[FISH_DRAMA3_CONFIG_FILE]);
    if problems.is_empty() {
        Ok(())
    } else {
        Err(Error::ModelLoadError(report(model_dir, &problems)))
    }
}

/// The full check, once after a download: the identity check, then the
/// SHA-256 of every pinned file.
pub fn verify_full(model_dir: &Path) -> Result<()> {
    let mut problems = identity_problems(model_dir, FISH_DRAMA3_FILES, &[]);
    if problems.is_empty() {
        problems = hash_problems(model_dir, FISH_DRAMA3_FILES);
    }
    if problems.is_empty() {
        Ok(())
    } else {
        Err(Error::DownloadError(report(model_dir, &problems)))
    }
}

fn report(model_dir: &Path, problems: &[String]) -> String {
    format!(
        "{} is not {FISH_DRAMA3_REPO}@{}: {}",
        model_dir.display(),
        &FISH_DRAMA3_REVISION[..8],
        problems.join("; ")
    )
}

/// Missing files, wrong sizes, and wrong hashes of the `hashed` files.
fn identity_problems(
    model_dir: &Path,
    files: &[FishDrama3PinnedFile],
    hashed: &[&str],
) -> Vec<String> {
    let mut problems = Vec::new();
    for file in files {
        let path = model_dir.join(file.name);
        // Follows symlinks, as a Hub cache snapshot links to its blobs.
        match std::fs::metadata(&path) {
            Ok(meta) if !meta.is_file() => problems.push(format!("{} is not a file", file.name)),
            Ok(meta) if meta.len() != file.bytes => problems.push(format!(
                "{} has {} bytes; the pin has {}",
                file.name,
                meta.len(),
                file.bytes
            )),
            Ok(_) => {
                if hashed.contains(&file.name) {
                    problems.extend(hash_problem(&path, file));
                }
            }
            Err(err) => problems.push(format!("{} is missing ({err})", file.name)),
        }
    }
    problems
}

fn hash_problems(model_dir: &Path, files: &[FishDrama3PinnedFile]) -> Vec<String> {
    files
        .iter()
        .filter_map(|file| hash_problem(&model_dir.join(file.name), file))
        .collect()
}

fn hash_problem(path: &Path, file: &FishDrama3PinnedFile) -> Option<String> {
    match sha256_file(path) {
        Ok(digest) if digest == file.sha256 => None,
        Ok(digest) => Some(format!(
            "{} hashes to {digest}; the pin has {}",
            file.name, file.sha256
        )),
        Err(err) => Some(format!("cannot hash {}: {err}", file.name)),
    }
}

fn sha256_file(path: &Path) -> std::io::Result<String> {
    let mut file = std::fs::File::open(path)?;
    let mut digest = Sha256::new();
    let mut buffer = vec![0u8; 8 << 20];
    loop {
        let read = file.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        digest.update(&buffer[..read]);
    }
    Ok(hex::encode(digest.finalize()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;
    use std::path::PathBuf;

    const PUBLISHED_CONFIG: &[u8] =
        include_bytes!("../../../../tests/fixtures/fish_drama3/config.json");

    /// A directory removed when dropped.
    struct TempDir(PathBuf);

    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    fn temp_dir() -> TempDir {
        let dir = std::env::temp_dir().join(format!(
            "izwi-fish-drama3-artifacts-{}",
            uuid::Uuid::new_v4()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        TempDir(dir)
    }

    #[test]
    fn the_manifest_is_the_published_bundle() {
        assert_eq!(FISH_DRAMA3_FILES.len(), 24);
        let names = FISH_DRAMA3_FILES
            .iter()
            .map(|file| file.name)
            .collect::<BTreeSet<_>>();
        assert_eq!(names.len(), FISH_DRAMA3_FILES.len());
        assert!(!names.contains(".gitattributes"));
        assert_eq!(
            FISH_DRAMA3_FILES.iter().map(|file| file.bytes).sum::<u64>(),
            FISH_DRAMA3_BUNDLE_BYTES
        );
        let shards = FISH_DRAMA3_FILES
            .iter()
            .filter(|file| file.name.ends_with(".safetensors"))
            .collect::<Vec<_>>();
        assert_eq!(shards.len(), 15);
        assert_eq!(
            shards.iter().map(|file| file.bytes).sum::<u64>(),
            39_603_551_368
        );
        for (index, shard) in shards.iter().enumerate() {
            assert_eq!(
                shard.name,
                format!("model-{:05}-of-00015.safetensors", index + 1)
            );
        }
        for file in FISH_DRAMA3_FILES {
            assert_eq!(file.sha256.len(), 64, "{}", file.name);
            assert!(
                file.sha256
                    .bytes()
                    .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b)),
                "{}",
                file.name
            );
        }
        // The committed fixture is the pinned config.
        let config = pinned_file(FISH_DRAMA3_CONFIG_FILE).unwrap();
        assert_eq!(config.bytes, PUBLISHED_CONFIG.len() as u64);
        assert_eq!(hex::encode(Sha256::digest(PUBLISHED_CONFIG)), config.sha256);
        assert_eq!(FISH_DRAMA3_REVISION.len(), 40);
    }

    /// A fake bundle: the real `config.json` and every other file sparse at
    /// its pinned size (no disk space is used).
    fn fake_bundle() -> TempDir {
        let dir = temp_dir();
        for file in FISH_DRAMA3_FILES {
            let path = dir.0.join(file.name);
            if file.name == FISH_DRAMA3_CONFIG_FILE {
                std::fs::write(&path, PUBLISHED_CONFIG).unwrap();
            } else {
                std::fs::File::create(&path)
                    .unwrap()
                    .set_len(file.bytes)
                    .unwrap();
            }
        }
        dir
    }

    #[test]
    fn identity_passes_at_the_pinned_sizes_and_fails_otherwise() {
        let dir = fake_bundle();
        verify_identity(&dir.0).unwrap();

        let shard = dir.0.join("model-00007-of-00015.safetensors");
        std::fs::File::options()
            .write(true)
            .open(&shard)
            .unwrap()
            .set_len(2_628_614_879)
            .unwrap();
        let err = verify_identity(&dir.0).unwrap_err().to_string();
        assert!(
            err.contains("model-00007-of-00015.safetensors has 2628614879 bytes"),
            "{err}"
        );
        std::fs::remove_file(&shard).unwrap();
        let err = verify_identity(&dir.0).unwrap_err().to_string();
        assert!(
            err.contains("model-00007-of-00015.safetensors is missing"),
            "{err}"
        );

        // A config of the right size with other bytes.
        let dir = fake_bundle();
        let mut config = PUBLISHED_CONFIG.to_vec();
        let at = config.iter().position(|&b| b == b'4').unwrap();
        config[at] = b'5';
        std::fs::write(dir.0.join(FISH_DRAMA3_CONFIG_FILE), config).unwrap();
        let err = verify_identity(&dir.0).unwrap_err().to_string();
        assert!(err.contains("config.json hashes to"), "{err}");
    }

    #[test]
    fn the_full_check_hashes_every_file() {
        let dir = temp_dir();
        let contents: [(&'static str, &[u8]); 3] =
            [("a.json", b"{}"), ("b.bin", &[7u8; 1000]), ("c.txt", b"")];
        let files = contents
            .iter()
            .map(|&(name, bytes)| {
                std::fs::write(dir.0.join(name), bytes).unwrap();
                let sha256: &'static str =
                    Box::leak(hex::encode(Sha256::digest(bytes)).into_boxed_str());
                pinned(name, bytes.len() as u64, sha256)
            })
            .collect::<Vec<_>>();
        assert!(identity_problems(&dir.0, &files, &["a.json"]).is_empty());
        assert!(hash_problems(&dir.0, &files).is_empty());
        let mut flipped = [7u8; 1000];
        flipped[500] = 8;
        std::fs::write(dir.0.join("b.bin"), flipped).unwrap();
        // Same size: identity cannot see it, the full check does.
        assert!(identity_problems(&dir.0, &files, &["a.json"]).is_empty());
        let problems = hash_problems(&dir.0, &files);
        assert_eq!(problems.len(), 1);
        assert!(problems[0].starts_with("b.bin hashes to"), "{problems:?}");
        // The full check runs the identity check first, and hashes nothing
        // when it fails.
        let bundle = fake_bundle();
        std::fs::remove_file(bundle.0.join("codec.pth")).unwrap();
        let err = verify_full(&bundle.0).unwrap_err();
        assert!(matches!(err, Error::DownloadError(_)), "{err}");
        assert!(err.to_string().contains("codec.pth is missing"), "{err}");
    }
}
