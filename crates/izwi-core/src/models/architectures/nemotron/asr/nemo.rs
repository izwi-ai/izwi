use std::fs::{self, File};
use std::io::{self, Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

use flate2::read::GzDecoder;
use tar::Archive;

use crate::error::{Error, Result};
use crate::model::ModelVariant;

use super::config::NemotronConfigInventory;

pub const NEMOTRON_NEMO_FILENAME: &str = "nemotron-3.5-asr-streaming-0.6b.nemo";

const OPTIONAL_TOKENIZER_SUFFIXES: &[&str] = &[
    "tokenizer.model",
    "tokenizer.vocab",
    "vocab.txt",
    "vocab.json",
    "merges.txt",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
];

#[derive(Debug, Clone)]
pub struct NemotronArtifacts {
    pub nemo_path: PathBuf,
    pub extracted_dir: PathBuf,
    pub model_config_path: PathBuf,
    pub checkpoint_path: PathBuf,
    pub tokenizer_paths: Vec<PathBuf>,
    pub config_inventory: NemotronConfigInventory,
}

pub fn ensure_nemotron_artifacts(
    model_dir: &Path,
    variant: ModelVariant,
) -> Result<NemotronArtifacts> {
    if variant != ModelVariant::Nemotron35AsrStreaming06B {
        return Err(Error::InvalidInput(format!(
            "Unsupported Nemotron ASR variant: {}",
            variant.dir_name()
        )));
    }

    let nemo_path = model_dir.join(NEMOTRON_NEMO_FILENAME);
    if !nemo_path.exists() {
        return Err(Error::ModelNotFound(format!(
            "Missing .nemo checkpoint for {} at {}",
            variant.dir_name(),
            nemo_path.display()
        )));
    }

    let extracted_dir = model_dir.join("nemotron-native");
    fs::create_dir_all(&extracted_dir).map_err(|e| {
        Error::ModelLoadError(format!(
            "Failed to create Nemotron cache directory {}: {}",
            extracted_dir.display(),
            e
        ))
    })?;

    let model_config_path = extracted_dir.join("model_config.yaml");
    let checkpoint_path = extracted_dir.join("model_weights.ckpt");

    if !model_config_path.exists() || !checkpoint_path.exists() {
        extract_nemotron_entries(
            &nemo_path,
            &extracted_dir,
            &model_config_path,
            &checkpoint_path,
        )?;
    }

    let tokenizer_paths = discover_tokenizer_paths(&extracted_dir)?;
    let config_contents = fs::read_to_string(&model_config_path).map_err(|e| {
        Error::ModelLoadError(format!(
            "Failed reading Nemotron config {}: {}",
            model_config_path.display(),
            e
        ))
    })?;
    let config_inventory = NemotronConfigInventory::from_yaml_str(&config_contents)?;

    Ok(NemotronArtifacts {
        nemo_path,
        extracted_dir,
        model_config_path,
        checkpoint_path,
        tokenizer_paths,
        config_inventory,
    })
}

fn extract_nemotron_entries(
    nemo_path: &Path,
    extracted_dir: &Path,
    model_config_path: &Path,
    checkpoint_path: &Path,
) -> Result<()> {
    let mut archive = open_nemo_archive(nemo_path)?;
    let mut found_config = false;
    let mut found_checkpoint = false;

    for entry in archive.entries().map_err(|e| {
        Error::ModelLoadError(format!(
            "Failed reading .nemo archive {}: {}",
            nemo_path.display(),
            e
        ))
    })? {
        let mut entry = entry.map_err(|e| {
            Error::ModelLoadError(format!(
                "Failed iterating .nemo archive {}: {}",
                nemo_path.display(),
                e
            ))
        })?;
        let entry_path = entry
            .path()
            .map_err(|e| {
                Error::ModelLoadError(format!(
                    "Failed reading archive entry path in {}: {}",
                    nemo_path.display(),
                    e
                ))
            })?
            .to_string_lossy()
            .into_owned();

        if entry_path.ends_with("model_config.yaml") {
            extract_entry_to_file(&mut entry, model_config_path)?;
            found_config = true;
            continue;
        }

        if entry_path.ends_with("model_weights.ckpt") {
            extract_entry_to_file(&mut entry, checkpoint_path)?;
            found_checkpoint = true;
            continue;
        }

        if let Some(filename) = tokenizer_asset_filename(&entry_path) {
            extract_entry_to_file(&mut entry, &extracted_dir.join(filename))?;
        }
    }

    let mut missing = Vec::new();
    if !found_config {
        missing.push("model_config.yaml");
    }
    if !found_checkpoint {
        missing.push("model_weights.ckpt");
    }
    if !missing.is_empty() {
        return Err(Error::ModelLoadError(format!(
            "Missing required files in .nemo archive {}: {}",
            nemo_path.display(),
            missing.join(", ")
        )));
    }

    Ok(())
}

fn tokenizer_asset_filename(entry_path: &str) -> Option<&'static str> {
    OPTIONAL_TOKENIZER_SUFFIXES
        .iter()
        .copied()
        .find(|suffix| entry_path.ends_with(suffix))
}

fn discover_tokenizer_paths(extracted_dir: &Path) -> Result<Vec<PathBuf>> {
    let mut paths = Vec::new();
    for suffix in OPTIONAL_TOKENIZER_SUFFIXES {
        let path = extracted_dir.join(suffix);
        if path.exists() {
            paths.push(path);
        }
    }
    Ok(paths)
}

fn open_nemo_archive(nemo_path: &Path) -> Result<Archive<Box<dyn Read>>> {
    let mut file = File::open(nemo_path).map_err(|e| {
        Error::ModelLoadError(format!(
            "Failed opening .nemo archive {}: {}",
            nemo_path.display(),
            e
        ))
    })?;
    let mut magic = [0u8; 2];
    let read = file.read(&mut magic).map_err(|e| {
        Error::ModelLoadError(format!(
            "Failed reading .nemo archive header {}: {}",
            nemo_path.display(),
            e
        ))
    })?;
    file.seek(SeekFrom::Start(0)).map_err(|e| {
        Error::ModelLoadError(format!(
            "Failed rewinding .nemo archive {}: {}",
            nemo_path.display(),
            e
        ))
    })?;

    let reader: Box<dyn Read> = if read == 2 && magic == [0x1f, 0x8b] {
        Box::new(GzDecoder::new(file))
    } else {
        Box::new(file)
    };
    Ok(Archive::new(reader))
}

fn extract_entry_to_file<R: Read>(entry: &mut tar::Entry<'_, R>, dest: &Path) -> Result<()> {
    let tmp_path = dest.with_extension("tmp");
    let mut tmp_file = File::create(&tmp_path).map_err(|e| {
        Error::ModelLoadError(format!(
            "Failed creating temp extraction file {}: {}",
            tmp_path.display(),
            e
        ))
    })?;

    io::copy(entry, &mut tmp_file).map_err(|e| {
        Error::ModelLoadError(format!(
            "Failed writing extracted .nemo entry to {}: {}",
            tmp_path.display(),
            e
        ))
    })?;

    fs::rename(&tmp_path, dest).map_err(|e| {
        Error::ModelLoadError(format!(
            "Failed moving extracted artifact into {}: {}",
            dest.display(),
            e
        ))
    })?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    use flate2::write::GzEncoder;
    use flate2::Compression;
    use std::io::Write;
    use tar::Builder;
    use uuid::Uuid;

    #[test]
    fn ensure_nemotron_artifacts_extracts_required_files_and_tokenizer_assets() {
        let temp_dir = std::env::temp_dir().join(format!("nemotron-nemo-{}", Uuid::new_v4()));
        fs::create_dir_all(&temp_dir).unwrap();
        let model_dir = temp_dir.join(ModelVariant::Nemotron35AsrStreaming06B.dir_name());
        fs::create_dir_all(&model_dir).unwrap();

        let nemo_path = model_dir.join(NEMOTRON_NEMO_FILENAME);
        let nemo_file = File::create(&nemo_path).unwrap();
        let encoder = GzEncoder::new(nemo_file, Compression::default());
        let mut builder = Builder::new(encoder);

        add_archive_file(
            &mut builder,
            "nested/model_config.yaml",
            br#"
model:
  _target_: nemo.collections.asr.models.EncDecRNNTBPEModel
  preprocessor:
    sample_rate: 16000
    features: 128
  encoder:
    n_layers: 24
    d_model: 512
    n_heads: 8
    att_context_size: [56, 13]
  prompt:
    prompt_dim: 128
"#,
        );
        add_archive_file(&mut builder, "nested/model_weights.ckpt", b"checkpoint");
        add_archive_file(&mut builder, "nested/tokenizer.vocab", b"<blank>\nhello\n");
        let encoder = builder.into_inner().unwrap();
        encoder.finish().unwrap();

        let artifacts =
            ensure_nemotron_artifacts(&model_dir, ModelVariant::Nemotron35AsrStreaming06B).unwrap();

        assert!(artifacts.model_config_path.exists());
        assert_eq!(fs::read(artifacts.checkpoint_path).unwrap(), b"checkpoint");
        assert_eq!(artifacts.config_inventory.sample_rate, Some(16_000));
        assert_eq!(artifacts.config_inventory.encoder_layers, Some(24));
        assert_eq!(artifacts.config_inventory.prompt_dim, Some(128));
        assert_eq!(artifacts.config_inventory.left_context_frames, Some(56));
        assert_eq!(artifacts.config_inventory.right_context_frames, vec![13]);
        assert_eq!(artifacts.tokenizer_paths.len(), 1);
        assert_eq!(
            fs::read_to_string(&artifacts.tokenizer_paths[0]).unwrap(),
            "<blank>\nhello\n"
        );

        fs::remove_dir_all(temp_dir).unwrap();
    }

    #[test]
    fn ensure_nemotron_artifacts_rejects_missing_checkpoint() {
        let temp_dir =
            std::env::temp_dir().join(format!("nemotron-nemo-missing-{}", Uuid::new_v4()));
        fs::create_dir_all(&temp_dir).unwrap();
        let model_dir = temp_dir.join(ModelVariant::Nemotron35AsrStreaming06B.dir_name());
        fs::create_dir_all(&model_dir).unwrap();

        let nemo_path = model_dir.join(NEMOTRON_NEMO_FILENAME);
        let mut builder = Builder::new(File::create(&nemo_path).unwrap());
        add_archive_file(
            &mut builder,
            "model_config.yaml",
            b"model:\n  preprocessor:\n    sample_rate: 16000\n",
        );
        builder.finish().unwrap();

        let err = ensure_nemotron_artifacts(&model_dir, ModelVariant::Nemotron35AsrStreaming06B)
            .unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("model_weights.ckpt"), "{msg}");

        fs::remove_dir_all(temp_dir).unwrap();
    }

    fn add_archive_file<W: Write>(builder: &mut Builder<W>, path: &str, contents: &[u8]) {
        let mut header = tar::Header::new_gnu();
        header.set_size(contents.len() as u64);
        header.set_mode(0o644);
        header.set_cksum();
        builder.append_data(&mut header, path, contents).unwrap();
    }
}
