use std::fs::{self, File};
use std::io::{self, Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

use flate2::read::GzDecoder;
use tar::Archive;

use crate::error::{Error, Result};
use crate::model::ModelVariant;

#[derive(Debug, Clone)]
pub struct SortformerArtifacts {
    pub nemo_path: PathBuf,
    pub extracted_dir: PathBuf,
    pub model_config_path: PathBuf,
    pub checkpoint_path: PathBuf,
}

pub fn ensure_sortformer_artifacts(
    model_dir: &Path,
    variant: ModelVariant,
) -> Result<SortformerArtifacts> {
    let nemo_filename = match variant {
        ModelVariant::DiarStreamingSortformer4SpkV21 => "diar_streaming_sortformer_4spk-v2.1.nemo",
        ModelVariant::Nemotron3Diarization => "Nemotron-3-Diarization.nemo",
        _ => {
            return Err(Error::InvalidInput(format!(
                "Unsupported Sortformer diarization variant: {}",
                variant.dir_name()
            )));
        }
    };

    let nemo_path = model_dir.join(nemo_filename);
    if !nemo_path.exists() {
        return Err(Error::ModelNotFound(format!(
            "Missing .nemo checkpoint for {} at {}",
            variant.dir_name(),
            nemo_path.display()
        )));
    }

    let extracted_dir = model_dir.join("sortformer-native");
    fs::create_dir_all(&extracted_dir).map_err(|e| {
        Error::ModelLoadError(format!(
            "Failed to create Sortformer cache directory {}: {}",
            extracted_dir.display(),
            e
        ))
    })?;

    let model_config_path = extracted_dir.join("model_config.yaml");
    let checkpoint_path = extracted_dir.join("model_weights.ckpt");
    if !model_config_path.exists() || !checkpoint_path.exists() {
        extract_required_entries(
            &nemo_path,
            &[
                ("model_config.yaml", &model_config_path),
                ("model_weights.ckpt", &checkpoint_path),
            ],
        )?;
    }

    Ok(SortformerArtifacts {
        nemo_path,
        extracted_dir,
        model_config_path,
        checkpoint_path,
    })
}

fn extract_required_entries(nemo_path: &Path, entries: &[(&str, &Path)]) -> Result<()> {
    let mut archive = open_nemo_archive(nemo_path)?;
    let mut extracted = vec![false; entries.len()];

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

        for (idx, (suffix, dest)) in entries.iter().enumerate() {
            if extracted[idx] || !entry_path.ends_with(suffix) {
                continue;
            }
            extract_entry_to_file(&mut entry, dest)?;
            extracted[idx] = true;
        }

        if extracted.iter().all(|done| *done) {
            return Ok(());
        }
    }

    let missing = entries
        .iter()
        .enumerate()
        .filter_map(|(idx, (suffix, _))| (!extracted[idx]).then_some(*suffix))
        .collect::<Vec<_>>();
    Err(Error::ModelLoadError(format!(
        "Missing required files in .nemo archive {}: {}",
        nemo_path.display(),
        missing.join(", ")
    )))
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
    fn ensure_sortformer_artifacts_extracts_required_files_from_gzipped_nemo() {
        let temp_dir = std::env::temp_dir().join(format!("sortformer-nemo-{}", Uuid::new_v4()));
        fs::create_dir_all(&temp_dir).unwrap();
        let model_dir = temp_dir.join(ModelVariant::DiarStreamingSortformer4SpkV21.dir_name());
        fs::create_dir_all(&model_dir).unwrap();

        let nemo_path = model_dir.join("diar_streaming_sortformer_4spk-v2.1.nemo");
        let nemo_file = File::create(&nemo_path).unwrap();
        let encoder = GzEncoder::new(nemo_file, Compression::default());
        let mut builder = Builder::new(encoder);

        add_archive_file(
            &mut builder,
            "nested/model_config.yaml",
            b"sample_rate: 16000\nmax_num_of_spks: 4\n",
        );
        add_archive_file(&mut builder, "nested/model_weights.ckpt", b"checkpoint");
        let encoder = builder.into_inner().unwrap();
        encoder.finish().unwrap();

        let artifacts =
            ensure_sortformer_artifacts(&model_dir, ModelVariant::DiarStreamingSortformer4SpkV21)
                .unwrap();

        assert_eq!(
            fs::read_to_string(artifacts.model_config_path).unwrap(),
            "sample_rate: 16000\nmax_num_of_spks: 4\n"
        );
        assert_eq!(fs::read(artifacts.checkpoint_path).unwrap(), b"checkpoint");

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
