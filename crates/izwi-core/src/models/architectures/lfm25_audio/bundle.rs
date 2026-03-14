use std::path::{Path, PathBuf};

use crate::backends::BackendKind;
use crate::error::{Error, Result};
use crate::models::shared::weights::gguf::{GgufLoader, GgufModelInfo};

const MAIN_GGUF: &str = "LFM2.5-Audio-1.5B-Q4_0.gguf";
const MMPROJ_GGUF: &str = "mmproj-LFM2.5-Audio-1.5B-Q4_0.gguf";
const TOKENIZER_GGUF: &str = "tokenizer-LFM2.5-Audio-1.5B-Q4_0.gguf";
const VOCODER_GGUF: &str = "vocoder-LFM2.5-Audio-1.5B-Q4_0.gguf";

#[derive(Debug, Clone)]
pub struct Lfm25AudioBundlePaths {
    pub main: PathBuf,
    pub mmproj: PathBuf,
    pub tokenizer: PathBuf,
    pub vocoder: PathBuf,
}

pub struct Lfm25AudioBundle {
    pub paths: Lfm25AudioBundlePaths,
    pub main: GgufLoader,
    pub mmproj: GgufLoader,
    pub tokenizer: GgufLoader,
    pub vocoder: GgufLoader,
}

#[derive(Clone)]
pub struct Lfm25AudioBundleInfo {
    pub main: GgufModelInfo,
    pub mmproj: GgufModelInfo,
    pub tokenizer: GgufModelInfo,
    pub vocoder: GgufModelInfo,
}

impl Lfm25AudioBundlePaths {
    pub fn from_model_dir(model_dir: &Path) -> Result<Self> {
        let paths = Self {
            main: model_dir.join(MAIN_GGUF),
            mmproj: model_dir.join(MMPROJ_GGUF),
            tokenizer: model_dir.join(TOKENIZER_GGUF),
            vocoder: model_dir.join(VOCODER_GGUF),
        };
        paths.ensure_exists()?;
        Ok(paths)
    }

    fn ensure_exists(&self) -> Result<()> {
        for path in [&self.main, &self.mmproj, &self.tokenizer, &self.vocoder] {
            if !path.exists() {
                return Err(Error::ModelNotFound(format!(
                    "Missing required LFM2.5 Audio GGUF bundle file: {}",
                    path.display()
                )));
            }
        }
        Ok(())
    }
}

impl Lfm25AudioBundle {
    pub fn load(model_dir: &Path, backend: BackendKind) -> Result<Self> {
        let paths = Lfm25AudioBundlePaths::from_model_dir(model_dir)?;
        let main = GgufLoader::from_path_with_backend(&paths.main, backend)?;
        let mmproj = GgufLoader::from_path_with_backend(&paths.mmproj, backend)?;
        let tokenizer = GgufLoader::from_path_with_backend(&paths.tokenizer, backend)?;
        let vocoder = GgufLoader::from_path_with_backend(&paths.vocoder, backend)?;

        Ok(Self {
            paths,
            main,
            mmproj,
            tokenizer,
            vocoder,
        })
    }

    pub fn info(&self) -> Lfm25AudioBundleInfo {
        Lfm25AudioBundleInfo {
            main: self.main.get_model_info(),
            mmproj: self.mmproj.get_model_info(),
            tokenizer: self.tokenizer.get_model_info(),
            vocoder: self.vocoder.get_model_info(),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::fs;

    use uuid::Uuid;

    use super::Lfm25AudioBundlePaths;

    fn make_temp_dir() -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("izwi-lfm25-audio-{}", Uuid::new_v4()));
        fs::create_dir_all(&dir).expect("create temp dir");
        dir
    }

    fn touch(path: &std::path::Path) {
        fs::write(path, b"").expect("write test file");
    }

    #[test]
    fn bundle_paths_require_all_four_q40_files() {
        let dir = make_temp_dir();
        touch(&dir.join(super::MAIN_GGUF));
        touch(&dir.join(super::MMPROJ_GGUF));
        touch(&dir.join(super::TOKENIZER_GGUF));
        touch(&dir.join(super::VOCODER_GGUF));

        let bundle = Lfm25AudioBundlePaths::from_model_dir(&dir).expect("bundle paths");
        assert!(bundle.main.ends_with(super::MAIN_GGUF));
        assert!(bundle.mmproj.ends_with(super::MMPROJ_GGUF));
        assert!(bundle.tokenizer.ends_with(super::TOKENIZER_GGUF));
        assert!(bundle.vocoder.ends_with(super::VOCODER_GGUF));

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn bundle_paths_fail_when_bundle_is_incomplete() {
        let dir = make_temp_dir();
        touch(&dir.join(super::MAIN_GGUF));
        touch(&dir.join(super::MMPROJ_GGUF));
        touch(&dir.join(super::TOKENIZER_GGUF));

        let err = Lfm25AudioBundlePaths::from_model_dir(&dir).expect_err("expected missing file");
        assert!(err.to_string().contains(super::VOCODER_GGUF));

        let _ = fs::remove_dir_all(&dir);
    }
}
