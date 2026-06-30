//! Fish S2 codec artifact boundary.
//!
//! The public S2 Pro checkpoint currently ships `codec.pth`. Izwi can validate
//! that the artifact is present, but native Rust/Candle codec loading still
//! needs either a full PyTorch state-dict reader or a safetensors codec package.

use std::path::{Path, PathBuf};

use crate::error::{Error, Result};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FishS2CodecSupport {
    PendingNativeLoader,
}

#[derive(Debug, Clone)]
pub struct FishS2CodecArtifact {
    pub path: PathBuf,
    pub support: FishS2CodecSupport,
}

impl FishS2CodecArtifact {
    pub fn load(model_dir: &Path) -> Result<Self> {
        let path = model_dir.join("codec.pth");
        if !path.exists() {
            return Err(Error::ModelLoadError(format!(
                "Fish S2 codec artifact missing: {}",
                path.display()
            )));
        }
        Ok(Self {
            path,
            support: FishS2CodecSupport::PendingNativeLoader,
        })
    }

    pub fn ensure_native_supported(&self) -> Result<()> {
        match self.support {
            FishS2CodecSupport::PendingNativeLoader => Err(Error::ModelLoadError(
                "Fish Audio S2 Pro codec.pth loading is not implemented in Rust/Candle yet"
                    .to_string(),
            )),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn codec_artifact_reports_pending_native_loader() {
        let dir = std::env::temp_dir().join(format!("izwi-fish-s2-codec-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("codec.pth"), [0u8]).unwrap();

        let codec = FishS2CodecArtifact::load(&dir).unwrap();
        assert_eq!(codec.support, FishS2CodecSupport::PendingNativeLoader);
        let err = codec.ensure_native_supported().unwrap_err();
        assert!(err
            .to_string()
            .contains("codec.pth loading is not implemented"));

        std::fs::remove_dir_all(dir).ok();
    }
}
