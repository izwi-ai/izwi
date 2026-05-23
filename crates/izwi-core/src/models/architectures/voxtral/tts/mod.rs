//! Native Voxtral TTS architecture support.
//!
//! Voxtral TTS is separate from Voxtral Realtime ASR. The checkpoint uses a
//! Mistral decoder backbone, a flow-matching acoustic transformer, preset voice
//! embeddings, and a neural codec decoder.

pub mod acoustic;
pub mod codec;
pub mod config;
pub mod model;
pub mod sampling;
pub mod tokenizer;
pub mod voice;

pub use acoustic::{
    apply_audio_token_offset, cfg_velocity_blend, fsq_code_to_unit, fsq_unit_to_code,
    padded_codebook_size, strip_audio_token_offset, AcousticCodeFrame, AcousticGenerationConfig,
    AudioCodeValue, AudioSpecialToken, FlowMatchingAudioTransformer, ACOUSTIC_CODEBOOK_OFFSET,
    AUDIO_SPECIAL_TOKEN_COUNT,
};
pub use codec::{VoxtralCodecConfig, VoxtralCodecTimeline, VOXTRAL_CODEC_CHUNK_FRAMES};
pub use config::{
    VoxtralTtsAcousticTransformerArgs, VoxtralTtsAudioEncodingArgs, VoxtralTtsAudioModelArgs,
    VoxtralTtsAudioTokenizerArgs, VoxtralTtsConfig, VoxtralTtsMultimodalConfig, DEFAULT_CFG_ALPHA,
    DEFAULT_N_DECODING_STEPS, VOXTRAL_TTS_MODEL_TYPE,
};
pub use model::{
    select_voxtral_tts_dtypes, VoxtralTtsAssets, VoxtralTtsDTypePlan, VoxtralTtsModel,
    VoxtralTtsOutput,
};
pub use sampling::VoxtralTtsGenerationParams;
pub use tokenizer::{VoxtralTtsPrompt, VoxtralTtsSpecialTokens, VoxtralTtsTokenizer};
pub use voice::{
    VoxtralVoiceCatalog, VoxtralVoiceEmbeddingLibrary, VoxtralVoiceEmbeddingShape,
    VoxtralVoiceInfo, VOXTRAL_TTS_BUILT_IN_VOICES,
};
