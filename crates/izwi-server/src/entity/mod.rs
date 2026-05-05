#![allow(dead_code)]

use sea_orm::entity::prelude::*;

macro_rules! empty_relation {
    () => {
        #[derive(Copy, Clone, Debug, EnumIter, DeriveRelation)]
        pub enum Relation {}

        impl ActiveModelBehavior for ActiveModel {}
    };
}

pub mod chat_threads {
    use super::*;
    #[derive(Clone, Debug, PartialEq, DeriveEntityModel)]
    #[sea_orm(table_name = "chat_threads")]
    pub struct Model {
        #[sea_orm(primary_key, auto_increment = false)]
        pub id: String,
        pub title: String,
        pub model_id: Option<String>,
        pub created_at: i64,
        pub updated_at: i64,
    }
    empty_relation!();
}

pub mod chat_messages {
    use super::*;
    #[derive(Clone, Debug, PartialEq, DeriveEntityModel)]
    #[sea_orm(table_name = "chat_messages")]
    pub struct Model {
        #[sea_orm(primary_key, auto_increment = false)]
        pub id: String,
        pub thread_id: String,
        pub role: String,
        pub content: String,
        pub content_parts: Option<String>,
        pub created_at: i64,
        pub tokens_generated: Option<i64>,
        pub generation_time_ms: Option<f64>,
    }
    empty_relation!();
}

pub mod voice_profiles {
    use super::*;
    #[derive(Clone, Debug, PartialEq, DeriveEntityModel)]
    #[sea_orm(table_name = "voice_profiles")]
    pub struct Model {
        #[sea_orm(primary_key, auto_increment = false)]
        pub id: String,
        pub name: String,
        pub system_prompt: String,
        pub observational_memory_enabled: i64,
        pub created_at: i64,
        pub updated_at: i64,
    }
    empty_relation!();
}

pub mod voice_sessions {
    use super::*;
    #[derive(Clone, Debug, PartialEq, DeriveEntityModel)]
    #[sea_orm(table_name = "voice_sessions")]
    pub struct Model {
        #[sea_orm(primary_key, auto_increment = false)]
        pub id: String,
        pub profile_id: String,
        pub mode: String,
        pub system_prompt: String,
        pub created_at: i64,
        pub updated_at: i64,
        pub ended_at: Option<i64>,
    }
    empty_relation!();
}

pub mod voice_turns {
    use super::*;
    #[derive(Clone, Debug, PartialEq, DeriveEntityModel)]
    #[sea_orm(table_name = "voice_turns")]
    pub struct Model {
        #[sea_orm(primary_key, auto_increment = false)]
        pub id: String,
        pub session_id: String,
        pub utterance_id: String,
        pub utterance_seq: i64,
        pub mode: String,
        pub status: String,
        pub status_reason: Option<String>,
        pub vad_end_reason: Option<String>,
        pub user_text: Option<String>,
        pub assistant_text: Option<String>,
        pub assistant_raw_text: Option<String>,
        pub language: Option<String>,
        pub audio_duration_secs: Option<f64>,
        pub asr_model_id: Option<String>,
        pub text_model_id: Option<String>,
        pub tts_model_id: Option<String>,
        pub s2s_model_id: Option<String>,
        pub speaker: Option<String>,
        pub created_at: i64,
        pub updated_at: i64,
    }
    empty_relation!();
}

pub mod voice_observations {
    use super::*;
    #[derive(Clone, Debug, PartialEq, DeriveEntityModel)]
    #[sea_orm(table_name = "voice_observations")]
    pub struct Model {
        #[sea_orm(primary_key, auto_increment = false)]
        pub id: String,
        pub profile_id: String,
        pub category: String,
        pub summary: String,
        pub canonical_summary: String,
        pub confidence: f64,
        pub source_turn_id: Option<String>,
        pub source_user_text: Option<String>,
        pub source_assistant_text: Option<String>,
        pub times_seen: i64,
        pub created_at: i64,
        pub updated_at: i64,
        pub forgotten_at: Option<i64>,
    }
    empty_relation!();
}

pub mod onboarding_state {
    use super::*;
    #[derive(Clone, Debug, PartialEq, DeriveEntityModel)]
    #[sea_orm(table_name = "onboarding_state")]
    pub struct Model {
        #[sea_orm(primary_key, auto_increment = false)]
        pub id: String,
        pub completed_at: Option<i64>,
        pub analytics_opt_in: i64,
    }
    empty_relation!();
}

pub mod transcription_records {
    use super::*;
    #[derive(Clone, Debug, PartialEq, DeriveEntityModel)]
    #[sea_orm(table_name = "transcription_records")]
    pub struct Model {
        #[sea_orm(primary_key, auto_increment = false)]
        pub id: String,
        pub created_at: i64,
        pub model_id: Option<String>,
        pub aligner_model_id: Option<String>,
        pub language: Option<String>,
        pub processing_status: String,
        pub processing_error: Option<String>,
        pub duration_secs: Option<f64>,
        pub processing_time_ms: f64,
        pub rtf: Option<f64>,
        pub audio_mime_type: String,
        pub audio_filename: Option<String>,
        pub audio_storage_path: String,
        pub transcription: String,
        pub segments_json: String,
        pub words_json: String,
        pub summary_status: String,
        pub summary_model_id: Option<String>,
        pub summary_text: Option<String>,
        pub summary_error: Option<String>,
        pub summary_updated_at: Option<i64>,
    }
    empty_relation!();
}

pub mod diarization_records {
    use super::*;
    #[derive(Clone, Debug, PartialEq, DeriveEntityModel)]
    #[sea_orm(table_name = "diarization_records")]
    pub struct Model {
        #[sea_orm(primary_key, auto_increment = false)]
        pub id: String,
        pub created_at: i64,
        pub model_id: Option<String>,
        pub asr_model_id: Option<String>,
        pub aligner_model_id: Option<String>,
        pub llm_model_id: Option<String>,
        pub processing_status: String,
        pub processing_error: Option<String>,
        pub min_speakers: Option<i64>,
        pub max_speakers: Option<i64>,
        pub min_speech_duration_ms: Option<f64>,
        pub min_silence_duration_ms: Option<f64>,
        pub enable_llm_refinement: i64,
        pub processing_time_ms: f64,
        pub duration_secs: Option<f64>,
        pub rtf: Option<f64>,
        pub speaker_count: i64,
        pub alignment_coverage: Option<f64>,
        pub unattributed_words: i64,
        pub llm_refined: i64,
        pub asr_text: String,
        pub raw_transcript: String,
        pub transcript: String,
        pub summary_status: String,
        pub summary_model_id: Option<String>,
        pub summary_text: Option<String>,
        pub summary_error: Option<String>,
        pub summary_updated_at: Option<i64>,
        pub segments_json: String,
        pub words_json: String,
        pub utterances_json: String,
        pub speaker_name_overrides_json: String,
        pub audio_mime_type: String,
        pub audio_filename: Option<String>,
        pub audio_storage_path: String,
    }
    empty_relation!();
}

pub mod speech_history_records {
    use super::*;
    #[derive(Clone, Debug, PartialEq, DeriveEntityModel)]
    #[sea_orm(table_name = "speech_history_records")]
    pub struct Model {
        #[sea_orm(primary_key, auto_increment = false)]
        pub id: String,
        pub created_at: i64,
        pub route_kind: String,
        pub processing_status: String,
        pub processing_error: Option<String>,
        pub model_id: Option<String>,
        pub speaker: Option<String>,
        pub language: Option<String>,
        pub saved_voice_id: Option<String>,
        pub speed: Option<f64>,
        pub input_text: String,
        pub voice_description: Option<String>,
        pub reference_text: Option<String>,
        pub generation_time_ms: f64,
        pub audio_duration_secs: Option<f64>,
        pub rtf: Option<f64>,
        pub tokens_generated: Option<i64>,
        pub audio_mime_type: String,
        pub audio_filename: Option<String>,
        pub audio_storage_path: String,
    }
    empty_relation!();
}

pub mod saved_voices {
    use super::*;
    #[derive(Clone, Debug, PartialEq, DeriveEntityModel)]
    #[sea_orm(table_name = "saved_voices")]
    pub struct Model {
        #[sea_orm(primary_key, auto_increment = false)]
        pub id: String,
        pub created_at: i64,
        pub updated_at: i64,
        pub name: String,
        pub reference_text: String,
        pub audio_mime_type: String,
        pub audio_filename: Option<String>,
        pub audio_storage_path: String,
        pub source_route_kind: Option<String>,
        pub source_record_id: Option<String>,
    }
    empty_relation!();
}

pub mod studio_projects {
    use super::*;
    #[derive(Clone, Debug, PartialEq, DeriveEntityModel)]
    #[sea_orm(table_name = "studio_projects")]
    pub struct Model {
        #[sea_orm(primary_key, auto_increment = false)]
        pub id: String,
        pub created_at: i64,
        pub updated_at: i64,
        pub name: String,
        pub source_filename: Option<String>,
        pub source_text: String,
        pub model_id: Option<String>,
        pub voice_mode: String,
        pub speaker: Option<String>,
        pub saved_voice_id: Option<String>,
        pub speed: Option<f64>,
    }
    empty_relation!();
}

pub mod studio_project_segments {
    use super::*;
    #[derive(Clone, Debug, PartialEq, DeriveEntityModel)]
    #[sea_orm(table_name = "studio_project_segments")]
    pub struct Model {
        #[sea_orm(primary_key, auto_increment = false)]
        pub id: String,
        pub project_id: String,
        pub position: i64,
        pub text: String,
        pub model_id: Option<String>,
        pub voice_mode: Option<String>,
        pub speaker: Option<String>,
        pub saved_voice_id: Option<String>,
        pub speech_record_id: Option<String>,
        pub updated_at: i64,
    }
    empty_relation!();
}

pub mod studio_project_folders {
    use super::*;
    #[derive(Clone, Debug, PartialEq, DeriveEntityModel)]
    #[sea_orm(table_name = "studio_project_folders")]
    pub struct Model {
        #[sea_orm(primary_key, auto_increment = false)]
        pub id: String,
        pub created_at: i64,
        pub updated_at: i64,
        pub name: String,
        pub parent_id: Option<String>,
        pub sort_order: i64,
    }
    empty_relation!();
}

pub mod studio_project_meta {
    use super::*;
    #[derive(Clone, Debug, PartialEq, DeriveEntityModel)]
    #[sea_orm(table_name = "studio_project_meta")]
    pub struct Model {
        #[sea_orm(primary_key, auto_increment = false)]
        pub project_id: String,
        pub folder_id: Option<String>,
        pub tags_json: String,
        pub default_export_format: String,
        pub last_render_job_id: Option<String>,
        pub last_rendered_at: Option<i64>,
    }
    empty_relation!();
}

pub mod studio_project_pronunciations {
    use super::*;
    #[derive(Clone, Debug, PartialEq, DeriveEntityModel)]
    #[sea_orm(table_name = "studio_project_pronunciations")]
    pub struct Model {
        #[sea_orm(primary_key, auto_increment = false)]
        pub id: String,
        pub project_id: String,
        pub source_text: String,
        pub replacement_text: String,
        pub locale: Option<String>,
        pub created_at: i64,
        pub updated_at: i64,
    }
    empty_relation!();
}

pub mod studio_project_snapshots {
    use super::*;
    #[derive(Clone, Debug, PartialEq, DeriveEntityModel)]
    #[sea_orm(table_name = "studio_project_snapshots")]
    pub struct Model {
        #[sea_orm(primary_key, auto_increment = false)]
        pub id: String,
        pub project_id: String,
        pub created_at: i64,
        pub label: Option<String>,
        pub project_json: String,
    }
    empty_relation!();
}

pub mod studio_project_render_jobs {
    use super::*;
    #[derive(Clone, Debug, PartialEq, DeriveEntityModel)]
    #[sea_orm(table_name = "studio_project_render_jobs")]
    pub struct Model {
        #[sea_orm(primary_key, auto_increment = false)]
        pub id: String,
        pub project_id: String,
        pub created_at: i64,
        pub updated_at: i64,
        pub status: String,
        pub error_message: Option<String>,
        pub queued_segment_ids_json: String,
    }
    empty_relation!();
}
