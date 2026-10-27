//! Maya 2 Native's languages and built-in voices, as its release lists them
//! (`voices/voices.json` at HF `1403b841`): every surface (engine, server,
//! CLI, UI metadata) reads them from here.

/// A language the model speaks: its request code and English name.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Maya2Language {
    pub code: &'static str,
    pub name: &'static str,
}

/// The twelve languages, in the order of the model's `<|lang:xx|>` tokens.
pub const MAYA2_LANGUAGES: [Maya2Language; 12] = [
    Maya2Language {
        code: "as",
        name: "Assamese",
    },
    Maya2Language {
        code: "bn",
        name: "Bengali",
    },
    Maya2Language {
        code: "en",
        name: "English",
    },
    Maya2Language {
        code: "gu",
        name: "Gujarati",
    },
    Maya2Language {
        code: "hi",
        name: "Hindi",
    },
    Maya2Language {
        code: "kn",
        name: "Kannada",
    },
    Maya2Language {
        code: "ml",
        name: "Malayalam",
    },
    Maya2Language {
        code: "mr",
        name: "Marathi",
    },
    Maya2Language {
        code: "or",
        name: "Odia",
    },
    Maya2Language {
        code: "pa",
        name: "Punjabi",
    },
    Maya2Language {
        code: "ta",
        name: "Tamil",
    },
    Maya2Language {
        code: "te",
        name: "Telugu",
    },
];

/// A built-in voice: a synthetic persona the model was trained to speak as
/// when its text starts with `{Name}: `. `languages` are those with at least
/// an hour of the voice's training audio; any other language is spoken
/// cross-lingually.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Maya2Voice {
    pub name: &'static str,
    pub gender: &'static str,
    pub age_band: &'static str,
    pub languages: &'static [&'static str],
    pub description: &'static str,
}

const ALL_BUT_AS: &[&str] = &[
    "bn", "en", "gu", "hi", "kn", "ml", "mr", "or", "pa", "ta", "te",
];
const ALL_BUT_AS_PA: &[&str] = &["bn", "en", "gu", "hi", "kn", "ml", "mr", "or", "ta", "te"];

/// The 20 voices of the release, in its order.
pub const MAYA2_VOICES: [Maya2Voice; 20] = [
    voice(
        "Ahana",
        "female",
        "young",
        ALL_BUT_AS,
        "Young female voice, bright, youthful",
    ),
    voice(
        "Vaidehi",
        "female",
        "adult",
        ALL_BUT_AS,
        "Adult female voice, calm, gentle",
    ),
    voice(
        "Malini",
        "female",
        "mature",
        ALL_BUT_AS,
        "Mature female voice, steady narrator",
    ),
    voice(
        "Charvi",
        "female",
        "young",
        ALL_BUT_AS,
        "Young female voice, energetic, upbeat",
    ),
    voice(
        "Jahnavi",
        "female",
        "adult",
        ALL_BUT_AS_PA,
        "Adult female voice, warm, friendly",
    ),
    voice(
        "Anirban",
        "male",
        "adult",
        ALL_BUT_AS,
        "Adult male voice, steady narrator",
    ),
    voice(
        "Nakul",
        "male",
        "young",
        ALL_BUT_AS,
        "Young male voice, bright, youthful",
    ),
    voice(
        "Vasant",
        "male",
        "mature",
        ALL_BUT_AS_PA,
        "Mature male voice, steady narrator",
    ),
    voice(
        "Siddhant",
        "male",
        "adult",
        ALL_BUT_AS,
        "Adult male voice, clear, professional",
    ),
    voice(
        "Satvik",
        "male",
        "young",
        ALL_BUT_AS,
        "Young male voice, warm, friendly",
    ),
    voice(
        "Abhay",
        "male",
        "adult",
        &["en", "hi"],
        "Adult male voice, natural conversational style",
    ),
    voice(
        "Nivedita",
        "female",
        "adult",
        &["en", "hi"],
        "Adult female voice, natural conversational style",
    ),
    voice(
        "Ishwari",
        "female",
        "adult",
        &["hi"],
        "Adult female voice, natural conversational style",
    ),
    voice(
        "Ritvik",
        "male",
        "adult",
        &["hi"],
        "Adult male voice, natural conversational style",
    ),
    voice(
        "Vrinda",
        "female",
        "young",
        &["hi"],
        "Young female voice, natural conversational style",
    ),
    voice(
        "Sreekar",
        "male",
        "adult",
        &["en", "hi", "te"],
        "Adult male voice, natural conversational style",
    ),
    voice(
        "Kayal",
        "female",
        "adult",
        &["en", "ta"],
        "Adult female voice, natural conversational style",
    ),
    voice(
        "Shaunak",
        "male",
        "young",
        &["en", "hi"],
        "Young male voice, natural conversational style",
    ),
    voice(
        "Akshat",
        "male",
        "adult",
        &["en", "hi"],
        "Adult male voice, natural conversational style",
    ),
    voice(
        "Tanushree",
        "female",
        "adult",
        &["en", "hi"],
        "Adult female voice, natural conversational style",
    ),
];

const fn voice(
    name: &'static str,
    gender: &'static str,
    age_band: &'static str,
    languages: &'static [&'static str],
    description: &'static str,
) -> Maya2Voice {
    Maya2Voice {
        name,
        gender,
        age_band,
        languages,
        description,
    }
}

/// The voice a request gets when it names none: a broad-coverage voice.
pub const MAYA2_DEFAULT_VOICE: &str = "Ahana";

/// OpenAI's voice names, each mapped to a fixed voice of similar register
/// (upstream maps them all to an unconditioned voice that changes with
/// every request). Only voices trained on 10+ languages are targets.
pub const MAYA2_OPENAI_VOICE_ALIASES: [(&str, &str); 13] = [
    ("alloy", "Vaidehi"),
    ("ash", "Siddhant"),
    ("ballad", "Vasant"),
    ("cedar", "Anirban"),
    ("coral", "Jahnavi"),
    ("echo", "Nakul"),
    ("fable", "Anirban"),
    ("marin", "Malini"),
    ("nova", "Ahana"),
    ("onyx", "Vasant"),
    ("sage", "Vaidehi"),
    ("shimmer", "Charvi"),
    ("verse", "Satvik"),
];

/// Inline tags the release names as performed best effort; any lowercase
/// `[tag]` passes through to the model, these are what the UI offers.
pub const MAYA2_SUGGESTED_TAGS: &[&str] = &[
    "[laughs]",
    "[sighs]",
    "[whispers]",
    "[sad]",
    "[excited]",
    "[curious]",
    "[angry]",
    "[surprised]",
];

/// The voice `name` selects: a voice name or an OpenAI alias, ignoring case.
pub fn maya2_voice(name: &str) -> Option<&'static Maya2Voice> {
    let name = name.trim();
    let target = MAYA2_OPENAI_VOICE_ALIASES
        .iter()
        .find(|(alias, _)| alias.eq_ignore_ascii_case(name))
        .map_or(name, |(_, voice)| voice);
    MAYA2_VOICES
        .iter()
        .find(|voice| voice.name.eq_ignore_ascii_case(target))
}

/// The language `code` names, ignoring case.
pub fn maya2_language(code: &str) -> Option<&'static Maya2Language> {
    let code = code.trim();
    MAYA2_LANGUAGES
        .iter()
        .find(|language| language.code.eq_ignore_ascii_case(code))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn aliases_target_broad_coverage_voices() {
        for (alias, target) in MAYA2_OPENAI_VOICE_ALIASES {
            let voice = maya2_voice(alias).unwrap_or_else(|| panic!("{alias} resolves"));
            assert_eq!(voice.name, target);
            assert!(voice.languages.len() >= 10, "{alias} → {target}");
        }
        assert_eq!(maya2_voice("ahana").unwrap().name, "Ahana");
        assert!(maya2_voice("Ahana2").is_none());
        assert!(maya2_voice(MAYA2_DEFAULT_VOICE).is_some());
    }

    #[test]
    fn voice_languages_are_model_languages() {
        for voice in MAYA2_VOICES {
            for code in voice.languages {
                assert!(
                    maya2_language(code).is_some(),
                    "{} speaks {code}",
                    voice.name
                );
            }
        }
        assert_eq!(maya2_language("TE").unwrap().name, "Telugu");
    }
}
