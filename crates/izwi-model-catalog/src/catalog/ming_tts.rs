//! Ming-3.0-tts-0.5B's built-in voices, languages and controls, as its
//! release documents them (`cookbooks/instruction_en.md` at GitHub
//! `c88c023e`): every surface (engine, server, CLI, UI metadata) reads them
//! from here.
//!
//! A built-in voice is a name the model learned, not a file. Some voices
//! exist only as emotion variants (`灵风信_生气`): the table lists each once
//! under its base name with the variants it has. Eight more names the model
//! knows imitate film, television and cartoon characters; they are left out
//! of this table on purpose and a request may still pass any name through.

/// A language the model speaks: its request code and English name.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MingTtsLanguage {
    pub code: &'static str,
    pub name: &'static str,
    /// Whether a request must name it (the five languages added in 3.0);
    /// Chinese and English are also spoken unprompted.
    pub must_be_named: bool,
}

pub const MING_TTS_LANGUAGES: [MingTtsLanguage; 7] = [
    MingTtsLanguage {
        code: "zh",
        name: "Chinese",
        must_be_named: false,
    },
    MingTtsLanguage {
        code: "en",
        name: "English",
        must_be_named: false,
    },
    MingTtsLanguage {
        code: "ja",
        name: "Japanese",
        must_be_named: true,
    },
    MingTtsLanguage {
        code: "ko",
        name: "Korean",
        must_be_named: true,
    },
    MingTtsLanguage {
        code: "de",
        name: "German",
        must_be_named: true,
    },
    MingTtsLanguage {
        code: "fr",
        name: "French",
        must_be_named: true,
    },
    MingTtsLanguage {
        code: "es",
        name: "Spanish",
        must_be_named: true,
    },
];

/// The emotions a request may name, in English.
pub const MING_TTS_EMOTIONS: [&str; 7] = [
    "neutral",
    "happy",
    "sad",
    "angry",
    "fearful",
    "disgusted",
    "surprised",
];

/// The levels speed, pitch and volume take.
pub const MING_TTS_LEVELS: [&str; 3] = ["low", "medium", "high"];

/// The dialects a request may name.
pub const MING_TTS_DIALECTS: [&str; 2] = ["cantonese", "sichuanese"];

/// The inline sounds the model performs, written exactly like this.
pub const MING_TTS_INLINE_TAGS: &[&str] = &[
    "[Breathing]",
    "[Cough]",
    "[Sigh]",
    "[Laughter]",
    "[Surprise-ah]",
    "[Surprise-oh]",
    "[Dissatisfaction-hnn]",
    "[Uhm]",
    "[Confirmation-en]",
    "[Question-ei]",
    "[Question-ah]",
    "[Question-en]",
    "[Question-oh]",
    "[Question-huh]",
];

/// One built-in voice.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MingTtsVoice {
    /// The name a request uses; for a voice without emotion variants it is
    /// also the name the model reads.
    pub name: &'static str,
    /// The language the voice was recorded in ("en" or "zh").
    pub language: &'static str,
    pub gender: &'static str,
    pub description: &'static str,
    /// (emotion, the name the model reads for it); empty for a voice with
    /// one delivery.
    pub variants: &'static [(&'static str, &'static str)],
}

impl MingTtsVoice {
    /// The name the model reads for this voice with `emotion` (`None` for
    /// its default delivery: neutral where it has one, else its first
    /// variant), or `None` when the voice has no such variant.
    pub fn model_name(&self, emotion: Option<&str>) -> Option<&'static str> {
        match (emotion, self.variants) {
            (None, []) => Some(self.name),
            (Some(_), []) => None,
            (None, variants) => Some(variants[0].1),
            (Some(emotion), variants) => variants
                .iter()
                .find(|(name, _)| name.eq_ignore_ascii_case(emotion.trim()))
                .map(|(_, model_name)| *model_name),
        }
    }

    /// The emotions this voice can be asked for.
    pub fn emotions(&self) -> impl Iterator<Item = &'static str> {
        self.variants.iter().map(|(emotion, _)| *emotion)
    }
}

/// The voice a request gets when it names none, by text language.
pub const MING_TTS_DEFAULT_VOICE: &str = "US_Emma";
pub const MING_TTS_DEFAULT_CHINESE_VOICE: &str = "灵小柔";

/// The built-in voices: 3 English, then the Chinese ones.
pub const MING_TTS_VOICES: [MingTtsVoice; 85] = [
    MingTtsVoice {
        name: "US_Emma",
        language: "en",
        gender: "female",
        description: "American English; assistants, narration and conversational product voice.",
        variants: &[],
    },
    MingTtsVoice {
        name: "UK_George",
        language: "en",
        gender: "male",
        description: "British English; audiobooks and formal announcements.",
        variants: &[],
    },
    MingTtsVoice {
        name: "UK_Charlotte",
        language: "en",
        gender: "female",
        description: "British English; audiobooks and measured narration.",
        variants: &[],
    },
    MingTtsVoice {
        name: "灵小安",
        language: "zh",
        gender: "female",
        description: "Sweet, crisp and energetic, with an approachable, cute anime-style delivery.",
        variants: &[],
    },
    MingTtsVoice {
        name: "灵语汐",
        language: "zh",
        gender: "female",
        description: "Clear, bright and youthful, with a sunny, energetic delivery.",
        variants: &[],
    },
    MingTtsVoice {
        name: "灵小萌",
        language: "zh",
        gender: "female",
        description: "Bright, crisp and brisk, with an energetic yet dependable delivery.",
        variants: &[],
    },
    MingTtsVoice {
        name: "灵小柔",
        language: "zh",
        gender: "female",
        description: "Clear, sweet and articulate, with polite, patient and professional intonation.",
        variants: &[],
    },
    MingTtsVoice {
        name: "灵小雅",
        language: "zh",
        gender: "female",
        description: "Sweet, bright and youthful, with an energetic delivery.",
        variants: &[],
    },
    MingTtsVoice {
        name: "花小呗",
        language: "zh",
        gender: "female",
        description: "Crisp and sweet, with a distinctly childlike, lively delivery.",
        variants: &[],
    },
    MingTtsVoice {
        name: "灵韵",
        language: "zh",
        gender: "female",
        description: "Bright and crisp, with brisk, dependable delivery.",
        variants: &[],
    },
    MingTtsVoice {
        name: "妩媚妲己",
        language: "zh",
        gender: "female",
        description: "Sweet and crisp, with light, rising intonation and an alluring delivery.",
        variants: &[],
    },
    MingTtsVoice {
        name: "灵婉",
        language: "zh",
        gender: "female",
        description: "Clear and sweet, with a youthful, intimate and soothing delivery.",
        variants: &[],
    },
    MingTtsVoice {
        name: "灵小月",
        language: "zh",
        gender: "female",
        description: "Sweet, clear and energetic, with cheerful, welcoming intonation.",
        variants: &[],
    },
    MingTtsVoice {
        name: "灵朵儿",
        language: "zh",
        gender: "female",
        description: "Warm, thoughtful and approachable, with an unhurried, soothing storytelling style.",
        variants: &[],
    },
    MingTtsVoice {
        name: "灵小锋",
        language: "zh",
        gender: "male",
        description: "Warm and resonant, with natural confidence and a balance of friendliness and maturity.",
        variants: &[],
    },
    MingTtsVoice {
        name: "灵小川",
        language: "zh",
        gender: "male",
        description: "Warm, mellow and resonant, with sensitive, expressive delivery.",
        variants: &[],
    },
    MingTtsVoice {
        name: "灵小甄",
        language: "zh",
        gender: "female",
        description: "Bright, crisp and lively, with a friendly, persuasive recommendation style.",
        variants: &[],
    },
    MingTtsVoice {
        name: "灵小知",
        language: "zh",
        gender: "female",
        description: "Bright and confident, with strong, articulate and persuasive delivery.",
        variants: &[],
    },
    MingTtsVoice {
        name: "灵小惠",
        language: "zh",
        gender: "female",
        description: "Very fast and penetrating, conveying urgency, enthusiasm and professional confidence.",
        variants: &[],
    },
    MingTtsVoice {
        name: "灵小晴",
        language: "zh",
        gender: "female",
        description: "Very fast yet clear, with a powerful, bright delivery that conveys urgency.",
        variants: &[],
    },
    MingTtsVoice {
        name: "灵小琪",
        language: "zh",
        gender: "male",
        description: "Sunny, friendly and natural, with an enthusiastic recommendation style.",
        variants: &[],
    },
    MingTtsVoice {
        name: "灵梦",
        language: "zh",
        gender: "female",
        description: "Very sweet and youthful, with playful, affectionate delivery.",
        variants: &[],
    },
    MingTtsVoice {
        name: "灵恬",
        language: "zh",
        gender: "female",
        description: "Honey-sweet, with a gently playful yet assured romantic delivery.",
        variants: &[],
    },
    MingTtsVoice {
        name: "灵小汐",
        language: "zh",
        gender: "female",
        description: "Clear and sweet, with relaxed, natural and endearingly innocent delivery.",
        variants: &[],
    },
    MingTtsVoice {
        name: "灵小棠",
        language: "zh",
        gender: "female",
        description: "Clear and warm, with calm pacing and a gentle, experienced storytelling style.",
        variants: &[],
    },
    MingTtsVoice {
        name: "灵小梨",
        language: "zh",
        gender: "female",
        description: "Sweet and clear, combining youthful playfulness with gentle, expressive narration.",
        variants: &[],
    },
    MingTtsVoice {
        name: "灵小芸",
        language: "zh",
        gender: "female",
        description: "Warm and thoughtful, with sincere, calm and sensitive delivery.",
        variants: &[],
    },
    MingTtsVoice {
        name: "灵空",
        language: "zh",
        gender: "male",
        description: "Low and resonant, with close, breathy delivery that conveys affection and reassurance.",
        variants: &[],
    },
    MingTtsVoice {
        name: "灵宇",
        language: "zh",
        gender: "male",
        description: "Youthful and resonant, with a confident smile and gentle teasing.",
        variants: &[],
    },
    MingTtsVoice {
        name: "灵朗",
        language: "zh",
        gender: "male",
        description: "Low and resonant, combining confident warmth with affectionate reassurance.",
        variants: &[],
    },
    MingTtsVoice {
        name: "灵楷",
        language: "zh",
        gender: "male",
        description: "Warm and resonant, with slow, soothing late-night-radio narration.",
        variants: &[],
    },
    MingTtsVoice {
        name: "灵旭",
        language: "zh",
        gender: "male",
        description: "Clear, youthful and scholarly, with calm, gentle storytelling.",
        variants: &[],
    },
    MingTtsVoice {
        name: "灵江",
        language: "zh",
        gender: "male",
        description: "Warm and resonant, with relaxed, heartfelt reflections on youth and enthusiasm.",
        variants: &[],
    },
    MingTtsVoice {
        name: "悲伤女声",
        language: "zh",
        gender: "female",
        description: "Gentle yet determined, with a thoughtful, intimate storytelling style.",
        variants: &[],
    },
    MingTtsVoice {
        name: "皮六",
        language: "zh",
        gender: "male",
        description: "Bright, youthful and confident, with strong contrast and dramatic emphasis.",
        variants: &[],
    },
    MingTtsVoice {
        name: "悟空",
        language: "zh",
        gender: "male",
        description: "Highly expressive and theatrical, with vivid but measured exaggeration.",
        variants: &[],
    },
    MingTtsVoice {
        name: "灵晓声",
        language: "zh",
        gender: "male",
        description: "Youthful and resonant, with calm confidence and an assertive, charismatic delivery.",
        variants: &[("neutral", "灵晓声_中性"), ("happy", "灵晓声_高兴"), ("sad", "灵晓声_悲伤"), ("disgusted", "灵晓声_厌恶"), ("surprised", "灵晓声_惊讶")],
    },
    MingTtsVoice {
        name: "灵风信",
        language: "zh",
        gender: "female",
        description: "Sweet and clear, with warm, concerned and reassuring delivery.",
        variants: &[("neutral", "灵风信_中性"), ("happy", "灵风信_高兴"), ("sad", "灵风信_悲伤"), ("angry", "灵风信_生气"), ("disgusted", "灵风信_厌恶")],
    },
    MingTtsVoice {
        name: "灵惜梦",
        language: "zh",
        gender: "female",
        description: "Bright and crisp, combining approachable warmth with independence and decisiveness.",
        variants: &[("neutral", "灵惜梦_中性"), ("happy", "灵惜梦_高兴"), ("sad", "灵惜梦_悲伤"), ("angry", "灵惜梦_生气"), ("fearful", "灵惜梦_恐惧"), ("disgusted", "灵惜梦_厌恶"), ("surprised", "灵惜梦_惊讶")],
    },
    MingTtsVoice {
        name: "灵逸风",
        language: "zh",
        gender: "male",
        description: "Fresh and youthful, with relaxed, casual and slightly evasive delivery.",
        variants: &[("neutral", "灵逸风_中性"), ("happy", "灵逸风_高兴"), ("sad", "灵逸风_悲伤"), ("fearful", "灵逸风_恐惧"), ("surprised", "灵逸风_惊讶")],
    },
    MingTtsVoice {
        name: "灵悦澜",
        language: "zh",
        gender: "female",
        description: "Sweet and crisp, with a smiling, approachable and lively delivery.",
        variants: &[("neutral", "灵悦澜_中性"), ("happy", "灵悦澜_高兴"), ("sad", "灵悦澜_悲伤"), ("angry", "灵悦澜_生气"), ("fearful", "灵悦澜_恐惧"), ("disgusted", "灵悦澜_厌恶")],
    },
    MingTtsVoice {
        name: "灵御心",
        language: "zh",
        gender: "female",
        description: "Bright, crisp and direct, with efficient, youthful energy.",
        variants: &[("neutral", "灵御心_中性"), ("happy", "灵御心_高兴"), ("angry", "灵御心_生气")],
    },
    MingTtsVoice {
        name: "灵语恬",
        language: "zh",
        gender: "female",
        description: "Clear and gentle, with attentive, earnest delivery.",
        variants: &[("neutral", "灵语恬_中性"), ("happy", "灵语恬_高兴"), ("sad", "灵语恬_悲伤"), ("angry", "灵语恬_生气"), ("fearful", "灵语恬_恐惧"), ("surprised", "灵语恬_惊讶")],
    },
    MingTtsVoice {
        name: "灵心恬",
        language: "zh",
        gender: "female",
        description: "Crisp and composed, with objective delivery and vivid storytelling.",
        variants: &[("neutral", "灵心恬_中性"), ("happy", "灵心恬_高兴"), ("sad", "灵心恬_悲伤"), ("angry", "灵心恬_生气"), ("fearful", "灵心恬_恐惧"), ("disgusted", "灵心恬_厌恶"), ("surprised", "灵心恬_惊讶")],
    },
    MingTtsVoice {
        name: "灵亦尘",
        language: "zh",
        gender: "male",
        description: "Warm and resonant, with natural, gentle concern and reassuring delivery.",
        variants: &[("neutral", "灵亦尘_中性"), ("fearful", "灵亦尘_恐惧")],
    },
    MingTtsVoice {
        name: "灵语尘",
        language: "zh",
        gender: "male",
        description: "Low, husky and world-weary, conveying helplessness and resignation.",
        variants: &[("neutral", "灵语尘_中性"), ("happy", "灵语尘_高兴"), ("sad", "灵语尘_悲伤"), ("fearful", "灵语尘_恐惧"), ("disgusted", "灵语尘_厌恶"), ("surprised", "灵语尘_惊讶")],
    },
    MingTtsVoice {
        name: "灵悦兮",
        language: "zh",
        gender: "female",
        description: "Bright and crisp, with lively, confident and action-oriented delivery.",
        variants: &[("neutral", "灵悦兮_中性"), ("happy", "灵悦兮_高兴"), ("sad", "灵悦兮_悲伤"), ("angry", "灵悦兮_生气"), ("fearful", "灵悦兮_恐惧"), ("disgusted", "灵悦兮_厌恶"), ("surprised", "灵悦兮_惊讶")],
    },
    MingTtsVoice {
        name: "灵若虚",
        language: "zh",
        gender: "female",
        description: "Clean and efficient, with thoughtful, objective yet warm delivery.",
        variants: &[("neutral", "灵若虚_中性"), ("happy", "灵若虚_高兴"), ("sad", "灵若虚_悲伤"), ("angry", "灵若虚_生气"), ("disgusted", "灵若虚_厌恶"), ("surprised", "灵若虚_惊讶")],
    },
    MingTtsVoice {
        name: "灵未央",
        language: "zh",
        gender: "female",
        description: "Sweet and crisp, with playful, rising intonation and confident explanations.",
        variants: &[("neutral", "灵未央_中性"), ("happy", "灵未央_高兴"), ("sad", "灵未央_悲伤"), ("angry", "灵未央_生气"), ("fearful", "灵未央_恐惧"), ("surprised", "灵未央_惊讶")],
    },
    MingTtsVoice {
        name: "灵梦瑶",
        language: "zh",
        gender: "female",
        description: "Soft and sweet, with a friendly, earnest and soothing explanatory style.",
        variants: &[("neutral", "灵梦瑶_中性"), ("happy", "灵梦瑶_高兴"), ("sad", "灵梦瑶_悲伤"), ("angry", "灵梦瑶_生气"), ("surprised", "灵梦瑶_惊讶")],
    },
    MingTtsVoice {
        name: "灵妙声",
        language: "zh",
        gender: "male",
        description: "Sweet and crisp, with standard pronunciation and a fresh, approachable, professional delivery.",
        variants: &[("neutral", "灵妙声_中性"), ("sad", "灵妙声_悲伤")],
    },
    MingTtsVoice {
        name: "灵言澈",
        language: "zh",
        gender: "male",
        description: "Warm, resonant and reassuring, with gentle, affectionate concern.",
        variants: &[("neutral", "灵言澈_中性"), ("happy", "灵言澈_高兴")],
    },
    MingTtsVoice {
        name: "灵嘉肴",
        language: "zh",
        gender: "male",
        description: "Bright and energetic, with precise articulation and rational, approachable confidence.",
        variants: &[("neutral", "灵嘉肴_中性"), ("fearful", "灵嘉肴_恐惧")],
    },
    MingTtsVoice {
        name: "灵孤影",
        language: "zh",
        gender: "male",
        description: "Warm and resonant, with composed, sincere appreciation and natural satisfaction.",
        variants: &[("neutral", "灵孤影_中性")],
    },
    MingTtsVoice {
        name: "灵雨汐",
        language: "zh",
        gender: "female",
        description: "Soft and sweet, with natural breathiness and an affectionate, relaxed youthful delivery.",
        variants: &[("neutral", "灵雨汐_中性"), ("happy", "灵雨汐_高兴"), ("sad", "灵雨汐_悲伤"), ("fearful", "灵雨汐_恐惧"), ("surprised", "灵雨汐_惊讶")],
    },
    MingTtsVoice {
        name: "灵信远",
        language: "zh",
        gender: "male",
        description: "Clear and warm, with a slight smile and audible breath, balancing youthful freshness and mature affection.",
        variants: &[("happy", "灵信远_高兴"), ("sad", "灵信远_悲伤"), ("angry", "灵信远_生气"), ("fearful", "灵信远_恐惧"), ("surprised", "灵信远_惊讶")],
    },
    MingTtsVoice {
        name: "灵思齐",
        language: "zh",
        gender: "male",
        description: "Sunny, bright and engaging, with sincere praise and approachable warmth.",
        variants: &[("happy", "灵思齐_高兴"), ("sad", "灵思齐_悲伤"), ("fearful", "灵思齐_恐惧"), ("disgusted", "灵思齐_厌恶"), ("surprised", "灵思齐_惊讶")],
    },
    MingTtsVoice {
        name: "灵叙柔",
        language: "zh",
        gender: "female",
        description: "Composed and dignified, with clear articulation and a balance of authority and gentle narration.",
        variants: &[("happy", "灵叙柔_高兴"), ("surprised", "灵叙柔_惊讶")],
    },
    MingTtsVoice {
        name: "灵绮木",
        language: "zh",
        gender: "female",
        description: "Professional yet approachable, with gentle, assured delivery.",
        variants: &[("happy", "灵绮木_高兴"), ("angry", "灵绮木_生气"), ("disgusted", "灵绮木_厌恶")],
    },
    MingTtsVoice {
        name: "灵渊霆",
        language: "zh",
        gender: "male",
        description: "Youthful, natural and relaxed, like a familiar friend making a casual invitation.",
        variants: &[("happy", "灵渊霆_高兴"), ("sad", "灵渊霆_悲伤"), ("angry", "灵渊霆_生气"), ("fearful", "灵渊霆_恐惧"), ("disgusted", "灵渊霆_厌恶"), ("surprised", "灵渊霆_惊讶")],
    },
    MingTtsVoice {
        name: "灵素心",
        language: "zh",
        gender: "female",
        description: "Sweet, crisp and energetic, with lively, curious intonation.",
        variants: &[("sad", "灵素心_悲伤"), ("disgusted", "灵素心_厌恶")],
    },
    MingTtsVoice {
        name: "灵采薇",
        language: "zh",
        gender: "female",
        description: "Warm and thoughtful, with clear articulation and calm, trustworthy delivery.",
        variants: &[("happy", "灵采薇_高兴"), ("angry", "灵采薇_生气"), ("disgusted", "灵采薇_厌恶")],
    },
    MingTtsVoice {
        name: "灵卓然",
        language: "zh",
        gender: "male",
        description: "Full and resonant, with precise pronunciation and calm, objective delivery.",
        variants: &[("sad", "灵卓然_悲伤"), ("fearful", "灵卓然_恐惧")],
    },
    MingTtsVoice {
        name: "灵浅忧",
        language: "zh",
        gender: "male",
        description: "Bright and crisp, with precise articulation and professional, objective confidence.",
        variants: &[("happy", "灵浅忧_高兴"), ("sad", "灵浅忧_悲伤"), ("fearful", "灵浅忧_恐惧"), ("disgusted", "灵浅忧_厌恶")],
    },
    MingTtsVoice {
        name: "灵昭野",
        language: "zh",
        gender: "male",
        description: "Clear and youthful, with sincere admiration and a warm, captivated delivery.",
        variants: &[("happy", "灵昭野_高兴"), ("sad", "灵昭野_悲伤"), ("disgusted", "灵昭野_厌恶")],
    },
    MingTtsVoice {
        name: "灵心澜",
        language: "zh",
        gender: "female",
        description: "Mature and composed, with assured, discerning and dignified delivery.",
        variants: &[("happy", "灵心澜_高兴"), ("sad", "灵心澜_悲伤")],
    },
    MingTtsVoice {
        name: "灵犀言",
        language: "zh",
        gender: "male",
        description: "Deep and full, with mature composure, firm articulation and an authoritative delivery.",
        variants: &[("sad", "灵犀言_悲伤"), ("angry", "灵犀言_生气")],
    },
    MingTtsVoice {
        name: "灵语冰",
        language: "zh",
        gender: "female",
        description: "Thoughtful and efficient, with precise articulation and objective, trustworthy delivery.",
        variants: &[("happy", "灵语冰_高兴"), ("sad", "灵语冰_悲伤"), ("surprised", "灵语冰_惊讶")],
    },
    MingTtsVoice {
        name: "灵惊鸿",
        language: "zh",
        gender: "male",
        description: "Brisk and direct, with natural, easygoing business confidence.",
        variants: &[("happy", "灵惊鸿_高兴"), ("sad", "灵惊鸿_悲伤"), ("angry", "灵惊鸿_生气"), ("disgusted", "灵惊鸿_厌恶"), ("surprised", "灵惊鸿_惊讶")],
    },
    MingTtsVoice {
        name: "灵惜蕊",
        language: "zh",
        gender: "female",
        description: "Clear and pleasant, with standard pronunciation and patient, professional warmth.",
        variants: &[("happy", "灵惜蕊_高兴"), ("sad", "灵惜蕊_悲伤"), ("angry", "灵惜蕊_生气"), ("fearful", "灵惜蕊_恐惧"), ("disgusted", "灵惜蕊_厌恶"), ("surprised", "灵惜蕊_惊讶")],
    },
    MingTtsVoice {
        name: "灵语萱",
        language: "zh",
        gender: "female",
        description: "Clear and sweet, with gentle yet firm affection, trust and reassurance.",
        variants: &[("happy", "灵语萱_高兴"), ("sad", "灵语萱_悲伤"), ("angry", "灵语萱_生气"), ("fearful", "灵语萱_恐惧"), ("disgusted", "灵语萱_厌恶")],
    },
    MingTtsVoice {
        name: "灵语烟",
        language: "zh",
        gender: "female",
        description: "Bright, crisp and lively, expressing surprise and everyday banter.",
        variants: &[("happy", "灵语烟_高兴"), ("sad", "灵语烟_悲伤"), ("angry", "灵语烟_生气"), ("fearful", "灵语烟_恐惧"), ("disgusted", "灵语烟_厌恶")],
    },
    MingTtsVoice {
        name: "灵律雅",
        language: "zh",
        gender: "female",
        description: "Efficient and grounded in urban life, with restrained, authentic frustration.",
        variants: &[("happy", "灵律雅_高兴"), ("angry", "灵律雅_生气"), ("fearful", "灵律雅_恐惧")],
    },
    MingTtsVoice {
        name: "灵惊涛",
        language: "zh",
        gender: "male",
        description: "Young male delivery with a distinctive Nanning accent, conveying humorous frustration at an overly cold internet cafe.",
        variants: &[("fearful", "灵惊涛_恐惧")],
    },
    MingTtsVoice {
        name: "灵寒陌",
        language: "zh",
        gender: "male",
        description: "Clear and youthful, with urgent, innocent attempts to clear up a misunderstanding.",
        variants: &[("happy", "灵寒陌_高兴"), ("disgusted", "灵寒陌_厌恶")],
    },
    MingTtsVoice {
        name: "灵决冥",
        language: "zh",
        gender: "male",
        description: "Resonant and precise, with calm, objective and engaging documentary-style delivery.",
        variants: &[("angry", "灵决冥_生气"), ("fearful", "灵决冥_恐惧"), ("disgusted", "灵决冥_厌恶")],
    },
    MingTtsVoice {
        name: "灵慈萱",
        language: "zh",
        gender: "female",
        description: "Crisp, youthful and natural, with a familiar, slightly urgent delivery.",
        variants: &[("sad", "灵慈萱_悲伤"), ("angry", "灵慈萱_生气"), ("fearful", "灵慈萱_恐惧")],
    },
    MingTtsVoice {
        name: "灵妙音",
        language: "zh",
        gender: "female",
        description: "Resonant and forceful, with efficient delivery and an unquestionable sense of authority.",
        variants: &[("happy", "灵妙音_高兴"), ("sad", "灵妙音_悲伤"), ("fearful", "灵妙音_恐惧"), ("disgusted", "灵妙音_厌恶"), ("surprised", "灵妙音_惊讶")],
    },
    MingTtsVoice {
        name: "灵断尘",
        language: "zh",
        gender: "male",
        description: "Youthful and conversational, with light questions and decisive refusals.",
        variants: &[("happy", "灵断尘_高兴"), ("sad", "灵断尘_悲伤"), ("fearful", "灵断尘_恐惧"), ("disgusted", "灵断尘_厌恶"), ("surprised", "灵断尘_惊讶")],
    },
    MingTtsVoice {
        name: "灵熙婉",
        language: "zh",
        gender: "female",
        description: "Sweet and clear, with a smiling, approachable and lively storytelling style.",
        variants: &[("happy", "灵熙婉_高兴"), ("sad", "灵熙婉_悲伤"), ("fearful", "灵熙婉_恐惧")],
    },
    MingTtsVoice {
        name: "灵旭扬",
        language: "zh",
        gender: "male",
        description: "Warm and resonant, with affectionate delivery and a hint of illness-related weakness and desire for attention.",
        variants: &[("happy", "灵旭扬_高兴"), ("sad", "灵旭扬_悲伤"), ("angry", "灵旭扬_生气"), ("fearful", "灵旭扬_恐惧"), ("disgusted", "灵旭扬_厌恶"), ("surprised", "灵旭扬_惊讶")],
    },
    MingTtsVoice {
        name: "灵阅川",
        language: "zh",
        gender: "male",
        description: "Deep and resonant, with experienced, composed and dignified storytelling.",
        variants: &[("happy", "灵阅川_高兴"), ("sad", "灵阅川_悲伤"), ("angry", "灵阅川_生气"), ("fearful", "灵阅川_恐惧"), ("disgusted", "灵阅川_厌恶")],
    },
    MingTtsVoice {
        name: "灵妩月",
        language: "zh",
        gender: "female",
        description: "Gentle and thoughtful, combining caring, rational advice with quiet strength.",
        variants: &[("happy", "灵妩月_高兴"), ("sad", "灵妩月_悲伤"), ("angry", "灵妩月_生气"), ("fearful", "灵妩月_恐惧"), ("disgusted", "灵妩月_厌恶"), ("surprised", "灵妩月_惊讶")],
    },
    MingTtsVoice {
        name: "灵汐月",
        language: "zh",
        gender: "female",
        description: "Sweet and soothing, with relaxed, heartfelt delivery conveying intimacy and trust.",
        variants: &[("happy", "灵汐月_高兴"), ("sad", "灵汐月_悲伤")],
    },
    MingTtsVoice {
        name: "灵黛嗔",
        language: "zh",
        gender: "female",
        description: "Warm, thoughtful and engaging, with sincere, confident recommendations.",
        variants: &[("happy", "灵黛嗔_高兴"), ("disgusted", "灵黛嗔_厌恶")],
    },
];

/// OpenAI's voice names, each mapped to one of the three English voices of
/// the same register, so a client written for OpenAI's speech API gets a
/// fixed voice instead of a name the model never learned.
pub const MING_TTS_OPENAI_VOICE_ALIASES: [(&str, &str); 13] = [
    ("alloy", "US_Emma"),
    ("ash", "UK_George"),
    ("ballad", "UK_George"),
    ("cedar", "UK_George"),
    ("coral", "UK_Charlotte"),
    ("echo", "UK_George"),
    ("fable", "UK_Charlotte"),
    ("marin", "US_Emma"),
    ("nova", "US_Emma"),
    ("onyx", "UK_George"),
    ("sage", "UK_Charlotte"),
    ("shimmer", "US_Emma"),
    ("verse", "UK_George"),
];

/// A built-in voice by its request name, by the name of one of its emotion
/// variants, or by an OpenAI voice name (case-insensitive for Latin names).
pub fn ming_tts_voice(name: &str) -> Option<&'static MingTtsVoice> {
    let name = name.trim();
    let name = MING_TTS_OPENAI_VOICE_ALIASES
        .iter()
        .find(|(alias, _)| alias.eq_ignore_ascii_case(name))
        .map_or(name, |(_, voice)| *voice);
    MING_TTS_VOICES.iter().find(|voice| {
        voice.name.eq_ignore_ascii_case(name)
            || voice.variants.iter().any(|(_, variant)| *variant == name)
    })
}

/// The name the model reads for request voice `name` and `emotion`.
///
/// A table voice resolves through its variants; the exact name of a variant
/// is itself; any other name is passed to the model as written (the model
/// knows names this table leaves out). `Err` carries the emotions the voice
/// does have when the one asked for is not among them.
pub fn ming_tts_model_voice_name(
    name: &str,
    emotion: Option<&str>,
) -> Result<String, Vec<&'static str>> {
    let name = name.trim();
    let Some(voice) = ming_tts_voice(name) else {
        return match emotion {
            None => Ok(name.to_string()),
            Some(_) => Err(Vec::new()),
        };
    };
    if emotion.is_none() {
        if let Some((_, variant)) = voice.variants.iter().find(|(_, variant)| *variant == name) {
            return Ok(variant.to_string());
        }
    }
    voice
        .model_name(emotion)
        .map(str::to_string)
        .ok_or_else(|| voice.emotions().collect())
}

pub fn ming_tts_language(code: &str) -> Option<&'static MingTtsLanguage> {
    let code = code.trim().to_ascii_lowercase().replace('_', "-");
    let primary = code.split('-').next().unwrap_or("");
    MING_TTS_LANGUAGES.iter().find(|language| {
        language.code == primary
            || language.name.eq_ignore_ascii_case(&code)
            || (language.code == "zh" && matches!(primary, "cmn" | "yue"))
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_table_has_no_duplicates_and_resolves_every_name() {
        let mut names = std::collections::HashSet::new();
        for voice in &MING_TTS_VOICES {
            assert!(names.insert(voice.name), "{} listed twice", voice.name);
            assert!(matches!(voice.language, "en" | "zh"));
            assert!(matches!(voice.gender, "female" | "male"));
            assert!(!voice.description.is_empty());
            assert_eq!(ming_tts_voice(voice.name), Some(voice));
            for (emotion, variant) in voice.variants {
                assert!(MING_TTS_EMOTIONS.contains(emotion), "{emotion}");
                assert!(variant.starts_with(voice.name), "{variant}");
                assert_eq!(ming_tts_voice(variant), Some(voice));
                assert!(names.insert(variant), "{variant} listed twice");
            }
        }
        assert_eq!(
            MING_TTS_VOICES
                .iter()
                .filter(|v| v.language == "en")
                .count(),
            3
        );
        assert!(ming_tts_voice(MING_TTS_DEFAULT_VOICE).is_some());
        assert!(ming_tts_voice(MING_TTS_DEFAULT_CHINESE_VOICE).is_some());
        assert!(ming_tts_voice("us_emma").is_some());
        for (alias, voice) in MING_TTS_OPENAI_VOICE_ALIASES {
            assert_eq!(ming_tts_voice(alias).unwrap().name, voice);
            assert_eq!(
                ming_tts_model_voice_name(alias, None),
                Ok(voice.to_string())
            );
        }
    }

    #[test]
    fn voices_resolve_to_the_names_the_model_reads() {
        assert_eq!(
            ming_tts_model_voice_name("US_Emma", None),
            Ok("US_Emma".into())
        );
        // A voice with one delivery has no emotions to ask for.
        assert_eq!(
            ming_tts_model_voice_name("US_Emma", Some("sad")),
            Err(vec![])
        );
        // A base name resolves to its neutral variant, or to an emotion.
        assert_eq!(
            ming_tts_model_voice_name("灵风信", None),
            Ok("灵风信_中性".into())
        );
        assert_eq!(
            ming_tts_model_voice_name("灵风信", Some("Angry")),
            Ok("灵风信_生气".into())
        );
        // A variant's own name is itself.
        assert_eq!(
            ming_tts_model_voice_name("灵风信_生气", None),
            Ok("灵风信_生气".into())
        );
        let missing = ming_tts_model_voice_name("灵风信", Some("fearful")).unwrap_err();
        assert!(missing.contains(&"happy") && !missing.contains(&"fearful"));
        // Names outside the table pass through.
        assert_eq!(
            ming_tts_model_voice_name("Some Other Voice", None),
            Ok("Some Other Voice".into())
        );
    }

    #[test]
    fn languages_resolve_by_code_or_name() {
        assert_eq!(ming_tts_language("de").unwrap().name, "German");
        assert_eq!(ming_tts_language("en-US").unwrap().code, "en");
        assert_eq!(ming_tts_language("Japanese").unwrap().code, "ja");
        assert_eq!(ming_tts_language("yue").unwrap().code, "zh");
        assert!(ming_tts_language("it").is_none());
        assert!(ming_tts_language("de").unwrap().must_be_named);
        assert!(!ming_tts_language("zh").unwrap().must_be_named);
    }
}
