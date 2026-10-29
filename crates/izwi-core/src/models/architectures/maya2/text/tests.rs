//! Tests of the Maya2 text normaliser.
//!
//! * `upstream_expectations`: every input → output pair upstream states
//!   (`maya3/tests/test_text.py`, the README rule tables, docstrings and
//!   comments) and `test_text.py`'s property tests, from
//!   `tests/fixtures/maya2/normaliser_expectations.json` (`check`: exact by
//!   default; `contains` for a stated fragment; `same_as_normaliser` = the
//!   wrapper equals the bare normaliser on the caps-folded text, with
//!   `dash` in the request language when `"dash": true`; `no_digits` /
//!   `no_digits_ends_with`; `absent` strings must not appear);
//! * `language_method_docstrings`: the examples the language classes'
//!   docstrings and comments state, cut from the source text;
//! * wrapper behaviour (tags, caps folding, whitespace, P1, the fallback,
//!   never failing);
//! * per-language smoke tests of the core matchers. Their expectations are
//!   DERIVED BY READING THE SOURCE, NOT BY RUNNING IT: the composition of each
//!   reading was worked out by hand from `normalize.py` / `indic_languages.py`
//!   and is assembled here from the raw table words of `tables.rs`.

use super::lang::{language_table, Lang, CODES};
use super::pyre::{is_alnum, is_digit};
use super::tables::{self as tb, LangData};
use super::{caps_fold, chars, normalise_text, normalise_with, py_strip, rules, st, tn};

const EXPECTATIONS: &str =
    include_str!("../../../../../tests/fixtures/maya2/normaliser_expectations.json");

/// The wrapper without the panic guard, so that a panic fails the test.
fn normalise_unguarded(text: &str, lang: &str) -> String {
    normalise_with(text, lang, |folded, seg, lang, before_tag| {
        super::normalise_piece(folded, seg, lang, before_tag).unwrap_or_else(|_| folded.to_string())
    })
}

fn collapse(s: &str) -> String {
    py_strip(&st(&super::pyre::re!(r"\s+").sub(&chars(s), " "))).to_string()
}

#[test]
fn upstream_expectations() {
    let rows: Vec<serde_json::Value> = serde_json::from_str(EXPECTATIONS).unwrap();
    assert!(rows.len() >= 140, "{} rows", rows.len());
    let mut failures = Vec::new();
    for row in &rows {
        let field = |k: &str| row[k].as_str().unwrap().to_string();
        let (lang, input, want, source) = (
            field("lang"),
            field("input"),
            field("output"),
            field("source"),
        );
        let got = normalise_unguarded(&input, &lang);
        let mut ok = match row.get("check").and_then(|c| c.as_str()) {
            None | Some("exact") => got == want,
            Some("contains") => got.contains(&want),
            Some("same_as_normaliser") => {
                let folded = st(&caps_fold(&chars(&input)));
                let mut plain = collapse(&tn::normalize(&folded, &lang).expect("normaliser"));
                if row.get("dash").is_some() {
                    let dash = tb::DASH
                        .iter()
                        .find(|(k, _)| *k == lang)
                        .map_or("dash", |(_, v)| *v);
                    plain = plain.replace(" dash ", &format!(" {dash} "));
                }
                got == plain
            }
            Some("no_digits") => !got.is_empty() && !got.chars().any(is_digit),
            Some("no_digits_ends_with") => {
                !got.is_empty() && !got.chars().any(is_digit) && got.ends_with(&want)
            }
            Some(other) => panic!("unknown check {other}"),
        };
        if let Some(absent) = row.get("absent").and_then(|a| a.as_array()) {
            ok &= absent.iter().all(|a| !got.contains(a.as_str().unwrap()));
        }
        if !ok {
            failures.push(format!(
                "{source} [{lang}] {input:?}\n   want {want:?}\n    got {got:?}"
            ));
        }
    }
    assert!(
        failures.is_empty(),
        "{} of {} failed:\n{}",
        failures.len(),
        rows.len(),
        failures.join("\n")
    );
}

fn t(code: &str) -> &'static Lang {
    language_table(code).unwrap()
}

#[test]
fn language_method_docstrings() {
    assert_eq!(t("bn").cardinal(100), "একশো", "indic_languages.py:205");
    assert_eq!(t("bn").cardinal(400), "চারশো", "indic_languages.py:205");
    assert_eq!(t("gu").cardinal(250), "બસો પચાસ", "indic_languages.py:366");
    assert_eq!(
        t("gu").cardinal(147),
        "એકસો સુડતાળીસ",
        "indic_languages.py:366"
    );
    assert_eq!(
        t("kn").cardinal(147),
        "ನೂರ ನಲವತ್ತೇಳು",
        "indic_languages.py:441"
    );
    assert_eq!(t("kn").cardinal(1000), "ಸಾವಿರ", "indic_languages.py:543");
    assert_eq!(
        t("kn").cardinal(1947),
        "ಸಾವಿರದ ಒಂಬೈನೂರ ನಲವತ್ತೇಳು",
        "indic_languages.py:543"
    );
    assert_eq!(
        t("ml").cardinal(20000),
        "ഇരുപതിനായിരം",
        "indic_languages.py:731"
    );
    assert_eq!(
        t("ml").cardinal(54000),
        "അമ്പത്തിനാലായിരം",
        "indic_languages.py:734"
    );
    assert_eq!(
        t("ml").cardinal(90000),
        "തൊണ്ണൂറായിരം",
        "indic_languages.py:735"
    );
    assert_eq!(t("mr").cardinal(200), "दोनशे", "indic_languages.py:911");
    assert_eq!(t("ta").cardinal(4000), "நான்காயிரம்", "indic_languages.py:1287");
    assert_eq!(
        t("ta").cardinal(54000),
        "ஐம்பத்துநான்காயிரம்",
        "indic_languages.py:1287"
    );
    assert_eq!(
        t("bn").year(1947),
        "উনিশশো সাতচল্লিশ",
        "indic_languages.py:214"
    );
    assert_eq!(t("bn").year(2026), "দুই হাজার ছাব্বিশ", "indic_languages.py:214");
    assert_eq!(
        t("gu").year(1947),
        "ઓગણીસો સુડતાળીસ",
        "indic_languages.py:373"
    );
    assert_eq!(
        t("gu").year(1857),
        "અઢારસો સત્તાવન",
        "indic_languages.py:374"
    );
    assert_eq!(
        t("kn").year(1947),
        "ಸಾವಿರದ ಒಂಬೈನೂರ ನಲವತ್ತೇಳು",
        "indic_languages.py:553"
    );
    assert_eq!(
        t("kn").year(2026),
        "ಎರಡು ಸಾವಿರದ ಇಪ್ಪತ್ತಾರು",
        "indic_languages.py:553"
    );
    assert_eq!(
        t("ml").year(1947),
        "ആയിരത്തി തൊള്ളായിരത്തി നാൽപ്പത്തിയേഴ്",
        "indic_languages.py:770"
    );
    assert_eq!(
        t("mr").year(1947),
        "एकोणीसशे सत्तेचाळीस",
        "indic_languages.py:923"
    );
    assert_eq!(
        t("mr").year(2026),
        "दोन हजार सव्वीस",
        "indic_languages.py:923"
    );
    assert_eq!(
        t("ta").year(1947),
        "ஆயிரத்துத் தொள்ளாயிரத்து நாற்பத்தேழு",
        "indic_languages.py:1316"
    );
    assert_eq!(
        t("ta").year(2026),
        "இரண்டாயிரத்து இருபத்தாறு",
        "indic_languages.py:1317"
    );
    assert_eq!(
        t("hi").ordinal(12, "वीं"),
        "बारहवीं",
        "indic_languages.py:100"
    );
    assert_eq!(t("hi").ordinal(1, "वीं"), "पहली", "indic_languages.py:103");
    assert_eq!(t("bn").ordinal(1, "লা"), "পহেলা", "indic_languages.py:224");
    assert_eq!(t("bn").ordinal(2, "রা"), "দোসরা", "indic_languages.py:224");
    assert_eq!(t("bn").ordinal(4, "ঠা"), "চৌঠা", "indic_languages.py:224");
    assert_eq!(t("bn").ordinal(5, "ই"), "পাঁচই", "indic_languages.py:225");
    assert_eq!(t("bn").ordinal(19, "শে"), "উনিশে", "indic_languages.py:225");
    assert_eq!(t("bn").ordinal(12, "তম"), "বারোতম", "indic_languages.py:225");
    assert_eq!(t("gu").ordinal(1, "લી"), "પહેલી", "indic_languages.py:353");
    assert_eq!(t("gu").ordinal(1, "લી"), "પહેલી", "indic_languages.py:386");
    assert_eq!(t("gu").ordinal(2, "જી"), "બીજી", "indic_languages.py:386");
    assert_eq!(t("gu").ordinal(4, "થા"), "ચોથા", "indic_languages.py:386");
    assert_eq!(t("gu").ordinal(12, "મી"), "બારમી", "indic_languages.py:391");
    assert_eq!(t("gu").ordinal(5, "મો"), "પાંચમો", "indic_languages.py:391");
    assert_eq!(
        t("mr").ordinal(12, "वा"),
        "बारावा",
        "indic_languages.py:933"
    );
    assert_eq!(
        t("mr").ordinal(12, "वी"),
        "बारावी",
        "indic_languages.py:933"
    );
    assert_eq!(
        t("pa").ordinal(12, "ਵੀਂ"),
        "ਬਾਰਾਂਵੀਂ",
        "indic_languages.py:1148"
    );
    assert_eq!(
        t("ta").ordinal(12, "ஆவது"),
        "பன்னிரண்டாவது",
        "indic_languages.py:1321"
    );
    assert_eq!(
        t("ta").ordinal(12, "ஆம்"),
        "பன்னிரண்டாம்",
        "indic_languages.py:1321"
    );
    assert_eq!(
        t("ta").ordinal(12, "வது"),
        "பன்னிரண்டாவது",
        "indic_languages.py:1321"
    );
    assert_eq!(
        t("ta").ordinal(12, "ம்"),
        "பன்னிரண்டாம்",
        "indic_languages.py:1321"
    );
    assert_eq!(t("kn").ordinal(2, ""), "ಎರಡನೇ", "indic_languages.py:564");
    assert_eq!(t("ml").ordinal(1, ""), "ഒന്നാം", "indic_languages.py:709");
    assert_eq!(t("ml").ordinal(10, ""), "പത്താം", "indic_languages.py:709");
    assert_eq!(t("ml").ordinal(2, ""), "രണ്ടാം", "indic_languages.py:782");
    assert_eq!(t("te").ordinal(2, ""), "రెండవ", "indic_languages.py:1527");
    assert_eq!(t("te").ordinal(8, ""), "ఎనిమిదవ", "indic_languages.py:1527");
    assert_eq!(t("te").ordinal(20, ""), "ఇరవయ్యవ", "indic_languages.py:1527");
    assert_eq!(t("te").ordinal(100, ""), "వందవ", "indic_languages.py:1527");
    assert_eq!(
        t("kn").decade(90, false),
        "ತೊಂಬತ್ತರ",
        "indic_languages.py:557"
    );
    assert_eq!(
        t("ml").decade(90, false),
        "തൊണ്ണൂറുകൾ",
        "indic_languages.py:774"
    );
    assert_eq!(
        t("bn").attach_postposition("টাকা", "এর").as_deref(),
        Some("টাকার"),
        "indic_languages.py:239"
    );
    assert_eq!(
        t("bn").attach_postposition("শতাংশ", "এ").as_deref(),
        Some("শতাংশে"),
        "indic_languages.py:239"
    );
    assert_eq!(
        t("bn").attach_postposition("টা", "মধ্যে").as_deref(),
        Some("টার মধ্যে"),
        "indic_languages.py:239"
    );
    assert_eq!(
        t("gu").attach_postposition("રૂપિયા", "માં").as_deref(),
        Some("રૂપિયામાં"),
        "indic_languages.py:397"
    );
    assert_eq!(
        t("kn").attach_postposition("ಐದು", "ಇಂದ").as_deref(),
        Some("ಐದರಿಂದ"),
        "indic_languages.py:572"
    );
    assert_eq!(
        t("kn").attach_postposition("ಮೂರು", "ಗೆ").as_deref(),
        Some("ಮೂರಕ್ಕೆ"),
        "indic_languages.py:572"
    );
    assert_eq!(
        t("kn").attach_postposition("ಒಂಬತ್ತು", "ನಲ್ಲಿ").as_deref(),
        Some("ಒಂಬತ್ತರಲ್ಲಿ"),
        "indic_languages.py:572"
    );
    assert_eq!(
        t("kn").attach_postposition("ನಿಮಿಷ", "ಕ್ಕೆ").as_deref(),
        Some("ನಿಮಿಷಕ್ಕೆ"),
        "indic_languages.py:573"
    );
    assert_eq!(
        t("kn").attach_postposition("ಸೊನ್ನೆ", "ಗೆ").as_deref(),
        Some("ಸೊನ್ನೆಗೆ"),
        "indic_languages.py:573"
    );
    assert_eq!(
        t("ml").attach_postposition("രൂപ", "ന്റെ").as_deref(),
        Some("രൂപയുടെ"),
        "indic_languages.py:790"
    );
    assert_eq!(
        t("ml").attach_postposition("ആറ്", "ൽ").as_deref(),
        Some("ആറിൽ"),
        "indic_languages.py:790"
    );
    assert_eq!(
        t("ml").attach_postposition("രൂപ", "ൽ").as_deref(),
        Some("രൂപയിൽ"),
        "indic_languages.py:790"
    );
    assert_eq!(
        t("mr").attach_postposition("रुपये", "चा").as_deref(),
        Some("रुपयांचा"),
        "indic_languages.py:941"
    );
    assert_eq!(
        t("mr").attach_postposition("मिनिटे", "ला").as_deref(),
        Some("मिनिटांना"),
        "indic_languages.py:941"
    );
    assert_eq!(
        t("or").attach_postposition("ଟଙ୍କା", "ରୁ").as_deref(),
        Some("ଟଙ୍କାରୁ"),
        "indic_languages.py:1048"
    );
    assert_eq!(
        t("te").attach_postposition("రూపాయలు", "గా").as_deref(),
        Some("రూపాయలుగా"),
        "indic_languages.py:1601"
    );
    assert_eq!(
        t("as").attach_postposition("টকা", "ৰ").as_deref(),
        Some("টকাৰ"),
        "lexicon.py:66"
    );
    assert_eq!(
        t("as").attach_postposition("মিনিট", "ত").as_deref(),
        Some("মিনিটত"),
        "lexicon.py:66"
    );
    assert_eq!(
        t("te")
            .attach_postposition("మూడు లక్షలు", "రూపాయలు")
            .as_deref(),
        Some("మూడు లక్షల రూపాయలు"),
        "indic_languages.py:1604"
    );
    assert_eq!(
        t("te").attach_postposition("ఒకటి", "litre").as_deref(),
        Some("ఒక litre"),
        "indic_languages.py:1629"
    );
    assert_eq!(
        t("te").attach_postposition("రూపాయలు", "గా").as_deref(),
        Some("రూపాయలుగా"),
        "indic_languages.py:1638"
    );
    assert_eq!(
        t("te").attach_postposition("శాతం", "కి").as_deref(),
        Some("శాతానికి"),
        "indic_languages.py:1653"
    );
    assert_eq!(
        t("te").attach_postposition("గంటలు", "కి").as_deref(),
        Some("గంటలకి"),
        "indic_languages.py:1656"
    );
    assert_eq!(
        t("te").attach_postposition("రూపాయలు", "ని").as_deref(),
        Some("రూపాయలను"),
        "indic_languages.py:1658"
    );
    assert_eq!(
        t("te").attach_postposition("గంటలు", "లో").as_deref(),
        Some("గంటలలో"),
        "indic_languages.py:1659"
    );
    assert_eq!(
        t("te").attach_postposition("ఐదు", "కి").as_deref(),
        Some("ఐదుకి"),
        "indic_languages.py:1661"
    );
}

#[test]
fn assamese_lexicon_tables() {
    // test_assamese_no_bengali_forms: no Bengali ra in the Assamese number words
    let ones = &t("as").ones;
    assert_eq!(ones.len(), 100);
    assert!(ones.iter().all(|w| !w.contains(tb::LEX_R)));
    // test_lexicon_off_restores_vendored_tables: Bengali keeps its own ra
    assert!(t("bn").cardinal(30).contains(tb::LEX_R));
}

// ---------------------------------------------------------------------------
// Wrapper behaviour
// ---------------------------------------------------------------------------

#[test]
fn caps_fold_keeps_acronyms() {
    // test_caps_fold_and_acronyms (`T._caps_fold`)
    assert_eq!(
        st(&caps_fold(&chars("ASHA OTP NASA KYC"))),
        "Asha OTP Nasa KYC"
    );
    assert_eq!(
        st(&caps_fold(&chars("RAHUL HDFC AEIOU THE"))),
        "Rahul HDFC Aeiou THE"
    );
    // no word boundary inside a Latin-Devanagari compound
    assert_eq!(st(&caps_fold(&chars("RAHULभाई"))), "RAHULभाई");
    assert_eq!(
        normalise_text("MEET RAHUL KUMAR AT NOON", "en"),
        "Meet Rahul Kumar AT Noon"
    );
}

#[test]
fn tags_are_protected() {
    assert_eq!(
        normalise_text("[laughs]5[sighs]", "en"),
        "[laughs]five[sighs]"
    );
    assert_eq!(normalise_text("  [whispers]  ", "hi"), "[whispers]");
    assert_eq!(
        normalise_text("[long pause] 7 [laughs]", "ta"),
        "[long pause] ஏழு [laughs]"
    );
    // not a tag (upper case): caps-folded, and the brackets are decorative junk
    assert_eq!(normalise_text("[LAUGHS] 2", "en"), "Laughs two");
    // a tag is never read, even with digits around it
    assert_eq!(normalise_text("1[laughs]2", "en"), "one[laughs]two");
}

#[test]
fn milliseconds_are_not_the_honorific() {
    // Not upstream: its abbreviation table reads every "ms" as "miss".
    assert_eq!(
        normalise_text("Time to first token can be as low as 50 ms.", "en"),
        "Time to first token can be as low as fifty milliseconds."
    );
    assert_eq!(
        normalise_text("a 200ms delay", "en"),
        "a two hundred milliseconds delay"
    );
    assert_eq!(
        normalise_text("a 50-ms budget", "en"),
        "a fifty milliseconds budget"
    );
    assert_eq!(
        normalise_text("TIMEOUT 500MS", "en"),
        "Timeout five hundred milliseconds"
    );
    assert_eq!(normalise_text("1 ms", "en"), "one millisecond");
    assert_eq!(
        normalise_text("1.5 ms", "en"),
        "one point five milliseconds"
    );
    // the honorific, and a word that merely starts with "ms", are untouched
    assert_eq!(
        normalise_text("Ms Smith is here", "en"),
        "miss Smith is here"
    );
    assert_eq!(
        normalise_text("Ms. Smith is here", "en"),
        "miss Smith is here"
    );
    assert_eq!(normalise_text("50 msec", "en"), "fifty msec");
    // other languages have no "miss" to avoid and keep upstream's reading
    assert!(!normalise_text("50 ms", "hi").contains("millisecond"));
}

#[test]
fn whitespace_is_collapsed() {
    assert_eq!(normalise_text(" \t a \n\n b \u{a0} c  ", "en"), "a b c");
    assert_eq!(normalise_text("", "en"), "");
    assert_eq!(normalise_text("   ", "hi"), "");
}

#[test]
fn unknown_language_keeps_the_text() {
    // test_fallback_returns_text_unchanged: "x 1" in "fr" -> "x 1"
    assert_eq!(
        normalise_text("RAHUL paid  25 [laughs]", "fr"),
        "Rahul paid 25 [laughs]"
    );
}

#[test]
fn a_failing_piece_is_kept_as_written() {
    // test_fallback_returns_text_unchanged with the normaliser raising
    let out = normalise_with("I have 25 apples [laughs] 3", "en", |folded, _, _, _| {
        folded.to_string()
    });
    assert_eq!(out, "I have 25 apples [laughs] 3");
}

#[test]
fn tag_comma_when_switched_on() {
    // test_tag_keeps_comma: off by default (in the fixture); with TAG_COMMA = True
    let on = |text: &str| {
        normalise_with(text, "hi", |folded, seg, lang, before_tag| {
            let pre = rules::pre(folded, lang).unwrap();
            let out = tn::normalize(&pre, lang).unwrap();
            rules::post(&out, seg, lang, before_tag, true)
        })
    };
    assert_eq!(
        on("नौ, आठ, सात, [curious] छह, पाँच"),
        "नौ, आठ, सात, [curious] छह, पाँच"
    );
    assert_eq!(
        on("आपका पेमेंट फेल हुआ था ना, [sighs] बारह मई को"),
        "आपका पेमेंट फेल हुआ था ना, [sighs] बारह मई को"
    );
    assert_eq!(on("ठीक है. [laughs] चलो"), "ठीक है. [laughs] चलो");
}

/// xorshift64*, for reproducible fuzz strings.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_f491_4f6c_dd1d)
    }

    fn pick<'a>(&mut self, items: &[&'a str]) -> &'a str {
        items[(self.next() % items.len() as u64) as usize]
    }
}

#[test]
fn never_fails_on_arbitrary_text() {
    const PIECES: &[&str] = &[
        "0",
        "7",
        "42",
        "1,250",
        "1,23,456.75",
        "3.14",
        "99.",
        "-5",
        "+91",
        "98204 51120",
        "560038",
        "4111 2233 4455 6677",
        "15-08-25",
        "12/03/2024",
        "2024-02-30",
        "31/02/2024",
        "11/26",
        "2023-24",
        "10:30",
        "25:99",
        "7:15 pm",
        "1:02:03",
        "₹",
        "₹95.41",
        "$3.5M",
        "€",
        "%",
        "50%",
        "3.5%-4%",
        "x²",
        "√16",
        "10^3",
        "a = b + 3",
        "<=",
        "≥",
        "°C",
        "45°F",
        "1/2",
        "3/4",
        "×",
        "÷",
        "±",
        "@",
        "#45",
        "_",
        "&",
        "...",
        "…",
        "(",
        ")",
        "[",
        "]",
        "[laughs]",
        "[sighs",
        "{",
        "}",
        "<b>",
        "${x}",
        "\"",
        "'",
        "‘Y’",
        "-",
        "–",
        "—",
        "/",
        "//",
        ":",
        ";",
        "!",
        "?",
        ".",
        ",",
        " ",
        "  ",
        "\t",
        "\n",
        "\u{0}",
        "\u{7}",
        "\u{1b}",
        "\u{7f}",
        "\u{200b}",
        "\u{200c}",
        "\u{200d}",
        "\u{feff}",
        "\u{e000}",
        "\u{f0000}",
        "😀",
        "👍🏽",
        "🇮🇳",
        "★",
        "→",
        "™",
        "a",
        "Z",
        "RAHUL",
        "OTP",
        "NEFT",
        "Kepler-186f",
        "WH-1000XM5",
        "MH12AB8821",
        "H2SO4",
        "Class XII",
        "Room 314",
        "1980s",
        "'90s",
        "21st",
        "12वीं",
        "१२३",
        "০৯৮",
        "௧௨",
        "൩",
        "٣",
        "３",
        "v4.5.2",
        "p.a.",
        "km/h",
        "100 MB/s",
        "No. 12",
        "Dr.",
        "https://www.example.com/a-b?x=1",
        "a.b@gmail.com",
        "rahul.deshmukh@okaxis",
        "irctc.co.in.",
        "@test_user",
        "पिन 4532",
        "ओटीपी",
        "रुपये",
        "टाका",
        "ரூபாய்",
        "ಗಂಟೆಗೆ",
        "గంటలు",
        "ਵਜੇ",
        "বজাত",
        "സമയം",
        "નંબર",
        "ଟଙ୍କା",
        "मराठी",
        "ಕನ್ನಡ",
        "తెలుగు",
        "தமிழ்",
        "മലയാളം",
        "ગુજરાતી",
        "ਪੰਜਾਬੀ",
        "অসমীয়া",
        "বাংলা",
        "ଓଡ଼ିଆ",
        "中文",
        "Ελληνικά",
        "ß",
        "İ",
        "ǅ",
        "Ⅻ",
        "½",
        "²",
    ];
    let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
    let mut texts: Vec<String> = (0..300)
        .map(|_| {
            let n = 1 + rng.next() % 12;
            (0..n)
                .map(|_| rng.pick(PIECES))
                .collect::<Vec<_>>()
                .join(if rng.next().is_multiple_of(3) { "" } else { " " })
        })
        .collect();
    texts.push("9".repeat(400));
    texts.push(format!("call {} now", "1234567890".repeat(4)));
    texts.push(format!("{} [laughs]", "1,".repeat(40)));
    texts.push("₹".repeat(50));
    for (i, text) in texts.iter().enumerate() {
        let lang = CODES[i % CODES.len()];
        for lang in [lang, "en"] {
            let out = normalise_unguarded(text, lang);
            if text.chars().any(is_alnum) {
                assert!(!out.is_empty(), "{lang}: {text:?} -> empty");
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Per-language smoke tests (DERIVED BY READING THE SOURCE, NOT BY RUNNING IT)
// ---------------------------------------------------------------------------

fn data(code: &str) -> &'static LangData {
    match code {
        "hi" => &tb::HI,
        "bn" => &tb::BN,
        "as" => &tb::AS,
        "gu" => &tb::GU,
        "kn" => &tb::KN,
        "ml" => &tb::ML,
        "mr" => &tb::MR,
        "or" => &tb::OR,
        "pa" => &tb::PA,
        "ta" => &tb::TA,
        _ => &tb::TE,
    }
}

/// `ONES[n]` as installed (Assamese with ৰ, lexicon L1).
fn w(code: &str, n: usize) -> String {
    let word = data(code).ones[n].to_string();
    if code == "as" {
        word.replace(tb::LEX_R, tb::LEX_RR)
    } else {
        word
    }
}

/// `SYM[key]` as installed (Assamese: ৰ and `_AS_SYM`; R9 paise words).
fn sym(code: &str, key: &str) -> String {
    let find = |table: &[(&str, &str)]| {
        table
            .iter()
            .find(|(k, _)| *k == key)
            .map(|(_, v)| v.to_string())
    };
    if code == "as" {
        if let Some(v) = find(tb::AS_SYM) {
            return v;
        }
    }
    let v = find(data(code).sym)
        .or_else(|| {
            (key == "paisa")
                .then(|| {
                    find(
                        tb::PAISE
                            .iter()
                            .filter(|(c, _)| *c == code)
                            .map(|(_, w)| ("paisa", *w))
                            .collect::<Vec<_>>()
                            .as_slice(),
                    )
                })
                .flatten()
        })
        .unwrap();
    if code == "as" {
        v.replace(tb::LEX_R, tb::LEX_RR)
    } else {
        v
    }
}

fn letter(code: &str, c: &str) -> String {
    let v = data(code).letters.iter().find(|(k, _)| *k == c).unwrap().1;
    if code == "as" {
        v.replace(tb::LEX_R, tb::LEX_RR)
    } else {
        v.to_string()
    }
}

fn hi_fold(s: &str) -> String {
    s.chars()
        .filter_map(
            |c| match tb::HINDI_NUKTA_FOLD.iter().find(|(k, _)| *k == c) {
                Some((_, to)) => *to,
                None => Some(c),
            },
        )
        .collect()
}

fn drop_last(s: &str) -> &str {
    super::lang::drop_last(s)
}

/// "1234": `cardinal`, as each language composes thousands and hundreds.
fn expect_1234(code: &str) -> String {
    let d = data(code);
    let (o1, o2, o34) = (w(code, 1), w(code, 2), w(code, 34));
    match code {
        "bn" | "mr" => format!("{o1} {} {o2}{} {o34}", d.thousand, d.hundred),
        "as" => format!("{o1} {} {} {o34}", d.thousand, tb::AS_HUNDREDS[2]),
        "gu" => format!("{o1} {} {} {o34}", d.thousand, tb::GU_HUNDREDS_FUSED[1].1),
        "kn" => format!("{} {} {o34}", tb::KN_THOUSAND_COMB, tb::KN_HUNDREDS_COMB[2]),
        "ml" => format!(
            "{}ത്തി {} {o34}",
            drop_last(tb::ML_TH_STAND[0].1),
            tb::ML_H_COMB[1].1
        ),
        "ta" => format!("ஆயிரத்து {} {o34}", tb::TA_HUNDREDS_COMB[2]),
        "te" => format!("వెయ్యి {o2} వందల {o34}"),
        _ => format!("{o1} {} {o2} {} {o34}", d.thousand, d.hundred),
    }
}

/// 2024 as a year (outside 1100-1999, so a cardinal everywhere).
fn expect_2024(code: &str) -> String {
    let (o2, o24) = (w(code, 2), w(code, 24));
    match code {
        "kn" => format!("{o2} {} {o24}", tb::KN_THOUSAND_COMB),
        "ml" => format!("{}ത്തി {o24}", drop_last(tb::ML_TH_STAND[1].1)),
        "ta" => format!("{}ாயிரத்து {o24}", drop_last(&o2)),
        "te" => format!("{o2} వేల {o24}"),
        _ => format!("{o2} {} {o24}", data(code).thousand),
    }
}

fn smoke_cases(code: &str) -> Vec<(&'static str, String)> {
    if code == "en" {
        return vec![
            ("1234", "one thousand two hundred thirty four".into()),
            ("₹12.50", "twelve rupees fifty paise".into()),
            ("12/03/2024", "twelve March two thousand twenty four".into()),
            ("10:30", "ten thirty".into()),
            (
                "9820451120",
                "nine, eight, two, zero, four, five, one, one, two, zero".into(),
            ),
            ("www.example.com", "W W W dot example dot com".into()),
            ("info@example.com", "info at example dot com".into()),
        ];
    }
    let d = data(code);
    let date = if code == "ta" {
        format!(
            "{} {} தேதி {}",
            d.months[2],
            tb::TAMIL_ORDINALS[12],
            expect_2024(code)
        )
    } else {
        format!("{} {} {}", w(code, 12), d.months[2], expect_2024(code))
    };
    let time = if code == "te" {
        format!("{} గంటల {} నిమిషాలు", w(code, 10), w(code, 30))
    } else {
        d.time_hm
            .replace("{h}", &w(code, 10))
            .replace("{m}", &w(code, 30))
    };
    let phone = "9820451120"
        .chars()
        .map(|c| w(code, c as usize - '0' as usize))
        .collect::<Vec<_>>()
        .join(", ");
    let (dot, ww) = (sym(code, "dot"), letter(code, "W"));
    vec![
        ("1234", expect_1234(code)),
        (
            "₹12.50",
            format!(
                "{} {} {} {}",
                w(code, 12),
                sym(code, "rupee"),
                w(code, 50),
                sym(code, "paisa")
            ),
        ),
        ("12/03/2024", date),
        ("10:30", time),
        ("9820451120", phone),
        (
            "www.example.com",
            format!("{ww} {ww} {ww} {dot} example {dot} com"),
        ),
        (
            "info@example.com",
            format!("info {} example {dot} com", sym(code, "at")),
        ),
    ]
}

#[test]
fn per_language_smoke() {
    let mut failures = Vec::new();
    for code in CODES {
        for (input, want) in smoke_cases(code) {
            let want = if code == "hi" { hi_fold(&want) } else { want };
            let got = normalise_text(input, code);
            if got != want {
                failures.push(format!(
                    "[{code}] {input:?}\n   want {want:?}\n    got {got:?}"
                ));
            }
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}
