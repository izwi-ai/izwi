//! A small backtracking matcher with Python `re` semantics, for the patterns
//! the normaliser was written against.
//!
//! The `regex` crate has no look-around or back-references, and its `\w`,
//! `\b` and `\s` follow UTS #18 (combining marks are word characters), while
//! Python's `str` patterns use `str.isalnum()` (letters and numbers only): a
//! Devanagari vowel sign ends a word for Python's `\b`, and upstream's
//! `(?!\w)` guards rely on that. So the patterns run here, on `char` slices
//! (Python indexes strings by code point), with:
//!
//! * `\w` = Python's `isalnum() or "_"`, `\d` = `isdecimal()`, `\s` =
//!   `isspace()`, `\b` / `\B` from that `\w`; `.` excludes `\n`;
//! * `^` only at 0, `$` at the end or before a final `\n`, `\A`, `\Z`;
//! * ordered alternation, greedy quantifiers with backtracking, `{m,n}`
//!   (a `{` that is not a quantifier is a literal, as in Python), groups,
//!   `(?:…)`, `(?P<name>…)`, `(?=…)`, `(?!…)`, fixed-width `(?<=…)` /
//!   `(?<!…)`, back-references `\1`…`\9`;
//! * `IGNORECASE` by simple case folding (a literal matches when the
//!   lower-cased characters agree; a set when the character, its lower or its
//!   upper case is in it);
//! * `finditer` / `sub` / `split` with Python 3.7+ empty-match rules.
//!
//! Lazy quantifiers, inline flags and unbounded repetition of a sub-pattern
//! that can match empty are rejected when a pattern is compiled (upstream
//! uses none of them).

use icu_properties::props::{GeneralCategory, NumericType};
use icu_properties::CodePointMapData;

/// Backtracking steps one match attempt may take. Upstream's patterns stay
/// far below it on real text; an adversarial input that would backtrack
/// exponentially stops here instead, and the normaliser keeps that piece as
/// written ([`take_exhausted`]).
const STEP_BUDGET: usize = 2_000_000;

thread_local! {
    static EXHAUSTED: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Whether a match on this thread ran out of steps since the last call
/// (and reset the flag).
pub(crate) fn take_exhausted() -> bool {
    EXHAUSTED.with(|flag| flag.replace(false))
}

// ---------------------------------------------------------------------------
// Python's character predicates
// ---------------------------------------------------------------------------

fn general_category(c: char) -> GeneralCategory {
    CodePointMapData::<GeneralCategory>::new().get(c)
}

fn numeric_type(c: char) -> NumericType {
    CodePointMapData::<NumericType>::new().get(c)
}

/// `str.isalpha()` for one character (categories Lu Ll Lt Lm Lo).
pub(crate) fn is_alpha(c: char) -> bool {
    if c.is_ascii() {
        return c.is_ascii_alphabetic();
    }
    matches!(
        general_category(c),
        GeneralCategory::UppercaseLetter
            | GeneralCategory::LowercaseLetter
            | GeneralCategory::TitlecaseLetter
            | GeneralCategory::ModifierLetter
            | GeneralCategory::OtherLetter
    )
}

/// `str.isdecimal()` for one character (`re`'s `\d`).
pub(crate) fn is_decimal(c: char) -> bool {
    if c.is_ascii() {
        return c.is_ascii_digit();
    }
    numeric_type(c) == NumericType::Decimal
}

/// `str.isdigit()` for one character (decimals plus e.g. superscripts).
pub(crate) fn is_digit(c: char) -> bool {
    if c.is_ascii() {
        return c.is_ascii_digit();
    }
    matches!(numeric_type(c), NumericType::Decimal | NumericType::Digit)
}

/// `str.isalnum()` for one character.
pub(crate) fn is_alnum(c: char) -> bool {
    if c.is_ascii() {
        return c.is_ascii_alphanumeric();
    }
    is_alpha(c) || numeric_type(c) != NumericType::None
}

/// `re`'s `\w` for `str` patterns: `isalnum()` or `_`.
pub(crate) fn is_word(c: char) -> bool {
    c == '_' || is_alnum(c)
}

/// `str.isspace()` for one character (`re`'s `\s`).
pub(crate) fn is_space(c: char) -> bool {
    c.is_whitespace() || ('\u{1c}'..='\u{1f}').contains(&c)
}

/// `unicodedata.category(c) in ("Cc", "Co")`.
pub(crate) fn is_control_or_private(c: char) -> bool {
    matches!(
        general_category(c),
        GeneralCategory::Control | GeneralCategory::PrivateUse
    )
}

/// `unicodedata.decimal(c)`: decimal digits are encoded in runs of ten
/// starting at zero, so the value is the distance to the run's start.
pub(crate) fn decimal_value(c: char) -> Option<u32> {
    if c.is_ascii_digit() {
        return Some(c as u32 - '0' as u32);
    }
    if !is_decimal(c) {
        return None;
    }
    let mut k = 0u32;
    let mut cp = c as u32;
    while cp > 0 {
        match char::from_u32(cp - 1) {
            Some(p) if is_decimal(p) => {
                k += 1;
                cp -= 1;
            }
            _ => break,
        }
    }
    Some(k % 10)
}

/// Simple lower-case mapping (`re.IGNORECASE`).
fn fold(c: char) -> char {
    if c.is_ascii() {
        return c.to_ascii_lowercase();
    }
    c.to_lowercase().next().unwrap_or(c)
}

fn upper1(c: char) -> char {
    if c.is_ascii() {
        return c.to_ascii_uppercase();
    }
    c.to_uppercase().next().unwrap_or(c)
}

// ---------------------------------------------------------------------------
// Parser
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq)]
enum ClassKind {
    Digit,
    Word,
    Space,
}

impl ClassKind {
    fn test(self, c: char) -> bool {
        match self {
            ClassKind::Digit => is_decimal(c),
            ClassKind::Word => is_word(c),
            ClassKind::Space => is_space(c),
        }
    }
}

#[derive(Clone, Debug)]
enum SetItem {
    Range(char, char),
    Class(ClassKind, bool),
}

#[derive(Clone, Debug)]
struct CharSet {
    negated: bool,
    items: Vec<SetItem>,
}

impl CharSet {
    fn raw(&self, c: char) -> bool {
        self.items.iter().any(|item| match *item {
            SetItem::Range(a, b) => a <= c && c <= b,
            SetItem::Class(kind, negated) => kind.test(c) != negated,
        })
    }

    fn test(&self, c: char, icase: bool) -> bool {
        let hit = self.raw(c) || (icase && (self.raw(fold(c)) || self.raw(upper1(c))));
        hit != self.negated
    }
}

#[derive(Clone, Debug)]
enum Node {
    Lit(char),
    Any,
    Set(CharSet),
    Group(Box<Node>, Option<usize>),
    Cat(Vec<Node>),
    Alt(Vec<Node>),
    Rep(Box<Node>, usize, Option<usize>),
    Bol,
    Eol,
    BeginText,
    EndText,
    WordB,
    NotWordB,
    Look(Box<Node>, bool, bool),
    Backref(usize),
}

struct Parser {
    p: Vec<char>,
    i: usize,
    groups: usize,
    names: Vec<(String, usize)>,
}

impl Parser {
    fn peek(&self) -> Option<char> {
        self.p.get(self.i).copied()
    }

    fn next(&mut self) -> char {
        let c = self.p[self.i];
        self.i += 1;
        c
    }

    fn eat(&mut self, s: &str) -> bool {
        let want: Vec<char> = s.chars().collect();
        if self.p[self.i..].starts_with(&want) {
            self.i += want.len();
            true
        } else {
            false
        }
    }

    fn alt(&mut self) -> Node {
        let mut alts = vec![self.cat()];
        while self.peek() == Some('|') {
            self.i += 1;
            alts.push(self.cat());
        }
        if alts.len() == 1 {
            alts.pop().unwrap()
        } else {
            Node::Alt(alts)
        }
    }

    fn cat(&mut self) -> Node {
        let mut items = Vec::new();
        while let Some(c) = self.peek() {
            if c == '|' || c == ')' {
                break;
            }
            let atom = self.atom();
            let atom = self.quantifier(atom);
            items.push(atom);
        }
        Node::Cat(items)
    }

    fn quantifier(&mut self, atom: Node) -> Node {
        let (min, max) = match self.peek() {
            Some('*') => {
                self.i += 1;
                (0, None)
            }
            Some('+') => {
                self.i += 1;
                (1, None)
            }
            Some('?') => {
                self.i += 1;
                (0, Some(1))
            }
            Some('{') => match self.braces() {
                Some(q) => q,
                None => return atom,
            },
            _ => return atom,
        };
        assert!(
            !matches!(self.peek(), Some('?') | Some('+')),
            "lazy/possessive quantifiers are not supported"
        );
        Node::Rep(Box::new(atom), min, max)
    }

    /// `{m}`, `{m,}`, `{,n}`, `{m,n}`; anything else leaves `{` a literal.
    fn braces(&mut self) -> Option<(usize, Option<usize>)> {
        let start = self.i;
        self.i += 1;
        let mut lo = String::new();
        while let Some(c) = self.peek().filter(|c| c.is_ascii_digit()) {
            lo.push(c);
            self.i += 1;
        }
        let q = if self.peek() == Some('}') {
            self.i += 1;
            lo.parse().ok().map(|n| (n, Some(n)))
        } else if self.peek() == Some(',') {
            self.i += 1;
            let mut hi = String::new();
            while let Some(c) = self.peek().filter(|c| c.is_ascii_digit()) {
                hi.push(c);
                self.i += 1;
            }
            if self.peek() == Some('}') {
                self.i += 1;
                let min = if lo.is_empty() {
                    0
                } else {
                    lo.parse().unwrap()
                };
                let max = if hi.is_empty() { None } else { hi.parse().ok() };
                Some((min, max))
            } else {
                None
            }
        } else {
            None
        };
        if q.is_none() {
            self.i = start;
        }
        q
    }

    fn atom(&mut self) -> Node {
        let c = self.next();
        match c {
            '(' => self.group(),
            '[' => Node::Set(self.set()),
            '.' => Node::Any,
            '^' => Node::Bol,
            '$' => Node::Eol,
            '\\' => self.escape(),
            '{' => Node::Lit('{'),
            _ => Node::Lit(c),
        }
    }

    fn group(&mut self) -> Node {
        let node = if self.eat("?:") {
            Node::Group(Box::new(self.alt()), None)
        } else if self.eat("?P<") {
            let mut name = String::new();
            while self.peek() != Some('>') {
                name.push(self.next());
            }
            self.i += 1;
            self.groups += 1;
            let idx = self.groups;
            self.names.push((name, idx));
            Node::Group(Box::new(self.alt()), Some(idx))
        } else if self.eat("?=") {
            Node::Look(Box::new(self.alt()), false, false)
        } else if self.eat("?!") {
            Node::Look(Box::new(self.alt()), false, true)
        } else if self.eat("?<=") {
            Node::Look(Box::new(self.alt()), true, false)
        } else if self.eat("?<!") {
            Node::Look(Box::new(self.alt()), true, true)
        } else {
            assert!(self.peek() != Some('?'), "unsupported group syntax");
            self.groups += 1;
            let idx = self.groups;
            Node::Group(Box::new(self.alt()), Some(idx))
        };
        assert_eq!(self.peek(), Some(')'), "unbalanced parenthesis");
        self.i += 1;
        node
    }

    fn hex(&mut self, digits: usize) -> char {
        let mut s = String::new();
        if digits == 0 {
            // \x{...}
            self.i += 1;
            while self.peek() != Some('}') {
                s.push(self.next());
            }
            self.i += 1;
        } else {
            for _ in 0..digits {
                s.push(self.next());
            }
        }
        char::from_u32(u32::from_str_radix(&s, 16).expect("hex escape")).expect("code point")
    }

    /// An escape that stands for one character (or a class) — shared by sets.
    fn char_escape(&mut self, c: char) -> Result<char, SetItem> {
        Ok(match c {
            'd' => return Err(SetItem::Class(ClassKind::Digit, false)),
            'D' => return Err(SetItem::Class(ClassKind::Digit, true)),
            'w' => return Err(SetItem::Class(ClassKind::Word, false)),
            'W' => return Err(SetItem::Class(ClassKind::Word, true)),
            's' => return Err(SetItem::Class(ClassKind::Space, false)),
            'S' => return Err(SetItem::Class(ClassKind::Space, true)),
            'n' => '\n',
            't' => '\t',
            'r' => '\r',
            'f' => '\u{0c}',
            'v' => '\u{0b}',
            'a' => '\u{07}',
            'u' => self.hex(4),
            'U' => self.hex(8),
            'x' if self.peek() == Some('{') => self.hex(0),
            'x' => self.hex(2),
            other => {
                assert!(
                    !other.is_ascii_alphanumeric(),
                    "unsupported escape \\{other}"
                );
                other
            }
        })
    }

    fn escape(&mut self) -> Node {
        let c = self.next();
        match c {
            'b' => Node::WordB,
            'B' => Node::NotWordB,
            'A' => Node::BeginText,
            'Z' => Node::EndText,
            '1'..='9' => Node::Backref(c as usize - '0' as usize),
            _ => match self.char_escape(c) {
                Ok(ch) => Node::Lit(ch),
                Err(item) => Node::Set(CharSet {
                    negated: false,
                    items: vec![item],
                }),
            },
        }
    }

    fn set_char(&mut self) -> Result<char, SetItem> {
        let c = self.next();
        if c == '\\' {
            let e = self.next();
            self.char_escape(e)
        } else {
            Ok(c)
        }
    }

    fn set(&mut self) -> CharSet {
        let negated = self.peek() == Some('^');
        if negated {
            self.i += 1;
        }
        let mut items = Vec::new();
        let mut first = true;
        loop {
            let c = self.peek().expect("unterminated set");
            if c == ']' && !first {
                self.i += 1;
                break;
            }
            first = false;
            match self.set_char() {
                Err(item) => items.push(item),
                Ok(lo) => {
                    if self.peek() == Some('-') && self.p.get(self.i + 1).is_some_and(|&n| n != ']')
                    {
                        self.i += 1;
                        match self.set_char() {
                            Ok(hi) => items.push(SetItem::Range(lo, hi)),
                            Err(item) => {
                                items.push(SetItem::Range(lo, lo));
                                items.push(SetItem::Range('-', '-'));
                                items.push(item);
                            }
                        }
                    } else {
                        items.push(SetItem::Range(lo, lo));
                    }
                }
            }
        }
        CharSet { negated, items }
    }
}

// ---------------------------------------------------------------------------
// Compiler
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
enum One {
    Char(char),
    Any,
    Set(usize),
}

#[derive(Clone, Debug)]
enum Inst {
    One(One),
    Split(usize, usize),
    Jmp(usize),
    Save(usize),
    Bol,
    Eol,
    BeginText,
    EndText,
    WordB,
    NotWordB,
    Look {
        prog: usize,
        behind: bool,
        negate: bool,
        width: usize,
    },
    Backref(usize),
    Rep {
        one: One,
        min: usize,
        max: usize,
    },
    Match,
}

/// A compiled pattern.
pub(crate) struct Regex {
    progs: Vec<Vec<Inst>>,
    sets: Vec<CharSet>,
    icase: bool,
    groups: usize,
}

fn width(node: &Node) -> Option<usize> {
    match node {
        Node::Lit(_) | Node::Any | Node::Set(_) => Some(1),
        Node::Group(inner, _) => width(inner),
        Node::Cat(items) => items.iter().map(width).sum(),
        Node::Alt(alts) => {
            let w = width(&alts[0])?;
            alts.iter().all(|a| width(a) == Some(w)).then_some(w)
        }
        Node::Rep(inner, min, max) => match (width(inner), max) {
            (Some(w), Some(max)) if max == min => Some(w * min),
            _ => None,
        },
        Node::Backref(_) => None,
        _ => Some(0),
    }
}

fn nullable(node: &Node) -> bool {
    match node {
        Node::Lit(_) | Node::Any | Node::Set(_) | Node::Backref(_) => false,
        Node::Group(inner, _) => nullable(inner),
        Node::Cat(items) => items.iter().all(nullable),
        Node::Alt(alts) => alts.iter().any(nullable),
        Node::Rep(inner, min, _) => *min == 0 || nullable(inner),
        _ => true,
    }
}

struct Compiler {
    progs: Vec<Vec<Inst>>,
    sets: Vec<CharSet>,
    icase: bool,
}

impl Compiler {
    fn one(&mut self, node: &Node) -> Option<One> {
        match node {
            Node::Lit(c) => Some(One::Char(if self.icase { fold(*c) } else { *c })),
            Node::Any => Some(One::Any),
            Node::Set(set) => {
                self.sets.push(set.clone());
                Some(One::Set(self.sets.len() - 1))
            }
            _ => None,
        }
    }

    fn emit(&mut self, node: &Node, out: &mut Vec<Inst>) {
        match node {
            Node::Lit(_) | Node::Any | Node::Set(_) => {
                let one = self.one(node).unwrap();
                out.push(Inst::One(one));
            }
            Node::Group(inner, idx) => {
                if let Some(i) = idx {
                    out.push(Inst::Save(2 * i));
                }
                self.emit(inner, out);
                if let Some(i) = idx {
                    out.push(Inst::Save(2 * i + 1));
                }
            }
            Node::Cat(items) => {
                for item in items {
                    self.emit(item, out);
                }
            }
            Node::Alt(alts) => {
                let mut jumps = Vec::new();
                for (k, alt) in alts.iter().enumerate() {
                    if k + 1 < alts.len() {
                        let split = out.len();
                        out.push(Inst::Split(split + 1, 0));
                        self.emit(alt, out);
                        jumps.push(out.len());
                        out.push(Inst::Jmp(0));
                        let next = out.len();
                        out[split] = Inst::Split(split + 1, next);
                    } else {
                        self.emit(alt, out);
                    }
                }
                let end = out.len();
                for j in jumps {
                    out[j] = Inst::Jmp(end);
                }
            }
            Node::Rep(inner, min, max) => {
                if let Some(one) = self.one(inner) {
                    out.push(Inst::Rep {
                        one,
                        min: *min,
                        max: max.unwrap_or(usize::MAX),
                    });
                    return;
                }
                for _ in 0..*min {
                    self.emit(inner, out);
                }
                match max {
                    None => {
                        assert!(!nullable(inner), "unbounded repeat of a nullable pattern");
                        let top = out.len();
                        out.push(Inst::Split(top + 1, 0));
                        self.emit(inner, out);
                        out.push(Inst::Jmp(top));
                        let end = out.len();
                        out[top] = Inst::Split(top + 1, end);
                    }
                    Some(max) => {
                        let mut splits = Vec::new();
                        for _ in *min..*max {
                            splits.push(out.len());
                            out.push(Inst::Split(0, 0));
                            self.emit(inner, out);
                        }
                        let end = out.len();
                        for s in splits {
                            out[s] = Inst::Split(s + 1, end);
                        }
                    }
                }
            }
            Node::Bol => out.push(Inst::Bol),
            Node::Eol => out.push(Inst::Eol),
            Node::BeginText => out.push(Inst::BeginText),
            Node::EndText => out.push(Inst::EndText),
            Node::WordB => out.push(Inst::WordB),
            Node::NotWordB => out.push(Inst::NotWordB),
            Node::Backref(g) => out.push(Inst::Backref(*g)),
            Node::Look(inner, behind, negate) => {
                let w = if *behind {
                    width(inner).expect("look-behind requires fixed-width pattern")
                } else {
                    0
                };
                let mut sub = Vec::new();
                self.emit(inner, &mut sub);
                sub.push(Inst::Match);
                self.progs.push(sub);
                out.push(Inst::Look {
                    prog: self.progs.len() - 1,
                    behind: *behind,
                    negate: *negate,
                    width: w,
                });
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Matching
// ---------------------------------------------------------------------------

/// A successful match: group spans in `char` indices (group 0 is the match).
#[derive(Clone, Debug)]
pub(crate) struct Match {
    spans: Vec<Option<(usize, usize)>>,
}

impl Match {
    pub(crate) fn start(&self) -> usize {
        self.spans[0].unwrap().0
    }

    pub(crate) fn end(&self) -> usize {
        self.spans[0].unwrap().1
    }

    pub(crate) fn span(&self, g: usize) -> Option<(usize, usize)> {
        self.spans.get(g).copied().flatten()
    }

    /// `m.group(g)`; `None` when the group did not take part.
    pub(crate) fn get(&self, text: &[char], g: usize) -> Option<String> {
        self.span(g).map(|(a, b)| text[a..b].iter().collect())
    }

    /// `m.group(g) or ""`.
    pub(crate) fn str(&self, text: &[char], g: usize) -> String {
        self.get(text, g).unwrap_or_default()
    }

    /// `m.start(g)` (Python gives -1 for an unmatched group).
    pub(crate) fn start_of(&self, g: usize) -> Option<usize> {
        self.span(g).map(|s| s.0)
    }
}

enum Frame {
    Alt(usize, usize),
    Restore(usize, Option<usize>),
    Rep {
        pc: usize,
        base: usize,
        k: usize,
        min: usize,
    },
}

impl Regex {
    /// Compiles `pattern` (Python syntax); panics on a pattern this matcher
    /// does not support — every pattern is fixed in the source, and the tests
    /// compile them all.
    pub(crate) fn new(pattern: &str) -> Regex {
        Self::build(pattern, false)
    }

    /// `re.compile(pattern, re.IGNORECASE)`.
    pub(crate) fn new_i(pattern: &str) -> Regex {
        Self::build(pattern, true)
    }

    fn build(pattern: &str, icase: bool) -> Regex {
        let mut parser = Parser {
            p: pattern.chars().collect(),
            i: 0,
            groups: 0,
            names: Vec::new(),
        };
        let node = parser.alt();
        assert!(
            parser.i == parser.p.len(),
            "unbalanced parenthesis in {pattern:?}"
        );
        let mut compiler = Compiler {
            progs: vec![Vec::new()],
            sets: Vec::new(),
            icase,
        };
        let mut main = Vec::new();
        compiler.emit(&node, &mut main);
        main.push(Inst::Match);
        compiler.progs[0] = main;
        Regex {
            progs: compiler.progs,
            sets: compiler.sets,
            icase,
            groups: parser.groups,
        }
    }

    fn one(&self, one: &One, c: char) -> bool {
        match *one {
            One::Char(lit) => {
                if self.icase {
                    fold(c) == lit
                } else {
                    c == lit
                }
            }
            One::Any => c != '\n',
            One::Set(i) => self.sets[i].test(c, self.icase),
        }
    }

    fn word_at(text: &[char], i: usize) -> bool {
        text.get(i).is_some_and(|&c| is_word(c))
    }

    /// Runs program `pi` anchored at `start`; returns the end. `reject_empty`
    /// makes an empty match at `start` fail (Python's `must_advance`),
    /// `require_end` one that stops short of the end (`fullmatch`).
    fn exec(
        &self,
        pi: usize,
        text: &[char],
        start: usize,
        slots: &mut [Option<usize>],
        reject_empty: bool,
        require_end: bool,
    ) -> Option<usize> {
        let prog = &self.progs[pi];
        let mut stack: Vec<Frame> = Vec::new();
        let mut pc = 0usize;
        let mut pos = start;
        let n = text.len();
        let mut steps = 0usize;
        loop {
            steps += 1;
            if steps > STEP_BUDGET {
                EXHAUSTED.with(|flag| flag.set(true));
                return None;
            }
            let ok = match &prog[pc] {
                Inst::Match => {
                    if (reject_empty && pos == start) || (require_end && pos != n) {
                        false
                    } else {
                        return Some(pos);
                    }
                }
                Inst::One(one) => {
                    if pos < n && self.one(one, text[pos]) {
                        pos += 1;
                        pc += 1;
                        continue;
                    }
                    false
                }
                Inst::Split(a, b) => {
                    stack.push(Frame::Alt(*b, pos));
                    pc = *a;
                    continue;
                }
                Inst::Jmp(a) => {
                    pc = *a;
                    continue;
                }
                Inst::Save(s) => {
                    stack.push(Frame::Restore(*s, slots[*s]));
                    slots[*s] = Some(pos);
                    pc += 1;
                    continue;
                }
                Inst::Bol => pos == 0,
                Inst::Eol => pos == n || (pos + 1 == n && text[pos] == '\n'),
                Inst::BeginText => pos == 0,
                Inst::EndText => pos == n,
                Inst::WordB | Inst::NotWordB => {
                    let before = pos > 0 && Self::word_at(text, pos - 1);
                    let after = Self::word_at(text, pos);
                    (before != after) == matches!(prog[pc], Inst::WordB)
                }
                Inst::Look {
                    prog: sub,
                    behind,
                    negate,
                    width,
                } => {
                    let at = if *behind {
                        pos.checked_sub(*width)
                    } else {
                        Some(pos)
                    };
                    let found = at.is_some_and(|at| {
                        let mut scratch = vec![None; slots.len()];
                        self.exec(*sub, text, at, &mut scratch, false, false)
                            .is_some()
                    });
                    found != *negate
                }
                Inst::Backref(g) => match (
                    slots.get(2 * g).copied().flatten(),
                    slots.get(2 * g + 1).copied().flatten(),
                ) {
                    (Some(a), Some(b)) => {
                        let len = b - a;
                        let same = pos + len <= n
                            && (0..len).all(|k| {
                                let (x, y) = (text[a + k], text[pos + k]);
                                if self.icase {
                                    fold(x) == fold(y)
                                } else {
                                    x == y
                                }
                            });
                        if same {
                            pos += len;
                            pc += 1;
                            continue;
                        }
                        false
                    }
                    _ => false,
                },
                Inst::Rep { one, min, max } => {
                    let mut k = 0usize;
                    while k < *max && pos + k < n && self.one(one, text[pos + k]) {
                        k += 1;
                    }
                    if k >= *min {
                        if k > *min {
                            stack.push(Frame::Rep {
                                pc: pc + 1,
                                base: pos,
                                k,
                                min: *min,
                            });
                        }
                        pos += k;
                        pc += 1;
                        continue;
                    }
                    false
                }
            };
            if ok {
                pc += 1;
                continue;
            }
            // backtrack
            loop {
                match stack.pop() {
                    None => return None,
                    Some(Frame::Restore(s, old)) => slots[s] = old,
                    Some(Frame::Alt(p, q)) => {
                        pc = p;
                        pos = q;
                        break;
                    }
                    Some(Frame::Rep {
                        pc: p,
                        base,
                        k,
                        min,
                    }) => {
                        let k = k - 1;
                        if k > min {
                            stack.push(Frame::Rep {
                                pc: p,
                                base,
                                k,
                                min,
                            });
                        }
                        pc = p;
                        pos = base + k;
                        break;
                    }
                }
            }
        }
    }

    fn attempt(&self, text: &[char], at: usize, reject_empty: bool) -> Option<Match> {
        self.attempt_opts(text, at, reject_empty, false)
    }

    fn attempt_opts(
        &self,
        text: &[char],
        at: usize,
        reject_empty: bool,
        require_end: bool,
    ) -> Option<Match> {
        let mut slots = vec![None; 2 * (self.groups + 1)];
        let end = self.exec(0, text, at, &mut slots, reject_empty, require_end)?;
        let mut spans = vec![Some((at, end))];
        for g in 1..=self.groups {
            spans.push(match (slots[2 * g], slots[2 * g + 1]) {
                (Some(a), Some(b)) => Some((a, b)),
                _ => None,
            });
        }
        Some(Match { spans })
    }

    fn search_inner(&self, text: &[char], from: usize, must_advance: bool) -> Option<Match> {
        (from..=text.len()).find_map(|at| self.attempt(text, at, must_advance && at == from))
    }

    /// `pattern.search(text, pos)`.
    pub(crate) fn search_from(&self, text: &[char], from: usize) -> Option<Match> {
        if from > text.len() {
            return None;
        }
        self.search_inner(text, from, false)
    }

    /// `pattern.search(text)`.
    pub(crate) fn search(&self, text: &[char]) -> Option<Match> {
        self.search_from(text, 0)
    }

    /// `pattern.match(text, pos)`: anchored at `pos`.
    pub(crate) fn match_at(&self, text: &[char], pos: usize) -> Option<Match> {
        if pos > text.len() {
            return None;
        }
        self.attempt(text, pos, false)
    }

    /// `pattern.fullmatch(text)`.
    pub(crate) fn fullmatch(&self, text: &[char]) -> Option<Match> {
        self.attempt_opts(text, 0, false, true)
    }

    /// `pattern.finditer(text)` (Python 3.7+ empty-match rules).
    pub(crate) fn find_iter(&self, text: &[char]) -> Vec<Match> {
        let mut out = Vec::new();
        let mut pos = 0;
        let mut must_advance = false;
        while pos <= text.len() {
            let Some(m) = self.search_inner(text, pos, must_advance) else {
                break;
            };
            must_advance = m.start() == m.end();
            pos = m.end();
            out.push(m);
        }
        out
    }

    /// `pattern.sub(repl, text)` with a callable `repl`.
    pub(crate) fn sub_with(
        &self,
        text: &[char],
        mut repl: impl FnMut(&Match) -> String,
    ) -> Vec<char> {
        let mut out = Vec::with_capacity(text.len());
        let mut last = 0;
        for m in self.find_iter(text) {
            out.extend_from_slice(&text[last..m.start()]);
            out.extend(repl(&m).chars());
            last = m.end();
        }
        out.extend_from_slice(&text[last..]);
        out
    }

    /// `pattern.sub(literal, text)` (a replacement without group references).
    pub(crate) fn sub(&self, text: &[char], literal: &str) -> Vec<char> {
        self.sub_with(text, |_| literal.to_string())
    }

    /// `pattern.split(text)`: captured groups are kept (`None` as "").
    pub(crate) fn split(&self, text: &[char]) -> Vec<String> {
        let mut out = Vec::new();
        let mut last = 0;
        for m in self.find_iter(text) {
            out.push(text[last..m.start()].iter().collect());
            for g in 1..=self.groups {
                out.push(m.str(text, g));
            }
            last = m.end();
        }
        out.push(text[last..].iter().collect());
        out
    }

    /// `pattern.findall(text)` for a pattern without groups.
    pub(crate) fn findall(&self, text: &[char]) -> Vec<String> {
        debug_assert_eq!(self.groups, 0);
        self.find_iter(text)
            .iter()
            .map(|m| m.str(text, 0))
            .collect()
    }
}

/// `re.escape` (Python 3.7+: only the special characters).
pub(crate) fn escape(word: &str) -> String {
    let mut out = String::with_capacity(word.len());
    for c in word.chars() {
        if "()[]{}?*+-|^$\\.&~# \t\n\r\u{0b}\u{0c}".contains(c) {
            out.push('\\');
        }
        out.push(c);
    }
    out
}

/// One compiled pattern per call site.
macro_rules! re {
    ($pat:expr) => {{
        static RE: ::std::sync::OnceLock<$crate::models::architectures::maya2::text::pyre::Regex> =
            ::std::sync::OnceLock::new();
        RE.get_or_init(|| $crate::models::architectures::maya2::text::pyre::Regex::new($pat))
    }};
}

/// `re!` with `re.IGNORECASE`.
macro_rules! re_i {
    ($pat:expr) => {{
        static RE: ::std::sync::OnceLock<$crate::models::architectures::maya2::text::pyre::Regex> =
            ::std::sync::OnceLock::new();
        RE.get_or_init(|| $crate::models::architectures::maya2::text::pyre::Regex::new_i($pat))
    }};
}

pub(crate) use {re, re_i};

#[cfg(test)]
mod tests {
    use super::*;

    fn c(s: &str) -> Vec<char> {
        s.chars().collect()
    }

    fn sub(p: &str, t: &str, r: &str) -> String {
        Regex::new(p).sub(&c(t), r).into_iter().collect()
    }

    #[test]
    fn exponential_backtracking_stops_at_the_step_budget() {
        take_exhausted();
        // (a+)+b over a run of a's backtracks exponentially without the budget.
        let pattern = Regex::new("(a+)+b");
        let started = std::time::Instant::now();
        assert!(pattern.search(&c(&"a".repeat(40))).is_none());
        assert!(take_exhausted());
        assert!(!take_exhausted(), "the flag resets once read");
        assert!(started.elapsed().as_secs() < 60);
        assert!(pattern.search(&c("aaab")).is_some());
        assert!(!take_exhausted());
    }

    #[test]
    fn python_word_class_excludes_combining_marks() {
        // "का": क (Lo) then ा (Mc) — Python's \b sits between them.
        let t = c("का");
        assert_eq!(Regex::new(r"\b").find_iter(&t).len(), 2);
        assert!(Regex::new(r"^\w\W$").search(&t).is_some());
        assert!(is_word('_') && is_word('२') && !is_word('\u{093e}'));
        assert_eq!(decimal_value('७'), Some(7));
        assert_eq!(decimal_value('\u{1d7d8}'), Some(0)); // math double-struck zero
        assert_eq!(decimal_value('²'), None);
        assert!(is_digit('²') && !is_decimal('²'));
    }

    #[test]
    fn lookaround_backrefs_and_alternation_order() {
        assert_eq!(
            sub(r"(?<![\d.])\d+(?!\.?\d)", "a 12 3.4 56.", "N"),
            "a N 3.4 N."
        );
        assert_eq!(sub(r"(?<!\d,)\d{3}", "1,234 567", "N"), "1,234 N");
        let t = c("12/05/2024 and 12/05-2024");
        let found = Regex::new(r"(\d{1,2})([/.\-])(\d{1,2})\2(\d{2,4})").find_iter(&t);
        assert_eq!(found.len(), 1);
        assert_eq!(found[0].str(&t, 4), "2024");
        assert_eq!(sub("ab|abc", "abc", "X"), "Xc");
        assert_eq!(sub(r"\d+(?=%)", "5% 6", "N"), "N% 6");
    }

    #[test]
    fn ignorecase_quantifier_braces_and_eol() {
        assert!(Regex::new_i("crores?").fullmatch(&c("Crores")).is_some());
        assert!(Regex::new("a{2}").fullmatch(&c("aa")).is_some());
        assert!(Regex::new("[{}]").fullmatch(&c("{")).is_some());
        assert!(Regex::new("x{h}").fullmatch(&c("x{h}")).is_some());
        assert_eq!(sub(r"[,\s]+$", "a, \n", "."), "a.");
        assert!(Regex::new("ab$").search(&c("ab\n")).is_some());
        assert!(Regex::new(r"ab\Z").search(&c("ab\n")).is_none());
    }

    #[test]
    fn empty_matches_follow_python_3_7() {
        assert_eq!(sub("x*", "abxd", "-"), "-a-b--d-");
        assert_eq!(sub(r"(?<=[A-Za-z])(?=[ऀ-෿])", "fiveన", " "), "five న");
        let parts = Regex::new(r"(\[[a-z ]+\])").split(&c("a [laughs] b"));
        assert_eq!(parts, vec!["a ", "[laughs]", " b"]);
    }
}
