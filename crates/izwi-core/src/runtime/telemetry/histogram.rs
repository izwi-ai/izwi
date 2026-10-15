//! Prometheus histograms and labelled counters for the text exposition.
//!
//! A histogram keeps one atomic counter per bucket, so observing is lock-free
//! and scrapes never sort samples. The exposition renders cumulative
//! `_bucket{le=...}` series, `_sum` and `_count`, which aggregate across
//! replicas unlike precomputed quantiles. Families key series by label values
//! and cap how many they keep: once full, new label sets share one `other`
//! series, so a stream of unexpected values cannot grow a scrape without
//! bound. Label values must never carry request or tenant identifiers.

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

/// Bucket upper bounds, in seconds unless the metric says otherwise.
pub mod buckets {
    /// Whole-request latency, from tens of milliseconds to minutes.
    pub const REQUEST_SECONDS: &[f64] = &[
        0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 20.0, 30.0, 60.0, 120.0, 300.0,
    ];
    /// Time to first token or first audio.
    pub const FIRST_OUTPUT_SECONDS: &[f64] = &[
        0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 10.0, 20.0,
    ];
    /// Gap between successive tokens of one request.
    pub const INTER_TOKEN_SECONDS: &[f64] = &[
        0.001, 0.0025, 0.005, 0.01, 0.015, 0.02, 0.03, 0.04, 0.05, 0.075, 0.1, 0.25, 0.5, 1.0,
    ];
    /// Waiting before execution starts.
    pub const QUEUE_SECONDS: &[f64] = &[
        0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0,
    ];
    /// Host work of one engine step phase.
    pub const STEP_SECONDS: &[f64] = &[
        0.0001, 0.00025, 0.0005, 0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25,
    ];
    /// Rows in one physical batch.
    pub const BATCH_ROWS: &[f64] = &[1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0];
    /// Tokens (or audio frames) in one physical batch.
    pub const BATCH_TOKENS: &[f64] = &[
        1.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0, 512.0, 1024.0, 2048.0, 4096.0, 8192.0,
    ];
    /// Conversational voice latencies.
    pub const VOICE_SECONDS: &[f64] = &[
        0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0,
    ];
}

/// Series a family keeps before new label sets share the `other` series.
pub const DEFAULT_MAX_SERIES: usize = 64;
const OVERFLOW_LABEL_VALUE: &str = "other";

/// One histogram series: cumulative counts are computed at render time from
/// per-bucket counts, so a scrape always sees monotone buckets and a `_count`
/// equal to the `+Inf` bucket.
#[derive(Debug)]
pub struct Histogram {
    bounds: &'static [f64],
    /// One counter per bound plus the `+Inf` overflow bucket.
    counts: Box<[AtomicU64]>,
    sum_bits: AtomicU64,
}

impl Histogram {
    pub fn new(bounds: &'static [f64]) -> Self {
        debug_assert!(bounds.windows(2).all(|pair| pair[0] < pair[1]));
        Self {
            bounds,
            counts: (0..=bounds.len()).map(|_| AtomicU64::new(0)).collect(),
            sum_bits: AtomicU64::new(0f64.to_bits()),
        }
    }

    /// Record one value. Negative values count as zero; NaN is ignored.
    pub fn observe(&self, value: f64) {
        if value.is_nan() {
            return;
        }
        let value = value.max(0.0);
        let index = self.bounds.partition_point(|bound| *bound < value);
        self.counts[index].fetch_add(1, Ordering::Relaxed);
        let mut current = self.sum_bits.load(Ordering::Relaxed);
        loop {
            let next = (f64::from_bits(current) + value).to_bits();
            match self.sum_bits.compare_exchange_weak(
                current,
                next,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => return,
                Err(observed) => current = observed,
            }
        }
    }

    pub fn observe_duration(&self, duration: Duration) {
        self.observe(duration.as_secs_f64());
    }

    pub fn snapshot(&self) -> HistogramSnapshot {
        let mut cumulative = 0u64;
        let buckets = self
            .counts
            .iter()
            .map(|count| {
                cumulative = cumulative.saturating_add(count.load(Ordering::Relaxed));
                cumulative
            })
            .collect::<Vec<_>>();
        HistogramSnapshot {
            bounds: self.bounds,
            count: cumulative,
            cumulative_buckets: buckets,
            sum: f64::from_bits(self.sum_bits.load(Ordering::Relaxed)),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct HistogramSnapshot {
    pub bounds: &'static [f64],
    /// Cumulative counts per bound, the last entry being `+Inf`.
    pub cumulative_buckets: Vec<u64>,
    pub count: u64,
    pub sum: f64,
}

/// A histogram metric with zero or more labels.
#[derive(Debug)]
pub struct HistogramFamily {
    name: &'static str,
    help: &'static str,
    labels: &'static [&'static str],
    bounds: &'static [f64],
    max_series: usize,
    series: Mutex<BTreeMap<Vec<String>, Arc<Histogram>>>,
}

impl HistogramFamily {
    pub const fn new(
        name: &'static str,
        help: &'static str,
        labels: &'static [&'static str],
        bounds: &'static [f64],
    ) -> Self {
        Self {
            name,
            help,
            labels,
            bounds,
            max_series: DEFAULT_MAX_SERIES,
            series: Mutex::new(BTreeMap::new()),
        }
    }

    pub const fn with_max_series(mut self, max_series: usize) -> Self {
        self.max_series = max_series;
        self
    }

    pub fn name(&self) -> &'static str {
        self.name
    }

    /// The series for `values`, one per declared label, created on first use.
    pub fn series(&self, values: &[&str]) -> Arc<Histogram> {
        debug_assert_eq!(values.len(), self.labels.len(), "{}", self.name);
        let mut series = self
            .series
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        let key = series_key(&series, values, self.max_series);
        series
            .entry(key)
            .or_insert_with(|| Arc::new(Histogram::new(self.bounds)))
            .clone()
    }

    pub fn observe(&self, values: &[&str], value: f64) {
        self.series(values).observe(value);
    }

    pub fn observe_duration(&self, values: &[&str], duration: Duration) {
        self.series(values).observe_duration(duration);
    }

    /// Append this family in the Prometheus text format. A family without
    /// labels always renders its one series, even before any observation.
    pub fn render(&self, out: &mut String) {
        let mut series = self
            .series
            .lock()
            .unwrap_or_else(|poison| poison.into_inner())
            .iter()
            .map(|(key, histogram)| (key.clone(), histogram.snapshot()))
            .collect::<Vec<_>>();
        if series.is_empty() && self.labels.is_empty() {
            series.push((Vec::new(), Histogram::new(self.bounds).snapshot()));
        }
        let _ = writeln!(out, "# HELP {} {}", self.name, self.help);
        let _ = writeln!(out, "# TYPE {} histogram", self.name);
        for (values, snapshot) in series {
            let labels = render_labels(self.labels, &values);
            let separator = if labels.is_empty() { "" } else { "," };
            for (index, cumulative) in snapshot.cumulative_buckets.iter().enumerate() {
                let le = snapshot
                    .bounds
                    .get(index)
                    .map(|bound| format_float(*bound))
                    .unwrap_or_else(|| "+Inf".to_string());
                let _ = writeln!(
                    out,
                    "{}_bucket{{{labels}{separator}le=\"{le}\"}} {cumulative}",
                    self.name
                );
            }
            let braces = braced(&labels);
            let _ = writeln!(
                out,
                "{}_sum{braces} {}",
                self.name,
                format_float(snapshot.sum)
            );
            let _ = writeln!(out, "{}_count{braces} {}", self.name, snapshot.count);
        }
    }
}

/// A monotonic counter metric with zero or more labels.
#[derive(Debug)]
pub struct CounterFamily {
    name: &'static str,
    help: &'static str,
    labels: &'static [&'static str],
    max_series: usize,
    series: Mutex<BTreeMap<Vec<String>, u64>>,
}

impl CounterFamily {
    pub const fn new(
        name: &'static str,
        help: &'static str,
        labels: &'static [&'static str],
    ) -> Self {
        Self {
            name,
            help,
            labels,
            max_series: DEFAULT_MAX_SERIES,
            series: Mutex::new(BTreeMap::new()),
        }
    }

    pub const fn with_max_series(mut self, max_series: usize) -> Self {
        self.max_series = max_series;
        self
    }

    pub fn add(&self, values: &[&str], amount: u64) {
        debug_assert_eq!(values.len(), self.labels.len(), "{}", self.name);
        let mut series = self
            .series
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        let key = series_key(&series, values, self.max_series);
        let total = series.entry(key).or_insert(0);
        *total = total.saturating_add(amount);
    }

    pub fn get(&self, values: &[&str]) -> u64 {
        let key = values
            .iter()
            .map(|value| value.to_string())
            .collect::<Vec<_>>();
        self.series
            .lock()
            .unwrap_or_else(|poison| poison.into_inner())
            .get(&key)
            .copied()
            .unwrap_or(0)
    }

    pub fn render(&self, out: &mut String) {
        let series = self
            .series
            .lock()
            .unwrap_or_else(|poison| poison.into_inner())
            .clone();
        let _ = writeln!(out, "# HELP {} {}", self.name, self.help);
        let _ = writeln!(out, "# TYPE {} counter", self.name);
        if series.is_empty() && self.labels.is_empty() {
            let _ = writeln!(out, "{} 0", self.name);
        }
        for (values, total) in series {
            let labels = render_labels(self.labels, &values);
            let _ = writeln!(out, "{}{} {total}", self.name, braced(&labels));
        }
    }
}

fn series_key<V>(series: &BTreeMap<Vec<String>, V>, values: &[&str], max: usize) -> Vec<String> {
    let key = values
        .iter()
        .map(|value| value.to_string())
        .collect::<Vec<_>>();
    if series.len() < max || series.contains_key(&key) {
        key
    } else {
        vec![OVERFLOW_LABEL_VALUE.to_string(); values.len()]
    }
}

fn render_labels(names: &[&str], values: &[String]) -> String {
    names
        .iter()
        .zip(values)
        .map(|(name, value)| format!("{name}=\"{}\"", escape_label_value(value)))
        .collect::<Vec<_>>()
        .join(",")
}

fn braced(labels: &str) -> String {
    if labels.is_empty() {
        String::new()
    } else {
        format!("{{{labels}}}")
    }
}

pub fn escape_label_value(value: &str) -> String {
    value
        .replace('\\', r"\\")
        .replace('"', r#"\""#)
        .replace('\n', r"\n")
}

fn format_float(value: f64) -> String {
    if value.is_finite() && value.fract() == 0.0 && value.abs() < 1e15 {
        format!("{value:.1}")
    } else {
        format!("{value}")
    }
}

/// One sample line of a Prometheus text exposition.
#[derive(Debug, Clone, PartialEq)]
pub struct ExpositionSample {
    pub name: String,
    pub labels: BTreeMap<String, String>,
    pub value: f64,
}

/// Parse the sample lines of a Prometheus text exposition. Comment lines are
/// checked only for a well-formed `# TYPE`; the result is for tests and
/// diagnostics, not a general client.
pub fn parse_exposition(text: &str) -> Result<Vec<ExpositionSample>, String> {
    let mut samples = Vec::new();
    for (number, line) in text.lines().enumerate() {
        let line = line.trim_end();
        if line.is_empty() {
            continue;
        }
        if let Some(comment) = line.strip_prefix('#') {
            let mut parts = comment.split_whitespace();
            if parts.next() == Some("TYPE") {
                let (Some(_), Some(kind)) = (parts.next(), parts.next()) else {
                    return Err(format!("line {}: malformed TYPE", number + 1));
                };
                if !matches!(
                    kind,
                    "counter" | "gauge" | "histogram" | "summary" | "untyped"
                ) {
                    return Err(format!("line {}: unknown type {kind}", number + 1));
                }
            }
            continue;
        }
        samples.push(parse_sample(line).map_err(|error| format!("line {}: {error}", number + 1))?);
    }
    Ok(samples)
}

fn parse_sample(line: &str) -> Result<ExpositionSample, String> {
    let (series, value) = line.rsplit_once(' ').ok_or("missing value")?;
    let value = match value {
        "+Inf" => f64::INFINITY,
        "-Inf" => f64::NEG_INFINITY,
        other => other
            .parse::<f64>()
            .map_err(|_| format!("bad value {other:?}"))?,
    };
    let (name, labels) = match series.split_once('{') {
        None => (series, BTreeMap::new()),
        Some((name, rest)) => {
            let body = rest.strip_suffix('}').ok_or("unterminated labels")?;
            (name, parse_labels(body)?)
        }
    };
    if name.is_empty()
        || !name
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == ':')
    {
        return Err(format!("bad metric name {name:?}"));
    }
    Ok(ExpositionSample {
        name: name.to_string(),
        labels,
        value,
    })
}

fn parse_labels(body: &str) -> Result<BTreeMap<String, String>, String> {
    let mut labels = BTreeMap::new();
    let mut chars = body.chars().peekable();
    while chars.peek().is_some() {
        let name = chars.by_ref().take_while(|c| *c != '=').collect::<String>();
        if chars.next() != Some('"') {
            return Err(format!("label {name:?} is not quoted"));
        }
        let mut value = String::new();
        loop {
            match chars.next().ok_or("unterminated label value")? {
                '\\' => match chars.next().ok_or("dangling escape")? {
                    'n' => value.push('\n'),
                    other => value.push(other),
                },
                '"' => break,
                other => value.push(other),
            }
        }
        if labels.insert(name.trim().to_string(), value).is_some() {
            return Err("duplicate label".to_string());
        }
        match chars.next() {
            None | Some(',') => {}
            Some(other) => return Err(format!("unexpected {other:?} after label")),
        }
    }
    Ok(labels)
}

/// Check every histogram in an exposition: buckets are cumulative and ordered
/// by `le`, the last bucket is `+Inf`, and `_count` equals it alongside a
/// `_sum`. Returns each series' `_count`, keyed by metric name and its
/// non-`le` labels rendered as `name{a="x",b="y"}`.
pub fn check_histograms(text: &str) -> Result<BTreeMap<String, u64>, String> {
    let samples = parse_exposition(text)?;
    let histograms = text
        .lines()
        .filter_map(|line| line.strip_prefix("# TYPE "))
        .filter_map(|rest| rest.strip_suffix(" histogram"))
        .collect::<Vec<_>>();
    let mut counts = BTreeMap::new();
    for name in histograms {
        let mut buckets = BTreeMap::<String, Vec<(f64, f64)>>::new();
        let mut sums = BTreeMap::new();
        let mut totals = BTreeMap::new();
        for sample in &samples {
            let mut labels = sample.labels.clone();
            let le = labels.remove("le");
            let key = series_label(name, &labels);
            if sample.name == format!("{name}_bucket") {
                let le = le.ok_or_else(|| format!("{key}: bucket without le"))?;
                let bound = if le == "+Inf" {
                    f64::INFINITY
                } else {
                    le.parse::<f64>()
                        .map_err(|_| format!("{key}: bad le {le:?}"))?
                };
                buckets.entry(key).or_default().push((bound, sample.value));
            } else if sample.name == format!("{name}_sum") {
                sums.insert(key, sample.value);
            } else if sample.name == format!("{name}_count") {
                totals.insert(key, sample.value);
            }
        }
        for (key, series) in buckets {
            if !series.windows(2).all(|pair| pair[0].0 < pair[1].0) {
                return Err(format!("{key}: le bounds are not increasing"));
            }
            if !series.windows(2).all(|pair| pair[0].1 <= pair[1].1) {
                return Err(format!("{key}: buckets are not cumulative"));
            }
            let (last_bound, last_count) = *series.last().expect("non-empty series");
            if last_bound != f64::INFINITY {
                return Err(format!("{key}: missing +Inf bucket"));
            }
            let total = *totals
                .get(&key)
                .ok_or_else(|| format!("{key}: missing _count"))?;
            if total != last_count {
                return Err(format!("{key}: _count {total} != +Inf bucket {last_count}"));
            }
            if !sums.contains_key(&key) {
                return Err(format!("{key}: missing _sum"));
            }
            counts.insert(key, total as u64);
        }
    }
    Ok(counts)
}

fn series_label(name: &str, labels: &BTreeMap<String, String>) -> String {
    if labels.is_empty() {
        return name.to_string();
    }
    let rendered = labels
        .iter()
        .map(|(label, value)| format!("{label}=\"{value}\""))
        .collect::<Vec<_>>()
        .join(",");
    format!("{name}{{{rendered}}}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn histogram_buckets_are_cumulative_with_sum_and_count() {
        let family = HistogramFamily::new(
            "izwi_test_seconds",
            "Test latency.",
            &["model"],
            &[0.1, 1.0],
        );
        for value in [0.05, 0.1, 0.5, 2.0, -1.0, f64::NAN] {
            family.observe(&["m"], value);
        }

        let mut text = String::new();
        family.render(&mut text);
        assert!(text.contains("# TYPE izwi_test_seconds histogram"));
        // 0.1 falls in le="0.1" (le is inclusive); -1 counts as 0; NaN is dropped.
        assert!(text.contains("izwi_test_seconds_bucket{model=\"m\",le=\"0.1\"} 3"));
        assert!(text.contains("izwi_test_seconds_bucket{model=\"m\",le=\"1.0\"} 4"));
        assert!(text.contains("izwi_test_seconds_bucket{model=\"m\",le=\"+Inf\"} 5"));
        let sum = parse_exposition(&text)
            .unwrap()
            .into_iter()
            .find(|sample| sample.name == "izwi_test_seconds_sum")
            .expect("sum sample");
        assert!((sum.value - 2.65).abs() < 1e-9);
        let counts = check_histograms(&text).expect("valid exposition");
        assert_eq!(counts["izwi_test_seconds{model=\"m\"}"], 5);
    }

    #[test]
    fn unlabelled_histogram_renders_before_any_observation() {
        let family = HistogramFamily::new("izwi_empty_seconds", "Empty.", &[], &[1.0]);
        let mut text = String::new();
        family.render(&mut text);
        assert_eq!(check_histograms(&text).unwrap()["izwi_empty_seconds"], 0);
    }

    #[test]
    fn families_cap_their_series_and_escape_label_values() {
        let family = HistogramFamily::new("izwi_capped_seconds", "Capped.", &["model"], &[1.0])
            .with_max_series(2);
        family.observe(&["a\"b"], 0.5);
        family.observe(&["c"], 0.5);
        family.observe(&["d"], 0.5);
        family.observe(&["e"], 0.5);
        family.observe(&["c"], 0.5);

        let mut text = String::new();
        family.render(&mut text);
        let counts = check_histograms(&text).unwrap();
        assert_eq!(counts.len(), 3);
        assert_eq!(counts["izwi_capped_seconds{model=\"a\"b\"}"], 1);
        assert_eq!(counts["izwi_capped_seconds{model=\"c\"}"], 2);
        assert_eq!(counts["izwi_capped_seconds{model=\"other\"}"], 2);
        assert!(text.contains(r#"model="a\"b""#));
    }

    #[test]
    fn counters_render_labelled_totals() {
        let family = CounterFamily::new("izwi_tokens_total", "Tokens.", &["model"]);
        family.add(&["m"], 3);
        family.add(&["m"], 4);
        let mut text = String::new();
        family.render(&mut text);
        assert!(text.contains("# TYPE izwi_tokens_total counter"));
        assert!(text.contains("izwi_tokens_total{model=\"m\"} 7"));
        assert_eq!(family.get(&["m"]), 7);
        parse_exposition(&text).unwrap();
    }

    #[test]
    fn checker_rejects_inconsistent_histograms() {
        let broken = "# TYPE h histogram\nh_bucket{le=\"1.0\"} 3\nh_bucket{le=\"+Inf\"} 2\nh_sum 1\nh_count 2\n";
        assert!(check_histograms(broken)
            .unwrap_err()
            .contains("not cumulative"));
        let miscounted = "# TYPE h histogram\nh_bucket{le=\"1.0\"} 1\nh_bucket{le=\"+Inf\"} 2\nh_sum 1\nh_count 3\n";
        assert!(check_histograms(miscounted).unwrap_err().contains("_count"));
        let no_inf = "# TYPE h histogram\nh_bucket{le=\"1.0\"} 1\nh_sum 1\nh_count 1\n";
        assert!(check_histograms(no_inf).unwrap_err().contains("+Inf"));
    }
}
