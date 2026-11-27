//! Family-neutral loading of chat image inputs: fetching a source's encoded
//! bytes (data URL, public http(s), or a local file under the configured media
//! root) and decoding them within bounded limits.

use std::env;
use std::fs::{self, File};
use std::io::{Cursor, Read};
use std::net::{IpAddr, SocketAddr, ToSocketAddrs};
use std::path::{Path, PathBuf};
use std::sync::{mpsc, Arc, Mutex, OnceLock};
use std::time::Duration;

#[cfg(unix)]
use rustix::fs::{Mode, OFlags};
#[cfg(unix)]
use std::path::Component;

use base64::Engine;
use image::{DynamicImage, ImageReader, Limits};

use crate::error::{Error, Result};
use crate::models::shared::chat::{ChatMediaInput, ChatMediaKind};

pub(crate) const MAX_MEDIA_INPUTS: usize = 4;
const MAX_ENCODED_IMAGE_BYTES: u64 = 16 * 1024 * 1024;
const MAX_IMAGE_DIMENSION: u32 = 8_192;
const MAX_IMAGE_PIXELS: u64 = 16 * 1024 * 1024;
pub(crate) const MAX_IMAGE_DECODER_ALLOC_BYTES: u64 = 256 * 1024 * 1024;
pub(crate) const IMAGE_HOST_PREPROCESS_WORKSPACE_BYTES: u64 = 64 * 1024 * 1024;
const MEDIA_CONNECT_TIMEOUT: Duration = Duration::from_secs(2);
const MEDIA_DNS_TIMEOUT: Duration = Duration::from_secs(2);
const MEDIA_REQUEST_TIMEOUT: Duration = Duration::from_secs(10);
const MEDIA_DNS_WORKERS: usize = 2;
const MEDIA_DNS_QUEUE_CAPACITY: usize = 16;
const MEDIA_DNS_MAX_ADDRESSES: usize = 16;
const LOCAL_MEDIA_ROOT_ENV: &str = "IZWI_MEDIA_LOCAL_ROOT";

/// Host and device bytes a request's media needs while it is prepared.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct MediaResourceEstimate {
    pub host_bytes: u64,
    pub backend_tensor_bytes: u64,
}

/// Host bytes to fetch and decode `media_inputs`: their encoded size plus the decoder
/// allocation and preprocessing workspace per image. Rejects more than MAX_MEDIA_INPUTS,
/// anything that is not an image, and an empty source. `family` names the model family
/// in error messages (e.g. "Qwen3.5").
pub(crate) fn image_media_host_bytes(media_inputs: &[ChatMediaInput], family: &str) -> Result<u64> {
    if media_inputs.len() > MAX_MEDIA_INPUTS {
        return Err(Error::InvalidInput(format!(
            "{family} accepts at most {MAX_MEDIA_INPUTS} media inputs per request"
        )));
    }

    let mut encoded_bytes = 0u64;
    for media in media_inputs {
        if media.kind != ChatMediaKind::Image {
            return Err(Error::InvalidInput(format!(
                "{family} video inputs are not implemented yet"
            )));
        }
        if media.source.trim().is_empty() {
            return Err(Error::InvalidInput(format!(
                "{family} image source cannot be empty"
            )));
        }
        encoded_bytes = encoded_bytes
            .checked_add(encoded_media_budget(&media.source)?)
            .ok_or_else(|| Error::Overloaded(format!("{family} media size overflow")))?;
    }

    let count = u64::try_from(media_inputs.len())
        .map_err(|_| Error::Overloaded(format!("{family} media count overflow")))?;
    let per_image_host = MAX_IMAGE_DECODER_ALLOC_BYTES
        .checked_add(IMAGE_HOST_PREPROCESS_WORKSPACE_BYTES)
        .ok_or_else(|| Error::Overloaded(format!("{family} media workspace overflow")))?;
    encoded_bytes
        .checked_add(
            count
                .checked_mul(per_image_host)
                .ok_or_else(|| Error::Overloaded(format!("{family} media workspace overflow")))?,
        )
        .ok_or_else(|| Error::Overloaded(format!("{family} media workspace overflow")))
}

fn encoded_media_budget(source: &str) -> Result<u64> {
    if source.starts_with("data:") {
        let (metadata, payload) = source
            .split_once(',')
            .ok_or_else(|| Error::InvalidInput("Invalid data URL image payload".to_string()))?;
        let payload_len = u64::try_from(payload.trim().len())
            .map_err(|_| Error::Overloaded("Qwen3.5 image payload overflow".to_string()))?;
        let decoded_upper = if metadata.contains(";base64") {
            payload_len
                .checked_add(3)
                .and_then(|value| value.checked_div(4))
                .and_then(|value| value.checked_mul(3))
                .ok_or_else(|| Error::Overloaded("Qwen3.5 image payload overflow".to_string()))?
        } else {
            payload_len
        };
        if decoded_upper > MAX_ENCODED_IMAGE_BYTES {
            return Err(Error::InvalidInput(format!(
                "Qwen3.5 encoded image exceeds the {} byte limit",
                MAX_ENCODED_IMAGE_BYTES
            )));
        }
        return Ok(decoded_upper);
    }

    // Remote and filesystem sources are not touched before admission. Reserve
    // the strict fetch cap and enforce the actual byte count while reading.
    Ok(MAX_ENCODED_IMAGE_BYTES)
}

/// Decodes encoded image bytes within the dimension, pixel and allocation limits.
pub(crate) fn decode_image(bytes: &[u8]) -> Result<DynamicImage> {
    let reader = bounded_image_reader(bytes)?;
    let (width, height) = reader.into_dimensions().map_err(|err| {
        Error::InvalidInput(format!("Failed to inspect image input dimensions: {err}"))
    })?;
    let pixels = u64::from(width)
        .checked_mul(u64::from(height))
        .ok_or_else(|| Error::InvalidInput("Qwen3.5 image dimensions overflow".to_string()))?;
    if width == 0 || height == 0 || pixels > MAX_IMAGE_PIXELS {
        return Err(Error::InvalidInput(format!(
            "Qwen3.5 image dimensions {width}x{height} exceed the {MAX_IMAGE_PIXELS} pixel limit"
        )));
    }

    bounded_image_reader(bytes)?
        .decode()
        .map_err(|err| Error::InvalidInput(format!("Failed to decode bounded image input: {err}")))
}

fn bounded_image_reader(bytes: &[u8]) -> Result<ImageReader<Cursor<&[u8]>>> {
    let mut reader = ImageReader::new(Cursor::new(bytes))
        .with_guessed_format()
        .map_err(|err| Error::InvalidInput(format!("Unknown image input format: {err}")))?;
    let mut limits = Limits::default();
    limits.max_image_width = Some(MAX_IMAGE_DIMENSION);
    limits.max_image_height = Some(MAX_IMAGE_DIMENSION);
    limits.max_alloc = Some(MAX_IMAGE_DECODER_ALLOC_BYTES);
    reader.limits(limits);
    Ok(reader)
}

/// Fetches one image source's encoded bytes (data URL, public http(s), or a local file
/// under IZWI_MEDIA_LOCAL_ROOT), bounded in size.
pub(crate) fn fetch_media_bytes(source: &str) -> Result<Vec<u8>> {
    if source.starts_with("data:") {
        let bytes = decode_data_url(source)?;
        return enforce_encoded_media_limit(bytes);
    }

    if Path::new(source).is_absolute() {
        return fetch_file_media_bytes(source);
    }

    if let Ok(url) = reqwest::Url::parse(source) {
        return match url.scheme() {
            "http" | "https" => fetch_remote_media_bytes(source),
            "file" => fetch_file_media_bytes(source),
            scheme => Err(Error::InvalidInput(format!(
                "Qwen3.5 media URL scheme {scheme:?} is not supported"
            ))),
        };
    }

    fetch_file_media_bytes(source)
}

fn fetch_remote_media_bytes(source: &str) -> Result<Vec<u8>> {
    let validated = validate_remote_media_url_with(source, resolve_media_host)?;
    // reqwest's rustls has no built-in crypto provider in this workspace.
    let _ = rustls::crypto::ring::default_provider().install_default();
    let client = reqwest::blocking::Client::builder()
        .connect_timeout(MEDIA_CONNECT_TIMEOUT)
        .timeout(MEDIA_REQUEST_TIMEOUT)
        .redirect(reqwest::redirect::Policy::none())
        // A process-level proxy would bypass the validated and pinned target
        // addresses, so media fetches deliberately never inherit proxy state.
        .no_proxy()
        .resolve_to_addrs(&validated.host, &validated.addresses)
        .build()?;
    let response = client.get(validated.url).send()?;
    if response.status().is_redirection() {
        return Err(Error::InvalidInput(
            "Remote media redirects are disabled".to_string(),
        ));
    }
    let response = response.error_for_status()?;
    if response
        .content_length()
        .is_some_and(|length| length > MAX_ENCODED_IMAGE_BYTES)
    {
        return Err(Error::InvalidInput(format!(
            "Qwen3.5 remote image exceeds the {} byte limit",
            MAX_ENCODED_IMAGE_BYTES
        )));
    }
    read_bounded(response)
}

fn fetch_file_media_bytes(source: &str) -> Result<Vec<u8>> {
    let root = configured_local_media_root()?;
    fetch_file_media_bytes_under_root(source, &root)
}

/// [`LOCAL_MEDIA_ROOT_ENV`] as set, read once; each request still validates it.
static LOCAL_MEDIA_ROOT: crate::env::ReadOnce<Option<PathBuf>> =
    crate::env::ReadOnce::new(|| env::var_os(LOCAL_MEDIA_ROOT_ENV).map(PathBuf::from));

fn configured_local_media_root() -> Result<PathBuf> {
    configured_local_media_root_from(LOCAL_MEDIA_ROOT.get())
}

fn configured_local_media_root_from(root: Option<PathBuf>) -> Result<PathBuf> {
    let root = root
        .filter(|root| !root.as_os_str().is_empty())
        .ok_or_else(|| {
            Error::InvalidInput(format!(
                "Local Qwen3.5 media is disabled; set {LOCAL_MEDIA_ROOT_ENV} to an explicit media root"
            ))
        })?;
    if !root.is_absolute() {
        return Err(Error::ConfigError(format!(
            "{LOCAL_MEDIA_ROOT_ENV} must be an absolute path"
        )));
    }
    let canonical = fs::canonicalize(&root).map_err(|error| {
        Error::ConfigError(format!(
            "Failed to canonicalize {LOCAL_MEDIA_ROOT_ENV}={}: {error}",
            root.display()
        ))
    })?;
    if !canonical.is_dir() {
        return Err(Error::ConfigError(format!(
            "{LOCAL_MEDIA_ROOT_ENV}={} is not a directory",
            canonical.display()
        )));
    }
    Ok(canonical)
}

fn local_media_source_path(source: &str) -> Result<PathBuf> {
    if !source.starts_with("file:") {
        return Ok(PathBuf::from(source));
    }

    let url = reqwest::Url::parse(source)
        .map_err(|error| Error::InvalidInput(format!("Invalid local media URL: {error}")))?;
    if url.scheme() != "file" || url.query().is_some() || url.fragment().is_some() {
        return Err(Error::InvalidInput(
            "Local media must use a plain file URL without query or fragment".to_string(),
        ));
    }
    url.to_file_path().map_err(|_| {
        Error::InvalidInput("Local media file URL could not be converted to a path".to_string())
    })
}

fn canonical_local_media_path(source: &str, canonical_root: &Path) -> Result<PathBuf> {
    let source_path = local_media_source_path(source)?;
    let candidate = if source_path.is_absolute() {
        source_path
    } else {
        canonical_root.join(source_path)
    };
    let canonical = fs::canonicalize(&candidate).map_err(|error| {
        Error::InvalidInput(format!("Failed to resolve local media path: {error}"))
    })?;
    if !canonical.starts_with(canonical_root) {
        return Err(Error::InvalidInput(format!(
            "Local media path escapes {LOCAL_MEDIA_ROOT_ENV}"
        )));
    }
    Ok(canonical)
}

fn fetch_file_media_bytes_under_root(source: &str, canonical_root: &Path) -> Result<Vec<u8>> {
    let path = canonical_local_media_path(source, canonical_root)?;
    let file = open_canonical_local_media(&path, canonical_root)?;
    let metadata = file.metadata()?;
    if !metadata.is_file() {
        return Err(Error::InvalidInput(
            "Qwen3.5 local media source must be a regular file".to_string(),
        ));
    }
    if metadata.len() > MAX_ENCODED_IMAGE_BYTES {
        return Err(Error::InvalidInput(format!(
            "Qwen3.5 local image exceeds the {} byte limit",
            MAX_ENCODED_IMAGE_BYTES
        )));
    }
    read_bounded(file)
}

#[cfg(unix)]
fn open_canonical_local_media(path: &Path, canonical_root: &Path) -> Result<File> {
    let relative = path.strip_prefix(canonical_root).map_err(|_| {
        Error::InvalidInput(format!("Local media path escapes {LOCAL_MEDIA_ROOT_ENV}"))
    })?;
    let components = relative
        .components()
        .map(|component| match component {
            Component::Normal(component) => Ok(component),
            _ => Err(Error::InvalidInput(
                "Local media path contains an invalid component".to_string(),
            )),
        })
        .collect::<Result<Vec<_>>>()?;
    let (file_name, directories) = components.split_last().ok_or_else(|| {
        Error::InvalidInput("Qwen3.5 local media source must name a file".to_string())
    })?;

    let directory_flags = OFlags::RDONLY | OFlags::DIRECTORY | OFlags::NOFOLLOW | OFlags::CLOEXEC;
    // `O_NOFOLLOW` on `open(canonical_root)` protects only the final path
    // component. Traverse from `/` so a concurrently swapped symlink in any
    // configured-root ancestor cannot redirect this descriptor outside the
    // canonical policy boundary.
    let mut directory = rustix::fs::open("/", directory_flags, Mode::empty()).map_err(|error| {
        Error::ConfigError(format!(
            "Failed to securely open the local media filesystem root: {error}"
        ))
    })?;
    for component in canonical_root.components() {
        match component {
            Component::RootDir => {}
            Component::Normal(component) => {
                directory =
                    rustix::fs::openat(&directory, component, directory_flags, Mode::empty())
                        .map_err(|error| {
                            Error::ConfigError(format!(
                                "Failed to securely open {LOCAL_MEDIA_ROOT_ENV}={}: {error}",
                                canonical_root.display()
                            ))
                        })?;
            }
            _ => {
                return Err(Error::ConfigError(format!(
                    "{LOCAL_MEDIA_ROOT_ENV} contains an invalid canonical component"
                )));
            }
        }
    }
    for component in directories {
        directory = rustix::fs::openat(&directory, *component, directory_flags, Mode::empty())
            .map_err(|error| {
                Error::InvalidInput(format!(
                    "Failed to securely resolve local media path: {error}"
                ))
            })?;
    }

    let file = rustix::fs::openat(
        &directory,
        *file_name,
        OFlags::RDONLY | OFlags::NOFOLLOW | OFlags::NONBLOCK | OFlags::CLOEXEC,
        Mode::empty(),
    )
    .map_err(|error| {
        Error::InvalidInput(format!("Failed to securely open local media file: {error}"))
    })?;
    Ok(File::from(file))
}

#[cfg(not(unix))]
fn open_canonical_local_media(_path: &Path, _canonical_root: &Path) -> Result<File> {
    // Canonicalize-then-open cannot close symlink replacement races on these
    // platforms. Keep local media fail-closed; data and public HTTPS sources
    // remain available on every backend.
    Err(Error::ConfigError(
        "Secure local Qwen3.5 media opens are not supported on this platform".to_string(),
    ))
}

#[derive(Debug)]
struct ValidatedRemoteMedia {
    url: reqwest::Url,
    host: String,
    addresses: Vec<SocketAddr>,
}

fn validate_remote_media_url_with<F>(source: &str, mut resolve: F) -> Result<ValidatedRemoteMedia>
where
    F: FnMut(&str, u16) -> std::io::Result<Vec<SocketAddr>>,
{
    let url = reqwest::Url::parse(source)
        .map_err(|error| Error::InvalidInput(format!("Invalid remote media URL: {error}")))?;
    if !matches!(url.scheme(), "http" | "https") {
        return Err(Error::InvalidInput(
            "Remote media must use http or https".to_string(),
        ));
    }
    if !url.username().is_empty() || url.password().is_some() {
        return Err(Error::InvalidInput(
            "Remote media URLs cannot contain credentials".to_string(),
        ));
    }
    if url.fragment().is_some() {
        return Err(Error::InvalidInput(
            "Remote media URLs cannot contain fragments".to_string(),
        ));
    }

    let host = url
        .host_str()
        .ok_or_else(|| Error::InvalidInput("Remote media URL is missing a host".to_string()))?
        .to_string();
    if host.len() > 253 {
        return Err(Error::InvalidInput(
            "Remote media host exceeds the 253-byte DNS limit".to_string(),
        ));
    }
    let normalized_host = host.trim_end_matches('.').to_ascii_lowercase();
    if normalized_host.is_empty()
        || host.ends_with('.')
        || normalized_host == "localhost"
        || normalized_host.ends_with(".localhost")
        || normalized_host.ends_with(".local")
        || normalized_host.ends_with(".home.arpa")
    {
        return Err(Error::InvalidInput(format!(
            "Remote media host {host:?} is not publicly routable"
        )));
    }

    let port = url
        .port_or_known_default()
        .ok_or_else(|| Error::InvalidInput("Remote media URL is missing a port".to_string()))?;
    let mut addresses = match host.parse::<IpAddr>() {
        Ok(address) => vec![SocketAddr::new(address, port)],
        Err(_) => resolve(&host, port).map_err(|error| {
            Error::InvalidInput(format!(
                "Failed to resolve remote media host {host:?}: {error}"
            ))
        })?,
    };
    if addresses.is_empty() {
        return Err(Error::InvalidInput(format!(
            "Remote media host {host:?} resolved to no addresses"
        )));
    }
    if addresses.len() > MEDIA_DNS_MAX_ADDRESSES {
        return Err(Error::InvalidInput(format!(
            "Remote media host {host:?} resolved to more than {MEDIA_DNS_MAX_ADDRESSES} addresses"
        )));
    }

    for address in &mut addresses {
        validate_public_media_ip(address.ip())?;
        address.set_port(port);
    }
    addresses.sort_unstable();
    addresses.dedup();

    Ok(ValidatedRemoteMedia {
        url,
        host,
        addresses,
    })
}

type MediaDnsResult = std::io::Result<Vec<SocketAddr>>;

struct MediaDnsJob {
    host: String,
    port: u16,
    response: mpsc::SyncSender<MediaDnsResult>,
}

struct BoundedMediaDnsResolver {
    jobs: mpsc::SyncSender<MediaDnsJob>,
}

impl BoundedMediaDnsResolver {
    fn start() -> std::result::Result<Self, String> {
        let (jobs, receiver) = mpsc::sync_channel::<MediaDnsJob>(MEDIA_DNS_QUEUE_CAPACITY);
        let receiver = Arc::new(Mutex::new(receiver));
        for worker_index in 0..MEDIA_DNS_WORKERS {
            let receiver = receiver.clone();
            std::thread::Builder::new()
                .name(format!("izwi-media-dns-{worker_index}"))
                .spawn(move || loop {
                    let job = {
                        let receiver = receiver.lock().unwrap_or_else(|poison| poison.into_inner());
                        receiver.recv()
                    };
                    let Ok(job) = job else {
                        break;
                    };
                    let result = (job.host.as_str(), job.port)
                        .to_socket_addrs()
                        .map(|addresses| addresses.take(MEDIA_DNS_MAX_ADDRESSES + 1).collect());
                    let _ = job.response.try_send(result);
                })
                .map_err(|error| format!("failed to start bounded media DNS worker: {error}"))?;
        }
        Ok(Self { jobs })
    }

    fn resolve(&self, host: &str, port: u16) -> std::io::Result<Vec<SocketAddr>> {
        let (response, result) = mpsc::sync_channel(1);
        self.jobs
            .try_send(MediaDnsJob {
                host: host.to_string(),
                port,
                response,
            })
            .map_err(|error| match error {
                mpsc::TrySendError::Full(_) => std::io::Error::new(
                    std::io::ErrorKind::WouldBlock,
                    "bounded media DNS queue is full",
                ),
                mpsc::TrySendError::Disconnected(_) => std::io::Error::new(
                    std::io::ErrorKind::BrokenPipe,
                    "bounded media DNS workers are unavailable",
                ),
            })?;
        recv_media_dns_result(result, MEDIA_DNS_TIMEOUT)
    }
}

fn recv_media_dns_result(
    result: mpsc::Receiver<MediaDnsResult>,
    timeout: Duration,
) -> std::io::Result<Vec<SocketAddr>> {
    result.recv_timeout(timeout).map_err(|error| match error {
        mpsc::RecvTimeoutError::Timeout => std::io::Error::new(
            std::io::ErrorKind::TimedOut,
            "remote media DNS resolution timed out",
        ),
        mpsc::RecvTimeoutError::Disconnected => std::io::Error::new(
            std::io::ErrorKind::BrokenPipe,
            "remote media DNS worker ended without a result",
        ),
    })?
}

fn bounded_media_dns_resolver() -> std::io::Result<&'static BoundedMediaDnsResolver> {
    static RESOLVER: OnceLock<std::result::Result<BoundedMediaDnsResolver, String>> =
        OnceLock::new();
    RESOLVER
        .get_or_init(BoundedMediaDnsResolver::start)
        .as_ref()
        .map_err(|error| std::io::Error::other(error.clone()))
}

fn resolve_media_host(host: &str, port: u16) -> std::io::Result<Vec<SocketAddr>> {
    bounded_media_dns_resolver()?.resolve(host, port)
}

fn validate_public_media_ip(address: IpAddr) -> Result<()> {
    if is_public_media_ip(address) {
        return Ok(());
    }
    Err(Error::InvalidInput(format!(
        "Remote media address {address} is not publicly routable"
    )))
}

/// Public-address policy shared with speech reference fetches
/// (`izwi_core::media_policy`).
fn is_public_media_ip(address: IpAddr) -> bool {
    crate::media_policy::is_public_ip(address)
}

fn read_bounded(mut reader: impl Read) -> Result<Vec<u8>> {
    read_with_limit(&mut reader, MAX_ENCODED_IMAGE_BYTES)
}

fn read_with_limit(mut reader: impl Read, max_bytes: u64) -> Result<Vec<u8>> {
    let limit = max_bytes
        .checked_add(1)
        .ok_or_else(|| Error::Overloaded("Qwen3.5 image byte limit overflow".to_string()))?;
    let mut bytes = Vec::new();
    reader.by_ref().take(limit).read_to_end(&mut bytes)?;
    if u64::try_from(bytes.len()).unwrap_or(u64::MAX) > max_bytes {
        return Err(Error::InvalidInput(format!(
            "Qwen3.5 encoded image exceeds the {max_bytes} byte limit"
        )));
    }
    if bytes.is_empty() {
        return Err(Error::InvalidInput(
            "Qwen3.5 encoded image is empty".to_string(),
        ));
    }
    Ok(bytes)
}

fn enforce_encoded_media_limit(bytes: Vec<u8>) -> Result<Vec<u8>> {
    if u64::try_from(bytes.len()).unwrap_or(u64::MAX) > MAX_ENCODED_IMAGE_BYTES {
        return Err(Error::InvalidInput(format!(
            "Qwen3.5 encoded image exceeds the {} byte limit",
            MAX_ENCODED_IMAGE_BYTES
        )));
    }
    if bytes.is_empty() {
        return Err(Error::InvalidInput(
            "Qwen3.5 encoded image is empty".to_string(),
        ));
    }
    Ok(bytes)
}

fn decode_data_url(data_url: &str) -> Result<Vec<u8>> {
    let _ = encoded_media_budget(data_url)?;
    let (_, payload) = data_url
        .split_once(',')
        .ok_or_else(|| Error::InvalidInput("Invalid data URL image payload".to_string()))?;
    if data_url[..data_url.find(',').unwrap_or_default()].contains(";base64") {
        base64::engine::general_purpose::STANDARD
            .decode(payload.trim())
            .map_err(|err| Error::InvalidInput(format!("Invalid base64 image payload: {err}")))
    } else {
        Ok(payload.as_bytes().to_vec())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::ImageFormat;
    use std::cell::Cell;

    #[test]
    fn decode_data_url_accepts_base64_payload() {
        let payload = base64::engine::general_purpose::STANDARD.encode(b"png");
        let data_url = format!("data:image/png;base64,{payload}");
        let decoded = decode_data_url(&data_url).expect("decode");
        assert_eq!(decoded, b"png");
    }

    #[test]
    fn image_media_host_bytes_is_bounded_and_rejects_unsupported_shapes() {
        let host_bytes = image_media_host_bytes(
            &[ChatMediaInput {
                kind: ChatMediaKind::Image,
                source: "https://example.invalid/image.png".to_string(),
            }],
            "Qwen3.5",
        )
        .unwrap();
        assert_eq!(
            host_bytes,
            MAX_ENCODED_IMAGE_BYTES
                + MAX_IMAGE_DECODER_ALLOC_BYTES
                + IMAGE_HOST_PREPROCESS_WORKSPACE_BYTES
        );
        assert_eq!(image_media_host_bytes(&[], "Qwen3.5").unwrap(), 0);

        let too_many = (0..=MAX_MEDIA_INPUTS)
            .map(|idx| ChatMediaInput {
                kind: ChatMediaKind::Image,
                source: format!("image-{idx}.png"),
            })
            .collect::<Vec<_>>();
        assert!(matches!(
            image_media_host_bytes(&too_many, "Qwen3.5"),
            Err(Error::InvalidInput(message))
                if message == "Qwen3.5 accepts at most 4 media inputs per request"
        ));
        assert!(matches!(
            image_media_host_bytes(
                &[ChatMediaInput {
                    kind: ChatMediaKind::Video,
                    source: "video.mp4".to_string(),
                }],
                "Qwen3.5"
            ),
            Err(Error::InvalidInput(message))
                if message == "Qwen3.5 video inputs are not implemented yet"
        ));
        assert!(matches!(
            image_media_host_bytes(
                &[ChatMediaInput {
                    kind: ChatMediaKind::Image,
                    source: "  ".to_string(),
                }],
                "Qwen3.5"
            ),
            Err(Error::InvalidInput(message)) if message == "Qwen3.5 image source cannot be empty"
        ));
    }

    #[test]
    fn bounded_reader_rejects_unknown_length_overflow() {
        let error = read_with_limit(Cursor::new(vec![7u8; 9]), 8)
            .expect_err("reader must stop one byte past its cap");
        assert!(matches!(error, Error::InvalidInput(_)));
    }

    #[test]
    fn local_media_is_default_deny_and_cannot_escape_its_canonical_root() {
        assert!(matches!(
            configured_local_media_root_from(None),
            Err(Error::InvalidInput(_))
        ));
        assert!(matches!(
            configured_local_media_root_from(Some(PathBuf::from("relative-media-root"))),
            Err(Error::ConfigError(_))
        ));

        let base = std::env::temp_dir().join(format!(
            "izwi-qwen35-media-root-test-{}",
            uuid::Uuid::new_v4()
        ));
        let root = base.join("root");
        let nested = root.join("nested");
        let outside = base.join("outside.png");
        fs::create_dir_all(&nested).unwrap();
        fs::write(nested.join("inside.png"), b"inside").unwrap();
        fs::write(&outside, b"outside").unwrap();

        let canonical_root =
            configured_local_media_root_from(Some(root.clone())).expect("canonical root");
        assert_eq!(
            fetch_file_media_bytes_under_root("nested/inside.png", &canonical_root).unwrap(),
            b"inside"
        );
        let file_url = reqwest::Url::from_file_path(nested.join("inside.png"))
            .unwrap()
            .to_string();
        assert_eq!(
            fetch_file_media_bytes_under_root(&file_url, &canonical_root).unwrap(),
            b"inside"
        );
        assert!(matches!(
            fetch_file_media_bytes_under_root("../outside.png", &canonical_root),
            Err(Error::InvalidInput(_))
        ));
        assert!(matches!(
            fetch_file_media_bytes_under_root(outside.to_str().unwrap(), &canonical_root),
            Err(Error::InvalidInput(_))
        ));

        #[cfg(unix)]
        {
            std::os::unix::fs::symlink(&outside, root.join("outside-link.png")).unwrap();
            assert!(matches!(
                fetch_file_media_bytes_under_root("outside-link.png", &canonical_root),
                Err(Error::InvalidInput(_))
            ));

            // Simulate an attacker replacing a canonicalized directory with a
            // symlink before open. Descriptor-relative O_NOFOLLOW traversal
            // must reject the swapped component instead of reading outside.
            let raced = root.join("raced");
            fs::create_dir_all(&raced).unwrap();
            fs::write(raced.join("image.png"), b"original").unwrap();
            let canonical_raced =
                canonical_local_media_path("raced/image.png", &canonical_root).unwrap();
            fs::rename(&raced, root.join("raced-original")).unwrap();
            let attacker = base.join("attacker");
            fs::create_dir_all(&attacker).unwrap();
            fs::write(attacker.join("image.png"), b"outside").unwrap();
            std::os::unix::fs::symlink(&attacker, &raced).unwrap();
            assert!(matches!(
                open_canonical_local_media(&canonical_raced, &canonical_root),
                Err(Error::InvalidInput(_))
            ));

            // `O_NOFOLLOW` must cover configured-root ancestors too, not only
            // descendants. A replacement symlink in an ancestor must not send
            // the open into an attacker-controlled lookalike root.
            let ancestor = base.join("ancestor");
            let ancestor_root = ancestor.join("root");
            fs::create_dir_all(&ancestor_root).unwrap();
            fs::write(ancestor_root.join("image.png"), b"original").unwrap();
            let canonical_ancestor_root =
                configured_local_media_root_from(Some(ancestor_root)).unwrap();
            let canonical_ancestor_image =
                canonical_local_media_path("image.png", &canonical_ancestor_root).unwrap();
            let original_ancestor = base.join("ancestor-original");
            fs::rename(&ancestor, &original_ancestor).unwrap();
            let attacker_ancestor = base.join("attacker-ancestor");
            fs::create_dir_all(attacker_ancestor.join("root")).unwrap();
            fs::write(attacker_ancestor.join("root/image.png"), b"outside").unwrap();
            std::os::unix::fs::symlink(&attacker_ancestor, &ancestor).unwrap();
            assert!(matches!(
                open_canonical_local_media(&canonical_ancestor_image, &canonical_ancestor_root),
                Err(Error::ConfigError(_))
            ));
        }

        // Non-regular descriptors are never treated as bounded image content.
        assert!(matches!(
            fetch_file_media_bytes_under_root("nested", &canonical_root),
            Err(Error::InvalidInput(_))
        ));

        fs::remove_dir_all(base).unwrap();
    }

    #[test]
    fn remote_media_rejects_non_public_and_mixed_dns_answers() {
        for address in [
            "0.0.0.0",
            "10.0.0.1",
            "100.64.0.1",
            "127.0.0.1",
            "169.254.169.254",
            "192.0.0.9",
            "192.0.2.1",
            "192.88.99.1",
            "198.18.0.1",
            "198.51.100.1",
            "203.0.113.1",
            "224.0.0.1",
            "240.0.0.1",
            "255.255.255.255",
            "::",
            "::1",
            "::ffff:127.0.0.1",
            "fc00::1",
            "fe80::1",
            "ff02::1",
            "2001::1",
            "2001:db8::1",
            "2002:7f00:1::1",
            "3fff::1",
        ] {
            assert!(
                !is_public_media_ip(address.parse().unwrap()),
                "{address} must not be accepted as a public media target"
            );
        }
        for address in ["93.184.216.34", "2606:4700:4700::1111"] {
            assert!(
                is_public_media_ip(address.parse().unwrap()),
                "{address} should remain a usable public media target"
            );
        }

        for source in [
            "http://127.0.0.1/image.png",
            "http://2130706433/image.png",
            "http://[::1]/image.png",
        ] {
            assert!(matches!(
                validate_remote_media_url_with(source, |_, _| Ok(Vec::new())),
                Err(Error::InvalidInput(_))
            ));
        }

        let error =
            validate_remote_media_url_with("https://images.example/image.png", |_, port| {
                Ok(vec![
                    SocketAddr::new("93.184.216.34".parse().unwrap(), port),
                    SocketAddr::new("10.0.0.2".parse().unwrap(), port),
                ])
            })
            .expect_err("one private DNS answer must reject the entire target");
        assert!(matches!(error, Error::InvalidInput(_)));

        let error =
            validate_remote_media_url_with("https://images.example/image.png", |_, port| {
                Ok((0..=MEDIA_DNS_MAX_ADDRESSES)
                    .map(|_| SocketAddr::new("93.184.216.34".parse().unwrap(), port))
                    .collect())
            })
            .expect_err("oversized DNS answer sets must be rejected");
        assert!(matches!(error, Error::InvalidInput(message) if message.contains("more than")));
    }

    #[test]
    fn remote_media_rejects_credentials_special_hosts_and_pins_validated_dns() {
        for source in [
            "https://user:secret@images.example/image.png",
            "https://images.example./image.png",
            "https://localhost/image.png",
            "https://host.local/image.png",
            "https://router.home.arpa/image.png",
        ] {
            assert!(matches!(
                validate_remote_media_url_with(source, |_, _| Ok(vec!["93.184.216.34:443"
                    .parse()
                    .unwrap()])),
                Err(Error::InvalidInput(_))
            ));
        }

        let resolver_calls = Cell::new(0usize);
        let validated = validate_remote_media_url_with(
            "https://images.example:8443/image.png?token=public",
            |host, _| {
                resolver_calls.set(resolver_calls.get() + 1);
                assert_eq!(host, "images.example");
                Ok(vec![
                    "93.184.216.34:1".parse().unwrap(),
                    "93.184.216.34:2".parse().unwrap(),
                    "[2606:4700:4700::1111]:3".parse().unwrap(),
                ])
            },
        )
        .expect("public DNS answers");

        assert_eq!(resolver_calls.get(), 1);
        assert_eq!(validated.host, "images.example");
        assert_eq!(validated.addresses.len(), 2);
        assert!(validated
            .addresses
            .contains(&"93.184.216.34:8443".parse().unwrap()));
        assert!(validated
            .addresses
            .contains(&"[2606:4700:4700::1111]:8443".parse().unwrap()));
    }

    #[test]
    fn media_dns_wait_has_a_hard_timeout() {
        let (pending_response, result) = mpsc::sync_channel::<MediaDnsResult>(1);
        let started = std::time::Instant::now();
        let error = recv_media_dns_result(result, Duration::from_millis(5))
            .expect_err("a stalled resolver must time out");
        assert_eq!(error.kind(), std::io::ErrorKind::TimedOut);
        assert!(started.elapsed() < Duration::from_secs(1));
        drop(pending_response);
    }

    #[test]
    fn image_decoder_rejects_dimension_bombs_before_full_decode() {
        let image = DynamicImage::new_rgb8(MAX_IMAGE_DIMENSION + 1, 1);
        let mut encoded = Cursor::new(Vec::new());
        image.write_to(&mut encoded, ImageFormat::Png).unwrap();

        let error = decode_image(encoded.get_ref()).expect_err("oversize dimensions must fail");
        assert!(matches!(error, Error::InvalidInput(_)));
    }
}
