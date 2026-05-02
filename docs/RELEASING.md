# Releasing Izwi

Releases are tag-driven.
Pushing a tag that matches `v*` triggers `.github/workflows/release.yml` automatically.
There is no manual trigger in the workflow right now.

## Files to update for a new version

For `v0.1.0-alpha-1`, update these files before tagging:

1. `Cargo.toml`
   - Set `[workspace.package].version = "0.1.0-alpha-1"`.
2. `crates/izwi-desktop/tauri.conf.json`
   - Set `"version": "0.1.0-alpha-1"`.
3. `ui/package.json`
   - Set `"version": "0.1.0-alpha-1"` to keep frontend package metadata aligned.
4. `crates/izwi-core/src/model/download.rs` (recommended)
   - Update the hardcoded `User-Agent` version string (`izwi/...`) to match.

## Release steps (example: v0.1.0-alpha-1)

1. Update the version files listed above.
2. Run local checks:

```bash
cargo check --workspace
npm ci --prefix ui
npm --prefix ui run build
```

3. Commit the release prep changes:

```bash
git add Cargo.toml crates/izwi-desktop/tauri.conf.json ui/package.json crates/izwi-core/src/model/download.rs
git commit -m "Prepare release v0.1.0-alpha-1"
```

4. Push the commit:

```bash
git push origin <your-branch>
```

5. Tag and push the release tag:

```bash
git tag -a v0.1.0-alpha-1 -m "Izwi v0.1.0-alpha-1"
git push origin v0.1.0-alpha-1
```

6. Verify release:
   - Open GitHub Actions and confirm the `Release` workflow runs for tag `v0.1.0-alpha-1`.
   - Confirm the `Backend Truth` workflow is green on the branch or merge commit that produced the tag.
   - Open the GitHub Releases page and confirm all assets are attached.
   - Confirm the release body includes the runtime support note describing the CPU-only native artifacts and Docker/source CUDA paths.

## What the release workflow builds

For each OS runner, the workflow builds:

1. Terminal bundle:
   - `izwi` (CLI)
   - `izwi-server`
   - `izwi-desktop` (desktop shell binary used by `izwi serve --mode desktop`)
2. Desktop installer bundle:
   - Linux: `.deb`
   - Windows: `NSIS .exe`
   - macOS: `.dmg`

## Current backend truth for release artifacts

GitHub Releases currently publish:

1. macOS assets as the Metal-capable path for Apple Silicon hosts.
2. Linux and Windows assets as CPU-only native artifacts with CPU-safe public entrypoints.

Linux and Windows public binary names must remain unchanged:

1. `izwi` / `izwi-server` on Linux
2. `izwi.exe` / `izwi-server.exe` on Windows

Linux and Windows native release artifacts must not contain `runtime/cuda`, CUDA shared libraries, or CUDA DLLs. The supported NVIDIA paths are:

1. The Docker CUDA image/profile on NVIDIA Linux hosts
2. Source builds using `cargo build --release --features cuda`

The `Backend Truth` and `Release` workflows are the pre-release guardrails for this contract:

1. CPU cargo checks run separately from CUDA cargo checks.
2. CPU Docker builds run separately from CUDA Docker builds.
3. Linux and Windows release jobs verify native artifacts are CPU-only, reject CUDA runtime payloads, and enforce a conservative per-file size cap.
4. Docker build jobs smoke-run the final CPU and CUDA images with `/usr/local/bin/izwi-server --help`.
5. A GPU-host runtime smoke test is still the next gap; until an NVIDIA runner exists, end-to-end CUDA execution remains a manual release check on an NVIDIA Linux host.

The desktop bundle build runs the UI build automatically via Tauri `beforeBuildCommand`, so `ui/dist` is rebuilt from source on each release run.

Linux `.deb` installs:

1. `/usr/bin/izwi`
2. `/usr/bin/izwi-server`

## Signing and notarization

macOS signing is optional. If no Apple signing identity is configured, macOS builds run unsigned.

When you are ready to sign/notarize, configure these repository secrets:

1. `APPLE_CERTIFICATE`
2. `APPLE_CERTIFICATE_PASSWORD`
3. `APPLE_SIGNING_IDENTITY`
4. `APPLE_ID`
5. `APPLE_PASSWORD`
6. `APPLE_TEAM_ID`

## Troubleshooting workflow parse errors

If GitHub shows an error like:

- `Unrecognized named-value: 'secrets'` inside an `if:` expression

Cause:

- GitHub Actions does not allow direct `secrets.*` usage in `if:` conditions.

Fix:

- Map secrets into environment variables (`env`) and reference `env.*` in `if:`.

## macOS install/uninstall

For user-facing macOS instructions, see `docs/MACOS.md`.
