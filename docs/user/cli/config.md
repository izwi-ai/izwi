# izwi config

Manage the local `config.toml` used by `izwi serve`.

---

## Synopsis

```bash
izwi config <COMMAND>
```

---

## Subcommands

| Command | Description |
|---------|-------------|
| `show` | Show the current config file, or the default template when no file exists |
| `set` | Set a typed configuration value |
| `get` | Get a configuration value, falling back to the built-in default |
| `edit` | Edit the config file in your default editor |
| `reset` | Remove the config file |
| `path` | Show the config file path |

---

## Supported Keys

`izwi config set` and `izwi config get` support these keys:

- `server.host`
- `server.port`
- `server.cors`
- `server.cors_origins`
- `models.dir`
- `runtime.backend`
- `runtime.max_batch_size`
- `runtime.threads`
- `runtime.max_concurrent`
- `runtime.timeout`
- `ui.enabled`
- `ui.dir`
- `defaults.model`
- `defaults.speaker`
- `defaults.format`

---

## izwi config set

Set a configuration value.

```bash
izwi config set <KEY> <VALUE>
```

Examples:

```bash
izwi config set server.host 127.0.0.1
izwi config set server.cors true
izwi config set server.cors_origins http://localhost:3000,https://example.com
izwi config set runtime.backend metal
izwi config set runtime.max_batch_size 16
izwi config set runtime.timeout 600
izwi config set ui.enabled false
```

`server.cors_origins` accepts either a comma-separated list or a TOML array literal.

---

## izwi config get

Get a specific configuration value.

```bash
izwi config get <KEY>
```

Examples:

```bash
izwi config get server.port
izwi config get runtime.backend
izwi config get ui.enabled
```

If a key is not set in the file, `izwi config get` prints the built-in default.

---

## izwi config show

Display the current configuration file.

```bash
izwi config show
```

If no file exists yet, `show` prints the default template instead.

---

## izwi config edit

Open the configuration file in your default editor.

```bash
izwi config edit
```

Uses `$EDITOR`, or falls back to `vi`.

---

## izwi config reset

Remove the configuration file.

```bash
izwi config reset
izwi config reset --yes
```

### Options

| Option | Description |
|--------|-------------|
| `-y, --yes` | Reset without confirmation |

---

## izwi config path

Show the configuration file path.

```bash
izwi config path
```

---

## Configuration File

The config file is TOML:

```toml
[server]
host = "0.0.0.0"
port = 8080
cors = false
cors_origins = ["http://localhost:3000"]

[models]
dir = "/path/to/models"

[runtime]
backend = "auto"
max_batch_size = 8
threads = 8
max_concurrent = 100
timeout = 300

[ui]
enabled = true
dir = "ui/dist"
```

Only keys you set need to be present in the file. `izwi serve` resolves runtime values in this order:

1. CLI flags
2. Environment variables
3. `config.toml`
4. Built-in defaults

### File Locations

| Platform | Path |
|----------|------|
| **macOS** | `~/Library/Application Support/izwi/config.toml` |
| **Linux** | `~/.config/izwi/config.toml` |
| **Windows** | `%APPDATA%\izwi\config.toml` |

---

## See Also

- [`izwi serve`](./serve.md) — Server options and runtime precedence
