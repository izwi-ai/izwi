---
title: "izwi completions"
description: "Generate shell completions for Bash, Zsh, Fish, PowerShell, and Elvish."
icon: "list-checks"
---
Generate shell completions.

---

## Synopsis

```bash
izwi completions <SHELL>
```

---

## Description

Generates shell completion scripts for tab-completion of Izwi commands.

---

## Arguments

| Argument | Description |
|----------|-------------|
| `<SHELL>` | Shell type: `bash`, `zsh`, `fish`, `powershell`, `elvish` |

---

## Installation

### Bash

```bash
izwi completions bash > ~/.local/share/bash-completion/completions/izwi
```

Or add to `.bashrc`:

```bash
eval "$(izwi completions bash)"
```

### Zsh

```bash
izwi completions zsh > ~/.zfunc/_izwi
```

Ensure `~/.zfunc` is in your `fpath`. Add to `.zshrc`:

```zsh
fpath=(~/.zfunc $fpath)
autoload -Uz compinit && compinit
```

### Fish

```bash
izwi completions fish > ~/.config/fish/completions/izwi.fish
```

### PowerShell

```powershell
izwi completions powershell >> $PROFILE
```

### Elvish

```bash
izwi completions elvish > ~/.elvish/lib/izwi.elv
```

---

## Usage

After installation, restart your shell or source your config file. Then use Tab to complete:

```bash
izwi <TAB>
izwi serve --<TAB>
izwi pull qwen3-<TAB>
```

---

## See Also

- [CLI Reference](/cli)
