#!/bin/bash
# Izwi CLI Installation Script
# Supports macOS and Linux

set -e

REPO_URL="https://github.com/izwi-ai/izwi"
INSTALL_DIR="${INSTALL_DIR:-$HOME/.izwi}"
BIN_DIR="${BIN_DIR:-$HOME/.local/bin}"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_banner() {
    echo -e "${BLUE}"
    echo " ___ ____ ____ "
    echo "|_ _|_  /_  /  High-performance audio inference"
    echo " | | / / / /   Text-to-Speech & Speech-to-Text"
    echo "|___/___/___|  Optimized for Apple Silicon & CUDA"
    echo -e "${NC}"
}

detect_platform() {
    local os=$(uname -s | tr '[:upper:]' '[:lower:]')
    local arch=$(uname -m)
    
    case "$arch" in
        x86_64) arch="x86_64" ;;
        amd64) arch="x86_64" ;;
        arm64) arch="aarch64" ;;
        aarch64) arch="aarch64" ;;
        *) echo "Unsupported architecture: $arch"; exit 1 ;;
    esac
    
    echo "${arch}-${os}"
}

check_requirements() {
    echo -e "${BLUE}Checking requirements...${NC}"
    
    # Check Rust
    if ! command -v rustc &> /dev/null; then
        echo -e "${YELLOW}Rust not found. Installing...${NC}"
        curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
        source "$HOME/.cargo/env"
    fi

    # Check Node/npm for desktop UI build
    if ! command -v npm &> /dev/null; then
        echo -e "${RED}npm is required to build desktop assets (ui/dist). Install Node.js 18+ and retry.${NC}"
        exit 1
    fi
    
    echo -e "${GREEN}✓ Requirements satisfied${NC}"
}

install_from_source() {
    echo -e "${BLUE}Building Izwi from source...${NC}"
    
    # Clone if not already in repo
    if [ ! -f "Cargo.toml" ]; then
        local temp_dir=$(mktemp -d)
        git clone "$REPO_URL" "$temp_dir/izwi"
        cd "$temp_dir/izwi"
    fi
    
    # Build UI assets (required by izwi-desktop compile-time Tauri config)
    if [ ! -d "ui/node_modules" ]; then
        npm ci --prefix ui
    fi
    npm --prefix ui run build

    # Build release binaries (desktop build expects izwi + izwi-server artifacts to exist)
    cargo build --release --bin izwi --bin izwi-server
    cargo build --release --bin izwi-desktop
    
    # Create bin directory
    mkdir -p "$BIN_DIR"
    
    # Copy binaries
    cp "target/release/izwi" "$BIN_DIR/"
    cp "target/release/izwi-server" "$BIN_DIR/"
    cp "target/release/izwi-desktop" "$BIN_DIR/"
    
    echo -e "${GREEN}✓ Binaries installed to $BIN_DIR${NC}"
}

setup_shell() {
    echo -e "${BLUE}Setting up shell integration...${NC}"
    
    # Detect shell
    local shell=$(basename "$SHELL")
    local config_file=""
    
    case "$shell" in
        bash) config_file="$HOME/.bashrc" ;;
        zsh) config_file="$HOME/.zshrc" ;;
        fish) config_file="$HOME/.config/fish/config.fish" ;;
        *) echo "Unknown shell: $shell"; return ;;
    esac
    
    # Add to PATH if not already there
    if [[ ":$PATH:" != *":$BIN_DIR:"* ]]; then
        echo "export PATH=\"$BIN_DIR:\$PATH\"" >> "$config_file"
        echo -e "${GREEN}✓ Added $BIN_DIR to PATH in $config_file${NC}"
        echo -e "${YELLOW}Please restart your shell or run: source $config_file${NC}"
    fi
    
    # Generate completions
    if [ -f "$BIN_DIR/izwi" ]; then
        case "$shell" in
            bash)
                "$BIN_DIR/izwi" completions bash > "$HOME/.izwi-completion.bash"
                echo "source ~/.izwi-completion.bash" >> "$config_file"
                ;;
            zsh)
                mkdir -p "$HOME/.zsh/completions"
                "$BIN_DIR/izwi" completions zsh > "$HOME/.zsh/completions/_izwi"
                ;;
            fish)
                mkdir -p "$HOME/.config/fish/completions"
                "$BIN_DIR/izwi" completions fish > "$HOME/.config/fish/completions/izwi.fish"
                ;;
        esac
        echo -e "${GREEN}✓ Shell completions installed${NC}"
    fi
}

print_usage() {
    echo ""
    echo -e "${GREEN}Installation complete!${NC}"
    echo ""
    echo "Quick start:"
    echo "  izwi --help              # Show help"
    echo "  izwi serve               # Start server mode"
    echo "  izwi serve --mode desktop # Start server + desktop app"
    echo "  izwi pull qwen3-tts-0.6b-base  # Download a model"
    echo "  izwi tts 'Hello world'   # Generate speech"
    echo ""
    echo "Documentation: https://github.com/izwi-ai/izwi"
}

main() {
    print_banner
    
    echo "Platform: $(detect_platform)"
    echo "Install directory: $INSTALL_DIR"
    echo "Binary directory: $BIN_DIR"
    echo ""
    
    check_requirements
    install_from_source
    setup_shell
    print_usage
}

# Run main function
main "$@"
