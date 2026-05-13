#!/bin/bash
# Izwi CLI Installation Script
# Supports macOS and Linux

set -e

REPO_URL="https://github.com/izwi-ai/izwi"
INSTALL_DIR="${INSTALL_DIR:-$HOME/.izwi}"
BIN_DIR="${BIN_DIR:-$HOME/.local/bin}"
BUILD_BACKEND="${IZWI_BUILD_BACKEND:-auto}"

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
    echo "|___/___/___|  Source install for CPU, Metal, or CUDA"
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

resolve_build_backend() {
    local requested=$(echo "$BUILD_BACKEND" | tr '[:upper:]' '[:lower:]')
    local os=$(uname -s | tr '[:upper:]' '[:lower:]')
    local arch=$(uname -m)

    case "$requested" in
        ""|auto)
            if [[ "$os" == "darwin" && ( "$arch" == "arm64" || "$arch" == "aarch64" ) ]]; then
                echo "metal"
            else
                echo "cpu"
            fi
            ;;
        cpu|metal|cuda)
            echo "$requested"
            ;;
        *)
            echo "Unsupported IZWI_BUILD_BACKEND value: $BUILD_BACKEND" >&2
            exit 1
            ;;
    esac
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

    if [[ "$SELECTED_BUILD_BACKEND" == "cuda" ]] && ! command -v nvcc &> /dev/null; then
        echo -e "${RED}CUDA build requested, but nvcc was not found in PATH.${NC}"
        echo -e "${YELLOW}Install the CUDA toolkit first or rerun with IZWI_BUILD_BACKEND=cpu.${NC}"
        exit 1
    fi
    
    echo -e "${GREEN}✓ Requirements satisfied${NC}"
}

install_from_source() {
    echo -e "${BLUE}Building Izwi from source...${NC}"
    echo -e "${BLUE}Selected backend: ${SELECTED_BUILD_BACKEND}${NC}"
    
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

    local cli_feature_args=()
    local server_feature_args=()

    case "$SELECTED_BUILD_BACKEND" in
        metal)
            cli_feature_args=(--features metal)
            ;;
        cuda)
            local cuda_features="${IZWI_CUDA_FEATURES:-cuda}"
            cli_feature_args=(--features "$cuda_features")
            server_feature_args=(--features "$cuda_features")
            ;;
    esac

    # Build release binaries (desktop build expects izwi + izwi-server artifacts to exist)
    cargo build --release --bin izwi "${cli_feature_args[@]}"
    cargo build --release --bin izwi-server "${server_feature_args[@]}"
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
    echo "  izwi version --full      # Show compiled backend support"
    echo "  izwi status --detailed   # Show selected runtime backend"
    echo ""
    echo "Install-time backend selection:"
    echo "  IZWI_BUILD_BACKEND=cpu ./scripts/install-cli.sh"
    echo "  IZWI_BUILD_BACKEND=metal ./scripts/install-cli.sh"
    echo "  IZWI_BUILD_BACKEND=cuda ./scripts/install-cli.sh"
    echo ""
    echo "Documentation: https://github.com/izwi-ai/izwi"
}

main() {
    print_banner

    SELECTED_BUILD_BACKEND="$(resolve_build_backend)"
    
    echo "Platform: $(detect_platform)"
    echo "Install directory: $INSTALL_DIR"
    echo "Binary directory: $BIN_DIR"
    echo "Build backend: $SELECTED_BUILD_BACKEND"
    echo ""
    
    check_requirements
    install_from_source
    setup_shell
    print_usage
}

# Run main function
main "$@"
