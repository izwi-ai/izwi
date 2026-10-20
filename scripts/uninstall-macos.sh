#!/usr/bin/env bash
# Izwi macOS full uninstall helper.
# Removes app bundle, CLI links, and local user data.

set -euo pipefail

APP_PATH="/Applications/Izwi.app"
IDENTIFIER="com.agentem.izwi.desktop"

PATH_LINKS=(
  "/opt/homebrew/bin/izwi"
  "/opt/homebrew/bin/izwi-server"
  "/usr/local/bin/izwi"
  "/usr/local/bin/izwi-server"
)

DB_FILES=(
  "$HOME/Library/Application Support/izwi/izwi.sqlite3"
  "$HOME/Library/Application Support/izwi/izwi.sqlite3-wal"
  "$HOME/Library/Application Support/izwi/izwi.sqlite3-shm"
)

MEDIA_PATHS=(
  "$HOME/Library/Application Support/izwi/media"
)

DATA_PATHS=(
  "$HOME/Library/Application Support/izwi"
  "$HOME/Library/Application Support/$IDENTIFIER"
  "$HOME/Library/Saved Application State/$IDENTIFIER.savedState"
  "$HOME/Library/Caches/$IDENTIFIER"
  "$HOME/Library/Caches/izwi"
  "$HOME/Library/WebKit/$IDENTIFIER"
  "$HOME/Library/HTTPStorages/$IDENTIFIER"
  "$HOME/Library/Preferences/$IDENTIFIER.plist"
  "$HOME/Library/Logs/$IDENTIFIER"
)

# "Open at login" (the autostart plugin's LaunchAgent, named after the app)
LAUNCH_AGENT="$HOME/Library/LaunchAgents/Izwi.plist"

echo "Stopping running Izwi processes..."
pkill -f "izwi-server|izwi serve|izwi-desktop|/Applications/Izwi\\.app|Izwi\\.app" >/dev/null 2>&1 || true

if [ -f "$LAUNCH_AGENT" ]; then
  echo "Removing the open-at-login item..."
  launchctl unload "$LAUNCH_AGENT" >/dev/null 2>&1 || true
  rm -f "$LAUNCH_AGENT"
fi

if [ -d "$APP_PATH" ]; then
  echo "Removing app bundle: $APP_PATH"
  sudo rm -rf "$APP_PATH"
fi

echo "Removing CLI links..."
for link in "${PATH_LINKS[@]}"; do
  if [ -L "$link" ]; then
    sudo rm -f "$link"
  fi
done

echo "Removing SQLite database files..."
for db_file in "${DB_FILES[@]}"; do
  rm -f "$db_file"
done

echo "Removing saved media files..."
for media_path in "${MEDIA_PATHS[@]}"; do
  rm -rf "$media_path"
done

echo "Removing user data..."
for path in "${DATA_PATHS[@]}"; do
  rm -rf "$path"
done

echo "Done. Izwi has been removed from this macOS user account."
