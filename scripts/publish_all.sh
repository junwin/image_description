#!/usr/bin/env bash
#
# publish_all.sh — publish one image (and its JSON sidecar) to every
# supported social platform with a single call.
#
# Platforms: Mastodon, Pixelfed, Bluesky, Tumblr, Facebook, Instagram.
#
# Usage:
#   publish_all.sh <image> [options]
#
set -uo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

usage() {
  cat <<'EOF'
Usage: publish_all.sh <image> [options]

Publish one image (and its JSON sidecar) to all supported platforms:
Mastodon, Pixelfed, Bluesky, Tumblr, Facebook, Instagram.

Options:
  --dry-run               Preview only; publish to nothing.
  --text <text>           Extra caption text (ignored by Pixelfed).
  --caption-field <name>  Sidecar field to use for the caption.
  --visibility <vis>      Mastodon visibility: public|unlisted|private|direct.
  --image-url <url>       Public HTTPS JPEG for Instagram (default: --ngrok tunnel).

Notes:
  - The image must be a JPEG for Facebook and Instagram.
  - Instagram uses an ngrok tunnel unless --image-url is supplied.
EOF
}

IMAGE=""
DRY_RUN=""
TEXT=""
CAPTION_FIELD=""
VISIBILITY=""
IMAGE_URL=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run)
      DRY_RUN="--dry-run"
      shift
      ;;
    --text)
      TEXT="${2:-}"
      shift 2
      ;;
    --caption-field)
      CAPTION_FIELD="${2:-}"
      shift 2
      ;;
    --visibility)
      VISIBILITY="${2:-}"
      shift 2
      ;;
    --image-url)
      IMAGE_URL="${2:-}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    -*)
      echo "Unknown option: $1" >&2
      usage
      exit 2
      ;;
    *)
      if [[ -z "$IMAGE" ]]; then
        IMAGE="$1"
      else
        echo "Unexpected extra argument: $1" >&2
        usage
        exit 2
      fi
      shift
      ;;
  esac
done

if [[ -z "$IMAGE" ]]; then
  echo "Error: an image path is required." >&2
  usage
  exit 2
fi

if [[ ! -f "$IMAGE" ]]; then
  echo "Error: image not found: $IMAGE" >&2
  exit 1
fi

case "${IMAGE,,}" in
  *.mp4|*.mov|*.mkv|*.avi|*.webm|*.m4v)
    echo "Error: $IMAGE looks like a video. Reels are published separately with instagram_reels_cli." >&2
    exit 2
    ;;
esac

cd "$REPO_DIR" || exit 1
# shellcheck disable=SC1091
source .venv/bin/activate

# Optional passthrough arguments (kept as arrays so empty args are safe).
TEXT_ARGS=()
[[ -n "$TEXT" ]] && TEXT_ARGS=(--text "$TEXT")
CAPTION_ARGS=()
[[ -n "$CAPTION_FIELD" ]] && CAPTION_ARGS=(--caption-field "$CAPTION_FIELD")
VIS_ARGS=()
[[ -n "$VISIBILITY" ]] && VIS_ARGS=(--visibility "$VISIBILITY")
DRY_ARGS=()
[[ -n "$DRY_RUN" ]] && DRY_ARGS=(--dry-run)

FAILURES=0

publish() {
  local label="$1"
  local module="$2"
  shift 2
  echo ""
  echo "=== $label ==="
  python -m "src.image_description.cli.$module" "$@"
  local rc=$?
  if [[ $rc -eq 0 ]]; then
    echo "OK: $label"
  else
    echo "FAILED: $label (exit $rc)" >&2
    FAILURES=$((FAILURES + 1))
  fi
}

publish "Mastodon"  mastodon_cli  post "$IMAGE" "${DRY_ARGS[@]}" "${TEXT_ARGS[@]}" "${CAPTION_ARGS[@]}" "${VIS_ARGS[@]}"
publish "Pixelfed"  pixelfed_cli  post "$IMAGE" "${DRY_ARGS[@]}" "${CAPTION_ARGS[@]}"
publish "Bluesky"   bluesky_cli   post "$IMAGE" "${DRY_ARGS[@]}" "${TEXT_ARGS[@]}" "${CAPTION_ARGS[@]}"
publish "Tumblr"    tumblr_cli    post "$IMAGE" "${DRY_ARGS[@]}" "${TEXT_ARGS[@]}" "${CAPTION_ARGS[@]}"
publish "Facebook"  facebook_cli  post "$IMAGE" "${DRY_ARGS[@]}" "${TEXT_ARGS[@]}" "${CAPTION_ARGS[@]}"

IG_SOURCE_ARGS=()
if [[ -n "$IMAGE_URL" ]]; then
  IG_SOURCE_ARGS=(--image-url "$IMAGE_URL")
else
  IG_SOURCE_ARGS=(--ngrok)
fi
publish "Instagram" instagram_cli post "$IMAGE" "${DRY_ARGS[@]}" "${IG_SOURCE_ARGS[@]}" "${TEXT_ARGS[@]}" "${CAPTION_ARGS[@]}"

echo ""
echo "=== Summary ==="
if [[ $FAILURES -eq 0 ]]; then
  echo "All platforms published successfully."
  exit 0
else
  echo "$FAILURES platform(s) failed." >&2
  exit 1
fi
