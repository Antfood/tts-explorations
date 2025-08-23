#!/usr/bin/env bash
set -euo pipefail

# ------------ Defaults ------------
IMAGE_NAME="${IMAGE_NAME:-tts-preprocess}"
IMAGE_TAG="${IMAGE_TAG:-latest}"
DOCKERHUB_USER="${DOCKERHUB_USER:-pbotsaris}"
GHCR_USER="${GHCR_USER:-$DOCKERHUB_USER}"
PLATFORMS="${PLATFORMS:-linux/amd64}"
DOCKERFILE="${DOCKERFILE:-docker/Dockerfile.preprocess}"
CONTEXT="${CONTEXT:-.}"
BUILD_ARGS="${BUILD_ARGS:-}"
NO_CACHE="${NO_CACHE:-0}"
USE_CACHE="${USE_CACHE:-0}"  # New: Control whether to use cache at all

usage() {
  cat <<'EOF'
Usage: $0 <build|push|clean|prune> [TAG] [FLAVOR]

FLAVOR: preprocess | espnet
Rules:
  - If only one extra arg is given, it must be a FLAVOR.
  - If two extra args are given, they are TAG and FLAVOR, respectively.
  - TAG defaults to "dev" for build and "latest" for push.

Examples:
  ./docker.sh build                 # prompts for flavor, tag=dev
  ./docker.sh build espnet          # flavor=espnet, tag=dev
  ./docker.sh build v0.3.1 espnet   # flavor=espnet, tag=v0.3.1
  ./docker.sh push                  # prompts for flavor, tag=latest
  ./docker.sh push v0.4.0 preprocess
  ./docker.sh clean                 # removes all Docker caches
  ./docker.sh prune                 # prunes buildx cache
  
  # Build without any cache:
  NO_CACHE=1 ./docker.sh build espnet
  
  # Build without using local cache dirs (but still use Docker's internal cache):
  USE_CACHE=0 ./docker.sh build espnet
  
  # Push to both Docker Hub and GitHub Container Registry:
  GHCR=1 ./docker.sh push v0.4.0 espnet

Environment Variables:
  DOCKERHUB_USER    Docker Hub username (default: pbotsaris)
  GHCR_USER         GitHub Container Registry username (default: same as DOCKERHUB_USER)
  IMAGE_NAME        Image name (default: depends on flavor)
  IMAGE_TAG         Default tag for push (default: latest)
  PLATFORMS         Build platforms (default: linux/amd64)
  DOCKERFILE        Path to Dockerfile (default: depends on flavor)
  CONTEXT           Build context path (default: .)
  BUILD_ARGS        Additional build arguments
  NO_CACHE          Set to 1 to disable all caching
  USE_CACHE         Set to 0 to disable local cache dirs (default: 1)
  GHCR              Set to 1 to also push to GitHub Container Registry
EOF
  exit 1
}

is_valid_flavor() { [[ "$1" == "preprocess" || "$1" == "espnet" ]]; }
is_valid_tag() {
  # Allow v1.2.3, 1.2.3, 2025.08.23, 1.0.0-rc1, latest, dev
  [[ "$1" =~ ^(dev|latest|v?[0-9]+(\.[0-9]+)*([._-][A-Za-z0-9]+)?)$ ]]
}

login_check() { 
  docker info >/dev/null 2>&1 || { echo "Docker is not running." >&2; exit 1; }
}

docker_hub_login() {
  echo "Checking Docker Hub login status..."
  if ! docker pull alpine:latest >/dev/null 2>&1; then
    echo "Not logged in to Docker Hub. Logging in..."
    docker login
  else
    echo "Already logged in to Docker Hub."
  fi
}

builder_setup() {
  echo "Setting up Docker buildx..."
  docker buildx create --name antfood --use >/dev/null 2>&1 || docker buildx use antfood
  
  # Only setup cache directories if USE_CACHE=1
  if [[ "$USE_CACHE" = "1" ]]; then
    echo "Setting up cache directories..."
    CACHE_DIR=".buildx-cache"
    CACHE_NEW=".buildx-cache-new"
    mkdir -p "$CACHE_DIR"
    echo "Cache setup complete."
  else
    echo "Skipping cache setup (USE_CACHE=0)"
  fi
}

cache_swap() { 
  if [[ "$USE_CACHE" = "1" && -d "${CACHE_NEW:-}" ]]; then 
    rm -rf "${CACHE_DIR:-}"
    mv "$CACHE_NEW" "$CACHE_DIR"
  fi
}

cleanup_on_fail() { 
  if [[ "$USE_CACHE" = "1" ]]; then
    echo "Build interrupted; cleaning temp cache dir..."
    rm -rf "${CACHE_NEW:-}" || true
  fi
}
trap cleanup_on_fail INT TERM

tag_list() {
  local t="${1:-$IMAGE_TAG}"
  local hub="docker.io/${DOCKERHUB_USER}/${IMAGE_NAME}:${t}"
  local tags=(-t "$hub")
  
  # Also tag without docker.io prefix for local use
  tags+=(-t "${DOCKERHUB_USER}/${IMAGE_NAME}:${t}")
  
  if [[ "${GHCR:-0}" = "1" ]]; then
    local ghcr="ghcr.io/${GHCR_USER}/${IMAGE_NAME}:${t}"
    tags+=(-t "$ghcr")
  fi
  printf '%s\n' "${tags[@]}"
}

build_flags() {
  local flags=(--platform "$PLATFORMS" -f "$DOCKERFILE" "$CONTEXT")
  
  # Add cache flags only if USE_CACHE=1
  if [[ "$USE_CACHE" = "1" ]]; then
    flags+=(--cache-from "type=local,src=$CACHE_DIR"
            --cache-to   "type=local,dest=$CACHE_NEW,mode=max")
  fi
  
  # Add no-cache flag if NO_CACHE=1
  [[ "$NO_CACHE" = "1" ]] && flags+=(--no-cache)
  
  if [[ -n "$BUILD_ARGS" ]]; then
    # shellcheck disable=SC2206
    flags+=($BUILD_ARGS)
  fi
  echo "${flags[@]}"
}

# --- helper to prompt only when interactive ---
prompt_flavor() {
  if [ -t 0 ]; then
    echo "No flavor provided. Please choose one:"
    echo "  1) preprocess"
    echo "  2) espnet"
    read -rp "Enter choice [1-2]: " choice
    case "$choice" in
      1) echo preprocess ;;
      2) echo espnet ;;
      *) echo "Invalid choice." >&2; exit 1 ;;
    esac
  else
    echo "No FLAVOR given and no TTY to prompt. Usage: $0 <cmd> [TAG] <preprocess|espnet>" >&2
    exit 2
  fi
}

clean_all_cache() {
  echo "Cleaning all Docker build caches..."
  
  # Remove local cache directories
  echo "Removing local cache directories..."
  rm -rf .buildx-cache* || true
  
  # Prune buildx cache
  echo "Pruning buildx cache..."
  docker buildx prune --all --force || true
  
  # Clean up dangling images
  echo "Removing dangling images..."
  docker image prune -f || true
  
  # Optional: Remove all build cache (more aggressive)
  echo "Removing all build cache..."
  docker builder prune -a -f || true
  
  echo "Cache cleanup complete!"
  echo ""
  echo "To see disk usage: docker system df"
  echo "For more aggressive cleanup: docker system prune -a"
}

# ------------ Parse ------------
cmd="${1:-}"; arg2="${2:-}"; arg3="${3:-}"
[[ -z "$cmd" ]] && usage

# Special case for commands that don't need flavor
if [[ "$cmd" == "clean" ]]; then
  clean_all_cache
  exit 0
elif [[ "$cmd" == "prune" ]]; then
  echo "Pruning buildx cache..."
  docker buildx prune --all --force
  echo "Prune complete."
  exit 0
elif [[ "$cmd" == "cache" ]]; then
  login_check
  echo "Setting up Docker buildx..."
  docker buildx create --name antfood --use >/dev/null 2>&1 || docker buildx use antfood
  echo "Buildx setup complete."
  exit 0
fi

override_tag=""
flavor=""

if [[ -n "$arg3" ]]; then
  is_valid_tag "$arg2" || { echo "Error: Invalid TAG: '$arg2'"; usage; }
  is_valid_flavor "$arg3" || { echo "Error: Invalid FLAVOR: '$arg3'. Must be 'preprocess' or 'espnet'"; exit 1; }
  override_tag="$arg2"; flavor="$arg3"
elif [[ -n "$arg2" ]]; then
  if is_valid_flavor "$arg2"; then
    flavor="$arg2"
  elif is_valid_tag "$arg2"; then
    override_tag="$arg2"
    flavor="$(prompt_flavor)"
  else
    echo "Error: Invalid argument: '$arg2'. Must be a valid TAG or FLAVOR (preprocess/espnet)"
    exit 1
  fi
else
  # Only prompt for build/push commands
  if [[ "$cmd" == "build" || "$cmd" == "push" ]]; then
    flavor="$(prompt_flavor)"
  else
    echo "Error: Unknown command: '$cmd'"
    usage
  fi
fi

# Flavor-specific defaults
case "$flavor" in
  preprocess)
    IMAGE_NAME="tts-preprocess"
    DOCKERFILE="docker/Dockerfile.preprocess"
    ;;
  espnet)
    IMAGE_NAME="tts-espnet"
    DOCKERFILE="docker/Dockerfile.espnet"
    ;;
esac

# ------------ Commands ------------

case "$cmd" in
  build)
    login_check
    builder_setup
    
    final_tag="${override_tag:-dev}"
    echo "Building (local load)… flavor=$flavor, tag=$final_tag"
    echo "Cache settings: USE_CACHE=$USE_CACHE, NO_CACHE=$NO_CACHE"
    
    docker buildx build $(tag_list "$final_tag") \
      $(build_flags) --load
    
    cache_swap
    echo "Build complete: ${DOCKERHUB_USER}/${IMAGE_NAME}:${final_tag} ($flavor)"
    echo ""
    echo "To push this image to Docker Hub, run:"
    echo "  docker push ${DOCKERHUB_USER}/${IMAGE_NAME}:${final_tag}"
    ;;
    
  push)
    login_check
    docker_hub_login  # Ensure we're logged in before pushing
    builder_setup
    
    final_tag="${override_tag:-$IMAGE_TAG}"
    echo "Building and pushing… flavor=$flavor, tag=$final_tag"
    echo "Cache settings: USE_CACHE=$USE_CACHE, NO_CACHE=$NO_CACHE"
    echo "Pushing to: Docker Hub (${DOCKERHUB_USER}/${IMAGE_NAME}:${final_tag})"
    
    if [[ "${GHCR:-0}" = "1" ]]; then
      echo "Also pushing to: GitHub Container Registry (ghcr.io/${GHCR_USER}/${IMAGE_NAME}:${final_tag})"
    fi
    
    docker buildx build $(tag_list "$final_tag") \
      $(build_flags) --push
    
    cache_swap
    echo ""
    echo "✅ Push complete!"
    echo "Docker Hub: https://hub.docker.com/r/${DOCKERHUB_USER}/${IMAGE_NAME}/tags"
    
    if [[ "${GHCR:-0}" = "1" ]]; then
      echo "GitHub: ghcr.io/${GHCR_USER}/${IMAGE_NAME}:${final_tag}"
    fi
    ;;
    
  *)
    usage
    ;;
esac
