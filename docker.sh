#!/usr/bin/env bash
set -euo pipefail

# Enhanced build+push script with cache management
# Supports cache options, size limits, and cleanup

usage() {
  echo "Usage: $0 <preprocess|espnet> [TAG] [OPTIONS]"
  echo ""
  echo "Arguments:"
  echo "  FLAVOR    Build flavor: 'preprocess' or 'espnet'"
  echo "  TAG       Image tag (default: latest)"
  echo ""
  echo "Options:"
  echo "  --no-cache           Disable build cache (slower but clean)"
  echo "  --cache-size SIZE    Set buildx cache size limit (e.g., 10GB, 5GB)"
  echo "  --cleanup-after      Clean buildx cache after build"
  echo "  --prune-before       Prune buildx cache before build"
  echo "  --show-cache         Show current cache usage"
  echo "  -h, --help           Show this help"
  echo ""
  echo "Examples:"
  echo "  $0 preprocess latest                    # Use default cache"
  echo "  $0 preprocess latest --no-cache         # No cache, clean build"
  echo "  $0 preprocess latest --cache-size 8GB   # Limit cache to 8GB"
  echo "  $0 preprocess --cleanup-after           # Clean up after build"
  echo "  $0 --show-cache                        # Just show cache status"
  echo "  $0 --prune-before preprocess           # Clean before building"
  exit 1
}

# Parse arguments
FLAVOR=""
TAG="latest"
USE_CACHE=true
CACHE_SIZE=""
CLEANUP_AFTER=false
PRUNE_BEFORE=false
SHOW_CACHE=false

while [[ $# -gt 0 ]]; do
  case $1 in
    preprocess|espnet)
      FLAVOR="$1"
      shift
      ;;
    --no-cache)
      USE_CACHE=false
      shift
      ;;
    --cache-size)
      CACHE_SIZE="$2"
      shift 2
      ;;
    --cleanup-after)
      CLEANUP_AFTER=true
      shift
      ;;
    --prune-before)
      PRUNE_BEFORE=true
      shift
      ;;
    --show-cache)
      SHOW_CACHE=true
      shift
      ;;
    -h|--help)
      usage
      ;;
    -*)
      echo "Unknown option: $1"
      usage
      ;;
    *)
      if [[ -z "$TAG" || "$TAG" == "latest" ]]; then
        TAG="$1"
      else
        echo "Unexpected argument: $1"
        usage
      fi
      shift
      ;;
  esac
done

# Function to show cache status
show_cache_status() {
  echo "📊 Current Docker cache usage:"
  docker system df
  echo ""
  echo "🔧 Buildx builders:"
  docker buildx ls
  echo ""
}

# Function to setup cache-limited builder
setup_cache_builder() {
  local size="$1"
  local builder_name="limited-cache-builder"
  
  echo "🔧 Setting up buildx builder with ${size} cache limit..."
  
  # Remove existing builder if it exists
  if docker buildx ls | grep -q "$builder_name"; then
    docker buildx rm "$builder_name" >/dev/null 2>&1 || true
  fi
  
  # Create new builder with size limit
  docker buildx create --name "$builder_name" \
    --driver-opt env.BUILDKIT_CACHE_MAX_SIZE="$size" >/dev/null
  
  docker buildx use "$builder_name"
  echo "✅ Using builder '$builder_name' with ${size} cache limit"
}

# Function to cleanup cache
cleanup_cache() {
  echo "🧹 Cleaning up buildx cache..."
  if [[ "$1" == "full" ]]; then
    docker buildx prune -a --all -f
    echo "✅ Full cache cleanup completed"
  else
    docker buildx prune -f --keep-storage "${1:-5GB}"
    echo "✅ Cache pruned, keeping ${1:-5GB}"
  fi
}

# Show cache if requested
if [[ "$SHOW_CACHE" == true ]]; then
  show_cache_status
  if [[ -z "$FLAVOR" ]]; then
    exit 0
  fi
fi

# Validate flavor
if [[ -z "$FLAVOR" ]]; then
  echo "Error: must specify flavor (preprocess or espnet)"
  usage
fi

if [[ "$FLAVOR" != "preprocess" && "$FLAVOR" != "espnet" ]]; then
  echo "Error: flavor must be 'preprocess' or 'espnet'"
  usage
fi

# Setup environment
export DOCKER_BUILDKIT=1

# Resolve repo root relative to this script
SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR" && (git rev-parse --show-toplevel 2>/dev/null || pwd))"
DOCKERHUB_USER="${DOCKERHUB_USER:-pbotsaris}"

# Set dockerfile and image name
case "$FLAVOR" in
  preprocess)
    IMAGE_NAME="tts-preprocess"
    DOCKERFILE="$REPO_ROOT/docker/Dockerfile.preprocess"
    ;;
  espnet)
    IMAGE_NAME="tts-espnet"
    DOCKERFILE="$REPO_ROOT/docker/Dockerfile.espnet"
    ;;
esac

# Sanity check
[[ -f "$DOCKERFILE" ]] || { echo "Missing Dockerfile: $DOCKERFILE"; exit 2; }

# Show initial cache status
show_cache_status

# Prune before if requested
if [[ "$PRUNE_BEFORE" == true ]]; then
  read -p "🗑️  Prune cache before build? [y/N]: " -n 1 -r
  echo ""
  if [[ $REPLY =~ ^[Yy]$ ]]; then
    cleanup_cache "5GB"
  fi
fi

# Setup cache-limited builder if size specified
if [[ -n "$CACHE_SIZE" ]]; then
  setup_cache_builder "$CACHE_SIZE"
fi

# Build configuration
BUILD_ARGS=(
  --platform linux/amd64
  -t "${DOCKERHUB_USER}/${IMAGE_NAME}:${TAG}"
  -f "$DOCKERFILE"
  "$REPO_ROOT"
  --push
)

# Add no-cache if requested
if [[ "$USE_CACHE" == false ]]; then
  BUILD_ARGS+=(--no-cache)
fi

# Build summary
echo "🚀 Building and pushing..."
echo "  flavor:      $FLAVOR"
echo "  tag:         $TAG"
echo "  dockerfile:  $DOCKERFILE"
echo "  context:     $REPO_ROOT"
echo "  image:       ${DOCKERHUB_USER}/${IMAGE_NAME}:${TAG}"
echo "  use cache:   $USE_CACHE"
if [[ -n "$CACHE_SIZE" ]]; then
  echo "  cache limit: $CACHE_SIZE"
fi
echo ""

# Confirm build
if [[ "$USE_CACHE" == false ]] || [[ -n "$CACHE_SIZE" ]]; then
  read -p "Continue with build? [Y/n]: " -n 1 -r
  echo ""
  if [[ $REPLY =~ ^[Nn]$ ]]; then
    echo "Build cancelled"
    exit 0
  fi
fi

# Execute build
echo "⏱️  Starting build..."
set -x
docker buildx build "${BUILD_ARGS[@]}"
set +x

echo ""
echo "✅ Build completed successfully!"
echo "📦 Pushed to Docker Hub:"
echo "    https://hub.docker.com/r/${DOCKERHUB_USER}/${IMAGE_NAME}/tags"

# Show final cache status
echo ""
show_cache_status

# Cleanup after if requested
if [[ "$CLEANUP_AFTER" == true ]]; then
  echo ""
  read -p "🧹 Clean up cache now? [y/N]: " -n 1 -r
  echo ""
  if [[ $REPLY =~ ^[Yy]$ ]]; then
    read -p "   Full cleanup or keep some cache? [f/k]: " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Ff]$ ]]; then
      cleanup_cache "full"
    else
      read -p "   How much to keep (e.g., 5GB): " KEEP_SIZE
      cleanup_cache "$KEEP_SIZE"
    fi
    
    # Show final status
    echo ""
    show_cache_status
  fi
fi

echo ""
echo "🎉 All done!"
