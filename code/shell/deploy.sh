#!/bin/bash
# =============================================================================
# iDAW Deployment Script
# Deploys the application to various targets
# =============================================================================

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

# =============================================================================
# Parse Arguments
# =============================================================================

DEPLOY_TARGET=""
VERSION=""
DRY_RUN=false

print_usage() {
    echo "Usage: $0 [TARGET] [OPTIONS]"
    echo ""
    echo "Targets:"
    echo "  docker        Build and push Docker image"
    echo "  github        Create GitHub release"
    echo "  local         Deploy locally for testing"
    echo "  server        Deploy to remote server via SSH"
    echo ""
    echo "Options:"
    echo "  --version     Specify version (default: from package.json)"
    echo "  --dry-run     Show what would be done without doing it"
    echo "  --help        Show this help message"
}

while [[ $# -gt 0 ]]; do
    case $1 in
        docker|github|local|server)
            DEPLOY_TARGET="$1"
            shift
            ;;
        --version)
            VERSION="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --help)
            print_usage
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            print_usage
            exit 1
            ;;
    esac
done

if [ -z "$DEPLOY_TARGET" ]; then
    echo -e "${RED}Error: No deployment target specified${NC}"
    print_usage
    exit 1
fi

# Get version from package.json if not specified
if [ -z "$VERSION" ]; then
    VERSION=$(grep '"version"' package.json | head -1 | awk -F'"' '{print $4}')
fi

echo -e "${BLUE}=====================================================${NC}"
echo -e "${BLUE}  iDAW Deployment${NC}"
echo -e "${BLUE}=====================================================${NC}"
echo ""
echo -e "Target:  ${YELLOW}$DEPLOY_TARGET${NC}"
echo -e "Version: ${YELLOW}$VERSION${NC}"
echo -e "Dry Run: ${YELLOW}$DRY_RUN${NC}"
echo ""

# =============================================================================
# Deploy Functions
# =============================================================================

deploy_docker() {
    echo -e "${YELLOW}Deploying to Docker...${NC}"

    # Check for Docker
    if ! command -v docker &> /dev/null; then
        echo -e "${RED}✗ Docker not found${NC}"
        exit 1
    fi

    DOCKER_REGISTRY="${DOCKER_REGISTRY:-}"
    DOCKER_IMAGE="${DOCKER_IMAGE:-idaw}"
    DOCKER_TAG="$DOCKER_IMAGE:$VERSION"

    if [ -n "$DOCKER_REGISTRY" ]; then
        DOCKER_TAG="$DOCKER_REGISTRY/$DOCKER_TAG"
    fi

    echo "Building Docker image: $DOCKER_TAG"

    if [ "$DRY_RUN" = true ]; then
        echo "[DRY RUN] Would run: docker build -t $DOCKER_TAG ."
        echo "[DRY RUN] Would run: docker push $DOCKER_TAG"
    else
        # Build image
        docker build -t "$DOCKER_TAG" .
        docker tag "$DOCKER_TAG" "$DOCKER_IMAGE:latest"

        echo -e "${GREEN}✓ Docker image built: $DOCKER_TAG${NC}"

        # Push if registry is configured
        if [ -n "$DOCKER_REGISTRY" ]; then
            echo "Pushing to registry..."
            docker push "$DOCKER_TAG"
            docker push "$DOCKER_IMAGE:latest"
            echo -e "${GREEN}✓ Image pushed to registry${NC}"
        fi
    fi
}

deploy_github() {
    echo -e "${YELLOW}Creating GitHub release...${NC}"

    # Check for gh CLI
    if ! command -v gh &> /dev/null; then
        echo -e "${RED}✗ GitHub CLI (gh) not found${NC}"
        echo "Install with: brew install gh"
        exit 1
    fi

    # Check authentication
    if ! gh auth status &> /dev/null; then
        echo -e "${RED}✗ Not authenticated with GitHub${NC}"
        echo "Run: gh auth login"
        exit 1
    fi

    RELEASE_TAG="v$VERSION"
    RELEASE_TITLE="iDAW $VERSION"

    # Check if release directory exists
    if [ ! -d "release" ]; then
        echo -e "${RED}✗ Release directory not found. Run build_all.sh first.${NC}"
        exit 1
    fi

    if [ "$DRY_RUN" = true ]; then
        echo "[DRY RUN] Would create release: $RELEASE_TAG"
        echo "[DRY RUN] Would upload files from: release/"
    else
        # Create tag if it doesn't exist
        if ! git rev-parse "$RELEASE_TAG" &> /dev/null; then
            echo "Creating git tag: $RELEASE_TAG"
            git tag -a "$RELEASE_TAG" -m "Release $VERSION"
            git push origin "$RELEASE_TAG"
        fi

        # Create release
        echo "Creating GitHub release..."
        gh release create "$RELEASE_TAG" \
            --title "$RELEASE_TITLE" \
            --notes "## iDAW $VERSION

### Changes
- See commit history for details

### Downloads
- macOS: Download the .dmg file
- Windows: Download the .msi file
- Linux: Download the .deb or .AppImage file

### Installation
1. Download the appropriate file for your platform
2. Run the installer
3. Launch iDAW from your applications" \
            release/* 2>/dev/null || echo "Some files may have failed to upload"

        echo -e "${GREEN}✓ GitHub release created: $RELEASE_TAG${NC}"
        echo "View at: https://github.com/$(gh repo view --json nameWithOwner -q .nameWithOwner)/releases/tag/$RELEASE_TAG"
    fi
}

deploy_local() {
    echo -e "${YELLOW}Deploying locally...${NC}"

    LOCAL_INSTALL_DIR="${LOCAL_INSTALL_DIR:-$HOME/Applications}"

    if [ "$DRY_RUN" = true ]; then
        echo "[DRY RUN] Would install to: $LOCAL_INSTALL_DIR"
    else
        mkdir -p "$LOCAL_INSTALL_DIR"

        # Detect platform and install
        case "$(uname -s)" in
            Darwin)
                # macOS
                DMG_FILE=$(find release -name "*.dmg" | head -1)
                if [ -n "$DMG_FILE" ]; then
                    echo "Installing from: $DMG_FILE"
                    hdiutil attach "$DMG_FILE" -quiet
                    APP_NAME=$(ls /Volumes/ | grep -i idaw | head -1)
                    if [ -n "$APP_NAME" ]; then
                        cp -r "/Volumes/$APP_NAME"/*.app "$LOCAL_INSTALL_DIR/"
                        hdiutil detach "/Volumes/$APP_NAME" -quiet
                        echo -e "${GREEN}✓ Installed to $LOCAL_INSTALL_DIR${NC}"
                    fi
                else
                    echo -e "${YELLOW}⚠ No DMG file found in release/${NC}"
                fi
                ;;
            Linux)
                # Linux
                DEB_FILE=$(find release -name "*.deb" | head -1)
                if [ -n "$DEB_FILE" ]; then
                    echo "Installing from: $DEB_FILE"
                    sudo dpkg -i "$DEB_FILE"
                    echo -e "${GREEN}✓ Installed via dpkg${NC}"
                else
                    APPIMAGE=$(find release -name "*.AppImage" | head -1)
                    if [ -n "$APPIMAGE" ]; then
                        cp "$APPIMAGE" "$LOCAL_INSTALL_DIR/"
                        chmod +x "$LOCAL_INSTALL_DIR/$(basename "$APPIMAGE")"
                        echo -e "${GREEN}✓ AppImage copied to $LOCAL_INSTALL_DIR${NC}"
                    fi
                fi
                ;;
            *)
                echo -e "${YELLOW}⚠ Manual installation required for this platform${NC}"
                ;;
        esac
    fi
}

deploy_server() {
    echo -e "${YELLOW}Deploying to remote server...${NC}"

    # Required environment variables
    if [ -z "$DEPLOY_HOST" ]; then
        echo -e "${RED}✗ DEPLOY_HOST not set${NC}"
        echo "Set with: export DEPLOY_HOST=user@hostname"
        exit 1
    fi

    DEPLOY_PATH="${DEPLOY_PATH:-/opt/idaw}"

    if [ "$DRY_RUN" = true ]; then
        echo "[DRY RUN] Would deploy to: $DEPLOY_HOST:$DEPLOY_PATH"
        echo "[DRY RUN] Would run: ssh $DEPLOY_HOST 'cd $DEPLOY_PATH && docker-compose up -d'"
    else
        echo "Copying files to $DEPLOY_HOST..."

        # Copy necessary files
        scp docker-compose.yml "$DEPLOY_HOST:$DEPLOY_PATH/"
        scp Dockerfile "$DEPLOY_HOST:$DEPLOY_PATH/"
        rsync -avz --exclude 'node_modules' --exclude 'venv' --exclude 'target' \
            . "$DEPLOY_HOST:$DEPLOY_PATH/"

        # Deploy with Docker Compose
        echo "Starting services..."
        ssh "$DEPLOY_HOST" "cd $DEPLOY_PATH && docker-compose pull && docker-compose up -d"

        echo -e "${GREEN}✓ Deployed to $DEPLOY_HOST${NC}"

        # Health check
        echo "Running health check..."
        sleep 5
        if ssh "$DEPLOY_HOST" "curl -s http://localhost:8000/health" | grep -q "healthy"; then
            echo -e "${GREEN}✓ Service is healthy${NC}"
        else
            echo -e "${YELLOW}⚠ Health check failed - check logs${NC}"
        fi
    fi
}

# =============================================================================
# Run Deployment
# =============================================================================

case $DEPLOY_TARGET in
    docker)
        deploy_docker
        ;;
    github)
        deploy_github
        ;;
    local)
        deploy_local
        ;;
    server)
        deploy_server
        ;;
esac

echo ""
echo -e "${GREEN}=====================================================${NC}"
echo -e "${GREEN}  Deployment Complete!${NC}"
echo -e "${GREEN}=====================================================${NC}"
