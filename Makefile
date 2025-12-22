.PHONY: all minimal standard full custom clean install help

# Default target
all: standard

# Build profiles
minimal: 
	@echo "Building minimal version..."
	@./build.sh profiles/minimal.profile

standard:
	@echo "Building standard version..."
	@./build.sh profiles/standard.profile

full:
	@echo "Building full-featured version..."
	@./build.sh profiles/full.profile

custom:
	@echo "Building custom version..."
	@./build.sh profiles/custom.profile

# Clean build artifacts
clean: 
	@echo "Cleaning dist directory..."
	@rm -rf dist/*
	@echo "Done."

# Install to ~/bin or /usr/local/bin
install: standard
	@echo "Installing git-update to ~/bin..."
	@mkdir -p ~/bin
	@cp dist/git-update.sh ~/bin/git-update
	@chmod +x ~/bin/git-update
	@echo "Done. Make sure ~/bin is in your PATH."

install-system: standard
	@echo "Installing git-update to /usr/local/bin (requires sudo)..."
	@sudo cp dist/git-update.sh /usr/local/bin/git-update
	@sudo chmod +x /usr/local/bin/git-update
	@echo "Done."

# Build all profiles
all-profiles: minimal standard full custom
	@echo "All profiles built successfully."

# Help
help:
	@echo "Git Updater Build System"
	@echo ""
	@echo "Targets:"
	@echo "  minimal       - Build minimal version (core only)"
	@echo "  standard      - Build standard version (recommended)"
	@echo "  full          - Build full-featured version (all modules)"
	@echo "  custom        - Build custom version (edit profiles/custom.profile)"
	@echo "  all-profiles  - Build all profiles"
	@echo "  clean         - Remove all built files"
	@echo "  install       - Install standard version to ~/bin"
	@echo "  install-system- Install standard version to /usr/local/bin"
	@echo "  help          - Show this help message"
	@echo ""
	@echo "Examples:"
	@echo "  make              # Build standard version"
	@echo "  make full         # Build full version"
	@echo "  make install      # Build and install to ~/bin"