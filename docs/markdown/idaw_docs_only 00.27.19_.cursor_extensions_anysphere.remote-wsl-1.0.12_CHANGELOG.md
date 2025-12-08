# Cursor Remote WSL Changelog

# 1.0.12
- Fix a performance issue where spinners continued forever after all reconnection attempts were exhausted

## 1.0.11
- Bug fixes and improvements

## v1.0.10
- Improve remote server download performance when launching Cursor via the WSL terminal. Requires Cursor 1.5.0+.
- Added a "Reveal in Explorer" option in the context menu.

## v1.0.9
- Adjusted timeouts to prevent connection failures for otherwise successful connections
- Fix an issue when using Docker over WSL with a non-default WSL distribution

## v1.0.8
- Source `~/.cursor-server/server-env-setup` (or, if that doesn't exist, `~/.vscode-server/server-env-setup`) when starting the server.
- When using Docker over WSL, fix an issue where symlinked `.gitconfig` and SSH known host files were ignored (requires Anysphere Remote Containers 1.0.16)

## v1.0.7
- Support shutting down docker containers when closing the editor (requires Anysphere Remote Containers version 1.0.13 or greater)

## v1.0.6
- Fix an issue where output from startup files would interfere with starting Docker over WSL.
- Switch to using non-login, non-interactive shells for server installation.
- Improve performance of remote server downloads and installations.

## v1.0.5
- Fix an issue where Docker over WSL did not respect environment variables specified in a `.bashrc` file, such as `SSH_AUTH_SOCK`.

## v1.0.4
- Support Docker inside WSL environments. Requires the Anysphere Remote Containers version 1.0.8 or greater.

## v1.0.3
- Bug fixes and improvements

## v1.0.2
- Add "Open Folder in WSL" command

## v1.0.1
- Fix an issue where launching Cursor from with WSL shell opened in a non-remote window. Now, running the `code` or `cursor` from within a WSL directory (in a WSL shell) will open as a remote window.

## v1.0.0
- Bug fixes and improvements

## v0.0.11

- Added prompt to reinstall the server on failed connections
- Removed the clean install when a stuck connection requires a server reboot and window reload. Now, just the running servers are killed. The server binaries are left in place.
- Added Kill Server and Reload Window Command
- Added Reinstall Server and Reload Window Command
- Added cleanup of old server binaries


## v0.0.10

- Added telemetry (enabled when privacy mode is disabled)


## v0.0.9

- Updated wording in error message
