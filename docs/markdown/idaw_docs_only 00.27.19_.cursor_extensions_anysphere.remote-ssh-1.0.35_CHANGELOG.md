# Cursor Remote SSH Changelog

# 1.0.35
- Add support for the `remote.SSH.preconnect` setting, which allows for a script to be executed before each SSH connection
- Manually prompt for the platform if automatic platform detection fails
- Fix a performance issue where spinners continued forever after all reconnection attempts were exhausted

# 1.0.34
- Fix an issue where `remote.SSH.defaultForwardedPorts` could not be configured at the workspace level

# 1.0.33
- Fix an issue where ports specified in the `LocalForward` option of a SSH config file were forwarded with incorrect port numbers

# 1.0.32
- Fix an issue (from version 1.0.31) where connections fail to start

# 1.0.31
- Add the setting `remote.SSH.serverPickPortsFromRange`, which ensures that the remote server ports are in the given range

# 1.0.30
- Lower the default connect timeout to 30 seconds (previously 3 minutes)

# 1.0.29
- Fix issues with X11 forwarding

# 1.0.28
- Bug fixes and improvements

## v1.0.27
- Reduce resource utilization for keeping the SSH tunnel alive
- Add a traffic monitor to monitor remote SSH traffic
- On windows clients, launch the SSH process using a `cmd` shell instead of powershell

## v1.0.26
- Fix an issue where periodic keep-alive pings to the remote command server are interrupted, causing future connection checks to fail.

## v1.0.25
- Fix an issue where automatic reconnections would fail due to failed process termination

## v1.0.24
- Add support for using an interactive terminal when establishing the SSH connection via the setting `remote.SSH.showLoginTerminal`.
- Add the setting `remote.SSH.logLevel`, which can be `trace` or `debug`. Defaults to debug, which is less verbose than previously.
- On failed SSH connections, added buttons to copy the logs to the clipboard and open the SSH config file.

## v1.0.23
- Add the setting `remote.SSH.defaultForwardedPorts`, which allows ports to be forwarded by default on all hosts
- Remove a connection check that caused timeouts and connection failures for otherwise successful connections
- Fix a bug where the `remote.SSH.configFile` argument was not used when calling the `ssh` executable

## v1.0.22
- Fix an issue where the `remote.localPortHost` setting was not respected when forwarding ports
- Add `code` and `cursor` to the `PATH` in the remote environment
- Forward ports for both `IPv4` and `IPv6`, and open ports on `localhost` instead of `127.0.0.1` locally

## v1.0.21
- (macOS / Linux Remotes) Fix stability with SSH agent forwarding.
- (macOS / Linux Remotes) Improve performance of remote server downloads.

## v1.0.20
- (macOS / Linux Remotes) Prefer using `/tmp` for temporary files if `XDG_RUNTIME_DIR` is set but not writable.

## v1.0.19
- (macOS / Linux Remotes) Increase the timeout to 90 seconds for checking the remote server ports

## v1.0.18
- (macOS / Linux Remotes) Move the log, token, and pid files to `/tmp` to eliminate issues when the server installation directory is on a slow disk
- Add the setting `remote.SSH.enableRemoteCommand`
- Add support for FreeBSD

## v1.0.17
- (macOS / Linux Remotes) Continue with establishing the connection even if the multiplex server fails to start. This feature is only required for Docker over SSH.

## v1.0.16
- Add config option `remote.SSH.serverShutdownTimeout` to change the default server shutdown timeout, after which the remote server will terminate due to inactivity. Default is 5 minutes (300 seconds). This setting requires Cursor 1.2 or greater.

## v1.0.15
- (Linux Clients) Explicitly unset 'ARGV0' when launching the SSH command to fix an issue with multiple jump hosts

## v1.0.14
- Refactor the reconnection logic to attempt to reuse existing SSH connections or socks connections, instead of always creating new ones.
- (Windows Remotes) Run Powershell with `-NoProfile`

## v1.0.13
- (macOS / Linux Remotes) Configured the installation script to run in a non-login shell

## v1.0.12
- (macOS / Linux Remotes) Fix a newline encoding issue that prevented the server from starting

## v1.0.11
- (macOS / Linux Clients): Always use a shell to launch the SSH client, and pass the command over stdin. Fixes issues where some hosts reject or truncate long command strings.
- Copy files over SSH instead instead of SCP

## v1.0.10
- (macOS / Linux Remotes) Fix an issue where the install script was running in a login shell, inheriting all bash options
- (macOS / Linux Remotes) Fix an issue where logfiles and lockfiles were not properly versioned

## v1.0.9
- (Windows Remotes) Fixed an issue where the fallback to SCP on failed downloads was not automatic

## v1.0.8
- Add an explicit error message when trying to use VSCode remote containers with Anysphere Remote SSH, as they are incompatible with each other
- Fallback to SCP'ing the server when the remote server fails to download directly

## v1.0.7
- Fix a bug where the Docker over SSH connection would terminate automatically after 5 minutes

## v1.0.6
- Remove the "Select Remote Platform" prompt for initial connection attempts. The new flow assumes that the remote system is running Linux/macOS
  and will only prompt if the initial connection attempt failed
- Show progress on remote server downloads
- Reset the connection timeout so long as downloads are making progress.

## v1.0.5
- (macOS / Linux Remotes) Automatically clear the lock file if it was left behind by a previous, failed connection attempt
- (macOS / Linux Remotes) Add a "Generate Connection Report" command to help diagnose connection issues
- (Windows clients) Replaced the SSH Askpass helper with a .bat script to resolve warnings from antivirus programs
- (Windows clients) Hide console host windows that appeared when using a SSH proxy or jump host


## v1.0.4
- Fix a bug, when using Remote Containers over SSH, spawned commands were not terminated when the window closed

## v1.0.3
- Fix a bug where private key arguments were not supported when manually pasting a connection string

## v1.0.2
- Add support for forwarding ports via SOCKS through the execServer. Helps improve the performance for Remote Containers over SSH (requires Anysphere Remote Containers v1.0.2 or greater)

## v1.0.1
- Fix an issue (introduced in v0.0.34) when connecting to remote hosts that default to Fish shells

## v1.0.0
- Simplified README

## v0.0.34
- Add support for Alpine linux remote extension hosts. Requires Cursor v0.50.5 or greater.

## v0.0.33
- Add support for cancelling connection attempts

## v0.0.32
- Upon temporary server disconnections, continuously attempt to reconnect for up to 5 minutes

## v0.0.31
- Fix an issue where the server fails to start on macOS remote hosts, due to a base64 decoding issue

## v0.0.30
- If the bundled NodeJS executable fails to run, fallback to using the system NodeJS executable (if available) and the system NodeJS version >= 20

## v0.0.29
- Increased default connection timeout to 180 seconds (up from 60 seconds)
- Fixed a race condition when starting the remote server, where it would fail to start on slow filesystems

## v0.0.28
- Implemented support for using a SSH remote as an intermediate resolver for connecting to a remote system

## v0.0.027
- Fixed an issue with server binaries failing to download

## v0.0.26
- Fixed a bug where SSH connection strings containing ports were not parsed properly in the "Remote-SSH: Connect to Host" command palette box

## v0.0.25
- Renamed `remote.SSH.serverDataFolderPath` to `remote.SSH.serverInstallPath` to match the `ms-vscode-remote.remote-ssh` extension.
- Fixed an bug where the `remote.SSH.serverInstallPath` setting did not apply to the `extension` and `data` sub-folders.
  Now, all remote server artifacts respect this setting.

## v0.0.24
- Added config options `remote.SSH.httpProxy`, `remote.SSH.httpsProxy`, and `remote.SSH.noProxy`, which will be used when downloading the remote server
  and during the remote sessions.

## v0.0.23

- Added prompt to reinstall the server on failed connections
- Added Kill Server and Reload Window Command
- Added Reinstall Server and Reload Window Command
- Moved cleanup of old server binaries to after the new server successfully launches
- Added config option `remote.SSH.serverDataFolderPath` for customizing the location of the remote server data folder

## v0.0.22

- Added telemetry (enabled when privacy mode is disabled)

## v0.0.21

- Switched installer script encoding to base64 to support `csh` SSH shells (for macOS / linux remote hosts)

## v0.0.20

- Added support for port forwarding
