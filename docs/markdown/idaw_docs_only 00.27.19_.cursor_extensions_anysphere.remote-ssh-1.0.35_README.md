# Cursor Remote - SSH

Connect to any remote machine with an SSH server and use it as your development environment. This extension enables seamless remote development with powerful features.

## Features

- Develop on remote machines with different operating systems or specialized hardware
- Switch between remote environments safely without affecting your local machine
- Access your development environment from any location
- Debug applications running on remote servers or cloud environments

All development work happens directly on the remote machine - no local source code required. Work with remote folders just as you would with local ones.

## Requirements

### Supported Platforms

- **Linux**: Debian 8+, Ubuntu 20.04+, CentOS/RHEL 7+, Alpine (*), FreeBSD (**).
- **Windows**: Windows 10+/Server 2016/2019 (1803+) with [OpenSSH Server](https://docs.microsoft.com/windows-server/administration/openssh/openssh_install_firstuse)
- **macOS**: 10.14+ (Mojave) with [Remote Login enabled](https://support.apple.com/guide/mac-help/allow-a-remote-computer-to-access-your-mac-mchlp1066/mac)

### System Requirements

- Remote host must have:
  - `bash` (macOS/Linux) or `powershell` (Windows)
  - `wget` or `curl`
  - SSH server with TCP Forwarding support

### *Alpine Linux Specific Notes

1. Requires Cursor v0.50.5 or newer
2. Install required packages:
   ```bash
   apk add bash libstdc++ openssh wget
   ```
3. Enable port forwarding:
   - Edit `/etc/ssh/sshd_config`
   - Set `AllowTcpForwarding yes`
   - Restart SSH: `service sshd restart`

### **FreeBSD Specific Notes

Cursor on FreeBSD requires `bash`, `wget`, and Linux Binary Compatibility.

1. Requires Cursor v0.50.5 or newer
2. Install required packages:
   ```bash
   sudo sysrc linux_enable="YES"
   sudo service linux start
   sudo pkg install bash wget linux_base-rl9
   ```

## Opening Remote Folders via the CLI

It is possible to open workspaces on a work machine directly via the `cursor` CLI via the following command:

```bash
cursor --folder-uri vscode-remote://ssh-remote+<hostname>/<folder_path>
```

The `hostname` should be the same entry that is in the SSH config file. The `folder_path` should be the complete path on the remote system. For example to open `/app` on `loginnode`:

```bash
cursor --folder-uri vscode-remote://ssh-remote+loginnode/app
```

If you need to specify additional arguments such as a user or a port, use this alternative syntax, where the `hostname` is a hex-encoded JSON string of the full connection uri. For example, to connect to `76.76.21.21` on port `22` as the `root` user:

```bash
SSH_CONF='{"hostName":"root@76.76.21.21 -p 22"}'
SSH_HEX_CONF=$(printf "$SSH_CONF" | od -A n -t x1 | tr -d '[\n\t ]')
cursor --folder-uri vscode-remote://ssh-remote+${SSH_HEX_CONF}/app
```


## Security Warning

⚠️ Only connect to trusted remote machines. A compromised remote system could potentially execute code on your local machine through the Remote-SSH connection.
