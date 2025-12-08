# Cursor Remote WSL

This extension enables you to use Cursor IDE on Windows to develop Linux applications using the [Windows Subsystem for Linux (WSL)](https://docs.microsoft.com/en-us/windows/wsl).

## What is WSL?

Windows Subsystem for Linux (WSL) allows you to run a complete Linux environment directly on Windows. Unlike traditional virtual machines, WSL provides:
- A lightweight Linux environment
- Direct access to Linux command-line tools
- The ability to run Linux applications
- No need for dual-booting or heavy virtualization

## Why Use This Extension?

When developing Linux applications on Windows, you face two main challenges:
1. Running and debugging Linux applications
2. Accessing Linux-specific development tools

This extension solves these problems by:
- Running Cursor's interface on Windows
- Executing all commands, extensions, and terminal operations in Linux
- Providing full IDE features (autocomplete, debugging, linting) using Linux tools

## Requirements

- Windows 11 or Windows 10 (version 21H1 or later)
- WSL 2 installed

## Best Practices

For optimal performance:
- Store your project files in the Linux file system (e.g., `/home/username/project`)
- Avoid storing files in the Windows file system (e.g., `C:\Users\username\project`)
- If your files are currently on Windows, move them to the Linux file system

For more details about working with files in WSL, refer to the [official WSL documentation](https://docs.microsoft.com/en-us/windows/wsl/faq).
