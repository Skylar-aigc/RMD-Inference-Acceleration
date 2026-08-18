# Security Policy

## Reporting a vulnerability

Please do not disclose security vulnerabilities in a public issue. Use
GitHub's **Security → Report a vulnerability** workflow in this repository to
send the maintainers a private report with impact, affected versions, and a
minimal reproducer.

## Safe model and data handling

- Load checkpoints and datasets only from sources you trust.
- Prefer `safetensors` for model weights.
- Never commit access tokens, private prompts, or generated user data.
- Install PyTorch, `torch-npu`, and accelerator libraries from their official
  distribution channels.

Only the latest release is supported with security updates.
