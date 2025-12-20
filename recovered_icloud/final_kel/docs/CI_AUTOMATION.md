# Boss/Worker CI Automation

This repo now includes a boss workflow that orchestrates two reusable workers:

- `worker-build.yml`: runs tests/builds and uploads optional artifacts.
- `worker-docker.yml`: builds a Docker image (with optional push, SBOM, smoke test).

The boss workflow (`boss.yml`) triggers both workers on `push`, `pull_request`, and manual `workflow_dispatch`.

## Boss workflow inputs (manual dispatch)
- `build_command`: command to build artifacts (blank = skip).
- `test_command`: command to run tests (blank = skip).
- `artifact_path`: path to upload as an Actions artifact (blank = skip).
- `docker_context`: Docker build context (default `.`).
- `dockerfile`: Dockerfile path (default `Dockerfile`).
- `image_name`: image name (defaults to `ghcr.io/<owner>/<repo>`).
- `image_tag`: image tag (defaults to `GITHUB_SHA` short).
- `push_image`: `true`/`false` (push to registry).
- `sbom`: `true`/`false` (generate SBOM via Anchore).
- `smoke_command`: optional container command for a smoke run (requires `push_image=false` so the image is loaded locally).

## Worker details
- **worker-build**: sets up Python (3.11) and optional Node if a version is provided. Runs `test_command`, then `build_command`, and uploads `artifact_path` if set.
- **worker-docker**: uses Buildx; login uses `GITHUB_TOKEN` to push to GHCR. Defaults to `ghcr.io/<owner>/<repo>:<sha7>`. SBOM is produced when `sbom=true`. Smoke test runs locally when `push=false` and `smoke_command` is provided.

## Secrets & permissions
- `GITHUB_TOKEN` (auto) is used for GHCR login when `push_image=true`.
- If pushing to another registry, set `image_name` with the full registry path and add an explicit login step or adjust `REGISTRY` in `worker-docker.yml`.

## How to run (examples)
- Manual: In GitHub Actions UI, select **Boss - Orchestrate Build & Docker**, provide inputs (e.g., `test_command="npm test"`, `build_command="npm run build"`, `artifact_path="dist/**"`, `push_image=false`), and dispatch.
- From CLI (GitHub CLI):  
  `gh workflow run boss.yml -f test_command="pytest" -f build_command="python -m build" -f artifact_path="dist/*" -f push_image=false`

## Notes
- Smoke testing is skipped when `push_image=true` because the image is not loaded locally; use `smoke_command` with `push_image=false` or add a pull/test step after the push as needed.
- SBOM artifacts are uploaded as `sbom-<tag>.spdx.json` when enabled.
