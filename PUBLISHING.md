# Publishing VELA to Comfy Registry / Manager

VELA uses the Comfy Registry identity declared in `pyproject.toml`:

- package: `comfyui-vela-h3`
- publisher: `speach1sdef178`
- display name: `ComfyUI VELA H3`
- first public version: `1.0.0`

## One-time setup

1. Create the separate GitHub repository `Speach1sdef178/ComfyUI-VELA-H3` and push this release tree to its default branch.
2. Sign in to the Comfy Registry and make sure the publisher ID is exactly `speach1sdef178`.
3. Create a Registry API key.
4. In the GitHub repository, add that key as the Actions secret `REGISTRY_ACCESS_TOKEN`.
5. Never commit the Registry key to the repository.

## Publish 1.0.0

The included `.github/workflows/publish.yml` uses `Comfy-Org/publish-node-action@main`. After the repository and secret exist, run **Publish to Comfy Registry** from GitHub Actions, or let a `pyproject.toml` push trigger it.

Before publishing, perform the release sanity run described in `RELEASE_VALIDATION.md` and commit the exact tested release contents.

## Future releases

Registry versions are versioned artifacts. For an update, bump `[project].version` in `pyproject.toml` (for example `1.0.1` or `1.1.0`), update `CHANGELOG.md`, validate the release, then publish the new version. Do not reuse the `1.0.0` number for changed contents after it has been published.
