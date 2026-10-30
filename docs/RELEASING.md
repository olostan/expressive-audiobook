# Releasing

Releases are made from version tags. A push to `main` runs the tests (CI) and nothing else; it never publishes.

The version lives in one place: `__version__` in `src/expressive_audiobook/__init__.py`. The package metadata and `audiobook --version` both read it.

## Make a release

1. Pick the next version ([semantic versioning](https://semver.org/): `0.2.0` for new features, `0.1.1` for fixes).
2. Change `__version__` in `src/expressive_audiobook/__init__.py`, commit, and push to `main`. Wait for CI to pass.
3. Tag that commit and push the tag:

   ```bash
   git tag -a v0.2.0 -m "expressive-audiobook 0.2.0"
   git push origin v0.2.0
   ```

The `Release` workflow then:

- checks the tag matches `__version__` (it fails if you forgot step 2),
- runs the tests and builds the sdist and wheel,
- creates the GitHub release with generated notes and the files attached (a tag like `v0.2.0-rc1` becomes a pre-release),
- updates the Homebrew formula in `olostan/homebrew-tap` (stable tags only), and
- optionally publishes to PyPI.

To check the build without releasing, run the `Release` workflow by hand from the Actions tab: it builds and tests but publishes nothing.

## One-time setup

- **Homebrew tap update.** Create a fine-grained personal access token with *Contents: read and write* on `olostan/homebrew-tap` only, and save it in this repository as the secret `HOMEBREW_TAP_TOKEN`. Without it the release still works and the tap is left as it is.
- **PyPI (optional).** Create the project on PyPI, add this repository's `release.yml` workflow (environment `pypi`) as a trusted publisher, then set the repository variable `PUBLISH_TO_PYPI` to `true`. No token is stored.

## If a release goes wrong

Delete the GitHub release and the tag (`gh release delete vX.Y.Z --cleanup-tag`), fix the problem, and tag again. Do not reuse a version number that has been on PyPI.
