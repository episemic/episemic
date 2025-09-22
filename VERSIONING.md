# Version Management with bump2version

This project uses [bump2version](https://pypi.org/project/bump2version/) for automated and consistent version management.

## Quick Commands

```bash
# Show current version
make version

# Bump patch version (0.1.1 → 0.1.2)
make bump-patch

# Bump minor version (0.1.2 → 0.2.0)
make bump-minor

# Bump major version (0.1.2 → 1.0.0)
make bump-major
```

## Combined Release Workflows

These commands bump the version AND upload to test PyPI:

```bash
# Patch release workflow
make release-patch

# Minor release workflow
make release-minor

# Major release workflow
make release-major
```

## What bump2version Does

When you run a version bump command, bump2version automatically:

1. **Updates version** in multiple files:
   - `pyproject.toml` - Package version
   - `episemic_core/__init__.py` - Module version
   - `.bumpversion.cfg` - Current version tracking

2. **Creates git commit** with descriptive message:
   ```
   Bump version: 0.1.1 → 0.1.2
   ```

3. **Creates git tag** for the new version:
   ```
   v0.1.2
   ```

## Semantic Versioning

This project follows [Semantic Versioning](https://semver.org/):

- **Patch** (0.1.1 → 0.1.2): Bug fixes, documentation updates
- **Minor** (0.1.2 → 0.2.0): New features, backwards compatible
- **Major** (0.1.2 → 1.0.0): Breaking changes

## Configuration

Version management is configured in `.bumpversion.cfg`:

```ini
[bumpversion]
current_version = 0.1.2
commit = True
tag = True
tag_name = v{new_version}
message = Bump version: {current_version} → {new_version}

[bumpversion:file:pyproject.toml]
search = version = "{current_version}"
replace = version = "{new_version}"

[bumpversion:file:episemic_core/__init__.py]
search = __version__ = "{current_version}"
replace = __version__ = "{new_version}"
```

## Manual Usage

You can also use bump2version directly:

```bash
# Install if not already available
pip install bump2version

# Bump version manually
bump2version patch
bump2version minor
bump2version major

# Dry run to see what would change
bump2version --dry-run patch

# Custom version
bump2version --new-version 1.0.0 major
```

## Release Process

1. **Make your changes** and commit them
2. **Choose version bump type** based on changes:
   - Bug fixes → `make bump-patch`
   - New features → `make bump-minor`
   - Breaking changes → `make bump-major`
3. **Test the release** with `make package-test`
4. **Upload to production** with `make package-prod` (when ready)

## Integration with CI/CD

The version tags can be used in CI/CD pipelines:

```bash
# Get latest tag
git describe --tags --abbrev=0

# Check if current commit is tagged
git describe --exact-match --tags HEAD
```

This enables automated releases based on version tags.