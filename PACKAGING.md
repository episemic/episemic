# Packaging Guide for Episemic-Core

This guide explains how to package and upload the episemic library to PyPI.

## Prerequisites

1. **PyPI Account**: Create accounts on both [test.pypi.org](https://test.pypi.org/account/register/) and [pypi.org](https://pypi.org/account/register/)

2. **API Tokens**: Generate API tokens for both test PyPI and production PyPI:
   - Test PyPI: https://test.pypi.org/manage/account/token/
   - Production PyPI: https://pypi.org/manage/account/token/

3. **Dependencies**: The build scripts will automatically install required dependencies (`build` and `twine`)

## Quick Start

### Option 1: Using Make Commands

```bash
# Build the package
make package-build

# Build and upload to test PyPI
make package-test

# Build and upload to production PyPI (after testing)
make package-prod
```

### Option 2: Using Scripts Directly

```bash
# Build only
./scripts/build.sh

# Build and upload to test PyPI
./scripts/package.sh test

# Build and upload to production PyPI
./scripts/package.sh prod
```

### Option 3: Step by Step

```bash
# 1. Build the package
./scripts/build.sh

# 2. Upload to test PyPI
./scripts/upload-test.sh

# 3. Test installation from test PyPI
pip install --index-url https://test.pypi.org/simple/ episemic

# 4. If everything works, upload to production PyPI
./scripts/upload.sh
```

## Recommended Workflow

1. **Update Version**: Update the version in `pyproject.toml` and `episemic_core/__init__.py`

2. **Run Tests**: Ensure all tests pass
   ```bash
   make test
   ```

3. **Build Package**:
   ```bash
   make package-build
   ```

4. **Test on Test PyPI**:
   ```bash
   make package-test
   ```

5. **Test Installation**:
   ```bash
   pip install --index-url https://test.pypi.org/simple/ episemic
   ```

6. **Upload to Production PyPI**:
   ```bash
   make package-prod
   ```

## Authentication

When prompted for credentials:
- **Username**: `__token__`
- **Password**: Your API token (including the `pypi-` prefix)

## Version Management

Before uploading:
1. Update version in `pyproject.toml`
2. Update version in `episemic_core/__init__.py`
3. Update any version references in documentation

## Troubleshooting

### Common Issues

1. **Version already exists**: PyPI doesn't allow overwriting existing versions. Increment the version number.

2. **Authentication failed**: Double-check your API token and ensure you're using `__token__` as the username.

3. **Missing files in package**: Check `MANIFEST.in` to ensure all necessary files are included.

4. **Build failed**: Ensure all dependencies are properly specified in `pyproject.toml`.

### Useful Commands

```bash
# Check what files will be included in the package
python -m build --sdist
tar -tzf dist/*.tar.gz

# Validate the package without uploading
python -m twine check dist/*

# Upload with verbose output for debugging
python -m twine upload --verbose dist/*
```

## Files Created

This packaging setup creates the following files:

- `scripts/build.sh` - Build the package
- `scripts/upload-test.sh` - Upload to test PyPI
- `scripts/upload.sh` - Upload to production PyPI
- `scripts/package.sh` - Complete workflow script
- `MANIFEST.in` - Specify files to include in package
- `LICENSE` - MIT license file
- `PACKAGING.md` - This guide

The `pyproject.toml` has been updated with:
- Author information
- License and classifiers
- Project URLs
- Build dependencies