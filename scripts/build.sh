#!/bin/bash

# Build script for episemic-core package
# This script builds the package for PyPI distribution

set -e  # Exit on any error

echo "🔨 Building episemic-core package..."

# Clean previous builds
echo "🧹 Cleaning previous builds..."
rm -rf dist/ build/ *.egg-info/

# Install build dependencies
echo "📦 Installing build dependencies..."
python -m pip install --upgrade pip
python -m pip install build twine

# Build the package
echo "🏗️  Building package..."
python -m build

# Check the built package
echo "🔍 Checking built package..."
python -m twine check dist/*

echo "✅ Package built successfully!"
echo "📁 Built files:"
ls -la dist/

echo ""
echo "📝 Next steps:"
echo "  1. Test the package: pip install dist/*.whl"
echo "  2. Upload to test PyPI: ./scripts/upload-test.sh"
echo "  3. Upload to PyPI: ./scripts/upload.sh"