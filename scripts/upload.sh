#!/bin/bash

# Upload script for production PyPI
# This script uploads the package to pypi.org

set -e  # Exit on any error

echo "🚀 Uploading episemic-core to production PyPI..."

# Check if dist directory exists
if [ ! -d "dist" ]; then
    echo "❌ dist/ directory not found. Run ./scripts/build.sh first."
    exit 1
fi

# Check if there are files to upload
if [ -z "$(ls -A dist/)" ]; then
    echo "❌ No files found in dist/. Run ./scripts/build.sh first."
    exit 1
fi

# Confirmation prompt
echo "⚠️  WARNING: This will upload to PRODUCTION PyPI!"
echo "   Make sure you have:"
echo "   1. ✅ Tested the package thoroughly"
echo "   2. ✅ Uploaded to test PyPI first"
echo "   3. ✅ Updated the version number"
echo "   4. ✅ Updated the changelog"
echo ""
read -p "Are you sure you want to proceed? (y/N): " confirm

if [[ $confirm != [yY] ]]; then
    echo "❌ Upload cancelled."
    exit 1
fi

# Install twine if not already installed
echo "📦 Ensuring twine is installed..."
python -m pip install --upgrade twine

# Upload to production PyPI
echo "📤 Uploading to production PyPI..."
echo "💡 You will be prompted for your PyPI credentials."
echo "   Username: __token__"
echo "   Password: your PyPI API token"
echo ""

python -m twine upload dist/*

echo ""
echo "✅ Upload completed!"
echo "🌐 Your package should be available at:"
echo "   https://pypi.org/project/episemic-core/"
echo ""
echo "🎉 Installation command for users:"
echo "   pip install episemic-core"