#!/bin/bash

# Upload script for test PyPI
# This script uploads the package to test.pypi.org

set -e  # Exit on any error

echo "🚀 Uploading episemic-core to test PyPI..."

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

# Install twine if not already installed
echo "📦 Ensuring twine is installed..."
python -m pip install --upgrade twine

# Upload to test PyPI
echo "📤 Uploading to test PyPI..."
echo "💡 You will be prompted for your test PyPI credentials."
echo "   Username: __token__"
echo "   Password: your test PyPI API token"
echo ""

python -m twine upload --repository testpypi dist/*

echo ""
echo "✅ Upload completed!"
echo "🌐 Your package should be available at:"
echo "   https://test.pypi.org/project/episemic-core/"
echo ""
echo "🧪 To test installation from test PyPI:"
echo "   pip install --index-url https://test.pypi.org/simple/ episemic-core"