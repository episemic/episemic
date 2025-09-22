#!/bin/bash

# Complete packaging script
# This script handles the full packaging workflow

set -e  # Exit on any error

echo "📦 Episemic-Core Complete Packaging Workflow"
echo "============================================"

# Function to show help
show_help() {
    echo "Usage: $0 [OPTION]"
    echo ""
    echo "Options:"
    echo "  build      Build the package only"
    echo "  test       Build and upload to test PyPI"
    echo "  prod       Build and upload to production PyPI"
    echo "  help       Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 build      # Just build the package"
    echo "  $0 test       # Build and upload to test PyPI"
    echo "  $0 prod       # Build and upload to production PyPI"
}

# Parse command line arguments
case "${1:-help}" in
    build)
        echo "🔨 Building package..."
        ./scripts/build.sh
        ;;
    test)
        echo "🧪 Building and uploading to test PyPI..."
        ./scripts/build.sh
        ./scripts/upload-test.sh
        ;;
    prod)
        echo "🚀 Building and uploading to production PyPI..."
        ./scripts/build.sh
        ./scripts/upload.sh
        ;;
    help|--help|-h)
        show_help
        ;;
    *)
        echo "❌ Unknown option: $1"
        echo ""
        show_help
        exit 1
        ;;
esac

echo ""
echo "✅ Packaging workflow completed!"