#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
xcodegen generate
mkdir -p Menia.xcodeproj/project.xcworkspace/xcshareddata/swiftpm
cp Package.resolved Menia.xcodeproj/project.xcworkspace/xcshareddata/swiftpm/Package.resolved
if ! xcrun metal --version; then
  xcodebuild -downloadComponent MetalToolchain
fi
