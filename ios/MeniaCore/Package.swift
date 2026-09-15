// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "MeniaCore",
    platforms: [.iOS(.v18), .macOS(.v15)],
    products: [.library(name: "MeniaCore", targets: ["MeniaCore"])],
    targets: [
        .target(name: "MeniaCore"),
        .testTarget(name: "MeniaCoreTests", dependencies: ["MeniaCore"])
    ]
)
