// swift-tools-version: 6.2
import PackageDescription
let package = Package(
    name: "MeniaKit",
    platforms: [.iOS(.v18), .macOS(.v15)],
    products: [.library(name: "MeniaKit", targets: ["MeniaKit"])],
    dependencies: [
        .package(path: "../MeniaCore"),
        .package(url: "https://github.com/ml-explore/mlx-swift", exact: "0.31.3"),
        .package(url: "https://github.com/ml-explore/mlx-swift-lm", exact: "3.31.3"),
        .package(url: "https://github.com/huggingface/swift-transformers", exact: "1.3.0")
    ],
    targets: [.target(name: "MeniaKit", dependencies: [
        .product(name: "MeniaCore", package: "MeniaCore"),
        .product(name: "MLX", package: "mlx-swift"),
        .product(name: "MLXLLM", package: "mlx-swift-lm"),
        .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
        .product(name: "MLXHuggingFace", package: "mlx-swift-lm"),
        .product(name: "Tokenizers", package: "swift-transformers")
    ], resources: [.process("Resources")])]
)
