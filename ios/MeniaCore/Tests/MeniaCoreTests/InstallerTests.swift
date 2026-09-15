import Foundation
import XCTest
@testable import MeniaCore

final class InstallerTests: XCTestCase {
    private func directory() throws -> URL {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        return url
    }

    /// Small inert fixtures test installation, never inference or weight validity.
    private func fixture(in parent: URL, type: String = "qwen3") throws -> URL {
        let url = parent.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        try Data("{\"model_type\":\"\(type)\",\"quantization\":{\"bits\":4}}".utf8)
            .write(to: url.appendingPathComponent("config.json"))
        for name in ["tokenizer.json", "tokenizer_config.json"] {
            try Data("{}".utf8).write(to: url.appendingPathComponent(name))
        }
        try Data(repeating: 42, count: 32).write(to: url.appendingPathComponent("model.safetensors"))
        return url
    }

    @MainActor func testImportFingerprintIsContentBasedAndChangesWithTokenizer() async throws {
        let root = try directory(); defer { try? FileManager.default.removeItem(at: root) }
        let source = try fixture(in: root)
        let installer = ModelInstaller()
        let first = try await installer.importDirectory(source, to: root.appendingPathComponent("installed"), progress: { _ in })
        let second = try await installer.importDirectory(source, to: root.appendingPathComponent("second"), progress: { _ in })
        XCTAssertEqual(first.fingerprint, second.fingerprint)
        try Data("{\"changed\":true}".utf8).write(to: source.appendingPathComponent("tokenizer_config.json"))
        let third = try await installer.importDirectory(source, to: root.appendingPathComponent("installed"), progress: { _ in })
        XCTAssertNotEqual(first.fingerprint, third.fingerprint)
        XCTAssertEqual(try ModelInstaller.descriptor(at: root.appendingPathComponent("installed")).fingerprint, third.fingerprint)
    }

    @MainActor func testMissingShardPreservesInstalledModel() async throws {
        let root = try directory(); defer { try? FileManager.default.removeItem(at: root) }
        let source = try fixture(in: root), destination = root.appendingPathComponent("installed")
        let installer = ModelInstaller()
        let original = try await installer.importDirectory(source, to: destination, progress: { _ in })
        try Data("{\"weight_map\":{\"tensor\":\"missing.safetensors\"}}".utf8)
            .write(to: source.appendingPathComponent("model.safetensors.index.json"))
        do { _ = try await installer.importDirectory(source, to: destination, progress: { _ in }); XCTFail("Missing shard accepted") }
        catch { XCTAssertEqual(try ModelInstaller.descriptor(at: destination).fingerprint, original.fingerprint) }
        let leftovers = try FileManager.default.contentsOfDirectory(atPath: root.path).filter { $0.hasPrefix("install-") }
        XCTAssertTrue(leftovers.isEmpty)
    }

    @MainActor func testCancellationPreservesInstalledModel() async throws {
        let root = try directory(); defer { try? FileManager.default.removeItem(at: root) }
        let source = try fixture(in: root), destination = root.appendingPathComponent("installed")
        let installer = ModelInstaller()
        let original = try await installer.importDirectory(source, to: destination, progress: { _ in })
        let task = Task {
            withUnsafeCurrentTask { $0?.cancel() }
            return try await installer.importDirectory(source, to: destination, progress: { _ in })
        }
        do { _ = try await task.value; XCTFail("Cancelled import completed") }
        catch { XCTAssertTrue(error is CancellationError) }
        XCTAssertEqual(try ModelInstaller.descriptor(at: destination).fingerprint, original.fingerprint)
    }

    @MainActor func testSymbolicLinkIsRejected() async throws {
        let root = try directory(); defer { try? FileManager.default.removeItem(at: root) }
        let source = try fixture(in: root)
        let weights = source.appendingPathComponent("model.safetensors")
        try FileManager.default.removeItem(at: weights)
        try FileManager.default.createSymbolicLink(at: weights, withDestinationURL: source.appendingPathComponent("config.json"))
        do { _ = try await ModelInstaller().importDirectory(source, to: root.appendingPathComponent("installed"), progress: { _ in }); XCTFail("Symlink accepted") }
        catch { XCTAssertFalse(FileManager.default.fileExists(atPath: root.appendingPathComponent("installed").path)) }
    }

    func testUnknownArchitectureAndTraversalIndexAreRejected() throws {
        let root = try directory(); defer { try? FileManager.default.removeItem(at: root) }
        let unknown = try fixture(in: root, type: "unknown")
        XCTAssertThrowsError(try ModelInstaller.checkStructure(at: unknown))
        let source = try fixture(in: root, type: "qwen3_5")
        XCTAssertNoThrow(try ModelInstaller.checkStructure(at: source))
        try Data("{\"weight_map\":{\"tensor\":\"../model.safetensors\"}}".utf8)
            .write(to: source.appendingPathComponent("model.safetensors.index.json"))
        XCTAssertThrowsError(try ModelInstaller.checkStructure(at: source))
    }

    func testRecommendedManifestPinsEveryFileAndLicense() throws {
        let url = try XCTUnwrap(Bundle.module.url(forResource: "qwen3-4b", withExtension: "json"))
        let manifest = try JSONDecoder().decode(ModelManifest.self, from: Data(contentsOf: url))
        XCTAssertEqual(manifest.repository, "Qwen/Qwen3-4B-MLX-4bit")
        XCTAssertEqual(manifest.revision.count, 40)
        XCTAssertTrue(manifest.files.contains { $0.name == "LICENSE" })
        XCTAssertTrue(manifest.files.allSatisfy { $0.sha256.count == 64 && $0.bytes > 0 })
        XCTAssertEqual(manifest.files.reduce(Int64(0)) { $0 + $1.bytes }, 2_153_298_402)
    }
}
