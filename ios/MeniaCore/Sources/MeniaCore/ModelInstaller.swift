import Foundation
import CryptoKit

public struct ModelDescriptor: Codable, Sendable {
    public let name: String
    public let fingerprint: String
    public let bytes: Int64
}

struct ModelManifest: Codable, Sendable {
    struct File: Codable, Sendable { let name: String; let bytes: Int64; let sha256: String }
    let repository: String
    let revision: String
    let displayName: String
    let files: [File]
}

public enum InstallError: LocalizedError {
    case invalidFile(String), downloadFailed, integrity(String), invalidModel
    public var errorDescription: String? {
        switch self {
        case .invalidModel: "Dossier invalide : modèle Qwen3 ou Qwen3.5 MLX 4 bits requis, avec tokenizer et poids complets."
        case .invalidFile(let name): "Fichier de modèle invalide : \(name)."
        case .downloadFailed: "Téléchargement impossible. Vérifie le Wi-Fi puis réessaie."
        case .integrity(let name): "Le fichier \(name) est incomplet ou ne correspond pas au modèle attendu."
        }
    }
}

/// Runs file work away from the main actor. Installation is staged, with rollback
/// if the final directory move fails. Only the explicit download action uses HTTP.
public actor ModelInstaller {
    public init() {}

    public func downloadRecommended(to destination: URL,
        progress: @Sendable (String) async -> Void) async throws -> ModelDescriptor {
        let manifest = try JSONDecoder().decode(ModelManifest.self,
            from: Data(contentsOf: Bundle.module.url(forResource: "qwen3-4b", withExtension: "json")!))
        let staging = try stagingDirectory(for: destination)
        defer { try? FileManager.default.removeItem(at: staging) }
        let configuration = URLSessionConfiguration.ephemeral
        configuration.allowsCellularAccess = false
        configuration.timeoutIntervalForRequest = 120
        configuration.timeoutIntervalForResource = 3600
        let session = URLSession(configuration: configuration)
        defer { session.invalidateAndCancel() }
        for (index, file) in manifest.files.enumerated() {
            try Task.checkCancellation()
            await progress("Téléchargement \(index + 1)/\(manifest.files.count) · \(file.name)")
            guard safeName(file.name),
                  let url = URL(string: "https://huggingface.co/\(manifest.repository)/resolve/\(manifest.revision)/\(file.name)")
            else { throw InstallError.invalidFile(file.name) }
            let (temporary, response) = try await session.download(from: url)
            defer { try? FileManager.default.removeItem(at: temporary) }
            guard (response as? HTTPURLResponse)?.statusCode == 200 else { throw InstallError.downloadFailed }
            try Task.checkCancellation()
            await progress("Vérification · \(file.name)")
            let digest = try hash(temporary)
            guard digest.bytes == file.bytes, digest.sha256 == file.sha256 else { throw InstallError.integrity(file.name) }
            try FileManager.default.moveItem(at: temporary, to: staging.appendingPathComponent(file.name))
        }
        let descriptor = try inspectAndFingerprint(staging, name: manifest.displayName)
        try install(staging, at: destination, descriptor: descriptor)
        return descriptor
    }

    public func importDirectory(_ source: URL, to destination: URL,
        progress: @Sendable (String) async -> Void) async throws -> ModelDescriptor {
        let staging = try stagingDirectory(for: destination)
        defer { try? FileManager.default.removeItem(at: staging) }
        let entries = try FileManager.default.contentsOfDirectory(at: source,
            includingPropertiesForKeys: [.isRegularFileKey, .isSymbolicLinkKey], options: [.skipsHiddenFiles])
        // Only known model data files; no nested directories, symlinks or executables.
        let allowed = entries.filter { ["json", "safetensors", "txt", "jinja"].contains($0.pathExtension) || $0.lastPathComponent == "LICENSE" }
        for file in allowed {
            try Task.checkCancellation()
            let values = try file.resourceValues(forKeys: [.isRegularFileKey, .isSymbolicLinkKey])
            guard values.isRegularFile == true, values.isSymbolicLink != true else { throw InstallError.invalidFile(file.lastPathComponent) }
            await progress("Copie · \(file.lastPathComponent)")
            try FileManager.default.copyItem(at: file, to: staging.appendingPathComponent(file.lastPathComponent))
        }
        await progress("Vérification du modèle importé…")
        let descriptor = try inspectAndFingerprint(staging, name: source.lastPathComponent)
        try install(staging, at: destination, descriptor: descriptor)
        return descriptor
    }

    public static func descriptor(at directory: URL) throws -> ModelDescriptor {
        try JSONDecoder().decode(ModelDescriptor.self,
            from: Data(contentsOf: directory.appendingPathComponent("menia-model.json")))
    }

    /// Call once at application startup, before allowing any new installation.
    /// Restore the previous directory if the process died during the final move.
    public static func recoverInterruptedInstall(at destination: URL) throws {
        let fm = FileManager.default
        let parent = destination.deletingLastPathComponent()
        let entries = try fm.contentsOfDirectory(at: parent, includingPropertiesForKeys: nil)
        func owned(_ url: URL, prefix: String) -> Bool {
            let name = url.lastPathComponent
            return name.hasPrefix(prefix) && UUID(uuidString: String(name.dropFirst(prefix.count))) != nil
        }
        let backups = entries.filter { owned($0, prefix: "previous-") }
        if !fm.fileExists(atPath: destination.path), !backups.isEmpty {
            // Ambiguous recovery is reported rather than silently choosing a model.
            guard backups.count == 1 else { throw InstallError.invalidFile("plusieurs modèles précédents à récupérer") }
            _ = try descriptor(at: backups[0])
            try checkStructure(at: backups[0])
            try fm.moveItem(at: backups[0], to: destination)
        }
        for entry in entries where owned(entry, prefix: "install-") || owned(entry, prefix: "previous-") {
            if fm.fileExists(atPath: entry.path) { try fm.removeItem(at: entry) }
        }
    }

    public static func checkStructure(at directory: URL) throws {
        let fm = FileManager.default
        let config = try JSONSerialization.jsonObject(with: Data(contentsOf: directory.appendingPathComponent("config.json"))) as? [String: Any]
        let quant = config?["quantization"] as? [String: Any]
        guard let type = config?["model_type"] as? String, ["qwen3", "qwen3_5", "qwen3_5_text"].contains(type),
              quant?["bits"] as? Int == 4,
              fm.fileExists(atPath: directory.appendingPathComponent("tokenizer.json").path),
              fm.fileExists(atPath: directory.appendingPathComponent("tokenizer_config.json").path)
        else { throw InstallError.invalidModel }
        let weights = try fm.contentsOfDirectory(at: directory, includingPropertiesForKeys: nil)
            .filter { $0.pathExtension == "safetensors" }
        guard !weights.isEmpty else { throw InstallError.invalidModel }
        let indexURL = directory.appendingPathComponent("model.safetensors.index.json")
        if fm.fileExists(atPath: indexURL.path) {
            let index = try JSONSerialization.jsonObject(with: Data(contentsOf: indexURL)) as? [String: Any]
            guard let map = index?["weight_map"] as? [String: String], !map.isEmpty,
                  map.values.allSatisfy({ safeName($0) && $0.hasSuffix(".safetensors") && fm.fileExists(atPath: directory.appendingPathComponent($0).path) })
            else { throw InstallError.invalidModel }
        }
    }

    private func stagingDirectory(for destination: URL) throws -> URL {
        let staging = destination.deletingLastPathComponent().appendingPathComponent("install-" + UUID().uuidString)
        try FileManager.default.createDirectory(at: staging, withIntermediateDirectories: true)
        return staging
    }

    private func inspectAndFingerprint(_ directory: URL, name: String) throws -> ModelDescriptor {
        try Self.checkStructure(at: directory)
        let files = try FileManager.default.contentsOfDirectory(at: directory, includingPropertiesForKeys: nil)
            .filter { $0.pathExtension == "safetensors" || ["config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja"].contains($0.lastPathComponent) }
            .sorted { $0.lastPathComponent < $1.lastPathComponent }
        var combined = SHA256()
        var bytes: Int64 = 0
        for file in files {
            let result = try hash(file)
            bytes += result.bytes
            combined.update(data: Data((file.lastPathComponent + ":" + result.sha256 + "\n").utf8))
        }
        return ModelDescriptor(name: String(name.prefix(120)), fingerprint: hex(combined.finalize()), bytes: bytes)
    }

    private func hash(_ url: URL) throws -> (bytes: Int64, sha256: String) {
        let handle = try FileHandle(forReadingFrom: url)
        defer { try? handle.close() }
        var hasher = SHA256()
        var bytes: Int64 = 0
        while let chunk = try handle.read(upToCount: 1024 * 1024), !chunk.isEmpty {
            try Task.checkCancellation()
            bytes += Int64(chunk.count)
            hasher.update(data: chunk)
        }
        return (bytes, hex(hasher.finalize()))
    }

    private func install(_ staging: URL, at destination: URL, descriptor: ModelDescriptor) throws {
        let fm = FileManager.default
        try JSONEncoder().encode(descriptor).write(to: staging.appendingPathComponent("menia-model.json"), options: .atomic)
        #if os(iOS)
        for file in try fm.contentsOfDirectory(at: staging, includingPropertiesForKeys: nil) {
            try fm.setAttributes([.protectionKey: FileProtectionType.complete], ofItemAtPath: file.path)
        }
        #endif
        try Task.checkCancellation()
        let backup = destination.deletingLastPathComponent().appendingPathComponent("previous-" + UUID().uuidString)
        let existed = fm.fileExists(atPath: destination.path)
        if existed { try fm.moveItem(at: destination, to: backup) }
        do { try fm.moveItem(at: staging, to: destination) }
        catch {
            if existed { try fm.moveItem(at: backup, to: destination) }
            throw error
        }
        if existed { try? fm.removeItem(at: backup) }
    }
}

private func safeName(_ name: String) -> Bool {
    !name.isEmpty && name != "." && name != ".." && !name.contains("/") && !name.contains("\\")
}

private func hex<D: Sequence>(_ digest: D) -> String where D.Element == UInt8 {
    digest.map { String(format: "%02x", $0) }.joined()
}
