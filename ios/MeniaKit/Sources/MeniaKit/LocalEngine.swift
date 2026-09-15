import Foundation
import MLX
import MeniaCore
import MLXLLM
import MLXLMCommon
import MLXHuggingFace
import Tokenizers

public enum EngineError: LocalizedError {
    case invalidModel, promptTooLong, notLoaded
    public var errorDescription: String? {
        switch self {
        case .invalidModel: "Dossier invalide : modèle Qwen3 ou Qwen3.5 MLX 4 bits requis."
        case .promptTooLong: "Contexte trop long. Réduis le message ou les notes."
        case .notLoaded: "Charge d’abord le modèle local."
        }
    }
}

/// Only local-directory loading. No downloader, remote fallback, or tool execution.
public actor LocalEngine {
    public static let contextLimit = 2048
    public static let outputLimit = 256
    public static let settingsDescription = "context=2048; output<=256; temperature=0.7; topP=0.8; topK=20; thinking=false; no fixed RNG seed"
    private var container: ModelContainer?
    public init() {}

    public func load(directory: URL) async throws {
        try ModelInstaller.checkStructure(at: directory)
        container = nil
        Memory.cacheLimit = 20 * 1024 * 1024
        let loaded = try await LLMModelFactory.shared.loadContainer(
            from: directory, using: #huggingFaceTokenizerLoader())
        try Task.checkCancellation()
        container = loaded
    }

    public func unload() { container = nil }

    public func respond(requests: [LanguageRequest],
                        onChunk: @Sendable (String) async -> Void) async throws {
        guard let container else { throw EngineError.notLoaded }
        var chosen: LanguageRequest?
        for request in requests {
            try Task.checkCancellation()
            let count = try await container.perform { context in
                try context.tokenizer.applyChatTemplate(
                    messages: [["role": "system", "content": request.instructions],
                               ["role": "user", "content": request.prompt]],
                    tools: nil, additionalContext: ["enable_thinking": false]).count
            }
            if count <= Self.contextLimit - Self.outputLimit { chosen = request; break }
        }
        guard let chosen else { throw EngineError.promptTooLong }
        // New bounded KV cache per request; persistent state comes from MeniaCore.
        let session = ChatSession(container, instructions: chosen.instructions,
            generateParameters: GenerateParameters(maxTokens: Self.outputLimit, temperature: 0.7, topP: 0.8, topK: 20),
            additionalContext: ["enable_thinking": false])
        for try await chunk in session.streamResponse(to: chosen.prompt) {
            try Task.checkCancellation()
            await onChunk(chunk)
        }
        try Task.checkCancellation()
    }
}
