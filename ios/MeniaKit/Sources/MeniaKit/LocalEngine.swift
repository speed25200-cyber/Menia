import Foundation
import MLXLLM
import MLXLMCommon
import MLXHuggingFace
import Tokenizers

public enum EngineError: LocalizedError {
    case invalidModel, promptTooLong, notLoaded
    public var errorDescription: String? {
        switch self {
        case .invalidModel: "Dossier invalide : modèle Qwen3 MLX 4 bits requis."
        case .promptTooLong: "Contexte trop long. Réduis le message ou les notes."
        case .notLoaded: "Charge d’abord le modèle local."
        }
    }
}

/// Only local-directory loading. No downloader, remote fallback, or tool execution.
public actor LocalEngine {
    private var container: ModelContainer?
    public init() {}

    public func load(directory: URL) async throws {
        let config = try JSONSerialization.jsonObject(
            with: Data(contentsOf: directory.appendingPathComponent("config.json"))) as? [String: Any]
        let quant = config?["quantization"] as? [String: Any]
        guard config?["model_type"] as? String == "qwen3", quant?["bits"] as? Int == 4 else {
            throw EngineError.invalidModel
        }
        container = nil
        let loaded = try await LLMModelFactory.shared.loadContainer(
            from: directory, using: #huggingFaceTokenizerLoader())
        try Task.checkCancellation()
        container = loaded
    }

    public func unload() { container = nil }

    public func respond(prompt: String, instructions: String,
                        onChunk: @Sendable (String) async -> Void) async throws {
        guard let container else { throw EngineError.notLoaded }
        let count = try await container.perform { context in
            let text = try context.tokenizer.applyChatTemplate(
                messages: [["role": "system", "content": instructions],
                           ["role": "user", "content": prompt]],
                tools: nil, additionalContext: ["enable_thinking": false])
            return text.count
        }
        guard count <= 1856 else { throw EngineError.promptTooLong }
        // New bounded session each request; only explicit notes supply persistent context.
        let session = ChatSession(container, instructions: instructions,
            generateParameters: GenerateParameters(maxTokens: 192, temperature: 0.6, topP: 0.95, topK: 20),
            additionalContext: ["enable_thinking": false])
        for try await chunk in session.streamResponse(to: prompt) {
            try Task.checkCancellation()
            await onChunk(chunk)
        }
        try Task.checkCancellation()
    }
}
