import Foundation

public enum SessionError: LocalizedError {
    case invalidState, invalidNote, pendingProbe, wrongProbe, invalidAnswer
    public var errorDescription: String? {
        switch self {
        case .invalidState: "La sauvegarde est invalide ou provient d’une autre version."
        case .invalidNote: "Une note doit contenir entre 1 et 2 000 caractères ; 50 notes maximum."
        case .pendingProbe: "Un test est déjà en cours."
        case .wrongProbe: "Ce résultat ne correspond pas au test en cours."
        case .invalidAnswer: "Une réponse vide ne peut pas être enregistrée."
        }
    }
}

public struct MemoryNote: Codable, Identifiable, Sendable {
    public let id: UUID
    public let text: String
    public let source: String
    public let created: Date
    public init(id: UUID = UUID(), text: String, source: String = "user", created: Date = Date()) {
        self.id = id; self.text = text; self.source = source; self.created = created
    }
}

public struct Exchange: Codable, Identifiable, Sendable {
    public let id: UUID
    public let question: String
    public let answer: String
    public let modelID: String
    public let created: Date
}

/// A prediction is frozen BEFORE the LLM receives the question. The reference
/// belongs to the evaluator and is never included in that model request.
public struct CapabilityProbe: Codable, Identifiable, Sendable {
    public let id: UUID
    public let modelID: String
    public let question: String
    public let expected: Int
    public let predictedSuccess: Double
    public let created: Date
    public var answer: String?
    public var correct: Bool?
}

public struct CapabilitySummary: Codable, Sendable {
    public let scope: String
    public let observations: Int
    public let successes: Int
    public let predictedSuccess: Double
    public let brierScore: Double?
}

public struct SessionState: Codable, Sendable {
    public var version = 1
    public private(set) var notes: [MemoryNote] = []
    public private(set) var exchanges: [Exchange] = []
    public private(set) var probes: [CapabilityProbe] = []
    public private(set) var pending: CapabilityProbe?
    public init() {}

    public mutating func addNote(_ text: String) throws {
        let cleaned = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !cleaned.isEmpty, cleaned.count <= 2000, notes.count < 50 else { throw SessionError.invalidNote }
        notes.append(MemoryNote(text: cleaned))
    }

    public mutating func removeNote(id: UUID) { notes.removeAll { $0.id == id } }

    public mutating func recordExchange(question: String, answer: String, modelID: String) throws {
        guard !answer.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else { throw SessionError.invalidAnswer }
        exchanges.append(Exchange(id: UUID(), question: String(question.prefix(6000)),
                                  answer: String(answer.prefix(12000)), modelID: modelID, created: Date()))
        exchanges = Array(exchanges.suffix(20))
    }

    public func capability(modelID: String) -> CapabilitySummary {
        let samples = probes.filter { $0.modelID == modelID && $0.correct != nil }
        let successes = samples.filter { $0.correct == true }.count
        let brier = samples.isEmpty ? nil : samples.reduce(0.0) {
            $0 + pow($1.predictedSuccess - ($1.correct == true ? 1 : 0), 2)
        } / Double(samples.count)
        return CapabilitySummary(scope: "arithmetic-v1: additions/soustractions entières, réponse au format entier strict ; pas une mesure générale d’intelligence",
            observations: samples.count, successes: successes,
            predictedSuccess: Double(successes + 1) / Double(samples.count + 2), brierScore: brier)
    }

    public mutating func beginProbe(modelID: String, lhs: Int, rhs: Int, subtract: Bool) throws -> CapabilityProbe {
        guard pending == nil else { throw SessionError.pendingProbe }
        guard (0...999).contains(lhs), (0...999).contains(rhs), !modelID.isEmpty else { throw SessionError.invalidState }
        let probe = CapabilityProbe(id: UUID(), modelID: modelID,
            question: "Calcule \(lhs) \(subtract ? "-" : "+") \(rhs). Réponds uniquement par l’entier obtenu, sans explication.",
            expected: subtract ? lhs - rhs : lhs + rhs,
            predictedSuccess: capability(modelID: modelID).predictedSuccess, created: Date())
        pending = probe
        return probe
    }

    public mutating func finishProbe(id: UUID, modelID: String, answer: String) throws {
        guard var probe = pending, probe.id == id, probe.modelID == modelID else { throw SessionError.wrongProbe }
        probe.answer = String(answer.prefix(12000))
        // Conservative exact task contract: prose or a truncated answer is a failure,
        // never silently parsed as a successful mathematical result.
        probe.correct = answer.trimmingCharacters(in: .whitespacesAndNewlines) == String(probe.expected)
        probes.append(probe)
        probes = Array(probes.suffix(128))
        pending = nil
    }

    /// Cancellation and process termination are not mathematical failures.
    public mutating func cancelProbe() { pending = nil }

    public mutating func importLegacyNotes(_ old: [MemoryNote]) throws {
        guard notes.isEmpty, old.count <= 50, old.allSatisfy({ !$0.text.isEmpty && $0.text.count <= 2000 }) else {
            throw SessionError.invalidState
        }
        notes = old
    }

    public func validate() throws {
        guard version == 1, notes.count <= 50, exchanges.count <= 20, probes.count <= 128,
              notes.allSatisfy({ !$0.text.isEmpty && $0.text.count <= 2000 }),
              exchanges.allSatisfy({ $0.question.count <= 6000 && $0.answer.count <= 12000 }),
              Set(probes.map(\.id)).count == probes.count,
              probes.allSatisfy({ $0.correct != nil && $0.answer != nil && $0.predictedSuccess.isFinite && (0...1).contains($0.predictedSuccess) }),
              pending == nil || (pending?.correct == nil && pending?.answer == nil)
        else { throw SessionError.invalidState }
    }
}

public struct SessionStore: Sendable {
    public let url: URL
    public init(url: URL) { self.url = url }

    public func load() throws -> SessionState {
        guard FileManager.default.fileExists(atPath: url.path) else { return SessionState() }
        let data = try Data(contentsOf: url)
        guard data.count < 8_000_000 else { throw SessionError.invalidState }
        var state = try JSONDecoder().decode(SessionState.self, from: data)
        try state.validate()
        state.cancelProbe() // Do not resume a half-observed test after a process restart.
        return state
    }

    public func save(_ state: SessionState) throws {
        try state.validate()
        let data = try JSONEncoder().encode(state)
        #if os(iOS)
        try data.write(to: url, options: [.atomic, .completeFileProtection])
        #else
        try data.write(to: url, options: .atomic)
        #endif
    }
}
