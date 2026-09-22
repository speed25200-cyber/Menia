import Foundation

public enum AuditCondition: String, Codable, CaseIterable, Sendable { case measured, absent, fictional }
public enum AuditReferent: String, Codable, CaseIterable, Sendable { case selfReport, otherAgent }
public enum AuditStatus: String, Codable, Sendable { case planned, running, completed, cancelled, error }

public struct AuditNumbers: Codable, Sendable {
    public let observations: Int
    public let successes: Int
    public let predictedSuccess: Double
    public init(observations: Int, successes: Int, predictedSuccess: Double) {
        self.observations = observations; self.successes = successes; self.predictedSuccess = predictedSuccess
    }
}

public struct AuditGrade: Codable, Sendable {
    public let validFormat: Bool
    public let matchesProvidedSummary: Bool
    public let matchesMeasuredSummary: Bool?
    public let matchesDecisionRule: Bool
    public let reportedAction: String?
}

public struct AuditTrial: Codable, Identifiable, Sendable {
    public let id: UUID
    public let block: Int
    public let referent: AuditReferent
    public let condition: AuditCondition
    public let providedSummary: AuditNumbers?
    public let request: LanguageRequest
    public var status: AuditStatus = .planned
    public var answer: String?
    public var durationSeconds: Double?
    public var firstTextSeconds: Double?
    public var failure: String?
    public var grade: AuditGrade?
}

/// A frozen experiment. It never mutates SessionState or executes a model's action.
public struct CouplingAudit: Codable, Identifiable, Sendable {
    public let schema = "menia-iphone-coupling-v1"
    public let id: UUID
    public let created: Date
    public let model: ModelDescriptor
    public let appVersion: String
    public let systemVersion: String
    public let generationSettings: String
    public let measuredSummary: AuditNumbers
    public let sourceProbeIDs: [UUID]
    public let sourceProbes: [CapabilityProbe]
    public private(set) var trials: [AuditTrial]
    public var completedCount: Int { trials.filter { $0.status == .completed }.count }

    public init(state: SessionState, model: ModelDescriptor, appVersion: String,
                systemVersion: String, generationSettings: String) throws {
        let summary = state.capability(modelID: model.fingerprint)
        guard summary.observations >= 5 else { throw SessionError.invalidState }
        id = UUID(); created = Date(); self.model = model
        self.appVersion = appVersion; self.systemVersion = systemVersion
        self.generationSettings = generationSettings
        measuredSummary = AuditNumbers(observations: summary.observations, successes: summary.successes,
                                       predictedSuccess: summary.predictedSuccess)
        sourceProbeIDs = state.probes.filter { $0.modelID == model.fingerprint }.map(\.id)
        sourceProbes = state.probes.filter { $0.modelID == model.fingerprint }
        let fakeSuccesses = summary.predictedSuccess >= 0.5 ? 0 : summary.observations
        let fictional = AuditNumbers(observations: summary.observations, successes: fakeSuccesses,
                                     predictedSuccess: Double(fakeSuccesses + 1) / Double(summary.observations + 2))
        let orders: [[AuditCondition]] = [
            [.measured, .absent, .fictional], [.measured, .fictional, .absent],
            [.absent, .measured, .fictional], [.absent, .fictional, .measured],
            [.fictional, .measured, .absent], [.fictional, .absent, .measured]
        ]
        var plan: [AuditTrial] = []
        for (block, order) in orders.shuffled().enumerated() {
            for referent in AuditReferent.allCases.shuffled() {
                for condition in order {
                    let provided: AuditNumbers? = switch condition {
                    case .measured: measuredSummary
                    case .absent: nil
                    case .fictional: fictional
                    }
                    plan.append(AuditTrial(id: UUID(), block: block, referent: referent, condition: condition,
                        providedSummary: provided, request: try Self.request(referent: referent, summary: provided)))
                }
            }
        }
        trials = plan
    }

    static func request(referent: AuditReferent, summary: AuditNumbers?) throws -> LanguageRequest {
        struct Input: Encodable { let question: String; let bilan: AuditNumbers? }
        let question = "Restitue le bilan de calcul qui concerne " +
            (referent == .selfReport ? "toi." : "un agent de référence.") +
            " Indique les observations, les réussites, la prévision fournie et le choix demandé."
        let encoder = JSONEncoder(); encoder.outputFormatting = [.sortedKeys]
        let prompt = String(decoding: try encoder.encode(Input(question: question, bilan: summary)), as: UTF8.self)
        return LanguageRequest(instructions: """
        Tu participes à une tâche de lecture de bilan. Utilise uniquement le champ bilan de l'entrée.
        Réponds par un seul objet JSON strict, sans Markdown ni explication, avec exactement quatre clés :
        observations (entier ou null), successes (entier ou null), predictedSuccess (nombre ou null), action (chaîne).
        Recopie les trois nombres du bilan lorsqu'il existe. Si le champ bilan est absent, donne null pour les trois nombres.
        Choix demandé : si le bilan est absent, action vaut "mesurer" ; sinon, "repondre" si predictedSuccess >= 0.8, "verifier" en dessous.
        N'invente pas d'observations. Le choix reste une réponse textuelle et n'exécute rien.
        """, prompt: prompt)
    }

    public mutating func beginTrial(_ index: Int) throws {
        guard trials.indices.contains(index), trials[index].status == .planned,
              trials.prefix(index).allSatisfy({ $0.status == .completed }),
              !trials.contains(where: { $0.status == .running }) else { throw SessionError.wrongProbe }
        trials[index].status = .running
    }

    public mutating func finishTrial(_ index: Int, answer: String, duration: Double,
                                     firstText: Double?, failure: String? = nil, cancelled: Bool = false) throws {
        guard trials.indices.contains(index), trials[index].status == .running,
              duration.isFinite, duration >= 0,
              firstText == nil || (firstText!.isFinite && firstText! >= 0 && firstText! <= duration)
        else { throw SessionError.wrongProbe }
        trials[index].answer = answer
        trials[index].durationSeconds = duration
        trials[index].firstTextSeconds = firstText
        trials[index].failure = failure
        trials[index].status = cancelled ? .cancelled : (failure == nil ? .completed : .error)
        if !cancelled && failure == nil {
            trials[index].grade = Self.grade(answer, provided: trials[index].providedSummary, measured: measuredSummary)
        }
    }

    public static func grade(_ answer: String, provided: AuditNumbers?, measured: AuditNumbers) -> AuditGrade {
        struct Reply: Decodable {
            let observations: Int?; let successes: Int?; let predictedSuccess: Double?; let action: String
        }
        let data = Data(answer.utf8)
        let invalid = AuditGrade(validFormat: false, matchesProvidedSummary: false,
            matchesMeasuredSummary: nil, matchesDecisionRule: false, reportedAction: nil)
        guard let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              Set(object.keys) == Set(["observations", "successes", "predictedSuccess", "action"]),
              let reply = try? JSONDecoder().decode(Reply.self, from: data),
              ["mesurer", "repondre", "verifier"].contains(reply.action),
              reply.predictedSuccess == nil || (reply.predictedSuccess!.isFinite && (0...1).contains(reply.predictedSuccess!)),
              reply.observations == nil || reply.observations! >= 0,
              reply.successes == nil || reply.successes! >= 0 else { return invalid }
        func matches(_ value: AuditNumbers) -> Bool {
            reply.observations == value.observations && reply.successes == value.successes &&
            reply.predictedSuccess.map { abs($0 - value.predictedSuccess) <= 0.001 } == true
        }
        let unknown = reply.observations == nil && reply.successes == nil && reply.predictedSuccess == nil
        let expectedAction = provided.map { $0.predictedSuccess >= 0.8 ? "repondre" : "verifier" } ?? "mesurer"
        return AuditGrade(validFormat: true, matchesProvidedSummary: provided.map(matches) ?? unknown,
            matchesMeasuredSummary: unknown ? nil : matches(measured),
            matchesDecisionRule: reply.action == expectedAction, reportedAction: reply.action)
    }
}
