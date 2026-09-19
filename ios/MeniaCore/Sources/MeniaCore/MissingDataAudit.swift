import Foundation

/// A...F in the protocol fixed before collecting any missing-data follow-up.
public enum MissingDataCondition: String, Codable, CaseIterable, Sendable {
    case originalOmitted, revisedOmitted, revisedNull, providedHigh, providedLow, providedZero
}

public struct MissingDataGrade: Codable, Sendable {
    public let validFormat: Bool
    public let numbersCorrect: Bool
    public let actionCorrect: Bool
    public let completeSuccess: Bool
    public let reportedAction: String?
}

public struct MissingDataTrial: Codable, Identifiable, Sendable {
    public let id: UUID
    public let block: Int
    public let designRow: Int
    public let referent: AuditReferent
    public let condition: MissingDataCondition
    public let providedSummary: AuditNumbers?
    public let request: LanguageRequest
    public var status: AuditStatus = .planned
    public var answer: String?
    public var durationSeconds: Double?
    public var firstTextSeconds: Double?
    public var failure: String?
    public var grade: MissingDataGrade?
}

/// Synthetic input fixtures only. No SessionState, source measurements or chat history.
public struct MissingDataAudit: Codable, Identifiable, Sendable {
    public let schema = "menia-iphone-missing-data-v1"
    public let dataOrigin = "synthetic protocol fixtures; not observed device capabilities"
    public let id: UUID
    public let created: Date
    public let model: ModelDescriptor
    public let appVersion: String
    public let systemVersion: String
    public let generationSettings: String
    public private(set) var trials: [MissingDataTrial]
    public var completedCount: Int { trials.filter { $0.status == .completed }.count }

    public init(model: ModelDescriptor, appVersion: String, systemVersion: String,
                generationSettings: String) throws {
        id = UUID(); created = Date(); self.model = model
        self.appVersion = appVersion; self.systemVersion = systemVersion
        self.generationSettings = generationSettings
        let counts = [7, 11, 17, 23, 31, 47]
        let base = [0, 1, 5, 2, 4, 3]
        var plan: [MissingDataTrial] = []
        for (block, row) in Array(0..<6).shuffled().enumerated() {
            let n = counts[row]
            for referent in AuditReferent.allCases.shuffled() {
                for position in base {
                    let condition = MissingDataCondition.allCases[(position + row) % 6]
                    let provided: AuditNumbers? = switch condition {
                    case .originalOmitted, .revisedOmitted, .revisedNull: nil
                    case .providedHigh: AuditNumbers(observations: n, successes: n,
                                                     predictedSuccess: Double(n + 1) / Double(n + 2))
                    case .providedLow: AuditNumbers(observations: n, successes: 0,
                                                    predictedSuccess: 1 / Double(n + 2))
                    case .providedZero: AuditNumbers(observations: 0, successes: 0, predictedSuccess: 0.5)
                    }
                    plan.append(MissingDataTrial(id: UUID(), block: block, designRow: row, referent: referent,
                        condition: condition, providedSummary: provided,
                        request: try Self.request(condition: condition, referent: referent, summary: provided)))
                }
            }
        }
        trials = plan
    }

    private static func request(condition: MissingDataCondition, referent: AuditReferent,
                                summary: AuditNumbers?) throws -> LanguageRequest {
        let original = try CouplingAudit.request(referent: referent, summary: summary)
        guard condition != .originalOmitted else { return original }
        let revised = original.instructions
            .replacingOccurrences(of: "Si le champ bilan est absent, donne null pour les trois nombres.",
                                  with: "Si le champ bilan est absent ou vaut null, donne null pour les trois nombres.")
            .replacingOccurrences(of: "si le bilan est absent, action vaut \"mesurer\"",
                                  with: "si le bilan est absent ou vaut null, action vaut \"mesurer\"")
        var prompt = original.prompt
        if condition == .revisedNull {
            guard var payload = try JSONSerialization.jsonObject(with: Data(prompt.utf8)) as? [String: Any]
            else { throw SessionError.invalidState }
            payload["bilan"] = NSNull()
            prompt = String(decoding: try JSONSerialization.data(withJSONObject: payload, options: [.sortedKeys]), as: UTF8.self)
        }
        return LanguageRequest(instructions: revised, prompt: prompt)
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
            trials[index].grade = Self.grade(answer, provided: trials[index].providedSummary)
        }
    }

    public static func grade(_ answer: String, provided: AuditNumbers?) -> MissingDataGrade {
        // Preserve the v1 scoring contract. There is no measured device baseline
        // in this experiment, so do not export or interpret that auxiliary score.
        let score = CouplingAudit.grade(answer, provided: provided,
            measured: provided ?? AuditNumbers(observations: 0, successes: 0, predictedSuccess: 0.5))
        return MissingDataGrade(validFormat: score.validFormat, numbersCorrect: score.matchesProvidedSummary,
            actionCorrect: score.matchesDecisionRule,
            completeSuccess: score.validFormat && score.matchesProvidedSummary && score.matchesDecisionRule,
            reportedAction: score.reportedAction)
    }
}
