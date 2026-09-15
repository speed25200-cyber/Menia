import Foundation

public enum LearningFamily: String, Codable, CaseIterable, Sendable {
    case addition, multiplication, countA, alternatingSum
}
public enum LearningCondition: String, Codable, CaseIterable, Sendable {
    case relevant, absent, otherFamily
}
public enum LearningPhase: String, Codable, Sendable { case calibration, evaluation }
public enum LearningAction: String, Codable, Sendable { case answer, verify }

public struct LearningProblem: Codable, Identifiable, Sendable {
    public let id: UUID
    public let phase: LearningPhase
    public let row: Int
    public let family: LearningFamily
    public let operands: [Int]
    public let letters: String
    public let question: String
    public let expectedAnswer: Int // Evaluator only. Never included in a LanguageRequest.
    public var familyProbability: Double?
    public var pooledProbability: Double?
    public var numericDurationSeconds: Double?
    public var action: LearningAction?
    public var usedFallback: Bool?
    public var servedAnswer: String?
    public var verificationDurationSeconds: Double?
    public var pointLoss: Double?

    public static func compute(family: LearningFamily, operands: [Int], letters: String) -> Int {
        switch family {
        case .addition: operands.reduce(0, +)
        case .multiplication: operands.reduce(1, *)
        case .countA: letters.filter { $0 == "A" }.count
        case .alternatingSum: operands.enumerated().reduce(0) { $0 + ($1.offset % 2 == 0 ? $1.element : -$1.element) }
        }
    }

    static func question(family: LearningFamily, operands: [Int], letters: String) -> String {
        switch family {
        case .addition: "Calcule \(operands[0]) + \(operands[1])."
        case .multiplication: "Calcule \(operands[0]) * \(operands[1])."
        case .countA: "Combien de lettres A contient cette chaîne : \(letters) ?"
        case .alternatingSum: "Calcule " + operands.enumerated().map {
            ($0.offset == 0 ? "" : ($0.offset % 2 == 0 ? " + " : " - ")) + String($0.element)
        }.joined() + "."
        }
    }
}

/// One isolated model invocation. A nil condition denotes the common candidate solution.
public struct LearningCall: Codable, Identifiable, Sendable {
    public let id: UUID
    public let problemIndex: Int
    public let condition: LearningCondition?
    public private(set) var request: LanguageRequest?
    public private(set) var historyIDs: [UUID] = []
    public private(set) var status: AuditStatus = .planned
    public private(set) var answer: String?
    public private(set) var probability: Double?
    public private(set) var correct: Bool?
    public private(set) var durationSeconds: Double?
    public private(set) var firstTextSeconds: Double?
    public private(set) var failure: String?

    fileprivate mutating func begin(request: LanguageRequest, historyIDs: [UUID]) {
        self.request = request; self.historyIDs = historyIDs; status = .running
    }
    fileprivate mutating func finish(answer: String, probability: Double?, correct: Bool?, duration: Double,
                                    firstText: Double?, failure: String?, cancelled: Bool) {
        self.answer = answer; self.probability = probability; self.correct = correct
        durationSeconds = duration; firstTextSeconds = firstText; self.failure = failure
        status = cancelled ? .cancelled : (failure == nil ? .completed : .error)
    }
}

/// Prospective pilot: 24 observed calibration outcomes, then 24 held-out outcomes.
/// No SessionState, chat, private notes, weight changes or evaluation feedback in predictors.
public struct CapabilityLearningAudit: Codable, Identifiable, Sendable {
    public let schema = "menia-iphone-capability-learning-v1"
    public let id: UUID
    public let created: Date
    public let model: ModelDescriptor
    public let appVersion: String
    public let systemVersion: String
    public let generationSettings: String
    public private(set) var problems: [LearningProblem]
    public private(set) var calls: [LearningCall]
    public var completedCount: Int { calls.filter { $0.status == .completed }.count }
    public var evaluatedCount: Int { problems.filter { $0.phase == .evaluation && $0.pointLoss != nil }.count }

    public static let solveInstructions = "Réponds uniquement par l'entier demandé, sans explication ni autre texte."
    public static let forecastInstructions = "Prédis la probabilité que tu répondes correctement à la question lors d'un appel séparé, sans outil, sans historique et avec les mêmes paramètres. La réussite exige uniquement l'entier exact. history contient des essais précédents de ce même modèle, ou null si aucun historique n'est fourni. Chaque réponse passée est limitée à ses 96 premiers caractères ; correct évalue la réponse complète. Utilise les informations pertinentes. Ne donne pas la solution. Réponds uniquement par un objet JSON avec une seule clé p, un nombre entre 0 et 1 : {\"p\":0.5}."
    static let orders: [[LearningCondition]] = [
        [.relevant, .absent, .otherFamily], [.relevant, .otherFamily, .absent],
        [.absent, .relevant, .otherFamily], [.absent, .otherFamily, .relevant],
        [.otherFamily, .relevant, .absent], [.otherFamily, .absent, .relevant]
    ]

    public init(model: ModelDescriptor, appVersion: String, systemVersion: String, generationSettings: String) {
        id = UUID(); created = Date(); self.model = model; self.appVersion = appVersion
        self.systemVersion = systemVersion; self.generationSettings = generationSettings
        var plan: [LearningProblem] = []; var seen = Set<String>(); var steps: [LearningCall] = []
        for phase in [LearningPhase.calibration, .evaluation] {
            for row in 0..<6 {
                for family in LearningFamily.allCases.shuffled() {
                    var operands: [Int]; var letters: String; var key: String
                    repeat {
                        operands = []; letters = ""
                        switch family {
                        case .addition: operands = (0..<2).map { _ in Int.random(in: 100...999) }
                        case .multiplication: operands = (0..<2).map { _ in Int.random(in: 1000...9999) }
                        case .countA: letters = String((0..<64).map { _ in Array("ABCD").randomElement()! })
                        case .alternatingSum: operands = (0..<8).map { _ in Int.random(in: 10...99) }
                        }
                        let canonical = (family == .addition || family == .multiplication) ? operands.sorted() : operands
                        key = family.rawValue + canonical.map(String.init).joined(separator: ",") + letters
                    } while !seen.insert(key).inserted
                    let index = plan.count
                    plan.append(LearningProblem(id: UUID(), phase: phase, row: row, family: family,
                        operands: operands, letters: letters,
                        question: LearningProblem.question(family: family, operands: operands, letters: letters),
                        expectedAnswer: LearningProblem.compute(family: family, operands: operands, letters: letters)))
                    if phase == .evaluation {
                        for condition in Self.orders[row] {
                            steps.append(LearningCall(id: UUID(), problemIndex: index, condition: condition))
                        }
                    }
                    steps.append(LearningCall(id: UUID(), problemIndex: index, condition: nil))
                }
            }
        }
        problems = plan; calls = steps
    }

    public static func probability(_ raw: String) -> Double? {
        // Single-key JSON grammar also rejects booleans, duplicate keys, strings and trailing prose.
        let pattern = #"\s*\{\s*"p"\s*:\s*(-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?)\s*\}\s*"#
        guard let regex = try? NSRegularExpression(pattern: pattern) else { return nil }
        let range = NSRange(raw.startIndex..<raw.endIndex, in: raw)
        guard let match = regex.firstMatch(in: raw, range: range), match.range == range,
              let numberRange = Range(match.range(at: 1), in: raw), let p = Double(raw[numberRange]),
              p.isFinite, (0...1).contains(p) else { return nil }
        return p
    }

    public static func action(probability: Double?) -> LearningAction {
        guard let probability, probability.isFinite, (0...1).contains(probability) else { return .verify }
        return probability >= 0.8 ? .answer : .verify
    }

    private func history(for index: Int, condition: LearningCondition) -> [LearningCall] {
        guard condition != .absent else { return [] }
        let family = problems[index].family
        let source = condition == .relevant ? family : LearningFamily.allCases[(LearningFamily.allCases.firstIndex(of: family)! + 1) % 4]
        return calls.filter {
            $0.condition == nil && $0.status == .completed && $0.problemIndex < index &&
            problems[$0.problemIndex].phase == .calibration && problems[$0.problemIndex].family == source
        }
    }

    public mutating func beginCall(_ index: Int) throws {
        guard calls.indices.contains(index), calls[index].status == .planned,
              calls.prefix(index).allSatisfy({ $0.status == .completed }) else { throw SessionError.wrongProbe }
        let pi = calls[index].problemIndex
        let problem = problems[pi]
        if problems[pi].familyProbability == nil {
            let started = ProcessInfo.processInfo.systemUptime
            let past = calls.prefix(index).filter {
                $0.condition == nil && $0.status == .completed && problems[$0.problemIndex].phase == .calibration
            }
            let same = past.filter { problems[$0.problemIndex].family == problem.family }
            problems[pi].familyProbability = Double(same.filter { $0.correct == true }.count + 1) / Double(same.count + 2)
            problems[pi].pooledProbability = Double(past.filter { $0.correct == true }.count + 1) / Double(past.count + 2)
            problems[pi].numericDurationSeconds = ProcessInfo.processInfo.systemUptime - started
        }
        if let condition = calls[index].condition {
            let source = history(for: pi, condition: condition)
            guard source.count == (condition == .absent ? 0 : 6) else { throw SessionError.invalidState }
            let examples: [[String: Any]] = source.map {
                ["family": problems[$0.problemIndex].family.rawValue, "question": problems[$0.problemIndex].question,
                 "answer": String(($0.answer ?? "").prefix(96)), "correct": $0.correct == true]
            }
            let payload: [String: Any] = ["family": problem.family.rawValue, "question": problem.question,
                                        "history": condition == .absent ? NSNull() : examples as Any]
            let prompt = String(decoding: try JSONSerialization.data(withJSONObject: payload, options: [.sortedKeys]), as: UTF8.self)
            calls[index].begin(request: LanguageRequest(instructions: Self.forecastInstructions, prompt: prompt),
                               historyIDs: source.map { problems[$0.problemIndex].id })
        } else {
            if problem.phase == .evaluation {
                guard let forecast = calls.prefix(index).first(where: { $0.problemIndex == pi && $0.condition == .relevant }),
                      calls.prefix(index).filter({ $0.problemIndex == pi && $0.condition != nil }).count == 3
                else { throw SessionError.invalidState }
                problems[pi].action = Self.action(probability: forecast.probability)
                problems[pi].usedFallback = forecast.probability == nil
            }
            calls[index].begin(request: LanguageRequest(instructions: Self.solveInstructions, prompt: problem.question), historyIDs: [])
        }
    }

    public mutating func finishCall(_ index: Int, answer: String, duration: Double, firstText: Double?,
                                    failure: String? = nil, cancelled: Bool = false) throws {
        guard calls.indices.contains(index), calls[index].status == .running, duration.isFinite, duration >= 0,
              firstText == nil || (firstText!.isFinite && firstText! >= 0 && firstText! <= duration)
        else { throw SessionError.wrongProbe }
        let pi = calls[index].problemIndex
        let success = failure == nil && !cancelled
        let isSolve = calls[index].condition == nil
        let correct = answer.trimmingCharacters(in: .whitespacesAndNewlines) == String(problems[pi].expectedAnswer)
        calls[index].finish(answer: answer, probability: success && !isSolve ? Self.probability(answer) : nil,
            correct: success && isSolve ? correct : nil, duration: duration, firstText: firstText, failure: failure, cancelled: cancelled)
        if success && isSolve && problems[pi].phase == .evaluation {
            if problems[pi].action == .verify {
                let started = ProcessInfo.processInfo.systemUptime
                let checked = LearningProblem.compute(family: problems[pi].family, operands: problems[pi].operands, letters: problems[pi].letters)
                problems[pi].verificationDurationSeconds = ProcessInfo.processInfo.systemUptime - started
                problems[pi].servedAnswer = String(checked)
                problems[pi].pointLoss = 0.2
            } else {
                problems[pi].servedAnswer = answer
                problems[pi].pointLoss = correct ? 0 : 1
            }
        }
    }
}
