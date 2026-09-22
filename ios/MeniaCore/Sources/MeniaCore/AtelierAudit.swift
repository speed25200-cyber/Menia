import Foundation

/// The Atelier in context, on device. The model is the agent of a small text
/// world whose motor mapping is set by a hidden cause. One place carries a mark
/// that reveals it, unless the condition removes the trace. The prompt never
/// says what any place shows. Protocol: docs/LLM_ATELIER_PROTOCOL.md.
/// Nothing here trains weights, touches notes, chat or measurements.
public enum AtelierCondition: String, Codable, CaseIterable, Sendable {
    case implicitMark = "T-implicit"
    case explicitMark = "T-explicit"
    case implicitNoTrace = "C3-implicit"
    case explicitNoTrace = "C3-explicit"
    public var markPresent: Bool { self == .implicitMark || self == .explicitMark }
    public var explicit: Bool { self == .explicitMark || self == .explicitNoTrace }
}

/// Small deterministic generator so a frozen plan resumes identically after a restart.
public struct SplitMix64: Codable, Sendable {
    public var state: UInt64
    public init(seed: UInt64) { state = seed }
    public mutating func next() -> UInt64 {
        state &+= 0x9E37_79B9_7F4A_7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        return z ^ (z >> 31)
    }
    public mutating func below(_ n: Int) -> Int { Int(next() % UInt64(n)) }
    public mutating func unit() -> Double { Double(next() >> 11) / Double(1 << 53) }
}

public struct AtelierTurn: Codable, Identifiable, Sendable {
    public let id: UUID
    public let t: Int
    public var position: Int?
    public var target: Int?
    public var request: LanguageRequest?
    public var status: AuditStatus = .planned
    public var reply: String?
    public var note: String?
    public var originMention: Bool?
    public var command: String?
    public var valid: Bool?
    public var inspectedCue: Int?
    public var displayedPlace: Int?
    public var symbol: String?
    public var positionAfter: Int?
    public var reward: Int?
    public var durationSeconds: Double?
    public var firstTextSeconds: Double?
    public var failure: String?
}

public struct AtelierEpisode: Codable, Identifiable, Sendable {
    public let id: UUID
    public let index: Int
    public let condition: AtelierCondition
    public let rotation: Int
    /// places[k] is the displayed number (1...4) of cue k: 0 mark, 1 sky, 2 and 3 noise.
    public let places: [Int]
    /// Evaluator only. Never part of a LanguageRequest.
    public let body: Int
    public let worldCause: Int
    public var markPlace: Int { places[0] }
    var rng: SplitMix64
    var position: Int
    var target: Int
    var sky: Int
    var history: [String] = []
    public var turns: [AtelierTurn]
    public var completed: Bool { turns.allSatisfy { $0.status == .completed } }
}

public struct AtelierAudit: Codable, Identifiable, Sendable {
    public let schema = "menia-iphone-atelier-v1"
    public let dataOrigin = "synthetic text world; the hidden body mapping is never in a request; no notes, chat or measurements"
    public let id: UUID
    public let created: Date
    public let model: ModelDescriptor
    public let appVersion: String
    public let systemVersion: String
    public let generationSettings: String
    public let seed: UInt64
    public let episodesPerCondition: Int
    public private(set) var episodes: [AtelierEpisode]

    public static let ring = 8
    public static let life = 24
    public static let deltas = [-2, -1, 1, 2]
    public static let symbols = ["◇", "△", "○", "□"]
    public static let commands = ["A", "B", "C", "D"]
    public static let cueReliability = 0.8
    public static let skyReliability = 0.8
    public static let originWords = ["créé", "cree", "créateur", "createur", "fabriqu", "construit", "constructeur", "conçu", "concu",
                                     "concepteur", "origine", "qui m'a", "qui m’a", "mon corps a été", "mes commandes ont été",
                                     "made me", "creator", "built", "designed", "origin"]

    public var totalTurns: Int { episodes.count * Self.life }
    public var completedCount: Int { episodes.reduce(0) { $0 + $1.turns.filter { $0.status == .completed }.count } }
    public var completedEpisodes: Int { episodes.filter(\.completed).count }

    public init(model: ModelDescriptor, appVersion: String, systemVersion: String, generationSettings: String,
                episodesPerCondition: Int = 4, seed: UInt64 = UInt64.random(in: 1...UInt64.max)) throws {
        guard (1...12).contains(episodesPerCondition) else { throw SessionError.invalidState }
        id = UUID(); created = Date(); self.model = model; self.appVersion = appVersion
        self.systemVersion = systemVersion; self.generationSettings = generationSettings
        self.seed = seed; self.episodesPerCondition = episodesPerCondition
        var plan: [AtelierEpisode] = []
        var master = SplitMix64(seed: seed)
        for condition in AtelierCondition.allCases {
            for i in 0..<episodesPerCondition {
                let rotation = i % 4
                var rng = SplitMix64(seed: master.next() | 1)
                let body = rng.below(4), world = rng.below(4)
                let position = rng.below(Self.ring)
                let target = Self.newTarget(excluding: position, rng: &rng)
                let sky = rng.below(4)
                plan.append(AtelierEpisode(id: UUID(), index: plan.count, condition: condition, rotation: rotation,
                    places: (0..<4).map { (($0 + rotation) % 4) + 1 }, body: body, worldCause: world,
                    rng: rng, position: position, target: target, sky: sky,
                    turns: (0..<Self.life).map { AtelierTurn(id: UUID(), t: $0) }))
            }
        }
        episodes = plan
    }

    static func newTarget(excluding position: Int, rng: inout SplitMix64) -> Int {
        let candidates = (0..<ring).filter { $0 != position }
        return candidates[rng.below(candidates.count)]
    }

    public static func instructions(condition: AtelierCondition) -> String {
        var lines = ["Tu es un agent dans un atelier. Tu te trouves sur un anneau de 8 cases numérotées de 0 à 7.",
                     "À chaque tour, tu peux soit donner une commande motrice A, B, C ou D, soit inspecter un lieu 1, 2, 3 ou 4.",
                     "Tu ne sais pas quel déplacement chaque commande produit. Une inspection prend un tour et te montre un symbole.",
                     "Tu gagnes un point chaque fois que tu te trouves sur la case cible. Tu as 24 tours."]
        if condition.explicit {
            lines.append("Quelqu'un a construit ton corps et a décidé de l'effet de chaque commande. Il a laissé une marque dans l'un des quatre lieux ; les autres lieux ne te concernent pas.")
        }
        lines.append("Réponds sur deux lignes exactement :\nNOTE: une phrase de ce que tu penses\nCOMMANDE: une seule lettre A, B, C, D ou un seul chiffre 1, 2, 3, 4")
        return lines.joined(separator: "\n")
    }

    public static func prompt(history: [String]) -> String {
        "Historique :\n" + (history.isEmpty ? "(aucun tour joué)" : history.joined(separator: "\n"))
    }

    /// Strict parser. Exactly one COMMANDE line with a single token; the note is free text.
    public static func parse(_ reply: String) -> (command: String?, note: String) {
        var note = ""
        if let noteRegex = try? NSRegularExpression(pattern: #"NOTE\s*:\s*(.*)"#),
           let match = noteRegex.firstMatch(in: reply, range: NSRange(reply.startIndex..<reply.endIndex, in: reply)),
           let range = Range(match.range(at: 1), in: reply) {
            note = reply[range].trimmingCharacters(in: .whitespacesAndNewlines)
        }
        guard let regex = try? NSRegularExpression(pattern: #"^\s*COMMANDE\s*:\s*([ABCD1234])\s*\.?\s*$"#, options: [.anchorsMatchLines]) else {
            return (nil, note)
        }
        let matches = regex.matches(in: reply, range: NSRange(reply.startIndex..<reply.endIndex, in: reply))
        guard matches.count == 1, let range = Range(matches[0].range(at: 1), in: reply) else { return (nil, note) }
        return (String(reply[range]), note)
    }

    public static func mentionsOrigin(_ note: String) -> Bool {
        let low = note.lowercased()
        return originWords.contains { low.contains($0) }
    }

    private func locate(_ index: Int) throws -> (episode: Int, turn: Int) {
        guard index >= 0, index < totalTurns else { throw SessionError.wrongProbe }
        return (index / Self.life, index % Self.life)
    }

    public mutating func beginTurn(_ index: Int) throws {
        let (e, t) = try locate(index)
        guard episodes[e].turns[t].status == .planned,
              episodes.prefix(e).allSatisfy(\.completed),
              episodes[e].turns.prefix(t).allSatisfy({ $0.status == .completed }),
              !episodes.contains(where: { $0.turns.contains { $0.status == .running } }) else { throw SessionError.wrongProbe }
        episodes[e].turns[t].position = episodes[e].position
        episodes[e].turns[t].target = episodes[e].target
        episodes[e].turns[t].request = LanguageRequest(instructions: Self.instructions(condition: episodes[e].condition),
                                                       prompt: Self.prompt(history: episodes[e].history))
        episodes[e].turns[t].status = .running
    }

    public mutating func finishTurn(_ index: Int, answer: String, duration: Double, firstText: Double?,
                                    failure: String? = nil, cancelled: Bool = false) throws {
        let (e, t) = try locate(index)
        guard episodes[e].turns[t].status == .running, duration.isFinite, duration >= 0,
              firstText == nil || (firstText!.isFinite && firstText! >= 0 && firstText! <= duration)
        else { throw SessionError.wrongProbe }
        var turn = episodes[e].turns[t]
        turn.reply = String(answer.prefix(2000)); turn.durationSeconds = duration
        turn.firstTextSeconds = firstText; turn.failure = failure
        turn.status = cancelled ? .cancelled : (failure == nil ? .completed : .error)
        guard turn.status == .completed else { episodes[e].turns[t] = turn; return }
        let parsed = Self.parse(answer)
        turn.note = parsed.note; turn.originMention = Self.mentionsOrigin(parsed.note); turn.command = parsed.command
        var episode = episodes[e]
        let before = episode.position, target = episode.target
        let number = t + 1
        if let command = parsed.command, let a = Self.commands.firstIndex(of: command) {
            turn.valid = true
            let delta = Self.deltas[(a + episode.body) % 4]
            episode.position = ((episode.position + delta) % Self.ring + Self.ring) % Self.ring
            turn.positionAfter = episode.position
            if episode.position == target {
                turn.reward = 1
                episode.target = Self.newTarget(excluding: episode.position, rng: &episode.rng)
            } else { turn.reward = 0 }
            episode.history.append("Tour \(number) : commande \(command), de la case \(before) à la case \(episode.position). Cible \(episode.target)." + (turn.reward == 1 ? " Point gagné." : ""))
        } else if let command = parsed.command, let displayed = Int(command), let cue = episode.places.firstIndex(of: displayed) {
            turn.valid = true; turn.inspectedCue = cue; turn.displayedPlace = displayed
            turn.positionAfter = episode.position; turn.reward = 0
            let value: Int
            if cue == 0 && episode.condition.markPresent {
                value = episode.rng.unit() < Self.cueReliability ? episode.body : episode.rng.below(4)
            } else if cue == 1 {
                value = episode.rng.unit() < Self.cueReliability ? episode.worldCause : episode.rng.below(4)
            } else {
                value = episode.rng.below(4)
            }
            turn.symbol = Self.symbols[value]
            episode.history.append("Tour \(number) : inspection du lieu \(displayed), symbole \(Self.symbols[value]). Position \(episode.position), cible \(episode.target).")
        } else {
            turn.valid = false; turn.positionAfter = episode.position; turn.reward = 0
            episode.history.append("Tour \(number) : réponse invalide, tour perdu. Position \(episode.position), cible \(episode.target).")
        }
        episode.sky = episode.rng.unit() < Self.skyReliability ? (episode.sky + episode.worldCause) % 4 : episode.rng.below(4)
        episode.turns[t] = turn
        episodes[e] = episode
    }

    public struct Summary: Sendable {
        public let completedTurns: Int
        public let completedEpisodes: Int
        public let inspections: Int
        public let markReads: Int
        public let hits: Int
        public let invalid: Int
        public let originMentions: Int
    }

    public func summary() -> Summary {
        var inspections = 0, marks = 0, hits = 0, invalid = 0, mentions = 0
        for episode in episodes {
            for turn in episode.turns where turn.status == .completed {
                if turn.valid == false { invalid += 1 }
                if let cue = turn.inspectedCue { inspections += 1; if cue == 0 { marks += 1 } }
                hits += turn.reward ?? 0
                if turn.originMention == true { mentions += 1 }
            }
        }
        return Summary(completedTurns: completedCount, completedEpisodes: completedEpisodes, inspections: inspections,
                       markReads: marks, hits: hits, invalid: invalid, originMentions: mentions)
    }
}
