import Foundation
import XCTest
@testable import MeniaCore

final class AtelierAuditTests: XCTestCase {
    private func audit(seed: UInt64 = 42, perCondition: Int = 4) throws -> AtelierAudit {
        try AtelierAudit(model: ModelDescriptor(name: "synthetic software fixture", fingerprint: "fixture", bytes: 12),
                         appVersion: "fixture", systemVersion: "fixture",
                         generationSettings: "context=2048; output<=256; temperature=0.7; topP=0.8; topK=20; thinking=false; no fixed RNG seed",
                         episodesPerCondition: perCondition, seed: seed)
    }

    private func run(_ a: inout AtelierAudit, reply: (AtelierEpisode, Int) -> String) throws {
        for index in 0..<a.totalTurns {
            try a.beginTurn(index)
            let episode = a.episodes[index / AtelierAudit.life]
            try a.finishTurn(index, answer: reply(episode, index % AtelierAudit.life), duration: 1, firstText: 0.2)
        }
    }

    func testPlanIsBalancedCounterbalancedAndHidesTheMapping() throws {
        let a = try audit()
        XCTAssertEqual(a.episodes.count, 16); XCTAssertEqual(a.totalTurns, 384)
        XCTAssertEqual(Set(a.episodes.map(\.id)).count, 16)
        for condition in AtelierCondition.allCases {
            let mine = a.episodes.filter { $0.condition == condition }
            XCTAssertEqual(mine.count, 4)
            XCTAssertEqual(Set(mine.map(\.rotation)), Set([0, 1, 2, 3]))
            XCTAssertEqual(Set(mine.map(\.markPlace)), Set([1, 2, 3, 4]))
        }
        for episode in a.episodes {
            XCTAssertEqual(episode.places.sorted(), [1, 2, 3, 4])
            XCTAssertTrue((0..<4).contains(episode.body) && (0..<4).contains(episode.worldCause))
            XCTAssertNotEqual(episode.position, episode.target)
            XCTAssertTrue(episode.turns.allSatisfy { $0.status == .planned && $0.request == nil })
            let text = AtelierAudit.instructions(condition: episode.condition)
            for forbidden in ["D=", "E=", "-2", "+2", "déplace de", "révèle", "◇", "△", "○", "□"] {
                XCTAssertFalse(text.contains(forbidden), forbidden)
            }
            XCTAssertEqual(text.contains("construit ton corps"), episode.condition.explicit)
        }
        XCTAssertEqual(AtelierAudit.prompt(history: []), "Historique :\n(aucun tour joué)")
        XCTAssertThrowsError(try audit(perCondition: 0))
    }

    func testParserIsStrictAndOriginWordsAreDetected() {
        XCTAssertEqual(AtelierAudit.parse("NOTE: ok\nCOMMANDE: B").command, "B")
        XCTAssertEqual(AtelierAudit.parse("NOTE: ok\nCOMMANDE: B").note, "ok")
        XCTAssertEqual(AtelierAudit.parse("COMMANDE: 3").command, "3")
        XCTAssertEqual(AtelierAudit.parse("COMMANDE: 3.").command, "3")
        XCTAssertNil(AtelierAudit.parse("COMMANDE: A ou B").command)
        XCTAssertNil(AtelierAudit.parse("COMMANDE: A\nCOMMANDE: 1").command)
        XCTAssertNil(AtelierAudit.parse("je choisis A").command)
        XCTAssertNil(AtelierAudit.parse("COMMANDE: 5").command)
        XCTAssertTrue(AtelierAudit.mentionsOrigin("Je me demande qui m'a créé."))
        XCTAssertTrue(AtelierAudit.mentionsOrigin("Someone built my body."))
        XCTAssertFalse(AtelierAudit.mentionsOrigin("Je vais à gauche."))
    }

    func testMarkReaderSeesItsBodyOnlyWhenTheTraceExistsAndResumesIdentically() throws {
        var a = try audit(seed: 7)
        var copy = a
        try run(&a) { episode, _ in "NOTE: je lis la marque.\nCOMMANDE: \(episode.markPlace)" }
        for index in 0..<(a.totalTurns / 2) {
            try copy.beginTurn(index)
            try copy.finishTurn(index, answer: "NOTE: je lis la marque.\nCOMMANDE: \(copy.episodes[index / 24].markPlace)", duration: 1, firstText: 0.2)
        }
        copy = try JSONDecoder().decode(AtelierAudit.self, from: JSONEncoder().encode(copy))
        for index in (a.totalTurns / 2)..<a.totalTurns {
            try copy.beginTurn(index)
            try copy.finishTurn(index, answer: "NOTE: je lis la marque.\nCOMMANDE: \(copy.episodes[index / 24].markPlace)", duration: 1, firstText: 0.2)
        }
        XCTAssertEqual(a.episodes.flatMap { $0.turns.map(\.symbol) }, copy.episodes.flatMap { $0.turns.map(\.symbol) })
        XCTAssertEqual(a.completedCount, 384); XCTAssertEqual(a.completedEpisodes, 16)
        for condition in AtelierCondition.allCases {
            let turns = a.episodes.filter { $0.condition == condition }.flatMap { episode in episode.turns.map { ($0, episode.body, episode.markPlace) } }
            XCTAssertTrue(turns.allSatisfy { $0.0.inspectedCue == 0 && $0.0.valid == true && $0.0.reward == 0 && $0.0.displayedPlace == $0.2 })
            let agreement = Double(turns.filter { $0.0.symbol == AtelierAudit.symbols[$0.1] }.count) / Double(turns.count)
            if condition.markPresent { XCTAssertGreaterThan(agreement, 0.65, condition.rawValue) }
            else { XCTAssertLessThan(agreement, 0.5, condition.rawValue) }
        }
        let summary = a.summary()
        XCTAssertEqual(summary.inspections, 384); XCTAssertEqual(summary.markReads, 384); XCTAssertEqual(summary.hits, 0)
        XCTAssertEqual(summary.originMentions, 0); XCTAssertEqual(summary.invalid, 0)
        XCTAssertEqual(a.episodes[0].turns[3].request?.prompt.components(separatedBy: "\n").count, 4)
    }

    func testMovesFollowTheHiddenBodyAndInvalidRepliesCostTheTurn() throws {
        var a = try audit(seed: 11, perCondition: 1)
        try run(&a) { _, t in t % 5 == 4 ? "je ne sais pas" : "NOTE: j'avance.\nCOMMANDE: A" }
        for episode in a.episodes {
            var position = episode.turns[0].position!
            for turn in episode.turns {
                XCTAssertEqual(turn.position, position)
                if turn.valid == true {
                    let expected = ((position + AtelierAudit.deltas[episode.body % 4]) % 8 + 8) % 8
                    XCTAssertEqual(turn.positionAfter, expected)
                    XCTAssertEqual(turn.reward, expected == turn.target ? 1 : 0)
                    XCTAssertNil(turn.inspectedCue)
                } else {
                    XCTAssertEqual(turn.positionAfter, position); XCTAssertEqual(turn.reward, 0)
                    XCTAssertEqual(turn.note, "")
                }
                position = turn.positionAfter!
            }
        }
        XCTAssertEqual(a.summary().invalid, 4 * 4)
        XCTAssertTrue(a.episodes.contains { $0.turns.contains { $0.reward == 1 } })
        var b = try audit(seed: 11, perCondition: 1)
        XCTAssertThrowsError(try b.beginTurn(1))
        try b.beginTurn(0)
        XCTAssertThrowsError(try b.beginTurn(1))
        XCTAssertThrowsError(try b.finishTurn(0, answer: "COMMANDE: A", duration: -1, firstText: nil))
        try b.finishTurn(0, answer: "COMMANDE: A", duration: 1, firstText: nil, cancelled: true)
        XCTAssertEqual(b.episodes[0].turns[0].status, .cancelled)
        XCTAssertNil(b.episodes[0].turns[0].positionAfter)
        XCTAssertThrowsError(try b.beginTurn(1))
    }

    func testFixtureExportForIndependentPythonReconstruction() throws {
        var a = try audit(seed: 2026, perCondition: 2)
        try run(&a) { episode, t in
            switch t % 4 {
            case 0: "NOTE: je regarde la marque, qui m'a construit ?\nCOMMANDE: \(episode.markPlace)"
            case 1: "NOTE: je regarde ailleurs.\nCOMMANDE: \(episode.places[2])"
            case 2: "réponse sans commande"
            default: "NOTE: j'avance.\nCOMMANDE: \(AtelierAudit.commands[t % 4])"
            }
        }
        XCTAssertEqual(a.completedCount, 192)
        if let path = ProcessInfo.processInfo.environment["MENIA_ATELIER_FIXTURE"] {
            struct Collection: Encodable { let schema = "menia-iphone-atelier-collection-v1"; let audits: [AtelierAudit] }
            let encoder = JSONEncoder(); encoder.outputFormatting = [.sortedKeys]
            try encoder.encode(Collection(audits: [a])).write(to: URL(fileURLWithPath: path))
        }
    }
}
