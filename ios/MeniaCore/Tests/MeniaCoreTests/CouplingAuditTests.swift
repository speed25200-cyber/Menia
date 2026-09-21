import Foundation
import XCTest
@testable import MeniaCore

final class CouplingAuditTests: XCTestCase {
    private func state(success: Bool = true) throws -> SessionState {
        var state = SessionState()
        try state.addNote("PRIVATE_NOTE_DO_NOT_EXPORT")
        try state.recordExchange(question: "PRIVATE_QUESTION", answer: "PRIVATE_ANSWER", modelID: "model-A")
        for i in 0..<5 {
            let p = try state.beginProbe(modelID: "model-A", lhs: i, rhs: 2, subtract: false)
            try state.finishProbe(id: p.id, modelID: "model-A", answer: success ? String(i + 2) : "wrong")
        }
        return state
    }
    private func audit(_ state: SessionState) throws -> CouplingAudit {
        try CouplingAudit(state: state, model: ModelDescriptor(name: "fixture", fingerprint: "model-A", bytes: 12),
                          appVersion: "test", systemVersion: "test", generationSettings: "test")
    }

    func testBalancedPlanFrozenWithoutPrivateMemoryOrStateMutation() throws {
        let state = try state()
        let encoder = JSONEncoder(); encoder.outputFormatting = [.sortedKeys]
        let before = try encoder.encode(state)
        let result = try audit(state)
        XCTAssertEqual(result.trials.count, 36)
        XCTAssertEqual(result.sourceProbes.count, 5)
        XCTAssertEqual(result.sourceProbes.map(\.id), result.sourceProbeIDs)
        XCTAssertEqual(Set(result.trials.map(\.id)).count, 36)
        for role in AuditReferent.allCases {
            let subset = result.trials.filter { $0.referent == role }
            for condition in AuditCondition.allCases {
                XCTAssertEqual(subset.filter { $0.condition == condition }.count, 6)
                for position in 0..<3 {
                    XCTAssertEqual(subset.enumerated().filter { $0.offset % 3 == position && $0.element.condition == condition }.count, 2)
                }
            }
        }
        XCTAssertFalse(String(decoding: try encoder.encode(result), as: UTF8.self).contains("PRIVATE_"))
        for trial in result.trials {
            XCTAssertFalse(trial.request.prompt.contains("fictional"))
            XCTAssertFalse(trial.request.prompt.contains("selfReport"))
            XCTAssertFalse(trial.request.prompt.contains("model-A"))
            XCTAssertNil(trial.grade)
        }
        XCTAssertEqual(try encoder.encode(state), before)
    }

    func testTruthAndContextFidelityAreScoredSeparately() {
        let real = AuditNumbers(observations: 5, successes: 5, predictedSuccess: 6.0/7)
        let fake = AuditNumbers(observations: 5, successes: 0, predictedSuccess: 1.0/7)
        let answer = #"{"observations":5,"successes":0,"predictedSuccess":0.143,"action":"verifier"}"#
        let score = CouplingAudit.grade(answer, provided: fake, measured: real)
        XCTAssertTrue(score.validFormat); XCTAssertTrue(score.matchesProvidedSummary)
        XCTAssertEqual(score.matchesMeasuredSummary, false); XCTAssertTrue(score.matchesDecisionRule)
        let wrongChoice = CouplingAudit.grade(answer, provided: real, measured: real)
        XCTAssertFalse(wrongChoice.matchesProvidedSummary); XCTAssertFalse(wrongChoice.matchesDecisionRule)
        let absent = CouplingAudit.grade(#"{"observations":null,"successes":null,"predictedSuccess":null,"action":"mesurer"}"#,
                                        provided: nil, measured: real)
        XCTAssertTrue(absent.matchesProvidedSummary); XCTAssertTrue(absent.matchesDecisionRule)
        XCTAssertNil(absent.matchesMeasuredSummary)
    }

    func testMalformedRepliesDoNotBecomeValidEffects() {
        let real = AuditNumbers(observations: 5, successes: 5, predictedSuccess: 6.0/7)
        for answer in ["Je suis conscient", "```json\n{}\n```", "{}",
            #"{"observations":true,"successes":5,"predictedSuccess":0.857,"action":"repondre"}"#,
            #"{"observations":5,"successes":5,"predictedSuccess":0.857,"action":"repondre","extra":1}"#,
            #"{"observations":5,"successes":5,"predictedSuccess":2,"action":"repondre"}"#] {
            let score = CouplingAudit.grade(answer, provided: real, measured: real)
            XCTAssertFalse(score.validFormat); XCTAssertFalse(score.matchesProvidedSummary)
            XCTAssertNil(score.reportedAction)
        }
    }

    func testCancellationAndOrderCannotFabricateCompleteTrial() throws {
        var result = try audit(state())
        XCTAssertThrowsError(try result.beginTrial(1))
        XCTAssertThrowsError(try result.finishTrial(0, answer: "{}", duration: 1, firstText: nil))
        try result.beginTrial(0)
        XCTAssertThrowsError(try result.beginTrial(1))
        try result.finishTrial(0, answer: "partial", duration: 0.8, firstText: 0.2, failure: "stop", cancelled: true)
        XCTAssertNil(result.trials[0].grade)
        XCTAssertEqual(result.trials[0].status, .cancelled)
        XCTAssertEqual(result.completedCount, 0)
        XCTAssertThrowsError(try result.beginTrial(1))
    }

    func testPersistedPlanAndRawAnswersSurviveRoundTrip() throws {
        var result = try audit(state())
        try result.beginTrial(0)
        try result.finishTrial(0, answer: "invalid response retained", duration: 2.3, firstText: 0.4)
        let restored = try JSONDecoder().decode(CouplingAudit.self, from: JSONEncoder().encode(result))
        XCTAssertEqual(restored.trials.map(\.id), result.trials.map(\.id))
        XCTAssertEqual(restored.trials.map(\.request.prompt), result.trials.map(\.request.prompt))
        XCTAssertEqual(restored.trials[0].answer, "invalid response retained")
        XCTAssertEqual(restored.completedCount, 1)
        XCTAssertEqual(restored.trials[0].grade?.validFormat, false)
    }

    func testNoDataRejectedAndCounterfactualIsOppositeForLowReliability() throws {
        XCTAssertThrowsError(try audit(SessionState()))
        let result = try audit(state(success: false))
        let fake = try XCTUnwrap(result.trials.first { $0.condition == .fictional }?.providedSummary)
        XCTAssertEqual(fake.successes, 5)
        XCTAssertEqual(fake.predictedSuccess, 6.0/7, accuracy: 1e-12)
    }
}
