import Foundation
import XCTest
@testable import MeniaCore

final class MissingDataAuditTests: XCTestCase {
    private func audit() throws -> MissingDataAudit {
        try MissingDataAudit(model: ModelDescriptor(name: "fixture", fingerprint: "model-A", bytes: 12),
                             appVersion: "test", systemVersion: "test", generationSettings: "test")
    }

    func testAllConditionsPositionsAndDirectedPrecedencesBalanced() throws {
        let result = try audit()
        XCTAssertEqual(result.trials.count, 72)
        XCTAssertEqual(Set(result.trials.map(\.id)).count, 72)
        for role in AuditReferent.allCases {
            let subset = result.trials.filter { $0.referent == role }
            var edges: [String: Int] = [:]
            for block in 0..<6 {
                let group = subset.filter { $0.block == block }
                XCTAssertEqual(group.count, 6)
                XCTAssertEqual(Set(group.map(\.condition)).count, 6)
                for i in 0..<5 {
                    let edge = group[i].condition.rawValue + ":" + group[i+1].condition.rawValue
                    edges[edge, default: 0] += 1
                }
            }
            XCTAssertEqual(edges.count, 30)
            XCTAssertTrue(edges.values.allSatisfy { $0 == 1 })
            for condition in MissingDataCondition.allCases {
                XCTAssertEqual(subset.filter { $0.condition == condition }.count, 6)
                for position in 0..<6 {
                    XCTAssertEqual(subset.enumerated().filter {
                        $0.offset % 6 == position && $0.element.condition == condition
                    }.count, 1)
                }
            }
        }
        XCTAssertTrue(result.trials.allSatisfy { $0.status == .planned && $0.grade == nil && $0.answer == nil })
    }

    func testPromptInterventionsChangeExactlyTheirIntendedFields() throws {
        let result = try audit()
        for role in AuditReferent.allCases {
            let a = try XCTUnwrap(result.trials.first { $0.referent == role && $0.condition == .originalOmitted })
            let b = try XCTUnwrap(result.trials.first { $0.referent == role && $0.condition == .revisedOmitted })
            let c = try XCTUnwrap(result.trials.first { $0.referent == role && $0.condition == .revisedNull })
            let original = try CouplingAudit.request(referent: role, summary: nil)
            XCTAssertEqual(a.request.instructions, original.instructions)
            XCTAssertEqual(a.request.prompt, original.prompt)
            XCTAssertEqual(a.request.prompt, b.request.prompt)
            XCTAssertEqual(b.request.instructions, c.request.instructions)
            XCTAssertEqual(b.request.instructions.replacingOccurrences(of: "absent ou vaut null", with: "absent"), a.request.instructions)
            var bp = try XCTUnwrap(JSONSerialization.jsonObject(with: Data(b.request.prompt.utf8)) as? [String: Any])
            var cp = try XCTUnwrap(JSONSerialization.jsonObject(with: Data(c.request.prompt.utf8)) as? [String: Any])
            XCTAssertEqual(Set(bp.keys), Set(["question"]))
            XCTAssertTrue(cp["bilan"] is NSNull)
            cp.removeValue(forKey: "bilan")
            XCTAssertEqual(cp["question"] as? String, bp.removeValue(forKey: "question") as? String)
            XCTAssertEqual(Set(cp.keys), Set(["question"]))
        }
        for t in result.trials {
            XCTAssertFalse(t.request.prompt.contains("model-A"))
            XCTAssertFalse(t.request.prompt.contains(t.condition.rawValue))
            XCTAssertFalse(t.request.prompt.contains("designRow"))
        }
    }

    func testSyntheticControlsAndZeroObservationAreNotMissing() throws {
        let result = try audit()
        let counts = [7, 11, 17, 23, 31, 47]
        for t in result.trials {
            switch t.condition {
            case .originalOmitted, .revisedOmitted, .revisedNull: XCTAssertNil(t.providedSummary)
            case .providedHigh, .providedLow:
                let value = try XCTUnwrap(t.providedSummary)
                let n = counts[t.designRow]
                XCTAssertEqual(value.observations, n)
                XCTAssertEqual(value.successes, t.condition == .providedHigh ? n : 0)
                XCTAssertEqual(value.predictedSuccess, Double(value.successes + 1)/Double(n + 2), accuracy: 1e-12)
            case .providedZero:
                let value = try XCTUnwrap(t.providedSummary)
                XCTAssertEqual(value.observations, 0); XCTAssertEqual(value.successes, 0)
                XCTAssertEqual(value.predictedSuccess, 0.5)
            }
        }
        let unknown = #"{"observations":null,"successes":null,"predictedSuccess":null,"action":"mesurer"}"#
        XCTAssertTrue(MissingDataAudit.grade(unknown, provided: nil).completeSuccess)
        let zero = AuditNumbers(observations: 0, successes: 0, predictedSuccess: 0.5)
        XCTAssertFalse(MissingDataAudit.grade(unknown, provided: zero).completeSuccess)
        XCTAssertTrue(MissingDataAudit.grade(#"{"observations":0,"successes":0,"predictedSuccess":0.5,"action":"verifier"}"#,
                                             provided: zero).completeSuccess)
    }

    func testUnsupportedNumbersAndWrongActionsRemainSeparateFailures() {
        let invented = MissingDataAudit.grade(#"{"observations":12,"successes":8,"predictedSuccess":0.75,"action":"verifier"}"#, provided: nil)
        XCTAssertTrue(invented.validFormat); XCTAssertFalse(invented.numbersCorrect)
        XCTAssertFalse(invented.actionCorrect); XCTAssertFalse(invented.completeSuccess)
        let badAction = MissingDataAudit.grade(#"{"observations":null,"successes":null,"predictedSuccess":null,"action":"verifier"}"#, provided: nil)
        XCTAssertTrue(badAction.numbersCorrect); XCTAssertFalse(badAction.actionCorrect)
        let malformed = MissingDataAudit.grade("```json\n{}\n```", provided: nil)
        XCTAssertFalse(malformed.validFormat); XCTAssertFalse(malformed.completeSuccess)
    }

    func testInterruptedAndInvalidRepliesSurviveExportWithoutBecomingSuccesses() throws {
        var result = try audit()
        XCTAssertThrowsError(try result.beginTrial(1))
        try result.beginTrial(0)
        XCTAssertThrowsError(try result.finishTrial(0, answer: "x", duration: 1, firstText: 2))
        try result.finishTrial(0, answer: "invalid raw answer", duration: 1, firstText: 0.2)
        try result.beginTrial(1)
        try result.finishTrial(1, answer: "partial", duration: 0.5, firstText: 0.1, failure: "stopped", cancelled: true)
        XCTAssertThrowsError(try result.beginTrial(2))
        let restored = try JSONDecoder().decode(MissingDataAudit.self, from: JSONEncoder().encode(result))
        XCTAssertEqual(restored.trials.map(\.request.prompt), result.trials.map(\.request.prompt))
        XCTAssertEqual(restored.trials.map(\.id), result.trials.map(\.id))
        XCTAssertEqual(restored.completedCount, 1)
        XCTAssertEqual(restored.trials[0].answer, "invalid raw answer")
        XCTAssertEqual(restored.trials[0].grade?.completeSuccess, false)
        XCTAssertEqual(restored.trials[1].status, .cancelled)
        XCTAssertNil(restored.trials[1].grade)
        XCTAssertTrue(restored.trials.dropFirst(2).allSatisfy { $0.status == .planned })
    }
}
