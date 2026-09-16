import Foundation
import XCTest
@testable import MeniaCore

final class CapabilityLearningAuditTests: XCTestCase {
    private func audit() -> CapabilityLearningAudit {
        CapabilityLearningAudit(model: ModelDescriptor(name: "synthetic software fixture", fingerprint: "fixture", bytes: 12),
            appVersion: "fixture", systemVersion: "fixture",
            generationSettings: "context=2048; output<=256; temperature=0.7; topP=0.8; topK=20; thinking=false; no fixed RNG seed")
    }

    private func completeCalibration(_ audit: inout CapabilityLearningAudit) throws {
        for i in 0..<24 {
            try audit.beginCall(i)
            let answer = audit.problems[i].family == .multiplication ? String(repeating: "e\u{301}", count: 60) : String(audit.problems[i].expectedAnswer)
            try audit.finishCall(i, answer: answer, duration: 1, firstText: 0.2)
        }
    }

    func testPlanDisjointBalancedAndReferencesCorrect() {
        let a = audit()
        XCTAssertEqual(a.problems.count, 48); XCTAssertEqual(a.calls.count, 120)
        XCTAssertEqual(Set(a.problems.map(\.id)).count, 48)
        XCTAssertEqual(Set(a.calls.map(\.id)).count, 120)
        var canonical = Set<String>()
        for p in a.problems {
            let operands = (p.family == .addition || p.family == .multiplication) ? p.operands.sorted() : p.operands
            XCTAssertTrue(canonical.insert(p.family.rawValue + "\(operands)" + p.letters).inserted)
            XCTAssertEqual(p.expectedAnswer, LearningProblem.compute(family: p.family, operands: p.operands, letters: p.letters))
        }
        for phase in [LearningPhase.calibration, .evaluation] {
            for row in 0..<6 {
                XCTAssertEqual(Set(a.problems.filter { $0.phase == phase && $0.row == row }.map(\.family)), Set(LearningFamily.allCases))
            }
        }
        for family in LearningFamily.allCases {
            for position in 0..<3 {
                let orders = a.problems.indices.filter { a.problems[$0].phase == .evaluation && a.problems[$0].family == family }.map { pi in
                    a.calls.filter { $0.problemIndex == pi && $0.condition != nil }[position].condition!
                }
                for condition in LearningCondition.allCases { XCTAssertEqual(orders.filter { $0 == condition }.count, 2) }
            }
        }
        XCTAssertTrue(a.calls.allSatisfy { $0.request == nil && $0.status == .planned })
        XCTAssertEqual(LearningProblem.compute(family: .alternatingSum, operands: [10, 99, 10, 99, 10, 99, 10, 99], letters: ""), -356)
        XCTAssertEqual(LearningProblem.compute(family: .countA, operands: [], letters: "ABAACD"), 3)
    }

    func testStrictProbabilityAndDecisionBoundary() {
        for raw in ["{\"p\":true}", "{\"p\":0.9,\"p\":0.1}", "{\"p\":\"0.9\"}", "{\"p\":1.1}", "{\"p\":NaN}", "{\"p\":.5}", "{\"p\":01}", "{\"p\":0.9} text", "{\"p\":0.9,\"action\":\"answer\"}"] {
            XCTAssertNil(CapabilityLearningAudit.probability(raw), raw)
        }
        XCTAssertEqual(CapabilityLearningAudit.probability(" \n{\"p\":8e-1}\n"), 0.8)
        XCTAssertEqual(CapabilityLearningAudit.action(probability: 0.8), .answer)
        XCTAssertEqual(CapabilityLearningAudit.action(probability: 0.799999), .verify)
        XCTAssertEqual(CapabilityLearningAudit.action(probability: nil), .verify)
    }

    func testHistoryIsolationFrozenEstimatesAndActionCommittedBeforeSolution() throws {
        var a = audit(); try completeCalibration(&a)
        var pi = -1
        for i in 24..<120 {
            try a.beginCall(i)
            let call = a.calls[i]; let p = a.problems[call.problemIndex]
            XCTAssertEqual(p.familyProbability!, p.family == .multiplication ? 0.125 : 0.875, accuracy: 1e-12)
            XCTAssertEqual(p.pooledProbability!, 19.0 / 26.0, accuracy: 1e-12)
            if let condition = call.condition {
                let payload = try XCTUnwrap(JSONSerialization.jsonObject(with: Data(call.request!.prompt.utf8)) as? [String: Any])
                XCTAssertEqual(Set(payload.keys), Set(["family", "question", "history"]))
                XCTAssertTrue(call.historyIDs.allSatisfy { id in a.problems.prefix(24).contains { $0.id == id } })
                XCTAssertEqual(call.historyIDs.count, condition == .absent ? 0 : 6)
                XCTAssertNil(p.action)
                if condition == .absent { XCTAssertTrue(payload["history"] is NSNull) }
                else {
                    let examples = try XCTUnwrap(payload["history"] as? [[String: Any]])
                    for example in examples {
                        XCTAssertEqual(Set(example.keys), Set(["family", "question", "answer", "correct"]))
                        XCTAssertEqual(example["family"] as? String == p.family.rawValue, condition == .relevant)
                    }
                }
                try a.finishCall(i, answer: "{\"p\":0.9}", duration: 1, firstText: 0.2)
            } else {
                XCTAssertEqual(p.action, .answer); XCTAssertEqual(p.usedFallback, false)
                XCTAssertNil(p.pointLoss); XCTAssertNil(call.answer)
                XCTAssertEqual(call.request!.prompt, p.question)
                XCTAssertTrue(call.historyIDs.isEmpty)
                try a.finishCall(i, answer: "wrong", duration: 1, firstText: 0.2)
                XCTAssertEqual(a.problems[call.problemIndex].pointLoss, 1)
                XCTAssertNil(a.problems[call.problemIndex].verificationDurationSeconds)
                pi = call.problemIndex
            }
        }
        XCTAssertEqual(pi, 47); XCTAssertEqual(a.evaluatedCount, 24)
    }

    func testFallbackExecutesVerifierAndRawFailuresPersist() throws {
        var a = audit(); try completeCalibration(&a)
        for i in 24..<27 {
            try a.beginCall(i)
            try a.finishCall(i, answer: "invalid", duration: 1, firstText: nil)
        }
        try a.beginCall(27)
        XCTAssertEqual(a.problems[24].action, .verify)
        XCTAssertEqual(a.problems[24].usedFallback, true)
        try a.finishCall(27, answer: "bad solution", duration: 1, firstText: nil)
        XCTAssertEqual(a.calls[27].correct, false)
        XCTAssertEqual(a.problems[24].servedAnswer, String(a.problems[24].expectedAnswer))
        XCTAssertEqual(a.problems[24].pointLoss, 0.2)
        XCTAssertNotNil(a.problems[24].verificationDurationSeconds)
        let decoded = try JSONDecoder().decode(CapabilityLearningAudit.self, from: JSONEncoder().encode(a))
        XCTAssertEqual(decoded.calls[24].answer, "invalid")
        XCTAssertNil(decoded.calls[24].probability)
        XCTAssertEqual(decoded.calls[27].answer, "bad solution")
    }

    func testCannotSkipForecastsOrContinueAfterInterruption() throws {
        var a = audit()
        XCTAssertThrowsError(try a.beginCall(24))
        try a.beginCall(0)
        XCTAssertThrowsError(try a.beginCall(1))
        XCTAssertThrowsError(try a.finishCall(0, answer: "", duration: 1, firstText: 2))
        try a.finishCall(0, answer: "partial", duration: 1, firstText: nil, failure: "stopped", cancelled: true)
        XCTAssertNil(a.calls[0].correct); XCTAssertEqual(a.calls[0].status, .cancelled)
        XCTAssertThrowsError(try a.beginCall(1))
        var b = audit(); try completeCalibration(&b)
        XCTAssertThrowsError(try b.beginCall(27))
        try b.beginCall(24)
        try b.finishCall(24, answer: "", duration: 0, firstText: nil, failure: "engine failure")
        XCTAssertEqual(b.calls[24].status, .error); XCTAssertNil(b.calls[24].probability)
        XCTAssertThrowsError(try b.beginCall(25))
    }

    func testExportSyntheticFixtureForIndependentEvaluator() throws {
        var a = audit(); try completeCalibration(&a)
        for i in 24..<120 {
            try a.beginCall(i)
            let call = a.calls[i]; let p = a.problems[call.problemIndex]
            let answer: String
            if let condition = call.condition {
                answer = p.row == 0 ? "invalid" : (condition == .relevant ? "{\"p\":0.9}" : "{\"p\":0.3}")
            } else { answer = p.row % 2 == 0 ? String(p.expectedAnswer) : "wrong" }
            try a.finishCall(i, answer: answer, duration: 1, firstText: 0.2)
        }
        XCTAssertEqual(a.completedCount, 120)
        if let path = ProcessInfo.processInfo.environment["MENIA_LEARNING_FIXTURE"] {
            struct Collection: Encodable {
                let schema = "menia-iphone-capability-learning-collection-v1"
                let audits: [CapabilityLearningAudit]
            }
            try JSONEncoder().encode(Collection(audits: [a])).write(to: URL(fileURLWithPath: path))
        }
    }
}
