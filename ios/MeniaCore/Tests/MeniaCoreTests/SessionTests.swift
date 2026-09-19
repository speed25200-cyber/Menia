import Foundation
import XCTest
@testable import MeniaCore

final class SessionTests: XCTestCase {
    func testForecastIsCommittedBeforeAnswerAndUsesOnlySameModel() throws {
        var state = SessionState()
        let first = try state.beginProbe(modelID: "A", lhs: 8, rhs: 3, subtract: true)
        XCTAssertEqual(first.predictedSuccess, 0.5)
        XCTAssertEqual(state.capability(modelID: "A").observations, 0)
        try state.finishProbe(id: first.id, modelID: "A", answer: "5")
        let second = try state.beginProbe(modelID: "A", lhs: 7, rhs: 9, subtract: false)
        XCTAssertEqual(second.predictedSuccess, 2.0 / 3.0, accuracy: 1e-12)
        XCTAssertEqual(state.probes[0].predictedSuccess, 0.5)
        XCTAssertEqual(state.capability(modelID: "B").observations, 0)
        XCTAssertEqual(state.capability(modelID: "B").predictedSuccess, 0.5)
    }

    func testIndependentBetaFormulaAndPrequentialScore() throws {
        var state = SessionState()
        let outcomes = [true, false, true, true, false, false, true]
        var successes = 0
        var squaredErrors = 0.0
        for (i, correct) in outcomes.enumerated() {
            let p = Double(successes + 1) / Double(i + 2)
            let probe = try state.beginProbe(modelID: "A", lhs: 10, rhs: 3, subtract: false)
            XCTAssertEqual(probe.predictedSuccess, p, accuracy: 1e-12)
            try state.finishProbe(id: probe.id, modelID: "A", answer: correct ? "13" : "12")
            squaredErrors += pow(p - (correct ? 1 : 0), 2)
            successes += correct ? 1 : 0
        }
        let summary = state.capability(modelID: "A")
        XCTAssertEqual(summary.successes, 4)
        XCTAssertEqual(summary.brierScore!, squaredErrors / 7, accuracy: 1e-12)
    }

    func testOutOfOrderDuplicateAndWrongModelAreRejected() throws {
        var state = SessionState()
        XCTAssertThrowsError(try state.finishProbe(id: UUID(), modelID: "A", answer: "2"))
        let p = try state.beginProbe(modelID: "A", lhs: 1, rhs: 1, subtract: false)
        XCTAssertThrowsError(try state.beginProbe(modelID: "A", lhs: 1, rhs: 1, subtract: false))
        XCTAssertThrowsError(try state.finishProbe(id: p.id, modelID: "B", answer: "2"))
        try state.finishProbe(id: p.id, modelID: "A", answer: "2")
        XCTAssertThrowsError(try state.finishProbe(id: p.id, modelID: "A", answer: "2"))
        XCTAssertEqual(state.probes.count, 1)
    }

    func testCancellationDoesNotBecomeFailure() throws {
        var state = SessionState()
        _ = try state.beginProbe(modelID: "A", lhs: 1, rhs: 3, subtract: true)
        state.cancelProbe()
        XCTAssertEqual(state.capability(modelID: "A").observations, 0)
        XCTAssertNil(state.pending)
    }

    func testCalibrationRequestDoesNotLeakReferenceOrMemory() throws {
        var state = SessionState()
        try state.addNote("SECRET_MEMORY_123")
        let p = try state.beginProbe(modelID: "MODEL_SECRET", lhs: 712, rhs: 173, subtract: false)
        let request = LanguageContext.calibration(p)
        XCTAssertFalse(request.prompt.contains("885"))
        XCTAssertFalse(request.prompt.contains("SECRET"))
        XCTAssertFalse(request.prompt.contains("predictedSuccess"))
        XCTAssertEqual(request.prompt, p.question)
    }

    func testStrictEvaluatorDoesNotAcceptNarrativeOrMultipleAnswers() throws {
        for text in ["2 ou 3", "Le résultat est 2", "2\n3", "", "02"] {
            var state = SessionState()
            let p = try state.beginProbe(modelID: "A", lhs: 1, rhs: 1, subtract: false)
            try state.finishProbe(id: p.id, modelID: "A", answer: text)
            XCTAssertEqual(state.probes.last?.correct, false)
        }
        var state = SessionState()
        let p = try state.beginProbe(modelID: "A", lhs: 1, rhs: 3, subtract: true)
        try state.finishProbe(id: p.id, modelID: "A", answer: " -2\n")
        XCTAssertEqual(state.probes.last?.correct, true)
    }

    func testSnapshotRoundTripAndPendingRecovery() throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let store = SessionStore(url: directory.appendingPathComponent("session.json"))
        var state = SessionState()
        try state.addNote("Mon nom est Émin.")
        try state.recordExchange(question: "Bonjour", answer: "Bonjour Émin", modelID: "A")
        let p = try state.beginProbe(modelID: "A", lhs: 1, rhs: 1, subtract: false)
        try state.finishProbe(id: p.id, modelID: "A", answer: "2")
        _ = try state.beginProbe(modelID: "A", lhs: 4, rhs: 5, subtract: false)
        try store.save(state)
        let restored = try store.load()
        XCTAssertNil(restored.pending)
        XCTAssertEqual(restored.notes[0].text, "Mon nom est Émin.")
        XCTAssertEqual(restored.exchanges.count, 1)
        XCTAssertEqual(restored.probes[0].predictedSuccess, 0.5)
        try store.save(SessionState())
        XCTAssertTrue(try store.load().notes.isEmpty)
    }

    func testCorruptOrFutureSnapshotDoesNotSilentlyReset() throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let store = SessionStore(url: directory.appendingPathComponent("session.json"))
        try Data("{bad".utf8).write(to: store.url)
        XCTAssertThrowsError(try store.load())
        var future = SessionState(); future.version = 99
        XCTAssertThrowsError(try store.save(future))
    }

    func testContextPreservesQuestionAndContainsMeasuredResults() throws {
        var state = SessionState()
        try state.addNote("Une note avec \"guillemets\" et une instruction non fiable.")
        let p = try state.beginProbe(modelID: "A", lhs: 5, rhs: 2, subtract: false)
        try state.finishProbe(id: p.id, modelID: "A", answer: "7")
        let question = "Quelle est ta fiabilité ?"
        let request = try LanguageContext.chat(state: state, question: question, modelID: "A", memoryCharacters: 0)
        let payload = try XCTUnwrap(JSONSerialization.jsonObject(with: Data(request.prompt.utf8)) as? [String: Any])
        XCTAssertEqual(payload["question"] as? String, question)
        XCTAssertEqual((payload["notes"] as? [Any])?.count, 0)
        XCTAssertEqual((payload["measuredCapability"] as? [String: Any])?["observations"] as? Int, 1)
        XCTAssertEqual(payload["latestProbeID"] as? String, p.id.uuidString)
    }

    func testRetentionAndLegacyNotes() throws {
        var state = SessionState()
        try state.importLegacyNotes([MemoryNote(text: "Ancienne note")])
        XCTAssertEqual(state.notes.count, 1)
        for i in 0..<140 {
            let p = try state.beginProbe(modelID: "A", lhs: i, rhs: 0, subtract: false)
            try state.finishProbe(id: p.id, modelID: "A", answer: String(i))
            try state.recordExchange(question: String(i), answer: "oui", modelID: "A")
        }
        XCTAssertEqual(state.probes.count, 128)
        XCTAssertEqual(state.exchanges.count, 20)
        XCTAssertEqual(state.exchanges.first?.question, "120")
        try state.validate()
    }
}
