import SwiftUI
import UIKit
import UniformTypeIdentifiers
import MeniaCore
import MeniaKit

@MainActor
final class MeniaController: ObservableObject {
    @Published var input = ""
    @Published var answer = ""
    @Published var currentQuestion = ""
    @Published private(set) var state = SessionState()
    @Published var status = "Télécharge le modèle sur Wi-Fi pour commencer."
    @Published private(set) var busy = false
    @Published private(set) var loaded = false
    @Published private(set) var model: ModelDescriptor?
    @Published private(set) var storageReady = false
    @Published var exportedFile: URL?
    private let engine = LocalEngine()
    private let installer = ModelInstaller()
    private var task: Task<Void, Never>?
    private var generation = UUID()
    private var firstChunk: Date?
    private let root: URL
    private let store: SessionStore
    private var modelURL: URL { root.appendingPathComponent("model", isDirectory: true) }
    private var legacyNotesURL: URL { root.appendingPathComponent("notes.json") }
    var capability: CapabilitySummary { state.capability(modelID: model?.fingerprint ?? "none") }

    init() {
        root = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("Menia", isDirectory: true)
        store = SessionStore(url: root.appendingPathComponent("session.json"))
        do {
            let fm = FileManager.default
            try fm.createDirectory(at: root, withIntermediateDirectories: true)
            var excluded = root
            var values = URLResourceValues(); values.isExcludedFromBackup = true
            try excluded.setResourceValues(values)
            let existed = fm.fileExists(atPath: store.url.path)
            var restored = try store.load()
            if !existed && fm.fileExists(atPath: legacyNotesURL.path) {
                try restored.importLegacyNotes(JSONDecoder().decode([MemoryNote].self, from: Data(contentsOf: legacyNotesURL)))
            }
            try store.save(restored)
            state = restored; storageReady = true
            if fm.fileExists(atPath: legacyNotesURL.path) { try fm.removeItem(at: legacyNotesURL) }
            model = try? ModelInstaller.descriptor(at: modelURL)
            if model != nil { status = "Modèle présent. Appuie sur Charger." }
            else if fm.fileExists(atPath: modelURL.path) { status = "Ancien modèle détecté : réimporte-le pour vérifier son identité." }
        } catch { status = "Mémoire indisponible : " + error.localizedDescription }
    }

    private func commit(_ edit: (inout SessionState) throws -> Void) throws {
        guard storageReady else { throw SessionError.invalidState }
        var updated = state
        try edit(&updated); try store.save(updated); state = updated
    }

    private func start(_ message: String) -> UUID {
        generation = UUID(); busy = true; status = message; firstChunk = nil
        UIApplication.shared.isIdleTimerDisabled = true
        return generation
    }

    private func finished() {
        busy = false; task = nil; UIApplication.shared.isIdleTimerDisabled = false
    }

    func saveNote() {
        guard !busy else { return }
        do { try commit { try $0.addNote(input) }; status = "Note enregistrée sur cet iPhone." }
        catch { status = error.localizedDescription }
    }

    func removeNote(_ id: UUID) {
        guard !busy else { return }
        do { try commit { $0.removeNote(id: id) } }
        catch { status = error.localizedDescription }
    }

    func install(source: URL? = nil) {
        guard !busy else { return }
        let id = start(source == nil ? "Téléchargement sur Wi-Fi…" : "Import du modèle…")
        loaded = false
        task = Task {
            defer { finished() }
            let scoped = source?.startAccessingSecurityScopedResource() ?? false
            defer { if scoped { source?.stopAccessingSecurityScopedResource() } }
            await engine.unload()
            do {
                let report: @Sendable (String) async -> Void = { [weak self] message in await self?.progress(message, id: id) }
                if let source { model = try await installer.importDirectory(source, to: modelURL, progress: report) }
                else { model = try await installer.downloadRecommended(to: modelURL, progress: report) }
                status = "Modèle vérifié et installé. Appuie sur Charger."
            } catch is CancellationError { status = "Installation interrompue. Tu peux réessayer." }
            catch { status = error.localizedDescription }
        }
    }

    private func progress(_ message: String, id: UUID) { if generation == id { status = message } }

    func load() {
        guard !busy, model != nil else { return }
        _ = start("Chargement local…"); loaded = false
        task = Task {
            defer { finished() }
            do {
                try await engine.load(directory: modelURL); try Task.checkCancellation()
                loaded = true; status = "Prêt · réponses calculées sur cet iPhone."
            } catch is CancellationError { await engine.unload(); status = "Chargement arrêté." }
            catch { status = error.localizedDescription }
        }
    }

    private func canGenerate() -> Bool {
        guard loaded, !busy, storageReady else { return false }
        guard [.nominal, .fair].contains(ProcessInfo.processInfo.thermalState) else {
            status = "Laisse refroidir l’iPhone avant de continuer."; return false
        }
        return true
    }

    func send() {
        let question = input.trimmingCharacters(in: .whitespacesAndNewlines)
        guard canGenerate(), let model, !question.isEmpty else { return }
        guard question.count <= 6000 else { status = "Raccourcis le message à 6 000 caractères maximum."; return }
        let id = start("Menia répond…")
        answer = ""; currentQuestion = question
        let started = Date()
        task = Task {
            defer { finished() }
            do {
                let requests = try [4000, 1500, 0].map {
                    try LanguageContext.chat(state: state, question: question, modelID: model.fingerprint, memoryCharacters: $0)
                }
                try await engine.respond(requests: requests) { [weak self] chunk in await self?.append(chunk, id: id) }
                try Task.checkCancellation()
                guard generation == id else { throw CancellationError() }
                let completed = answer
                try commit { try $0.recordExchange(question: question, answer: completed, modelID: model.fingerprint) }
                let latency = firstChunk.map { String(format: " · premier texte %.1f s", $0.timeIntervalSince(started)) } ?? ""
                status = String(format: "Terminé en %.1f s", Date().timeIntervalSince(started)) + latency
                answer = ""; currentQuestion = ""; input = ""
            } catch is CancellationError { status = "Arrêté · réponse partielle non mémorisée." }
            catch { status = error.localizedDescription }
        }
    }

    func calibrate() {
        guard canGenerate(), let model else { return }
        let id = start("Préparation des tests…")
        answer = ""; currentQuestion = ""
        task = Task {
            defer { finished() }
            do {
                for index in 1...5 {
                    try Task.checkCancellation()
                    let lhs = Int.random(in: 0...999), rhs = Int.random(in: 0...999), subtract = Bool.random()
                    try commit { _ = try $0.beginProbe(modelID: model.fingerprint, lhs: lhs, rhs: rhs, subtract: subtract) }
                    guard let probe = state.pending else { throw SessionError.wrongProbe }
                    status = "Test de calcul \(index)/5 · prévision enregistrée."
                    answer = ""; currentQuestion = probe.question
                    try await engine.respond(requests: [LanguageContext.calibration(probe)]) { [weak self] chunk in
                        await self?.append(chunk, id: id)
                    }
                    try Task.checkCancellation()
                    guard generation == id else { throw CancellationError() }
                    let result = answer
                    try commit { try $0.finishProbe(id: probe.id, modelID: model.fingerprint, answer: result) }
                }
                answer = ""; currentQuestion = ""
                status = "5 tests terminés. Ces résultats seront fournis aux prochaines réponses."
            } catch {
                let failure = error
                do { try commit { $0.cancelProbe() } }
                catch { status = "Impossible de sauvegarder l’arrêt : " + error.localizedDescription; return }
                status = failure is CancellationError ? "Tests interrompus ; seuls les tests terminés sont conservés." : failure.localizedDescription
            }
        }
    }

    private func append(_ chunk: String, id: UUID) {
        guard generation == id else { return }
        if firstChunk == nil, !chunk.isEmpty { firstChunk = Date() }
        answer += chunk
    }

    func stop() {
        generation = UUID(); task?.cancel(); status = busy ? "Arrêt en cours…" : "Arrêté."
    }

    func handleMemoryPressure() {
        stop(); loaded = false
        Task { await engine.unload() }
    }

    func wipe() {
        guard !busy else { return }
        do {
            try store.save(SessionState())
            if FileManager.default.fileExists(atPath: legacyNotesURL.path) { try FileManager.default.removeItem(at: legacyNotesURL) }
            if let exportedFile { try? FileManager.default.removeItem(at: exportedFile) }
            exportedFile = nil; state = SessionState(); storageReady = true
            input = ""; answer = ""; currentQuestion = ""; status = "Notes, échanges et tests effacés."
        } catch { status = "Échec de l’effacement : " + error.localizedDescription }
    }

    func exportTests() {
        guard !busy else { return }
        do {
            struct Report: Encodable {
                let schema = "menia-iphone-capability-v1"
                let model: ModelDescriptor?
                let probes: [CapabilityProbe]
                let summary: CapabilitySummary
                let scope = "Calcul élémentaire seulement. Aucun score de conscience."
            }
            let encoder = JSONEncoder(); encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
            let data = try encoder.encode(Report(model: model, probes: state.probes, summary: capability))
            let url = root.appendingPathComponent("tests-menia.json")
            try data.write(to: url, options: [.atomic, .completeFileProtection])
            exportedFile = url; status = "Rapport prêt à partager ; tests uniquement, sans notes ni échanges."
        } catch { status = error.localizedDescription }
    }

    func deleteModel() {
        guard !busy else { return }
        _ = start("Suppression du modèle…"); loaded = false
        task = Task {
            defer { finished() }
            await engine.unload()
            do {
                if FileManager.default.fileExists(atPath: modelURL.path) { try FileManager.default.removeItem(at: modelURL) }
                model = nil; status = "Modèle supprimé."
            } catch { status = error.localizedDescription }
        }
    }
}

@main
struct MeniaApp: App {
    @StateObject private var controller = MeniaController()
    @Environment(\.scenePhase) private var phase
    @State private var importer = false
    @State private var confirmWipe = false
    @State private var confirmModelDelete = false
    var body: some Scene {
        WindowGroup {
            NavigationStack {
                ScrollView {
                    VStack(alignment: .leading, spacing: 20) {
                        Text("Une mémoire qui reste. Des capacités que l’on mesure.").font(.title2).bold()
                        Text("Prototype expérimental").font(.caption).foregroundStyle(.secondary)
                        GroupBox("Modèle sur cet iPhone") {
                            VStack(alignment: .leading, spacing: 12) {
                                Text(controller.model?.name ?? "Aucun modèle installé")
                                if controller.model == nil {
                                    Button("Télécharger Qwen3-4B · 2,15 Go") { controller.install() }.disabled(controller.busy)
                                    Text("Wi-Fi requis pour le téléchargement. Les réponses fonctionnent ensuite hors ligne.").font(.caption).foregroundStyle(.secondary)
                                }
                                HStack {
                                    Button("Importer un dossier") { importer = true }.disabled(controller.busy)
                                    Button("Charger") { controller.load() }.disabled(controller.busy || controller.loaded || controller.model == nil)
                                }.buttonStyle(.bordered)
                            }.frame(maxWidth: .infinity, alignment: .leading)
                        }
                        HStack {
                            if controller.busy { ProgressView() }
                            Text(controller.status).font(.callout)
                        }.accessibilityElement(children: .combine)
                        ForEach(controller.state.exchanges.suffix(5)) { turn in
                            VStack(alignment: .leading, spacing: 8) {
                                Text(turn.question).font(.headline)
                                Text(turn.answer).textSelection(.enabled)
                            }.frame(maxWidth: .infinity, alignment: .leading).padding()
                                .background(.quaternary, in: RoundedRectangle(cornerRadius: 14))
                        }
                        if !controller.currentQuestion.isEmpty {
                            Text(controller.currentQuestion).font(.headline)
                            Text(controller.answer).textSelection(.enabled)
                        }
                        TextField("Parle à Menia…", text: $controller.input, axis: .vertical)
                            .lineLimit(3...8).textFieldStyle(.roundedBorder).disabled(controller.busy)
                        HStack {
                            Button("Envoyer") { controller.send() }.disabled(controller.busy || !controller.loaded || !controller.storageReady || controller.input.isEmpty)
                            Button("Arrêter", role: .destructive) { controller.stop() }.disabled(!controller.busy)
                        }.buttonStyle(.borderedProminent)
                        Text("Les 20 derniers échanges restent sur cet appareil. Le contexte reçoit un extrait récent selon la place disponible.").font(.caption).foregroundStyle(.secondary)
                        GroupBox("Capacités mesurées") {
                            VStack(alignment: .leading, spacing: 10) {
                                if controller.capability.observations == 0 { Text("Aucun test pour ce modèle.") }
                                else {
                                    Text("\(controller.capability.successes) réussites / \(controller.capability.observations) tests")
                                    Text(String(format: "Prévision du prochain test : %.0f %%", 100 * controller.capability.predictedSuccess))
                                }
                                Text("Petites additions et soustractions, réponse entière stricte. Cette mesure ne vaut pas pour les autres tâches.").font(.caption).foregroundStyle(.secondary)
                                Button("Tester mes capacités · 5 calculs") { controller.calibrate() }.disabled(controller.busy || !controller.loaded || !controller.storageReady)
                                ForEach(controller.state.probes.filter { $0.modelID == controller.model?.fingerprint }.suffix(3)) { probe in
                                    Text("\(probe.correct == true ? "✓" : "✗") \(probe.question) Réponse : \(probe.answer ?? "")").font(.caption)
                                }
                                Button("Préparer le rapport des tests") { controller.exportTests() }.disabled(controller.busy)
                                if let file = controller.exportedFile { ShareLink("Partager le rapport", item: file) }
                            }.frame(maxWidth: .infinity, alignment: .leading)
                        }
                        GroupBox("Notes à conserver · \(controller.state.notes.count)/50") {
                            VStack(alignment: .leading, spacing: 10) {
                                Button("Mémoriser le texte saisi") { controller.saveNote() }.disabled(controller.busy || !controller.storageReady || controller.input.isEmpty || controller.input.count > 2000 || controller.state.notes.count >= 50)
                                ForEach(controller.state.notes) { note in
                                    HStack {
                                        Text(note.text).font(.callout); Spacer()
                                        Button(role: .destructive) { controller.removeNote(note.id) } label: { Image(systemName: "trash") }
                                            .accessibilityLabel("Supprimer cette note").disabled(controller.busy)
                                    }
                                }
                            }.frame(maxWidth: .infinity, alignment: .leading)
                        }
                        Button("Effacer la mémoire et les tests", role: .destructive) { confirmWipe = true }.disabled(controller.busy)
                        Button("Supprimer le modèle", role: .destructive) { confirmModelDelete = true }.disabled(controller.busy)
                    }.padding(20)
                }.navigationTitle("Menia")
                    .fileImporter(isPresented: $importer, allowedContentTypes: [.folder]) { result in
                        switch result {
                        case .success(let url): controller.install(source: url)
                        case .failure(let error): controller.status = error.localizedDescription
                        }
                    }
                    .confirmationDialog("Effacer les notes, échanges et tests conservés sur cet appareil ?", isPresented: $confirmWipe) {
                        Button("Effacer", role: .destructive) { controller.wipe() }
                    }
                    .confirmationDialog("Supprimer les fichiers du modèle local ?", isPresented: $confirmModelDelete) {
                        Button("Supprimer", role: .destructive) { controller.deleteModel() }
                    }
            }
            .onChange(of: phase) { _, value in if value != .active && controller.busy { controller.stop() } }
            .onReceive(NotificationCenter.default.publisher(for: ProcessInfo.thermalStateDidChangeNotification)) { _ in
                if [.serious, .critical].contains(ProcessInfo.processInfo.thermalState) { controller.stop() }
            }
            .onReceive(NotificationCenter.default.publisher(for: UIApplication.didReceiveMemoryWarningNotification)) { _ in controller.handleMemoryPressure() }
        }
    }
}
