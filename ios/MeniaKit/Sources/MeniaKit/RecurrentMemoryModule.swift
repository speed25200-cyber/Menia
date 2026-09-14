import Foundation

/// Tiny symbolic research module. Not connected to chat and not a consciousness detector.
public struct RecurrentMemoryModule {
    public enum ModelError: Error { case invalidCheckpoint, invalidSymbol }
    private struct Parameters: Decodable {
        let W: [[Double]]
        let U: [[Double]]
        let G: [[Double]]
        let b: [Double]
        let bg: [Double]
        let V: [[Double]]
        let bo: [Double]
    }
    private struct Checkpoint: Decodable {
        let format: String
        let version: Int
        let hidden: Int
        let parameters: Parameters
    }
    private let parameters: Parameters
    public private(set) var state: [Double]

    public init(data: Data) throws {
        guard data.count <= 4_000_000 else { throw ModelError.invalidCheckpoint }
        let model = try JSONDecoder().decode(Checkpoint.self, from: data)
        let h = model.hidden
        let p = model.parameters
        func matrix(_ values: [[Double]], _ rows: Int, _ columns: Int) -> Bool {
            values.count == rows && values.allSatisfy { $0.count == columns && $0.allSatisfy(\.isFinite) }
        }
        guard model.format == "menia-recurrent-memory", model.version == 1,
              (4...128).contains(h), matrix(p.W, 5, h), matrix(p.G, 5, h),
              matrix(p.U, h, h), matrix(p.V, h, 4), p.b.count == h,
              p.bg.count == h, p.bo.count == 4,
              (p.b + p.bg + p.bo).allSatisfy(\.isFinite) else { throw ModelError.invalidCheckpoint }
        parameters = p
        state = Array(repeating: 0, count: h)
    }

    public mutating func reset() { state = Array(repeating: 0, count: state.count) }

    /// Feed a symbol or nil for an unobserved step. Returns four class probabilities.
    public mutating func step(symbol: Int?) throws -> [Double] {
        var x = Array(repeating: 0.0, count: 5)
        if let symbol {
            guard (0..<4).contains(symbol) else { throw ModelError.invalidSymbol }
            x[symbol] = 1
            x[4] = 1
        }
        let p = parameters
        let previous = state
        for j in state.indices {
            var a = p.b[j]
            var z = p.bg[j]
            for i in 0..<5 { a += x[i] * p.W[i][j]; z += x[i] * p.G[i][j] }
            for i in previous.indices { a += previous[i] * p.U[i][j] }
            let gate = 1 / (1 + exp(-min(30, max(-30, z))))
            state[j] = (1-gate) * previous[j] + gate * tanh(a)
        }
        var logits = p.bo
        for j in 0..<4 {
            for i in state.indices { logits[j] += state[i] * p.V[i][j] }
        }
        let maximum = logits.max()!
        let values = logits.map { exp($0-maximum) }
        let sum = values.reduce(0,+)
        return values.map { $0/sum }
    }
}
