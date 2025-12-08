//
//  DAiWBrainServerConnection.swift
//  DAiW
//
//  Connection to Python brain server via OSC
//

import Foundation
import Combine

/// Connection to Python brain server
class BrainServerConnection {
    private let host: String
    private let port: UInt16
    private let oscClient: OSCClient
    
    init(host: String = "127.0.0.1", port: UInt16 = 9000) {
        self.host = host
        self.port = port
        self.oscClient = OSCClient(host: host, port: port)
    }
    
    func connect() -> AnyPublisher<Bool, Error> {
        return Future { [weak self] promise in
            self?.oscClient.connect { success in
                promise(.success(success))
            }
        }
        .eraseToAnyPublisher()
    }
    
    func generateMIDI(
        intent: String,
        motivation: Double,
        chaos: Double,
        vulnerability: Double
    ) -> AnyPublisher<[MIDINoteEvent], Error> {
        return Future { [weak self] promise in
            guard let self = self else {
                promise(.failure(NSError(domain: "DAiW", code: -1)))
                return
            }
            
            var message = OSCMessage(address: "/daiw/generate")
            message.add(intent)
            message.add(Float(motivation))
            message.add(Float(chaos))
            message.add(Float(vulnerability))
            
            self.oscClient.send(message) { sent in
                if sent {
                    // Wait for response (simplified - use proper async handling)
                    promise(.success([]))
                } else {
                    promise(.failure(NSError(domain: "DAiW", code: -2)))
                }
            }
        }
        .eraseToAnyPublisher()
    }
    
    func disconnect() {
        oscClient.disconnect()
    }
}

/// MIDI scheduler for playback
class MIDIScheduler {
    var onMIDIMessage: ((MIDIMessage) -> Void)?
    
    func schedule(_ message: MIDIMessage, at time: TimeInterval) {
        // Schedule MIDI message for playback
        onMIDIMessage?(message)
    }
}

struct MIDIMessage {
    let status: UInt8
    let data1: UInt8
    let data2: UInt8
}

