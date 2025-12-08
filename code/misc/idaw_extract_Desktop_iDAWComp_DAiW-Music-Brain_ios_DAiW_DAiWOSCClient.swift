//
//  DAiWOSCClient.swift
//  DAiW
//
//  OSC client for communicating with Python brain server
//

import Foundation
import Network
import Combine

/// OSC message structure
struct OSCMessage {
    let address: String
    var arguments: [Any] = []
    
    mutating func add(_ value: Any) {
        arguments.append(value)
    }
}

/// OSC client for sending messages to brain server
class OSCClient {
    private let host: String
    private let port: UInt16
    private let timeout: TimeInterval
    private var connection: NWConnection?
    
    init(host: String, port: UInt16, timeout: TimeInterval = 5.0) {
        self.host = host
        self.port = port
        self.timeout = timeout
    }
    
    func connect(completion: @escaping (Bool) -> Void) {
        let endpoint = NWEndpoint.hostPort(
            host: NWEndpoint.Host(host),
            port: NWEndpoint.Port(integerLiteral: port)
        )
        
        connection = NWConnection(host: endpoint.host, port: endpoint.port, using: .udp)
        
        connection?.stateUpdateHandler = { state in
            switch state {
            case .ready:
                completion(true)
            case .failed, .cancelled:
                completion(false)
            default:
                break
            }
        }
        
        connection?.start(queue: .global())
    }
    
    func send(_ message: OSCMessage, completion: ((Bool) -> Void)? = nil) {
        guard let connection = connection else {
            completion?(false)
            return
        }
        
        // Serialize OSC message (simplified - use proper OSC library in production)
        let data = serializeOSCMessage(message)
        
        connection.send(
            content: data,
            completion: .contentProcessed { error in
                completion?(error == nil)
            }
        )
    }
    
    func disconnect() {
        connection?.cancel()
        connection = nil
    }
    
    private func serializeOSCMessage(_ message: OSCMessage) -> Data {
        // Simplified OSC serialization
        // In production, use a proper OSC library like SwiftOSC
        var data = Data()
        
        // Address
        data.append(message.address.data(using: .utf8)!)
        data.append(0)  // Null terminator
        
        // Type tag
        var typeTag = ","
        for arg in message.arguments {
            if arg is String {
                typeTag += "s"
            } else if arg is Float || arg is Double {
                typeTag += "f"
            } else if arg is Int {
                typeTag += "i"
            }
        }
        data.append(typeTag.data(using: .utf8)!)
        data.append(0)
        
        // Arguments
        for arg in message.arguments {
            if let str = arg as? String {
                data.append(str.data(using: .utf8)!)
                data.append(0)
            } else if let float = arg as? Float {
                data.append(contentsOf: withUnsafeBytes(of: float.bigEndian) { Array($0) })
            }
        }
        
        return data
    }
}

/// OSC receiver for receiving messages from brain server
class OSCReceiver {
    private let port: UInt16
    private var listener: NWListener?
    var onMessage: ((OSCMessage) -> Void)?
    
    init(port: UInt16) {
        self.port = port
    }
    
    func start() {
        let parameters = NWParameters.udp
        parameters.allowLocalEndpointReuse = true
        
        listener = try? NWListener(parameters: parameters, on: NWEndpoint.Port(integerLiteral: port))
        
        listener?.newConnectionHandler = { connection in
            connection.start(queue: .global())
            self.receive(on: connection)
        }
        
        listener?.start(queue: .global())
    }
    
    func stop() {
        listener?.cancel()
        listener = nil
    }
    
    private func receive(on connection: NWConnection) {
        connection.receive { [weak self] data, _, isComplete, error in
            if let data = data, let message = self?.parseOSCMessage(data) {
                DispatchQueue.main.async {
                    self?.onMessage?(message)
                }
            }
            
            if !isComplete && error == nil {
                self?.receive(on: connection)
            }
        }
    }
    
    private func parseOSCMessage(_ data: Data) -> OSCMessage? {
        // Simplified OSC parsing
        // In production, use a proper OSC library
        guard let addressString = String(data: data, encoding: .utf8) else {
            return nil
        }
        
        let components = addressString.split(separator: "\0")
        guard components.count >= 2 else { return nil }
        
        let address = String(components[0])
        // Parse type tag and arguments (simplified)
        
        return OSCMessage(address: address, arguments: [])
    }
}

