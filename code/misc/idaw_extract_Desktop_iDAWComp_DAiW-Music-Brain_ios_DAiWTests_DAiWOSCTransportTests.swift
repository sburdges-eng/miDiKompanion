//
//  DAiWOSCTransportTests.swift
//  DAiWTests
//
//  Tests for OSC transport communication with Python brain server
//

import XCTest
import Network
@testable import DAiW

class DAiWOSCTransportTests: XCTestCase {
    
    var oscClient: OSCClient?
    let brainServerHost = "127.0.0.1"
    let brainServerPort: UInt16 = 9000
    let receivePort: UInt16 = 9001
    
    override func setUp() {
        super.setUp()
        // Initialize OSC client for testing
        oscClient = OSCClient(host: brainServerHost, port: brainServerPort)
    }
    
    override func tearDown() {
        oscClient?.disconnect()
        oscClient = nil
        super.tearDown()
    }
    
    // MARK: - Connection Tests
    
    func testOSCConnection() {
        // Test that OSC client can connect to brain server
        let expectation = XCTestExpectation(description: "OSC connection established")
        
        oscClient?.connect { success in
            XCTAssertTrue(success, "OSC client should connect successfully")
            expectation.fulfill()
        }
        
        wait(for: [expectation], timeout: 5.0)
    }
    
    func testOSCConnectionFailure() {
        // Test connection failure handling
        let invalidClient = OSCClient(host: "192.0.2.0", port: 9999) // Invalid IP
        let expectation = XCTestExpectation(description: "OSC connection fails gracefully")
        
        invalidClient.connect { success in
            XCTAssertFalse(success, "Invalid connection should fail")
            expectation.fulfill()
        }
        
        wait(for: [expectation], timeout: 5.0)
    }
    
    // MARK: - Message Sending Tests
    
    func testSendGenerateRequest() {
        // Test sending /daiw/generate message
        let expectation = XCTestExpectation(description: "Generate request sent")
        
        oscClient?.connect { [weak self] success in
            guard success, let self = self else {
                XCTFail("Failed to connect")
                return
            }
            
            let message = OSCMessage(address: "/daiw/generate")
            message.add("I feel deep grief")
            message.add(Float(7.0))  // motivation
            message.add(Float(5.0))  // chaos
            message.add(Float(6.0))  // vulnerability
            
            self.oscClient?.send(message) { sent in
                XCTAssertTrue(sent, "Message should be sent successfully")
                expectation.fulfill()
            }
        }
        
        wait(for: [expectation], timeout: 10.0)
    }
    
    func testSendPing() {
        // Test sending /daiw/ping message
        let expectation = XCTestExpectation(description: "Ping sent")
        
        oscClient?.connect { [weak self] success in
            guard success, let self = self else {
                XCTFail("Failed to connect")
                return
            }
            
            let message = OSCMessage(address: "/daiw/ping")
            self.oscClient?.send(message) { sent in
                XCTAssertTrue(sent, "Ping should be sent successfully")
                expectation.fulfill()
            }
        }
        
        wait(for: [expectation], timeout: 10.0)
    }
    
    // MARK: - Message Receiving Tests
    
    func testReceiveMIDIResult() {
        // Test receiving /daiw/result message with MIDI data
        let expectation = XCTestExpectation(description: "MIDI result received")
        
        let receiver = OSCReceiver(port: receivePort)
        receiver.onMessage = { message in
            if message.address == "/daiw/result" {
                XCTAssertTrue(message.arguments.count > 0, "Result should contain data")
                if let jsonString = message.arguments[0] as? String {
                    // Parse JSON and verify structure
                    let data = jsonString.data(using: .utf8)!
                    let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
                    XCTAssertNotNil(json, "Result should be valid JSON")
                    expectation.fulfill()
                }
            }
        }
        
        receiver.start()
        
        // Send generate request to trigger response
        oscClient?.connect { [weak self] success in
            guard success, let self = self else { return }
            let message = OSCMessage(address: "/daiw/generate")
            message.add("test")
            message.add(Float(5.0))
            message.add(Float(5.0))
            message.add(Float(5.0))
            self.oscClient?.send(message)
        }
        
        wait(for: [expectation], timeout: 15.0)
        receiver.stop()
    }
    
    func testReceivePong() {
        // Test receiving /daiw/pong response
        let expectation = XCTestExpectation(description: "Pong received")
        
        let receiver = OSCReceiver(port: receivePort)
        receiver.onMessage = { message in
            if message.address == "/daiw/pong" {
                XCTAssertTrue(message.arguments.count > 0, "Pong should contain data")
                expectation.fulfill()
            }
        }
        
        receiver.start()
        
        // Send ping to trigger pong
        oscClient?.connect { [weak self] success in
            guard success, let self = self else { return }
            let message = OSCMessage(address: "/daiw/ping")
            self.oscClient?.send(message)
        }
        
        wait(for: [expectation], timeout: 10.0)
        receiver.stop()
    }
    
    // MARK: - Error Handling Tests
    
    func testInvalidMessageFormat() {
        // Test handling of invalid message format
        let expectation = XCTestExpectation(description: "Invalid message handled")
        
        oscClient?.connect { [weak self] success in
            guard success, let self = self else { return }
            
            // Send message with wrong argument types
            let message = OSCMessage(address: "/daiw/generate")
            message.add(123)  // Wrong type (should be String)
            
            self.oscClient?.send(message) { sent in
                // Server should handle gracefully (may accept or reject)
                expectation.fulfill()
            }
        }
        
        wait(for: [expectation], timeout: 10.0)
    }
    
    func testNetworkTimeout() {
        // Test network timeout handling
        let timeoutClient = OSCClient(host: "192.0.2.0", port: 9999, timeout: 2.0)
        let expectation = XCTestExpectation(description: "Timeout handled")
        
        timeoutClient.connect { success in
            XCTAssertFalse(success, "Connection should timeout")
            expectation.fulfill()
        }
        
        wait(for: [expectation], timeout: 5.0)
    }
}

