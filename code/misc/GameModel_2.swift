//
//  GameModel.swift
//  Dart Strike
//
//  Core game logic and data model
//

import Foundation
import Combine

// MARK: - Player Model
class Player: Identifiable, ObservableObject {
    let id = UUID()
    @Published var name: String
    @Published var frames: [[Int?]]  // 10 frames, each with up to 3 throws
    @Published var scores: [Int?]    // Cumulative scores for each frame
    @Published var currentFrame: Int = 0
    @Published var currentThrow: Int = 0
    
    init(name: String) {
        self.name = name
        // Initialize 10 frames, each can have up to 3 throws (10th frame)
        self.frames = Array(repeating: [nil, nil, nil], count: 10)
        self.scores = Array(repeating: nil, count: 10)
    }
    
    var isComplete: Bool {
        return currentFrame >= 10
    }
}

// MARK: - Pin Model
struct Pin: Identifiable {
    let id: Int
    var isStanding: Bool = true
    
    var pinNumber: Int {
        return id
    }
}

// MARK: - Game Model
class GameModel: ObservableObject {
    @Published var players: [Player] = []
    @Published var currentPlayerIndex: Int = 0
    @Published var pins: [Pin] = []
    @Published var gameStarted: Bool = false
    @Published var gameOver: Bool = false
    
    init() {
        resetPins()
    }
    
    // MARK: - Player Management
    
    func addPlayer(name: String) {
        let player = Player(name: name)
        players.append(player)
    }
    
    func startGame() {
        guard !players.isEmpty else { return }
        gameStarted = true
        gameOver = false
        currentPlayerIndex = 0
        resetPins()
    }
    
    func resetGame() {
        players.removeAll()
        currentPlayerIndex = 0
        gameStarted = false
        gameOver = false
        resetPins()
    }
    
    var currentPlayer: Player? {
        guard currentPlayerIndex < players.count else { return nil }
        return players[currentPlayerIndex]
    }
    
    // MARK: - Pin Management
    
    func resetPins() {
        pins = (1...10).map { Pin(id: $0) }
    }
    
    func togglePin(_ pin: Pin) {
        guard let index = pins.firstIndex(where: { $0.id == pin.id }) else { return }
        pins[index].isStanding.toggle()
    }
    
    var pinsKnockedDown: Int {
        return pins.filter { !$0.isStanding }.count
    }
    
    // MARK: - Game Logic
    
    func submitThrow() {
        guard let player = currentPlayer, !player.isComplete else {
            nextPlayer()
            return
        }
        
        let pinsDown = pinsKnockedDown
        let frameIdx = player.currentFrame
        
        // Handle 10th frame special rules
        if frameIdx == 9 {
            handle10thFrame(player: player, pinsDown: pinsDown)
        } else {
            handleRegularFrame(player: player, pinsDown: pinsDown)
        }
        
        calculateAllScores()
    }
    
    private func handleRegularFrame(player: Player, pinsDown: Int) {
        let frameIdx = player.currentFrame
        
        if player.currentThrow == 0 {
            // First throw
            player.frames[frameIdx][0] = pinsDown
            
            if pinsDown == 10 {
                // Strike! Move to next frame
                player.currentFrame += 1
                player.currentThrow = 0
                nextPlayer()
            } else {
                player.currentThrow = 1
            }
        } else {
            // Second throw
            player.frames[frameIdx][1] = pinsDown
            player.currentFrame += 1
            player.currentThrow = 0
            nextPlayer()
        }
        
        resetPins()
    }
    
    private func handle10thFrame(player: Player, pinsDown: Int) {
        if player.currentThrow == 0 {
            // First throw
            player.frames[9][0] = pinsDown
            player.currentThrow = 1
            if pinsDown == 10 {
                resetPins()  // Strike - reset for bonus
            }
        } else if player.currentThrow == 1 {
            // Second throw
            player.frames[9][1] = pinsDown
            let first = player.frames[9][0] ?? 0
            
            if first == 10 || (first + pinsDown == 10) {
                // Strike or spare - get third throw
                player.currentThrow = 2
                resetPins()
            } else {
                // No bonus - done
                player.currentFrame = 10
                player.currentThrow = 0
                nextPlayer()
            }
        } else if player.currentThrow == 2 {
            // Third throw (bonus)
            player.frames[9][2] = pinsDown
            player.currentFrame = 10
            player.currentThrow = 0
            nextPlayer()
        }
        
        if player.currentThrow == 0 || player.currentFrame >= 10 {
            // Only reset if we're done with this player
        }
    }
    
    private func nextPlayer() {
        currentPlayerIndex += 1
        
        if currentPlayerIndex >= players.count {
            currentPlayerIndex = 0
        }
        
        // Skip completed players
        var attempts = 0
        while currentPlayer?.isComplete == true && attempts < players.count {
            currentPlayerIndex = (currentPlayerIndex + 1) % players.count
            attempts += 1
        }
        
        // Check if game is over
        if players.allSatisfy({ $0.isComplete }) {
            endGame()
        } else {
            resetPins()
        }
    }
    
    private func endGame() {
        gameStarted = false
        gameOver = true
    }
    
    var winner: Player? {
        guard gameOver else { return nil }
        return players.max(by: { ($0.scores[9] ?? 0) < ($1.scores[9] ?? 0) })
    }
    
    // MARK: - Scoring
    
    func calculateAllScores() {
        for player in players {
            var cumulative = 0
            
            for i in 0..<10 {
                let frame = player.frames[i]
                guard let first = frame[0] else {
                    player.scores[i] = nil
                    continue
                }
                
                if i == 9 {
                    // 10th frame - just add all throws
                    let score = (first) + (frame[1] ?? 0) + (frame[2] ?? 0)
                    cumulative += score
                    player.scores[i] = cumulative
                } else {
                    if let score = calculateFrameScore(player: player, frameIdx: i) {
                        cumulative += score
                        player.scores[i] = cumulative
                    } else {
                        player.scores[i] = nil
                    }
                }
            }
        }
    }
    
    private func calculateFrameScore(player: Player, frameIdx: Int) -> Int? {
        let frame = player.frames[frameIdx]
        guard let first = frame[0] else { return nil }
        
        // Strike
        if first == 10 {
            let nextFrame = player.frames[frameIdx + 1]
            guard let nextFirst = nextFrame[0] else { return nil }
            
            if nextFirst == 10 && frameIdx < 8 {
                // Next is also strike
                let nextNext = player.frames[frameIdx + 2]
                guard let nextNextFirst = nextNext[0] else { return nil }
                return 10 + 10 + nextNextFirst
            } else if nextFirst == 10 && frameIdx == 8 {
                // Next is 10th frame
                guard let nextSecond = nextFrame[1] else { return nil }
                return 10 + nextFirst + nextSecond
            } else {
                guard let nextSecond = nextFrame[1] else { return nil }
                return 10 + nextFirst + nextSecond
            }
        }
        
        guard let second = frame[1] else { return nil }
        
        // Spare
        if first + second == 10 {
            let nextFrame = player.frames[frameIdx + 1]
            guard let nextFirst = nextFrame[0] else { return nil }
            return 10 + nextFirst
        }
        
        // Open frame
        return first + second
    }
}
