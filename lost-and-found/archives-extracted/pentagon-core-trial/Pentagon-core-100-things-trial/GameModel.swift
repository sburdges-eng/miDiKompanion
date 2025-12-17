//
//  GameModel.swift
//  Dart Strike
//
//  Core game logic with traditional bowling scoring
//

import Foundation

// MARK: - Pin Model
struct Pin: Identifiable, Codable {
    let id: Int  // Pin number (1-10)
    var isStanding: Bool
    
    init(id: Int) {
        self.id = id
        self.isStanding = true
    }
}

// MARK: - Frame Model
struct Frame: Codable {
    var firstThrow: Int?
    var secondThrow: Int?
    var thirdThrow: Int?  // Only used in 10th frame
    var score: Int?
    
    var isComplete: Bool {
        if firstThrow == nil { return false }
        if firstThrow == 10 { return true }  // Strike
        if secondThrow == nil { return false }
        return true
    }
    
    var isStrike: Bool {
        firstThrow == 10
    }
    
    var isSpare: Bool {
        guard let first = firstThrow, let second = secondThrow else { return false }
        return first + second == 10 && first != 10
    }
}

// MARK: - Player Model
struct Player: Identifiable, Codable {
    let id: UUID
    var name: String
    var frames: [Frame]
    var currentFrameIndex: Int
    var currentThrow: Int  // 1 or 2 (or 3 for 10th frame)
    
    init(name: String) {
        self.id = UUID()
        self.name = name
        self.frames = Array(repeating: Frame(), count: 10)
        self.currentFrameIndex = 0
        self.currentThrow = 1
    }
    
    var totalScore: Int {
        return frames.compactMap { $0.score }.reduce(0, +)
    }
    
    var isGameComplete: Bool {
        return currentFrameIndex >= 10
    }
}

// MARK: - Game Model
class GameModel: ObservableObject {
    @Published var players: [Player]
    @Published var currentPlayerIndex: Int
    @Published var pins: [Pin]
    @Published var gameStarted: Bool
    @Published var gameComplete: Bool
    
    private let maxPins = 10
    
    init() {
        self.players = []
        self.currentPlayerIndex = 0
        self.pins = (1...10).map { Pin(id: $0) }
        self.gameStarted = false
        self.gameComplete = false
    }
    
    // MARK: - Game Setup
    func startNewGame(playerNames: [String]) {
        players = playerNames.map { Player(name: $0) }
        currentPlayerIndex = 0
        resetPins()
        gameStarted = true
        gameComplete = false
    }
    
    func addPlayer(name: String) {
        players.append(Player(name: name))
    }
    
    // MARK: - Pin Management
    func resetPins() {
        pins = (1...10).map { Pin(id: $0) }
    }
    
    func togglePin(_ pinId: Int) {
        if let index = pins.firstIndex(where: { $0.id == pinId }) {
            pins[index].isStanding.toggle()
        }
    }
    
    func getPinsKnockedDown() -> Int {
        return pins.filter { !$0.isStanding }.count
    }
    
    // MARK: - Throw Submission
    func submitThrow() {
        guard currentPlayerIndex < players.count else { return }
        
        let pinsDown = getPinsKnockedDown()
        var player = players[currentPlayerIndex]
        let frameIndex = player.currentFrameIndex
        
        guard frameIndex < 10 else {
            // Game complete for this player
            checkGameComplete()
            return
        }
        
        var frame = player.frames[frameIndex]
        
        // Handle 10th frame special rules
        if frameIndex == 9 {
            handle10thFrame(player: &player, frame: &frame, pinsDown: pinsDown)
        } else {
            handleRegularFrame(player: &player, frame: &frame, pinsDown: pinsDown)
        }
        
        players[currentPlayerIndex] = player
        calculateScores()
    }
    
    private func handleRegularFrame(player: inout Player, frame: inout Frame, pinsDown: Int) {
        let frameIndex = player.currentFrameIndex
        
        if player.currentThrow == 1 {
            frame.firstThrow = pinsDown
            
            if pinsDown == 10 {
                // Strike - move to next frame
                player.frames[frameIndex] = frame
                moveToNextFrame(player: &player)
                resetPins()
            } else {
                // Not a strike - continue to second throw
                player.currentThrow = 2
                player.frames[frameIndex] = frame
            }
        } else {
            // Second throw
            frame.secondThrow = pinsDown
            player.frames[frameIndex] = frame
            moveToNextFrame(player: &player)
            resetPins()
        }
    }
    
    private func handle10thFrame(player: inout Player, frame: inout Frame, pinsDown: Int) {
        if player.currentThrow == 1 {
            frame.firstThrow = pinsDown
            player.currentThrow = 2
            
            if pinsDown == 10 {
                resetPins()  // Strike - reset for bonus throws
            }
            
            player.frames[9] = frame
        } else if player.currentThrow == 2 {
            frame.secondThrow = pinsDown
            
            // Check if player gets a third throw
            let firstThrow = frame.firstThrow ?? 0
            let secondThrow = frame.secondThrow ?? 0
            
            if firstThrow == 10 || (firstThrow + secondThrow == 10) {
                // Strike or spare - get bonus throw
                player.currentThrow = 3
                resetPins()
            } else {
                // No bonus throw - move to next player
                moveToNextFrame(player: &player)
                resetPins()
            }
            
            player.frames[9] = frame
        } else {
            // Third throw (bonus)
            frame.thirdThrow = pinsDown
            player.frames[9] = frame
            moveToNextFrame(player: &player)
            resetPins()
        }
    }
    
    private func moveToNextFrame(player: inout Player) {
        player.currentFrameIndex += 1
        player.currentThrow = 1
        
        // Move to next player if current player finished their turn
        moveToNextPlayer()
    }
    
    private func moveToNextPlayer() {
        currentPlayerIndex += 1
        
        if currentPlayerIndex >= players.count {
            // All players finished this round
            currentPlayerIndex = 0
            
            // Check if all players are done
            if players.allSatisfy({ $0.isGameComplete }) {
                gameComplete = true
            }
        }
        
        // Skip players who are done
        while currentPlayerIndex < players.count && players[currentPlayerIndex].isGameComplete {
            currentPlayerIndex += 1
            if currentPlayerIndex >= players.count {
                currentPlayerIndex = 0
            }
        }
        
        resetPins()
    }
    
    private func checkGameComplete() {
        if players.allSatisfy({ $0.isGameComplete }) {
            gameComplete = true
        }
    }
    
    // MARK: - Score Calculation
    func calculateScores() {
        for playerIndex in 0..<players.count {
            var cumulativeScore = 0
            
            for frameIndex in 0..<10 {
                let frame = players[playerIndex].frames[frameIndex]
                
                if frameIndex == 9 {
                    // 10th frame scoring
                    let first = frame.firstThrow ?? 0
                    let second = frame.secondThrow ?? 0
                    let third = frame.thirdThrow ?? 0
                    cumulativeScore += first + second + third
                    players[playerIndex].frames[frameIndex].score = cumulativeScore
                } else {
                    // Regular frame scoring
                    if let score = calculateFrameScore(playerIndex: playerIndex, frameIndex: frameIndex) {
                        cumulativeScore += score
                        players[playerIndex].frames[frameIndex].score = cumulativeScore
                    } else {
                        players[playerIndex].frames[frameIndex].score = nil
                    }
                }
            }
        }
    }
    
    private func calculateFrameScore(playerIndex: Int, frameIndex: Int) -> Int? {
        let frame = players[playerIndex].frames[frameIndex]
        
        guard let firstThrow = frame.firstThrow else { return nil }
        
        // Strike
        if firstThrow == 10 {
            let nextFrame = frameIndex + 1
            guard nextFrame < 10 else { return nil }
            
            let next = players[playerIndex].frames[nextFrame]
            guard let nextFirst = next.firstThrow else { return nil }
            
            if nextFirst == 10 {
                // Next frame is also a strike
                let nextNextFrame = frameIndex + 2
                if nextNextFrame < 10 {
                    guard let nextNextFirst = players[playerIndex].frames[nextNextFrame].firstThrow else { return nil }
                    return 10 + nextFirst + nextNextFirst
                } else {
                    // Next frame is 10th frame
                    guard let nextSecond = next.secondThrow else { return nil }
                    return 10 + nextFirst + nextSecond
                }
            } else {
                // Next frame is not a strike
                guard let nextSecond = next.secondThrow else { return nil }
                return 10 + nextFirst + nextSecond
            }
        }
        
        // Spare
        guard let secondThrow = frame.secondThrow else { return nil }
        
        if firstThrow + secondThrow == 10 {
            let nextFrame = frameIndex + 1
            guard nextFrame < 10 else { return nil }
            
            let next = players[playerIndex].frames[nextFrame]
            guard let nextFirst = next.firstThrow else { return nil }
            return 10 + nextFirst
        }
        
        // Open frame
        return firstThrow + secondThrow
    }
    
    // MARK: - Current Player Helper
    var currentPlayer: Player? {
        guard currentPlayerIndex < players.count else { return nil }
        return players[currentPlayerIndex]
    }
    
    // MARK: - Game State
    func resetGame() {
        players = []
        currentPlayerIndex = 0
        resetPins()
        gameStarted = false
        gameComplete = false
    }
}
