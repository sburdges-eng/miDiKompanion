//
//  GameView.swift
//  Bulling
//
//  Main game play view with interactive pins
//

import SwiftUI

struct GameView: View {
    @EnvironmentObject var gameModel: GameModel
    @State private var showScorecard = false
    
    var body: some View {
        VStack(spacing: 0) {
            // Header with current player info
            currentPlayerHeader
            
            // Main content
            ScrollView {
                VStack(spacing: 20) {
                    // Pin layout area
                    pinArea
                    
                    // Submit button
                    submitButton
                    
                    // Mini scorecard
                    miniScorecard
                }
                .padding()
            }
            
            // Bottom controls
            bottomControls
        }
        .background(Color(red: 0.95, green: 0.97, blue: 1.0))
        .sheet(isPresented: $showScorecard) {
            ScorecardView()
                .environmentObject(gameModel)
        }
    }
    
    var currentPlayerHeader: some View {
        VStack(spacing: 8) {
            if let player = gameModel.currentPlayer {
                HStack {
                    Image(systemName: "person.circle.fill")
                        .font(.title2)
                        .foregroundColor(.orange)
                    
                    VStack(alignment: .leading, spacing: 2) {
                        Text(player.name)
                            .font(.headline)
                        Text("Frame \(player.currentFrame + 1), Ball \(player.currentThrow + 1)")
                            .font(.caption)
                            .foregroundColor(.secondary)
                    }
                    
                    Spacer()
                    
                    Text("\(gameModel.pinsKnockedDown) pins")
                        .font(.title3)
                        .fontWeight(.bold)
                        .foregroundColor(.red)
                }
                .padding()
                .background(Color.white)
                .shadow(color: Color.black.opacity(0.1), radius: 3)
            }
        }
    }
    
    var pinArea: some View {
        VStack(spacing: 20) {
            Text("Tap pins to knock down")
                .font(.subheadline)
                .foregroundColor(.secondary)
            
            // Bowling pin layout
            ZStack {
                // Background (lane)
                RoundedRectangle(cornerRadius: 16)
                    .fill(
                        LinearGradient(
                            gradient: Gradient(colors: [
                                Color(red: 0.85, green: 0.7, blue: 0.5),
                                Color(red: 0.75, green: 0.6, blue: 0.4)
                            ]),
                            startPoint: .top,
                            endPoint: .bottom
                        )
                    )
                    .frame(height: 350)
                
                // Pins in standard 10-pin layout
                VStack(spacing: 15) {
                    // Row 1 (back) - 4 pins
                    HStack(spacing: 20) {
                        ForEach([7, 8, 9, 10], id: \.self) { pinId in
                            if let pin = gameModel.pins.first(where: { $0.id == pinId }) {
                                PinView(pin: pin) {
                                    gameModel.togglePin(pin)
                                }
                            }
                        }
                    }
                    
                    // Row 2 - 3 pins
                    HStack(spacing: 20) {
                        ForEach([4, 5, 6], id: \.self) { pinId in
                            if let pin = gameModel.pins.first(where: { $0.id == pinId }) {
                                PinView(pin: pin) {
                                    gameModel.togglePin(pin)
                                }
                            }
                        }
                    }
                    
                    // Row 3 - 2 pins
                    HStack(spacing: 20) {
                        ForEach([2, 3], id: \.self) { pinId in
                            if let pin = gameModel.pins.first(where: { $0.id == pinId }) {
                                PinView(pin: pin) {
                                    gameModel.togglePin(pin)
                                }
                            }
                        }
                    }
                    
                    // Row 4 (front) - 1 pin
                    HStack(spacing: 20) {
                        if let pin = gameModel.pins.first(where: { $0.id == 1 }) {
                            PinView(pin: pin) {
                                gameModel.togglePin(pin)
                            }
                        }
                    }
                }
                .padding(.vertical, 30)
            }
        }
    }
    
    var submitButton: some View {
        Button(action: {
            gameModel.submitThrow()
        }) {
            HStack {
                Image(systemName: "checkmark.circle.fill")
                Text("Submit Throw")
                    .fontWeight(.semibold)
            }
            .frame(maxWidth: .infinity)
            .padding()
            .background(Color.green)
            .foregroundColor(.white)
            .cornerRadius(12)
        }
    }
    
    var miniScorecard: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Text("Scores")
                    .font(.headline)
                Spacer()
                Button(action: { showScorecard = true }) {
                    Text("Full Scorecard")
                        .font(.caption)
                        .foregroundColor(.blue)
                }
            }
            
            ForEach(gameModel.players) { player in
                HStack {
                    Text(player.name)
                        .font(.body)
                    Spacer()
                    Text("\(player.scores[9] ?? player.scores.compactMap { $0 }.last ?? 0)")
                        .font(.title3)
                        .fontWeight(.bold)
                        .foregroundColor(.blue)
                }
                .padding(.vertical, 4)
            }
        }
        .padding()
        .background(Color.white)
        .cornerRadius(12)
        .shadow(color: Color.black.opacity(0.05), radius: 3)
    }
    
    var bottomControls: some View {
        HStack(spacing: 12) {
            Button(action: {
                gameModel.resetPins()
            }) {
                HStack {
                    Image(systemName: "arrow.counterclockwise")
                    Text("Reset Pins")
                        .font(.caption)
                }
                .padding(.horizontal, 16)
                .padding(.vertical, 10)
                .background(Color.orange.opacity(0.2))
                .foregroundColor(.orange)
                .cornerRadius(8)
            }
            
            Spacer()
            
            Button(action: {
                gameModel.resetGame()
            }) {
                HStack {
                    Image(systemName: "stop.circle")
                    Text("End Game")
                        .font(.caption)
                }
                .padding(.horizontal, 16)
                .padding(.vertical, 10)
                .background(Color.red.opacity(0.2))
                .foregroundColor(.red)
                .cornerRadius(8)
            }
        }
        .padding()
        .background(Color.white)
        .shadow(color: Color.black.opacity(0.1), radius: 3)
    }
}

// Individual pin view
struct PinView: View {
    let pin: Pin
    let onTap: () -> Void
    
    var body: some View {
        Button(action: onTap) {
            ZStack {
                // Pin shape
                BowlingPinShape()
                    .fill(
                        pin.isStanding ?
                        LinearGradient(
                            gradient: Gradient(colors: [Color.white, Color(red: 0.95, green: 0.95, blue: 0.95)]),
                            startPoint: .top,
                            endPoint: .bottom
                        ) :
                        LinearGradient(
                            gradient: Gradient(colors: [Color.red.opacity(0.8), Color.red.opacity(0.6)]),
                            startPoint: .top,
                            endPoint: .bottom
                        )
                    )
                    .frame(width: 35, height: 50)
                    .overlay(
                        BowlingPinShape()
                            .stroke(pin.isStanding ? Color.black.opacity(0.3) : Color.red, lineWidth: 2)
                            .frame(width: 35, height: 50)
                    )
                
                // Pin number
                Text("\(pin.pinNumber)")
                    .font(.system(size: 14, weight: .bold))
                    .foregroundColor(pin.isStanding ? .black : .white)
            }
            .shadow(color: Color.black.opacity(pin.isStanding ? 0.2 : 0.4), radius: 3)
            .scaleEffect(pin.isStanding ? 1.0 : 0.9)
            .rotationEffect(.degrees(pin.isStanding ? 0 : 90))
            .animation(.spring(response: 0.3), value: pin.isStanding)
        }
    }
}

struct GameView_Previews: PreviewProvider {
    static var previews: some View {
        GameView()
            .environmentObject(GameModel())
    }
}
