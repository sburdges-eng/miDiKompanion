//
//  GameView.swift
//  Bulling
//
//  Main game play view with interactive pins for macOS
//

import SwiftUI

struct GameView: View {
    @EnvironmentObject var gameModel: GameModel
    @State private var showScorecard = false

    var body: some View {
        HSplitView {
            // Left side - Pin area
            VStack(spacing: 20) {
                // Current player header
                currentPlayerHeader

                Spacer()

                // Pin layout area
                pinArea

                Spacer()

                // Submit button
                submitButton

                // Bottom controls
                bottomControls
            }
            .frame(minWidth: 450)
            .padding()
            .background(Color(red: 0.95, green: 0.97, blue: 1.0))

            // Right side - Scorecard
            ScorecardView()
                .frame(minWidth: 350)
        }
    }

    var currentPlayerHeader: some View {
        VStack(spacing: 8) {
            if let player = gameModel.currentPlayer {
                HStack(spacing: 16) {
                    // Player info
                    HStack {
                        Image(systemName: "person.circle.fill")
                            .font(.title)
                            .foregroundColor(.orange)

                        VStack(alignment: .leading, spacing: 2) {
                            Text(player.name)
                                .font(.title2)
                                .fontWeight(.bold)
                            Text("Frame \(player.currentFrame + 1), Ball \(player.currentThrow + 1)")
                                .font(.subheadline)
                                .foregroundColor(.secondary)
                        }
                    }

                    Spacer()

                    // Pins knocked down
                    VStack(alignment: .trailing, spacing: 2) {
                        Text("\(gameModel.pinsKnockedDown)")
                            .font(.system(size: 36, weight: .bold))
                            .foregroundColor(.red)
                        Text("pins down")
                            .font(.caption)
                            .foregroundColor(.secondary)
                    }
                }
                .padding()
                .background(Color.white)
                .cornerRadius(12)
                .shadow(color: Color.black.opacity(0.1), radius: 5)
            }
        }
    }

    var pinArea: some View {
        VStack(spacing: 20) {
            Text("Click pins to knock them down")
                .font(.subheadline)
                .foregroundColor(.secondary)

            // Bowling pin layout
            ZStack {
                // Background (lane)
                RoundedRectangle(cornerRadius: 20)
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
                    .frame(width: 380, height: 400)

                // Pins in standard 10-pin layout
                VStack(spacing: 20) {
                    // Row 1 (back) - 4 pins
                    HStack(spacing: 25) {
                        ForEach([7, 8, 9, 10], id: \.self) { pinId in
                            if let pin = gameModel.pins.first(where: { $0.id == pinId }) {
                                PinView(pin: pin) {
                                    gameModel.togglePin(pin)
                                }
                            }
                        }
                    }

                    // Row 2 - 3 pins
                    HStack(spacing: 25) {
                        ForEach([4, 5, 6], id: \.self) { pinId in
                            if let pin = gameModel.pins.first(where: { $0.id == pinId }) {
                                PinView(pin: pin) {
                                    gameModel.togglePin(pin)
                                }
                            }
                        }
                    }

                    // Row 3 - 2 pins
                    HStack(spacing: 25) {
                        ForEach([2, 3], id: \.self) { pinId in
                            if let pin = gameModel.pins.first(where: { $0.id == pinId }) {
                                PinView(pin: pin) {
                                    gameModel.togglePin(pin)
                                }
                            }
                        }
                    }

                    // Row 4 (front) - 1 pin
                    HStack(spacing: 25) {
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
            .frame(width: 200)
            .padding()
            .background(Color.green)
            .foregroundColor(.white)
            .cornerRadius(12)
        }
        .buttonStyle(.plain)
        .keyboardShortcut(.return, modifiers: [])
    }

    var bottomControls: some View {
        HStack(spacing: 20) {
            Button(action: {
                gameModel.knockDownAll()
            }) {
                HStack {
                    Image(systemName: "burst.fill")
                    Text("Strike!")
                }
                .padding(.horizontal, 20)
                .padding(.vertical, 10)
                .background(Color.orange.opacity(0.2))
                .foregroundColor(.orange)
                .cornerRadius(8)
            }
            .buttonStyle(.plain)

            Button(action: {
                gameModel.resetPins()
            }) {
                HStack {
                    Image(systemName: "arrow.counterclockwise")
                    Text("Reset Pins")
                }
                .padding(.horizontal, 20)
                .padding(.vertical, 10)
                .background(Color.blue.opacity(0.2))
                .foregroundColor(.blue)
                .cornerRadius(8)
            }
            .buttonStyle(.plain)

            Spacer()

            Button(action: {
                gameModel.resetGame()
            }) {
                HStack {
                    Image(systemName: "stop.circle")
                    Text("End Game")
                }
                .padding(.horizontal, 20)
                .padding(.vertical, 10)
                .background(Color.red.opacity(0.2))
                .foregroundColor(.red)
                .cornerRadius(8)
            }
            .buttonStyle(.plain)
        }
        .padding()
        .background(Color.white)
        .cornerRadius(12)
        .shadow(color: Color.black.opacity(0.05), radius: 3)
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
                    .frame(width: 45, height: 65)
                    .overlay(
                        BowlingPinShape()
                            .stroke(pin.isStanding ? Color.black.opacity(0.3) : Color.red, lineWidth: 2)
                            .frame(width: 45, height: 65)
                    )

                // Pin number
                Text("\(pin.pinNumber)")
                    .font(.system(size: 16, weight: .bold))
                    .foregroundColor(pin.isStanding ? .black : .white)
            }
            .shadow(color: Color.black.opacity(pin.isStanding ? 0.2 : 0.4), radius: 4)
            .scaleEffect(pin.isStanding ? 1.0 : 0.85)
            .rotationEffect(.degrees(pin.isStanding ? 0 : 90))
            .animation(.spring(response: 0.3), value: pin.isStanding)
        }
        .buttonStyle(.plain)
        .onHover { hovering in
            if hovering && pin.isStanding {
                NSCursor.pointingHand.push()
            } else {
                NSCursor.pop()
            }
        }
    }
}

struct GameView_Previews: PreviewProvider {
    static var previews: some View {
        GameView()
            .environmentObject(GameModel())
    }
}
