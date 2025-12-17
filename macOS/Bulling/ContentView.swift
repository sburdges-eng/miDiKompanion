//
//  ContentView.swift
//  Bulling
//
//  Main navigation and game setup view for macOS
//

import SwiftUI

struct ContentView: View {
    @EnvironmentObject var gameModel: GameModel
    @State private var showingAddPlayer = false
    @State private var newPlayerName = ""
    @State private var selectedPlayerIndex: Int?

    var body: some View {
        ZStack {
            // Background
            Color(red: 0.95, green: 0.97, blue: 1.0)
                .ignoresSafeArea()

            if gameModel.gameStarted {
                GameView()
            } else if gameModel.gameOver {
                gameOverView
            } else {
                setupView
            }
        }
        .frame(minWidth: 800, minHeight: 600)
    }

    var setupView: some View {
        HStack(spacing: 0) {
            // Left side - Logo and branding
            VStack(spacing: 30) {
                Spacer()

                BullHeadLogo()
                    .frame(width: 180, height: 180)
                    .shadow(color: Color.black.opacity(0.2), radius: 10)

                VStack(spacing: 8) {
                    Text("BULLING")
                        .font(.system(size: 36, weight: .bold, design: .rounded))
                        .foregroundColor(Color(red: 0.3, green: 0.2, blue: 0.1))

                    Text("Strike & Score!")
                        .font(.title3)
                        .foregroundColor(.secondary)
                }

                Spacer()
            }
            .frame(maxWidth: .infinity)
            .background(
                LinearGradient(
                    gradient: Gradient(colors: [
                        Color(red: 0.95, green: 0.9, blue: 0.85),
                        Color(red: 0.9, green: 0.85, blue: 0.8)
                    ]),
                    startPoint: .top,
                    endPoint: .bottom
                )
            )

            // Right side - Player setup
            VStack(spacing: 24) {
                Text("Game Setup")
                    .font(.title)
                    .fontWeight(.bold)
                    .padding(.top, 30)

                // Players list
                VStack(alignment: .leading, spacing: 12) {
                    HStack {
                        Text("Players")
                            .font(.headline)
                        Spacer()
                        Text("\(gameModel.players.count) / 8")
                            .font(.caption)
                            .foregroundColor(.secondary)
                    }

                    if gameModel.players.isEmpty {
                        VStack(spacing: 12) {
                            Image(systemName: "person.3")
                                .font(.system(size: 40))
                                .foregroundColor(.secondary.opacity(0.5))
                            Text("No players added yet")
                                .foregroundColor(.secondary)
                        }
                        .frame(maxWidth: .infinity)
                        .padding(.vertical, 40)
                    } else {
                        List(selection: $selectedPlayerIndex) {
                            ForEach(Array(gameModel.players.enumerated()), id: \.element.id) { index, player in
                                HStack {
                                    Image(systemName: "person.circle.fill")
                                        .foregroundColor(.blue)
                                        .font(.title2)
                                    Text(player.name)
                                        .font(.body)
                                    Spacer()
                                }
                                .padding(.vertical, 4)
                                .tag(index)
                            }
                            .onDelete { indexSet in
                                for index in indexSet {
                                    gameModel.removePlayer(at: index)
                                }
                            }
                        }
                        .frame(height: 200)
                        .cornerRadius(8)
                    }
                }
                .padding()
                .background(Color.white)
                .cornerRadius(12)
                .shadow(color: Color.black.opacity(0.05), radius: 5)

                // Add player section
                HStack(spacing: 12) {
                    TextField("Player name", text: $newPlayerName)
                        .textFieldStyle(.roundedBorder)
                        .onSubmit {
                            addPlayer()
                        }

                    Button(action: addPlayer) {
                        Image(systemName: "plus.circle.fill")
                            .font(.title2)
                    }
                    .buttonStyle(.plain)
                    .foregroundColor(.blue)
                    .disabled(newPlayerName.isEmpty || gameModel.players.count >= 8)
                }
                .padding(.horizontal)

                Spacer()

                // Action buttons
                VStack(spacing: 12) {
                    if !gameModel.players.isEmpty {
                        Button(action: { gameModel.startGame() }) {
                            HStack {
                                Image(systemName: "play.fill")
                                Text("Start Game")
                                    .fontWeight(.semibold)
                            }
                            .frame(maxWidth: .infinity)
                            .padding()
                            .background(Color.green)
                            .foregroundColor(.white)
                            .cornerRadius(12)
                        }
                        .buttonStyle(.plain)
                    }

                    if !gameModel.players.isEmpty {
                        Button(action: { gameModel.resetGame() }) {
                            HStack {
                                Image(systemName: "trash")
                                Text("Clear All Players")
                            }
                            .font(.caption)
                            .foregroundColor(.red)
                        }
                        .buttonStyle(.plain)
                    }
                }
                .padding()
                .padding(.bottom, 20)
            }
            .frame(maxWidth: .infinity)
            .padding(.horizontal, 30)
        }
    }

    var gameOverView: some View {
        VStack(spacing: 30) {
            Spacer()

            BullHeadLogo()
                .frame(width: 120, height: 120)

            Text("Game Over!")
                .font(.system(size: 40, weight: .bold))
                .foregroundColor(Color(red: 0.3, green: 0.2, blue: 0.1))

            if let winner = gameModel.winner {
                VStack(spacing: 8) {
                    Text("Winner")
                        .font(.headline)
                        .foregroundColor(.secondary)

                    Text(winner.name)
                        .font(.system(size: 32, weight: .bold))
                        .foregroundColor(.orange)

                    Text("\(winner.scores[9] ?? 0) points")
                        .font(.title)
                        .foregroundColor(.blue)
                }
                .padding()
                .background(Color.white)
                .cornerRadius(16)
                .shadow(color: Color.black.opacity(0.1), radius: 10)
            }

            // Final scores
            VStack(spacing: 8) {
                Text("Final Scores")
                    .font(.headline)

                ForEach(gameModel.players.sorted(by: { ($0.scores[9] ?? 0) > ($1.scores[9] ?? 0) })) { player in
                    HStack {
                        Text(player.name)
                        Spacer()
                        Text("\(player.scores[9] ?? 0)")
                            .fontWeight(.bold)
                    }
                    .padding(.horizontal)
                }
            }
            .padding()
            .background(Color.white.opacity(0.8))
            .cornerRadius(12)
            .frame(maxWidth: 300)

            Button(action: { gameModel.resetGame() }) {
                HStack {
                    Image(systemName: "arrow.counterclockwise")
                    Text("New Game")
                        .fontWeight(.semibold)
                }
                .frame(width: 200)
                .padding()
                .background(Color.blue)
                .foregroundColor(.white)
                .cornerRadius(12)
            }
            .buttonStyle(.plain)

            Spacer()
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .background(
            LinearGradient(
                gradient: Gradient(colors: [
                    Color(red: 0.95, green: 0.9, blue: 0.85),
                    Color(red: 0.9, green: 0.85, blue: 0.8)
                ]),
                startPoint: .top,
                endPoint: .bottom
            )
        )
    }

    private func addPlayer() {
        let trimmedName = newPlayerName.trimmingCharacters(in: .whitespaces)
        guard !trimmedName.isEmpty, gameModel.players.count < 8 else { return }
        gameModel.addPlayer(name: trimmedName)
        newPlayerName = ""
    }
}

struct ContentView_Previews: PreviewProvider {
    static var previews: some View {
        ContentView()
            .environmentObject(GameModel())
    }
}
