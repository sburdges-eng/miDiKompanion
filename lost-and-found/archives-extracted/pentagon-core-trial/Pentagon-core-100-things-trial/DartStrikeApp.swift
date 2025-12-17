//
//  DartStrikeApp.swift
//  Dart Strike
//
//  Main app entry point
//

import SwiftUI

@main
struct DartStrikeApp: App {
    @StateObject private var game = GameModel()
    @Environment(\.scenePhase) private var scenePhase
    @State private var showLoadAlert = false
    
    var body: some Scene {
        WindowGroup {
            ContentView(game: game, showLoadAlert: $showLoadAlert)
                .onAppear {
                    checkForSavedGame()
                }
                .onChange(of: scenePhase) { newPhase in
                    handleScenePhaseChange(newPhase)
                }
                .alert("Resume Game?", isPresented: $showLoadAlert) {
                    Button("Resume") {
                        game.loadGame()
                    }
                    Button("New Game") {
                        game.deleteSavedGame()
                        game.resetGame()
                    }
                } message: {
                    if let lastPlayed = PersistenceManager.shared.getLastPlayedDate() {
                        Text("You have a saved game from \(formatDate(lastPlayed)). Would you like to resume?")
                    }
                }
        }
    }
    
    private func checkForSavedGame() {
        if game.hasSavedGame() && !game.gameStarted {
            showLoadAlert = true
        }
    }
    
    private func handleScenePhaseChange(_ phase: ScenePhase) {
        switch phase {
        case .background, .inactive:
            // Auto-save when app goes to background
            if game.gameStarted && !game.gameComplete {
                game.saveGame()
            }
        case .active:
            break
        @unknown default:
            break
        }
    }
    
    private func formatDate(_ date: Date) -> String {
        let formatter = DateFormatter()
        formatter.dateStyle = .medium
        formatter.timeStyle = .short
        return formatter.string(from: date)
    }
}

// MARK: - Content View
struct ContentView: View {
    @ObservedObject var game: GameModel
    @Binding var showLoadAlert: Bool
    
    var body: some View {
        ZStack {
            if game.gameStarted {
                GameView(game: game)
            } else {
                PlayerSetupView(game: game)
            }
        }
    }
}

// MARK: - Preview
struct ContentView_Previews: PreviewProvider {
    static var previews: some View {
        ContentView(game: GameModel(), showLoadAlert: .constant(false))
    }
}
