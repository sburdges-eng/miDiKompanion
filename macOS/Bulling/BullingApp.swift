//
//  BullingApp.swift
//  Bulling
//
//  Main application entry point for macOS
//

import SwiftUI

@main
struct BullingApp: App {
    @StateObject private var gameModel = GameModel()
    @State private var showSplash = true

    var body: some Scene {
        WindowGroup {
            if showSplash {
                SplashScreen()
                    .onAppear {
                        DispatchQueue.main.asyncAfter(deadline: .now() + 2.0) {
                            withAnimation {
                                showSplash = false
                            }
                        }
                    }
            } else {
                ContentView()
                    .environmentObject(gameModel)
            }
        }
        .windowStyle(.hiddenTitleBar)
        .defaultSize(width: 900, height: 700)
        .commands {
            CommandGroup(replacing: .newItem) {}
            CommandMenu("Game") {
                Button("New Game") {
                    gameModel.resetGame()
                }
                .keyboardShortcut("n", modifiers: .command)

                Divider()

                Button("Reset Pins") {
                    gameModel.resetPins()
                }
                .keyboardShortcut("r", modifiers: .command)
            }
        }
    }
}
