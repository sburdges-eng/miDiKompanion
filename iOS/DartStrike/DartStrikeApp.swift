//
//  DartStrikeApp.swift
//  Dart Strike
//
//  Main application entry point for iOS
//

import SwiftUI

@main
struct DartStrikeApp: App {
    @StateObject private var gameModel = GameModel()
    
    var body: some Scene {
        WindowGroup {
            ContentView()
                .environmentObject(gameModel)
        }
    }
}
