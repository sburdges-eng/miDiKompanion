//
//  BullingApp.swift
//  Bulling
//
//  Main application entry point for iOS
//
//  ⚠️ PERSONAL USE ONLY
//  This software is licensed for personal, non-commercial use only.
//  See LICENSE.txt for complete terms and conditions.
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
    }
}
