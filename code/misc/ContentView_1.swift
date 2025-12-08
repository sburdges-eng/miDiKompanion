//
//  ContentView.swift
//  Bulling
//
//  Main navigation and game setup view
//

import SwiftUI

struct ContentView: View {
    @EnvironmentObject var gameModel: GameModel
    @State private var showingAddPlayer = false
    @State private var newPlayerName = ""
    
    var body: some View {
        NavigationView {
            ZStack {
                // Background
                Color(red: 0.95, green: 0.97, blue: 1.0)
                    .ignoresSafeArea()
                
                if gameModel.gameStarted {
                    GameView()
                } else {
                    setupView
                }
            }
            .navigationTitle("🐂 Bulling")
            .navigationBarTitleDisplayMode(.inline)
        }
    }
    
    var setupView: some View {
        VStack(spacing: 20) {
            // Logo area
            VStack {
                BullHeadLogo()
                    .frame(width: 100, height: 100)
                    .shadow(color: Color.black.opacity(0.1), radius: 5)
                
                Text("Strike & Score!")
                    .font(.subheadline)
                    .foregroundColor(.secondary)
                    .padding(.top, 8)
            }
            .padding(.top, 40)
            
            Spacer()
            
            // Players list
            VStack(alignment: .leading, spacing: 12) {
                Text("Players (\(gameModel.players.count))")
                    .font(.headline)
                    .foregroundColor(.primary)
                
                if gameModel.players.isEmpty {
                    Text("No players added yet")
                        .foregroundColor(.secondary)
                        .italic()
                        .frame(maxWidth: .infinity, alignment: .center)
                        .padding()
                } else {
                    ForEach(gameModel.players) { player in
                        HStack {
                            Image(systemName: "person.circle.fill")
                                .foregroundColor(.blue)
                            Text(player.name)
                                .font(.body)
                            Spacer()
                        }
                        .padding(12)
                        .background(Color.white)
                        .cornerRadius(8)
                        .shadow(color: Color.black.opacity(0.05), radius: 2)
                    }
                }
            }
            .padding()
            .frame(maxWidth: .infinity)
            .background(Color.white.opacity(0.5))
            .cornerRadius(12)
            .padding(.horizontal)
            
            Spacer()
            
            // Action buttons
            VStack(spacing: 12) {
                Button(action: { showingAddPlayer = true }) {
                    HStack {
                        Image(systemName: "plus.circle.fill")
                        Text("Add Player")
                            .fontWeight(.semibold)
                    }
                    .frame(maxWidth: .infinity)
                    .padding()
                    .background(Color.blue)
                    .foregroundColor(.white)
                    .cornerRadius(12)
                }
                
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
                }
            }
            .padding(.horizontal)
            .padding(.bottom, 30)
        }
        .alert("Add Player", isPresented: $showingAddPlayer) {
            TextField("Player Name", text: $newPlayerName)
            Button("Cancel", role: .cancel) {
                newPlayerName = ""
            }
            Button("Add") {
                if !newPlayerName.isEmpty {
                    gameModel.addPlayer(name: newPlayerName)
                    newPlayerName = ""
                }
            }
        }
    }
}

struct ContentView_Previews: PreviewProvider {
    static var previews: some View {
        ContentView()
            .environmentObject(GameModel())
    }
}
