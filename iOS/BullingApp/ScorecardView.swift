//
//  ScorecardView.swift
//  Bulling
//
//  Detailed scorecard view showing all frames and scores
//

import SwiftUI

struct ScorecardView: View {
    @EnvironmentObject var gameModel: GameModel
    @Environment(\.dismiss) var dismiss
    
    var body: some View {
        NavigationView {
            ScrollView {
                VStack(spacing: 20) {
                    // Header
                    VStack(spacing: 8) {
                        BullHeadLogo()
                            .frame(width: 60, height: 60)
                        
                        Text("Scorecard")
                            .font(.title2)
                            .fontWeight(.bold)
                        
                        if gameModel.gameOver {
                            if let winner = gameModel.winner {
                                VStack(spacing: 4) {
                                    Text("🏆 Winner!")
                                        .font(.headline)
                                        .foregroundColor(.orange)
                                    Text(winner.name)
                                        .font(.title)
                                        .fontWeight(.bold)
                                    Text("\(winner.scores[9] ?? 0) points")
                                        .font(.title3)
                                        .foregroundColor(.blue)
                                }
                                .padding()
                                .background(Color.orange.opacity(0.1))
                                .cornerRadius(12)
                            }
                        }
                    }
                    .padding()
                    
                    // Scorecard table
                    VStack(spacing: 0) {
                        // Frame headers
                        HStack(spacing: 2) {
                            Text("Player")
                                .font(.caption)
                                .fontWeight(.bold)
                                .frame(width: 80, alignment: .leading)
                            
                            ForEach(1...10, id: \.self) { frame in
                                Text("\(frame)")
                                    .font(.caption)
                                    .fontWeight(.bold)
                                    .frame(width: 50)
                            }
                            
                            Text("Total")
                                .font(.caption)
                                .fontWeight(.bold)
                                .frame(width: 60)
                        }
                        .padding(.vertical, 8)
                        .background(Color.gray.opacity(0.2))
                        
                        // Player rows
                        ForEach(gameModel.players) { player in
                            PlayerScoreRow(player: player)
                                .background(
                                    gameModel.currentPlayer?.id == player.id ?
                                    Color.blue.opacity(0.1) : Color.clear
                                )
                        }
                    }
                    .background(Color.white)
                    .cornerRadius(12)
                    .shadow(color: Color.black.opacity(0.1), radius: 5)
                    .padding(.horizontal)
                }
                .padding(.vertical)
            }
            .background(Color(red: 0.95, green: 0.97, blue: 1.0))
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button("Done") {
                        dismiss()
                    }
                }
            }
        }
    }
}

struct PlayerScoreRow: View {
    @ObservedObject var player: Player
    
    var body: some View {
        HStack(spacing: 2) {
            // Player name
            Text(player.name)
                .font(.caption)
                .fontWeight(.semibold)
                .frame(width: 80, alignment: .leading)
                .lineLimit(1)
            
            // Frames 1-10
            ForEach(0..<10, id: \.self) { frameIdx in
                FrameCell(
                    frame: player.frames[frameIdx],
                    score: player.scores[frameIdx],
                    is10thFrame: frameIdx == 9
                )
            }
            
            // Total score
            Text("\(player.scores[9] ?? player.scores.compactMap { $0 }.last ?? 0)")
                .font(.caption)
                .fontWeight(.bold)
                .foregroundColor(.blue)
                .frame(width: 60)
        }
        .padding(.vertical, 8)
    }
}

struct FrameCell: View {
    let frame: [Int?]
    let score: Int?
    let is10thFrame: Bool
    
    var body: some View {
        VStack(spacing: 2) {
            // Throws
            HStack(spacing: 1) {
                if is10thFrame {
                    // 10th frame has 3 boxes
                    Text(formatThrow(frame[0], isStrike: true))
                        .font(.system(size: 9))
                        .frame(width: 15)
                    Text(formatThrow10th(frame[1], prev: frame[0]))
                        .font(.system(size: 9))
                        .frame(width: 15)
                    Text(formatThrow10th(frame[2], prev: frame[0] == 10 ? nil : frame[1]))
                        .font(.system(size: 9))
                        .frame(width: 15)
                } else {
                    // Regular frames have 2 boxes
                    Text(formatThrow(frame[0], isStrike: true))
                        .font(.system(size: 9))
                        .frame(width: 20)
                    Text(formatThrow(frame[1], prev: frame[0]))
                        .font(.system(size: 9))
                        .frame(width: 20)
                }
            }
            .frame(height: 18)
            .background(Color.gray.opacity(0.1))
            
            // Score
            Text(score != nil ? "\(score!)" : "")
                .font(.caption)
                .fontWeight(.bold)
                .frame(width: 50, height: 20)
        }
        .frame(width: 50)
        .overlay(
            Rectangle()
                .stroke(Color.gray.opacity(0.3), lineWidth: 1)
        )
    }
    
    func formatThrow(_ throwValue: Int?, isStrike: Bool = false, prev: Int? = nil) -> String {
        guard let throwValue = throwValue else { return "" }
        if isStrike && throwValue == 10 { return "X" }
        if let prev = prev, prev + throwValue == 10 { return "/" }
        if throwValue == 0 { return "-" }
        return "\(throwValue)"
    }

    func formatThrow10th(_ throwValue: Int?, prev: Int?) -> String {
        guard let throwValue = throwValue else { return "" }
        if throwValue == 10 { return "X" }
        if let prev = prev, prev + throwValue == 10 { return "/" }
        if throwValue == 0 { return "-" }
        return "\(throwValue)"
    }
}

struct ScorecardView_Previews: PreviewProvider {
    static var previews: some View {
        ScorecardView()
            .environmentObject(GameModel())
    }
}
