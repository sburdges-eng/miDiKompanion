//
//  ScorecardView.swift
//  Bulling
//
//  Detailed scorecard view showing all frames and scores for macOS
//

import SwiftUI

struct ScorecardView: View {
    @EnvironmentObject var gameModel: GameModel

    var body: some View {
        VStack(spacing: 0) {
            // Header
            HStack {
                BullHeadLogo()
                    .frame(width: 40, height: 40)

                VStack(alignment: .leading, spacing: 2) {
                    Text("Scorecard")
                        .font(.headline)
                    if gameModel.gameOver {
                        Text("Game Complete")
                            .font(.caption)
                            .foregroundColor(.green)
                    }
                }

                Spacer()

                if gameModel.gameOver, let winner = gameModel.winner {
                    HStack(spacing: 4) {
                        Image(systemName: "trophy.fill")
                            .foregroundColor(.orange)
                        Text(winner.name)
                            .fontWeight(.bold)
                    }
                }
            }
            .padding()
            .background(Color.white)

            Divider()

            // Scorecard table
            ScrollView {
                VStack(spacing: 0) {
                    // Frame headers
                    HStack(spacing: 0) {
                        Text("Player")
                            .font(.caption)
                            .fontWeight(.bold)
                            .frame(width: 80, alignment: .leading)
                            .padding(.leading, 8)

                        ForEach(1...10, id: \.self) { frame in
                            Text("\(frame)")
                                .font(.caption)
                                .fontWeight(.bold)
                                .frame(width: 48)
                        }

                        Text("Total")
                            .font(.caption)
                            .fontWeight(.bold)
                            .frame(width: 50)
                    }
                    .padding(.vertical, 8)
                    .background(Color.gray.opacity(0.2))

                    // Player rows
                    ForEach(gameModel.players) { player in
                        PlayerScoreRow(player: player, isCurrentPlayer: gameModel.currentPlayer?.id == player.id)
                        Divider()
                    }
                }
            }
            .background(Color.white)
        }
        .background(Color(red: 0.95, green: 0.97, blue: 1.0))
    }
}

struct PlayerScoreRow: View {
    @ObservedObject var player: Player
    var isCurrentPlayer: Bool

    var body: some View {
        HStack(spacing: 0) {
            // Player name
            HStack {
                if isCurrentPlayer {
                    Circle()
                        .fill(Color.green)
                        .frame(width: 8, height: 8)
                }
                Text(player.name)
                    .font(.caption)
                    .fontWeight(.semibold)
                    .lineLimit(1)
            }
            .frame(width: 80, alignment: .leading)
            .padding(.leading, 8)

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
                .frame(width: 50)
        }
        .padding(.vertical, 6)
        .background(isCurrentPlayer ? Color.blue.opacity(0.1) : Color.clear)
    }
}

struct FrameCell: View {
    let frame: [Int?]
    let score: Int?
    let is10thFrame: Bool

    var body: some View {
        VStack(spacing: 1) {
            // Throws
            HStack(spacing: 1) {
                if is10thFrame {
                    // 10th frame has 3 boxes
                    ThrowBox(text: formatThrow(frame[0], isStrike: true), width: 14)
                    ThrowBox(text: formatThrow10th(frame[1], prev: frame[0]), width: 14)
                    ThrowBox(text: formatThrow10th(frame[2], prev: frame[0] == 10 ? nil : frame[1]), width: 14)
                } else {
                    // Regular frames have 2 boxes
                    ThrowBox(text: formatThrow(frame[0], isStrike: true), width: 22)
                    ThrowBox(text: formatThrow(frame[1], prev: frame[0]), width: 22)
                }
            }
            .frame(height: 16)

            // Score
            Text(score != nil ? "\(score!)" : "")
                .font(.system(size: 11, weight: .bold))
                .frame(width: 46, height: 18)
        }
        .frame(width: 48)
        .overlay(
            Rectangle()
                .stroke(Color.gray.opacity(0.3), lineWidth: 0.5)
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

struct ThrowBox: View {
    let text: String
    let width: CGFloat

    var body: some View {
        Text(text)
            .font(.system(size: 10, weight: text == "X" || text == "/" ? .bold : .regular))
            .foregroundColor(text == "X" ? .red : (text == "/" ? .orange : .primary))
            .frame(width: width, height: 14)
            .background(Color.gray.opacity(0.1))
    }
}

struct ScorecardView_Previews: PreviewProvider {
    static var previews: some View {
        ScorecardView()
            .environmentObject(GameModel())
    }
}
