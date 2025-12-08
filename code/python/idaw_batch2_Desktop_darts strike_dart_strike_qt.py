#!/usr/bin/env python3
"""
Dart Strike - Bowling Scoring App (Python/Qt6 Version)
Traditional 10-pin bowling with proper scoring
"""

import sys
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QFrame, QGridLayout, QLineEdit,
    QMessageBox, QScrollArea, QInputDialog
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont


class Pin(QPushButton):
    """Interactive bowling pin button"""

    def __init__(self, pin_id, parent=None):
        super().__init__(parent)
        self.pin_id = pin_id
        self.standing = True
        self.setFixedSize(50, 50)
        self.update_style()
        self.clicked.connect(self.toggle)

    def toggle(self):
        self.standing = not self.standing
        self.update_style()

    def update_style(self):
        if self.standing:
            self.setStyleSheet("""
                QPushButton {
                    background-color: #FFFFFF;
                    border: 3px solid #1D1D1F;
                    border-radius: 25px;
                    font-size: 16px;
                    font-weight: bold;
                    color: #1D1D1F;
                }
                QPushButton:hover {
                    background-color: #F5F5F7;
                    border-color: #007AFF;
                }
            """)
        else:
            self.setStyleSheet("""
                QPushButton {
                    background-color: #FF3B30;
                    border: 3px solid #CC2F27;
                    border-radius: 25px;
                    font-size: 16px;
                    font-weight: bold;
                    color: white;
                }
                QPushButton:hover {
                    background-color: #FF6961;
                }
            """)
        self.setText(str(self.pin_id))

    def reset(self):
        self.standing = True
        self.update_style()


class Player:
    """Player with 10 frames of bowling data"""

    def __init__(self, name):
        self.name = name
        self.frames = [[None, None, None] for _ in range(10)]  # [throw1, throw2, throw3 for 10th]
        self.scores = [None] * 10
        self.current_frame = 0
        self.current_throw = 0

    def is_complete(self):
        return self.current_frame >= 10


class DartStrikeApp(QMainWindow):
    """Main Dart Strike Application"""

    def __init__(self):
        super().__init__()
        self.players = []
        self.current_player_index = 0
        self.pins = []
        self.game_started = False

        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("Dart Strike - Bowling Scorer")
        self.setGeometry(100, 100, 900, 700)
        self.setStyleSheet("background-color: #F5F5F7;")

        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setSpacing(15)
        layout.setContentsMargins(20, 20, 20, 20)

        # Title
        title = QLabel("🎳 DART STRIKE")
        title.setFont(QFont("Helvetica Neue", 32, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("color: #1D1D1F; padding: 10px;")
        layout.addWidget(title)

        # Status bar
        self.status_label = QLabel("Add players to start a new game")
        self.status_label.setFont(QFont("Helvetica Neue", 14))
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setStyleSheet("""
            background-color: #007AFF;
            color: white;
            padding: 10px;
            border-radius: 8px;
        """)
        layout.addWidget(self.status_label)

        # Main content area
        content = QHBoxLayout()

        # Left side - Pin layout
        pin_frame = QFrame()
        pin_frame.setStyleSheet("""
            QFrame {
                background-color: #D4A574;
                border-radius: 12px;
                padding: 20px;
            }
        """)
        pin_layout = QVBoxLayout(pin_frame)

        pin_title = QLabel("Click pins to knock down")
        pin_title.setFont(QFont("Helvetica Neue", 12))
        pin_title.setAlignment(Qt.AlignCenter)
        pin_title.setStyleSheet("color: #1D1D1F; background: transparent;")
        pin_layout.addWidget(pin_title)

        # Pin triangle layout (standard 10-pin)
        self.create_pin_layout(pin_layout)

        # Submit throw button
        self.submit_btn = QPushButton("Submit Throw")
        self.submit_btn.setFont(QFont("Helvetica Neue", 14, QFont.Bold))
        self.submit_btn.setStyleSheet("""
            QPushButton {
                background-color: #34C759;
                color: white;
                border: none;
                border-radius: 8px;
                padding: 15px 30px;
            }
            QPushButton:hover {
                background-color: #2DB550;
            }
            QPushButton:disabled {
                background-color: #86868B;
            }
        """)
        self.submit_btn.clicked.connect(self.submit_throw)
        self.submit_btn.setEnabled(False)
        pin_layout.addWidget(self.submit_btn)

        content.addWidget(pin_frame, 1)

        # Right side - Scorecard
        score_frame = QFrame()
        score_frame.setStyleSheet("""
            QFrame {
                background-color: white;
                border-radius: 12px;
                padding: 15px;
            }
        """)
        score_layout = QVBoxLayout(score_frame)

        score_title = QLabel("Scorecard")
        score_title.setFont(QFont("Helvetica Neue", 16, QFont.Bold))
        score_title.setStyleSheet("color: #1D1D1F;")
        score_layout.addWidget(score_title)

        # Scrollable scorecard
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("border: none;")

        self.scorecard_widget = QWidget()
        self.scorecard_layout = QVBoxLayout(self.scorecard_widget)
        scroll.setWidget(self.scorecard_widget)
        score_layout.addWidget(scroll)

        content.addWidget(score_frame, 2)

        layout.addLayout(content)

        # Control buttons
        controls = QHBoxLayout()

        add_player_btn = QPushButton("Add Player")
        add_player_btn.setStyleSheet("""
            QPushButton {
                background-color: #007AFF;
                color: white;
                border: none;
                border-radius: 8px;
                padding: 12px 24px;
                font-size: 13px;
            }
            QPushButton:hover {
                background-color: #0056CC;
            }
        """)
        add_player_btn.clicked.connect(self.add_player)
        controls.addWidget(add_player_btn)

        start_btn = QPushButton("Start Game")
        start_btn.setStyleSheet("""
            QPushButton {
                background-color: #34C759;
                color: white;
                border: none;
                border-radius: 8px;
                padding: 12px 24px;
                font-size: 13px;
            }
            QPushButton:hover {
                background-color: #2DB550;
            }
        """)
        start_btn.clicked.connect(self.start_game)
        controls.addWidget(start_btn)

        reset_btn = QPushButton("New Game")
        reset_btn.setStyleSheet("""
            QPushButton {
                background-color: #FF9500;
                color: white;
                border: none;
                border-radius: 8px;
                padding: 12px 24px;
                font-size: 13px;
            }
            QPushButton:hover {
                background-color: #CC7700;
            }
        """)
        reset_btn.clicked.connect(self.reset_game)
        controls.addWidget(reset_btn)

        controls.addStretch()

        self.pins_down_label = QLabel("Pins Down: 0")
        self.pins_down_label.setFont(QFont("Helvetica Neue", 14, QFont.Bold))
        self.pins_down_label.setStyleSheet("color: #FF3B30;")
        controls.addWidget(self.pins_down_label)

        layout.addLayout(controls)

    def create_pin_layout(self, parent_layout):
        """Create 10-pin bowling layout"""
        pin_container = QWidget()
        pin_container.setStyleSheet("background: transparent;")
        grid = QGridLayout(pin_container)
        grid.setSpacing(10)

        # Standard 10-pin layout:
        #       7  8  9  10
        #         4  5  6
        #           2  3
        #             1

        positions = [
            (3, 1.5, 1),   # Pin 1 - front
            (2, 1, 2),     # Pin 2
            (2, 2, 3),     # Pin 3
            (1, 0.5, 4),   # Pin 4
            (1, 1.5, 5),   # Pin 5
            (1, 2.5, 6),   # Pin 6
            (0, 0, 7),     # Pin 7 - back left
            (0, 1, 8),     # Pin 8
            (0, 2, 9),     # Pin 9
            (0, 3, 10),    # Pin 10 - back right
        ]

        self.pins = []
        for row, col, pin_id in positions:
            pin = Pin(pin_id)
            pin.clicked.connect(self.update_pins_count)
            self.pins.append(pin)
            grid.addWidget(pin, row, int(col * 2), 1, 2)

        parent_layout.addWidget(pin_container)

    def update_pins_count(self):
        """Update the pins knocked down counter"""
        count = sum(1 for pin in self.pins if not pin.standing)
        self.pins_down_label.setText(f"Pins Down: {count}")

    def add_player(self):
        """Add a new player"""
        name, ok = QInputDialog.getText(self, "Add Player", "Enter player name:")
        if ok and name.strip():
            self.players.append(Player(name.strip()))
            self.update_scorecard()
            self.status_label.setText(f"Added {name}. {len(self.players)} player(s) ready.")

    def start_game(self):
        """Start the game"""
        if not self.players:
            QMessageBox.warning(self, "No Players", "Add at least one player first!")
            return

        self.game_started = True
        self.current_player_index = 0
        self.submit_btn.setEnabled(True)
        self.reset_pins()
        self.update_status()
        self.update_scorecard()

    def reset_game(self):
        """Reset for a new game"""
        self.players = []
        self.current_player_index = 0
        self.game_started = False
        self.submit_btn.setEnabled(False)
        self.reset_pins()
        self.status_label.setText("Add players to start a new game")
        self.update_scorecard()

    def reset_pins(self):
        """Reset all pins to standing"""
        for pin in self.pins:
            pin.reset()
        self.update_pins_count()

    def get_pins_knocked(self):
        """Get count of knocked pins"""
        return sum(1 for pin in self.pins if not pin.standing)

    def submit_throw(self):
        """Submit the current throw"""
        if not self.game_started:
            return

        player = self.players[self.current_player_index]
        if player.is_complete():
            self.next_player()
            return

        pins_down = self.get_pins_knocked()
        frame_idx = player.current_frame
        throw_idx = player.current_throw

        # Handle 10th frame special rules
        if frame_idx == 9:
            self.handle_10th_frame(player, pins_down)
        else:
            self.handle_regular_frame(player, pins_down)

        self.calculate_all_scores()
        self.update_scorecard()
        self.update_status()

    def handle_regular_frame(self, player, pins_down):
        """Handle throws in frames 1-9"""
        frame_idx = player.current_frame

        if player.current_throw == 0:
            # First throw
            player.frames[frame_idx][0] = pins_down

            if pins_down == 10:
                # Strike!
                player.current_frame += 1
                player.current_throw = 0
                self.next_player()
            else:
                player.current_throw = 1
        else:
            # Second throw
            player.frames[frame_idx][1] = pins_down
            player.current_frame += 1
            player.current_throw = 0
            self.next_player()

        self.reset_pins()

    def handle_10th_frame(self, player, pins_down):
        """Handle 10th frame with bonus throws"""
        frame = player.frames[9]

        if player.current_throw == 0:
            # First throw
            frame[0] = pins_down
            player.current_throw = 1
            if pins_down == 10:
                self.reset_pins()  # Strike - reset for bonus

        elif player.current_throw == 1:
            # Second throw
            frame[1] = pins_down
            first = frame[0]

            if first == 10 or (first + pins_down == 10):
                # Strike or spare - get third throw
                player.current_throw = 2
                self.reset_pins()
            else:
                # No bonus - done
                player.current_frame = 10
                player.current_throw = 0
                self.next_player()

        elif player.current_throw == 2:
            # Third throw (bonus)
            frame[2] = pins_down
            player.current_frame = 10
            player.current_throw = 0
            self.next_player()

        if player.current_throw != 0 or player.current_frame < 10:
            pass  # Don't reset if continuing in 10th
        else:
            self.reset_pins()

    def next_player(self):
        """Move to next player"""
        self.current_player_index += 1

        if self.current_player_index >= len(self.players):
            self.current_player_index = 0

        # Skip completed players
        attempts = 0
        while self.players[self.current_player_index].is_complete() and attempts < len(self.players):
            self.current_player_index = (self.current_player_index + 1) % len(self.players)
            attempts += 1

        # Check if game is over
        if all(p.is_complete() for p in self.players):
            self.game_over()
        else:
            self.reset_pins()

    def game_over(self):
        """Handle game completion"""
        self.game_started = False
        self.submit_btn.setEnabled(False)

        # Find winner
        winner = max(self.players, key=lambda p: p.scores[9] or 0)
        self.status_label.setText(f"🎉 Game Over! Winner: {winner.name} with {winner.scores[9]} points!")
        self.status_label.setStyleSheet("""
            background-color: #34C759;
            color: white;
            padding: 10px;
            border-radius: 8px;
        """)

    def calculate_all_scores(self):
        """Calculate scores for all players"""
        for player in self.players:
            cumulative = 0

            for i in range(10):
                frame = player.frames[i]
                first = frame[0]

                if first is None:
                    player.scores[i] = None
                    continue

                if i == 9:
                    # 10th frame - just add all throws
                    score = (first or 0) + (frame[1] or 0) + (frame[2] or 0)
                    cumulative += score
                    player.scores[i] = cumulative
                else:
                    score = self.calculate_frame_score(player, i)
                    if score is not None:
                        cumulative += score
                        player.scores[i] = cumulative
                    else:
                        player.scores[i] = None

    def calculate_frame_score(self, player, frame_idx):
        """Calculate score for a single frame"""
        frame = player.frames[frame_idx]
        first = frame[0]
        second = frame[1]

        if first is None:
            return None

        # Strike
        if first == 10:
            next_frame = player.frames[frame_idx + 1]
            next_first = next_frame[0]

            if next_first is None:
                return None

            if next_first == 10 and frame_idx < 8:
                # Next is also strike
                next_next = player.frames[frame_idx + 2]
                if next_next[0] is None:
                    return None
                return 10 + 10 + next_next[0]
            elif next_first == 10 and frame_idx == 8:
                # Next is 10th frame
                if next_frame[1] is None:
                    return None
                return 10 + next_first + next_frame[1]
            else:
                if next_frame[1] is None:
                    return None
                return 10 + next_first + next_frame[1]

        if second is None:
            return None

        # Spare
        if first + second == 10:
            next_frame = player.frames[frame_idx + 1]
            if next_frame[0] is None:
                return None
            return 10 + next_frame[0]

        # Open frame
        return first + second

    def update_status(self):
        """Update the status display"""
        if not self.game_started:
            return

        player = self.players[self.current_player_index]
        frame_num = player.current_frame + 1
        throw_num = player.current_throw + 1

        self.status_label.setText(f"🎳 {player.name} - Frame {frame_num}, Throw {throw_num}")
        self.status_label.setStyleSheet("""
            background-color: #007AFF;
            color: white;
            padding: 10px;
            border-radius: 8px;
        """)

    def update_scorecard(self):
        """Update the scorecard display"""
        # Clear existing
        for i in reversed(range(self.scorecard_layout.count())):
            widget = self.scorecard_layout.itemAt(i).widget()
            if widget:
                widget.deleteLater()

        if not self.players:
            no_players = QLabel("No players yet")
            no_players.setFont(QFont("Helvetica Neue", 12))
            no_players.setStyleSheet("color: #86868B;")
            self.scorecard_layout.addWidget(no_players)
            return

        # Header row
        header = QWidget()
        header_layout = QHBoxLayout(header)
        header_layout.setSpacing(2)

        name_label = QLabel("Player")
        name_label.setFixedWidth(80)
        name_label.setFont(QFont("Helvetica Neue", 10, QFont.Bold))
        header_layout.addWidget(name_label)

        for i in range(1, 11):
            frame_label = QLabel(str(i))
            frame_label.setFixedWidth(45)
            frame_label.setAlignment(Qt.AlignCenter)
            frame_label.setFont(QFont("Helvetica Neue", 10, QFont.Bold))
            header_layout.addWidget(frame_label)

        total_label = QLabel("Total")
        total_label.setFixedWidth(50)
        total_label.setAlignment(Qt.AlignCenter)
        total_label.setFont(QFont("Helvetica Neue", 10, QFont.Bold))
        header_layout.addWidget(total_label)

        self.scorecard_layout.addWidget(header)

        # Player rows
        for idx, player in enumerate(self.players):
            is_current = idx == self.current_player_index and self.game_started

            row = QWidget()
            row_layout = QHBoxLayout(row)
            row_layout.setSpacing(2)

            if is_current:
                row.setStyleSheet("background-color: #E3F2FD; border-radius: 4px;")

            name = QLabel(player.name[:8])
            name.setFixedWidth(80)
            name.setFont(QFont("Helvetica Neue", 10, QFont.Bold if is_current else QFont.Normal))
            row_layout.addWidget(name)

            for i in range(10):
                frame_widget = self.create_frame_widget(player, i)
                row_layout.addWidget(frame_widget)

            total = player.scores[9] if player.scores[9] else "-"
            total_lbl = QLabel(str(total))
            total_lbl.setFixedWidth(50)
            total_lbl.setAlignment(Qt.AlignCenter)
            total_lbl.setFont(QFont("Helvetica Neue", 12, QFont.Bold))
            total_lbl.setStyleSheet("color: #007AFF;")
            row_layout.addWidget(total_lbl)

            self.scorecard_layout.addWidget(row)

        self.scorecard_layout.addStretch()

    def create_frame_widget(self, player, frame_idx):
        """Create a widget for a single frame"""
        frame = player.frames[frame_idx]
        score = player.scores[frame_idx]

        widget = QFrame()
        widget.setFixedWidth(45)
        widget.setStyleSheet("""
            QFrame {
                background-color: #F5F5F7;
                border: 1px solid #E5E5E5;
                border-radius: 4px;
            }
        """)

        layout = QVBoxLayout(widget)
        layout.setSpacing(0)
        layout.setContentsMargins(2, 2, 2, 2)

        # Throws row
        throws_widget = QWidget()
        throws_layout = QHBoxLayout(throws_widget)
        throws_layout.setSpacing(1)
        throws_layout.setContentsMargins(0, 0, 0, 0)

        if frame_idx < 9:
            # Regular frame - 2 throws
            t1 = self.format_throw(frame[0], is_strike=True)
            t2 = self.format_throw(frame[1], prev=frame[0])

            for t in [t1, t2]:
                lbl = QLabel(t)
                lbl.setAlignment(Qt.AlignCenter)
                lbl.setFont(QFont("Helvetica Neue", 9))
                throws_layout.addWidget(lbl)
        else:
            # 10th frame - 3 throws
            t1 = self.format_throw(frame[0], is_strike=True)
            t2 = self.format_throw_10th(frame[1], frame[0])
            t3 = self.format_throw_10th(frame[2], frame[1] if frame[0] != 10 else None)

            for t in [t1, t2, t3]:
                lbl = QLabel(t)
                lbl.setAlignment(Qt.AlignCenter)
                lbl.setFont(QFont("Helvetica Neue", 8))
                throws_layout.addWidget(lbl)

        layout.addWidget(throws_widget)

        # Score
        score_lbl = QLabel(str(score) if score is not None else "")
        score_lbl.setAlignment(Qt.AlignCenter)
        score_lbl.setFont(QFont("Helvetica Neue", 10, QFont.Bold))
        layout.addWidget(score_lbl)

        return widget

    def format_throw(self, throw, is_strike=False, prev=None):
        """Format a throw for display"""
        if throw is None:
            return ""
        if is_strike and throw == 10:
            return "X"
        if prev is not None and prev + throw == 10:
            return "/"
        if throw == 0:
            return "-"
        return str(throw)

    def format_throw_10th(self, throw, prev):
        """Format 10th frame throws"""
        if throw is None:
            return ""
        if throw == 10:
            return "X"
        if prev is not None and prev + throw == 10:
            return "/"
        if throw == 0:
            return "-"
        return str(throw)


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    window = DartStrikeApp()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
