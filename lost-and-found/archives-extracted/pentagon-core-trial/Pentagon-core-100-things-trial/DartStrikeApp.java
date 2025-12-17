package com.dartstrike;

import com.dartstrike.controllers.GameViewController;
import com.dartstrike.controllers.PlayerSetupController;
import com.dartstrike.models.GameModel;
import com.dartstrike.utils.PersistenceManager;
import javafx.application.Application;
import javafx.fxml.FXMLLoader;
import javafx.scene.Parent;
import javafx.scene.Scene;
import javafx.scene.control.Alert;
import javafx.scene.control.ButtonType;
import javafx.stage.Stage;

import java.io.IOException;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.Optional;

/**
 * Dart Strike JavaFX Application
 * Main entry point
 */
public class DartStrikeApp extends Application {
    
    private GameModel gameModel;
    private Stage primaryStage;
    
    @Override
    public void start(Stage primaryStage) {
        this.primaryStage = primaryStage;
        this.gameModel = new GameModel();
        
        primaryStage.setTitle("🎯 Dart Strike");
        primaryStage.setWidth(800);
        primaryStage.setHeight(600);
        
        // Check for saved game
        if (PersistenceManager.hasSavedGame()) {
            showResumeGameDialog();
        } else {
            showPlayerSetup();
        }
        
        // Auto-save on exit
        primaryStage.setOnCloseRequest(event -> {
            if (gameModel.gameStartedProperty().get() && !gameModel.gameCompleteProperty().get()) {
                PersistenceManager.saveGame(gameModel);
            }
        });
        
        primaryStage.show();
    }
    
    private void showResumeGameDialog() {
        long lastPlayed = PersistenceManager.getLastPlayedTime();
        String dateStr = new SimpleDateFormat("MMM dd, yyyy HH:mm").format(new Date(lastPlayed));
        
        Alert alert = new Alert(Alert.AlertType.CONFIRMATION);
        alert.setTitle("Resume Game?");
        alert.setHeaderText("Saved Game Found");
        alert.setContentText("You have a saved game from " + dateStr + ". Would you like to resume?");
        
        ButtonType resumeBtn = new ButtonType("Resume");
        ButtonType newGameBtn = new ButtonType("New Game");
        alert.getButtonTypes().setAll(resumeBtn, newGameBtn);
        
        Optional<ButtonType> result = alert.showAndWait();
        
        if (result.isPresent() && result.get() == resumeBtn) {
            PersistenceManager.GameState state = PersistenceManager.loadGame();
            if (state != null) {
                PersistenceManager.applyGameState(gameModel, state);
                showGameView();
            } else {
                showPlayerSetup();
            }
        } else {
            PersistenceManager.deleteSavedGame();
            showPlayerSetup();
        }
    }
    
    private void showPlayerSetup() {
        try {
            FXMLLoader loader = new FXMLLoader(getClass().getResource("/fxml/PlayerSetupView.fxml"));
            PlayerSetupController controller = new PlayerSetupController(gameModel);
            loader.setController(controller);
            Parent root = loader.load();
            
            Scene scene = new Scene(root, 600, 500);
            primaryStage.setScene(scene);
            primaryStage.setTitle("Dart Strike - Setup");
        } catch (IOException e) {
            showError("Failed to load player setup: " + e.getMessage());
            e.printStackTrace();
        }
    }
    
    private void showGameView() {
        try {
            FXMLLoader loader = new FXMLLoader(getClass().getResource("/fxml/GameView.fxml"));
            GameViewController controller = new GameViewController(gameModel);
            loader.setController(controller);
            Parent root = loader.load();
            
            Scene scene = new Scene(root, 800, 600);
            primaryStage.setScene(scene);
            primaryStage.setTitle("Dart Strike - Game");
        } catch (IOException e) {
            showError("Failed to load game view: " + e.getMessage());
            e.printStackTrace();
        }
    }
    
    private void showError(String message) {
        Alert alert = new Alert(Alert.AlertType.ERROR);
        alert.setTitle("Error");
        alert.setHeaderText(null);
        alert.setContentText(message);
        alert.showAndWait();
    }
    
    public static void main(String[] args) {
        launch(args);
    }
}
