#pragma once
/*
 * main_window.h - Qt Main Window (Alternative GUI)
 * ===============================================
 *
 * CONNECTIONS (for Cursor Graph):
 * - GUI Layer: Qt-based main window (alternative to JUCE PluginEditor)
 * - Platform Layer: Qt framework (QMainWindow)
 * - UI Layer: Provides Qt-based GUI alternative to JUCE interface
 *
 * Purpose: Qt-based main window for alternative GUI implementation.
 *          Note: Main plugin uses JUCE PluginEditor, this is an alternative Qt-based interface.
 *
 * Features:
 * - Qt main window
 * - Menu creation
 * - Toolbar creation
 * - UI setup
 */

#include <QMainWindow>
#include <QWidget>

namespace kelly {

class MainWindow : public QMainWindow {
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = nullptr);
    ~MainWindow() override = default;

private:
    void setupUi();
    void createMenus();
    void createToolbar();
};

} // namespace kelly
