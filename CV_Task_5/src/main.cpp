/**
 * @file main.cpp
 * @brief Native Qt Widgets entry point for the PCA Eigenfaces desktop app.
 */

#include "ui/MainWindow.h"

#include <QApplication>

int main(int argc, char* argv[])
{
    QApplication app(argc, argv);

    facerecog::MainWindow window;
    window.show();

    return app.exec();
}
