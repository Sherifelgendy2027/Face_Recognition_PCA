#ifndef MAINWINDOW_H
#define MAINWINDOW_H

/**
 * @file MainWindow.h
 * @brief Step 5: Qt Widgets UI for PCA face detection, recognition, and ROC plotting.
 */

#include "../pipeline/FaceDetector.h"
#include "../pipeline/FaceRecognizer.h"
#include "../pipeline/PCAPipeline.h"
#include "ROCWidget.h"

#include <QImage>
#include <QMainWindow>
#include <QPointF>

#include <memory>
#include <utility>
#include <string>
#include <vector>

class QLabel;
class QPushButton;
class QTextEdit;
class QDoubleSpinBox;
class QSpinBox;
class QResizeEvent;

namespace facerecog {

class MainWindow : public QMainWindow {
public:
    explicit MainWindow(QWidget* parent = nullptr);

protected:
    void resizeEvent(QResizeEvent* event) override;

private:
    void buildUi();
    void connectSignals();

    void trainModel();
    void loadTestImage();
    void runDetectionAndRecognition();

    void updateImageView();
    void annotateDetections(const std::vector<FaceDetection>& detections,
                            const std::vector<RecognitionResult>& recognitions);
    void appendLog(const QString& message);

    QString defaultDatasetPath() const;
    std::pair<std::vector<QPointF>, double> buildRocCurve() const;

private:
    PCAPipeline m_pca;
    std::unique_ptr<FaceDetector> m_detector;
    std::unique_ptr<FaceRecognizer> m_recognizer;

    QString m_datasetPath;
    QString m_testImagePath;

    // Raw, unscaled target image used by the math pipeline.
    // Never replace this with a QLabel-scaled pixmap/image.
    QImage m_rawTargetImage;

    // UI-only copies. These can be converted, annotated, and scaled for display.
    QImage m_displayImage;
    QImage m_annotatedImage;

    QLabel* m_imageLabel = nullptr;
    QLabel* m_statusLabel = nullptr;
    QPushButton* m_trainButton = nullptr;
    QPushButton* m_loadImageButton = nullptr;
    QPushButton* m_runButton = nullptr;
    QTextEdit* m_log = nullptr;
    QDoubleSpinBox* m_varianceSpin = nullptr;
    QDoubleSpinBox* m_thresholdSpin = nullptr;
    QSpinBox* m_strideSpin = nullptr;
    QDoubleSpinBox* m_maxScaleSpin = nullptr;
    QSpinBox* m_knnSpin = nullptr;
    ROCWidget* m_rocWidget = nullptr;
};

} // namespace facerecog

#endif // MAINWINDOW_H
