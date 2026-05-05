#ifndef MAINWINDOW_H
#define MAINWINDOW_H

/**
 * @file MainWindow.h
 * @brief Step 5: Qt Widgets UI for PCA face detection, recognition, and ROC plotting.
 */

#include "../pipeline/FaceDetector.h"
#include "../pipeline/FaceRecognizer.h"
#include "../pipeline/PCAPipeline.h"
#include "DetectionWorker.h"
#include "ROCWidget.h"

#include <QImage>
#include <QMainWindow>
#include <QPointF>

#include <memory>
#include <utility>
#include <string>
#include <vector>

class QLabel;
class QProgressBar;
class QPushButton;
class QTextEdit;
class QSlider;
class QResizeEvent;
class QThread;

namespace facerecog {

class MainWindow : public QMainWindow {
public:
    explicit MainWindow(QWidget* parent = nullptr);
    ~MainWindow() override;

protected:
    void resizeEvent(QResizeEvent* event) override;

private:
    void buildUi();
    void connectSignals();

    void trainModel();
    void loadTestImage();
    void runDetectionAndRecognition();
    void generatePerformanceReport();

    void updateImageView();
    void annotateDetections(const std::vector<FaceDetection>& detections,
                            const std::vector<RecognitionResult>& recognitions);
    void appendLog(const QString& message);
    void setDetectionUiRunning(bool running);

    QString defaultDatasetPath() const;
    std::pair<std::vector<QPointF>, double> buildRocCurve() const;
    std::pair<int, int> leaveOneOutRecognitionScore(int knnK) const;
    QString buildPerformanceReportText() const;

    // ── Slider value accessors (apply integer-to-real scaling) ──
    double varianceValue() const;     ///< Slider 50..99 → 0.50..0.99
    double thresholdValue() const;    ///< Slider 1..50  → 1.0..50.0
    int    strideValue() const;       ///< Slider 4..64  → 4..64
    double maxScaleValue() const;     ///< Slider 10..50 → 1.0..5.0
    int    knnValue() const;          ///< Slider 1..15  → 1..15

private:
    PCAPipeline m_pca;
    std::unique_ptr<FaceDetector> m_detector;
    std::unique_ptr<FaceRecognizer> m_recognizer;

    QThread* m_detectionThread = nullptr;
    bool m_detectionRunning = false;

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
    QPushButton* m_reportButton = nullptr;
    QProgressBar* m_detectionProgressBar = nullptr;
    QTextEdit* m_log = nullptr;

    // Parameter sliders + companion value labels
    QSlider* m_varianceSlider   = nullptr;
    QLabel*  m_varianceLabel    = nullptr;
    QSlider* m_thresholdSlider  = nullptr;
    QLabel*  m_thresholdLabel   = nullptr;
    QSlider* m_strideSlider     = nullptr;
    QLabel*  m_strideLabel      = nullptr;
    QSlider* m_maxScaleSlider   = nullptr;
    QLabel*  m_maxScaleLabel    = nullptr;
    QSlider* m_knnSlider        = nullptr;
    QLabel*  m_knnLabel         = nullptr;

    ROCWidget* m_rocWidget = nullptr;
};

} // namespace facerecog

#endif // MAINWINDOW_H
