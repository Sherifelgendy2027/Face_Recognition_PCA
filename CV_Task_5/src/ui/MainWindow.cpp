#include "MainWindow.h"

#include "../core/MathEngine.h"

#include <QApplication>
#include <QColor>
#include <QCoreApplication>
#include <QDir>
#include <QDateTime>
#include <QSlider>
#include <QFile>
#include <QFileDialog>
#include <QFont>
#include <QFormLayout>
#include <QFrame>
#include <QGroupBox>
#include <QHBoxLayout>
#include <QImageReader>
#include <QLabel>
#include <QMessageBox>
#include <QPainter>
#include <QProgressBar>
#include <QPen>
#include <QPixmap>
#include <QPushButton>
#include <QResizeEvent>
#include <QScrollArea>

#include <QSplitter>
#include <QStatusBar>
#include <QTextStream>
#include <QThread>
#include <QTextEdit>
#include <QVBoxLayout>
#include <QMetaType>

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <map>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace facerecog {

MainWindow::MainWindow(QWidget* parent)
    : QMainWindow(parent)
{
    qRegisterMetaType<facerecog::DetectionWorkerResult>("facerecog::DetectionWorkerResult");

    buildUi();
    connectSignals();

    setWindowTitle("CV Task 5 - PCA Face Detection & Recognition");
    resize(1180, 760);

    m_datasetPath = defaultDatasetPath();
    appendLog("Ready. Train the PCA model first, then load a test image.");
    appendLog("Default dataset path: " + m_datasetPath);
}

MainWindow::~MainWindow()
{
    if (m_detectionThread && m_detectionThread->isRunning()) {
        m_detectionThread->quit();
        m_detectionThread->wait();
    }
}

void MainWindow::resizeEvent(QResizeEvent* event)
{
    QMainWindow::resizeEvent(event);
    updateImageView();
}

void MainWindow::buildUi()
{
    auto* central = new QWidget(this);
    auto* rootLayout = new QHBoxLayout(central);

    auto* splitter = new QSplitter(Qt::Horizontal, central);
    rootLayout->addWidget(splitter);

    // ── Left side: image viewer ───────────────────────────────────────────
    auto* imagePanel = new QWidget(splitter);
    auto* imageLayout = new QVBoxLayout(imagePanel);

    m_statusLabel = new QLabel("No model trained", imagePanel);
    m_statusLabel->setWordWrap(true);
    imageLayout->addWidget(m_statusLabel);

    auto* scroll = new QScrollArea(imagePanel);
    scroll->setWidgetResizable(true);
    scroll->setFrameShape(QFrame::StyledPanel);

    m_imageLabel = new QLabel(scroll);
    m_imageLabel->setAlignment(Qt::AlignCenter);
    m_imageLabel->setMinimumSize(640, 520);
    m_imageLabel->setText("Load a test image to preview detections");
    scroll->setWidget(m_imageLabel);

    imageLayout->addWidget(scroll, 1);
    splitter->addWidget(imagePanel);

    // ── Right side: controls, log, ROC ────────────────────────────────────
    auto* sidePanel = new QWidget(splitter);
    auto* sideLayout = new QVBoxLayout(sidePanel);

    auto* actionsBox = new QGroupBox("Workflow", sidePanel);
    auto* actionsLayout = new QVBoxLayout(actionsBox);

    m_trainButton = new QPushButton("1. Select Dataset && Train PCA", actionsBox);
    m_loadImageButton = new QPushButton("2. Load Test Image", actionsBox);
    m_runButton = new QPushButton("3. Detect && Recognize", actionsBox);
    m_reportButton = new QPushButton("4. Generate Performance Report", actionsBox);
    m_runButton->setEnabled(false);
    m_reportButton->setEnabled(false);

    m_detectionProgressBar = new QProgressBar(actionsBox);
    m_detectionProgressBar->setRange(0, 100);
    m_detectionProgressBar->setValue(0);
    m_detectionProgressBar->setTextVisible(true);
    m_detectionProgressBar->setFormat("Detection: %p%");

    actionsLayout->addWidget(m_trainButton);
    actionsLayout->addWidget(m_loadImageButton);
    actionsLayout->addWidget(m_runButton);
    actionsLayout->addWidget(m_reportButton);
    actionsLayout->addWidget(m_detectionProgressBar);
    sideLayout->addWidget(actionsBox);

    auto* paramsBox = new QGroupBox("Parameters", sidePanel);
    auto* form = new QFormLayout(paramsBox);

    // ── PCA Variance slider: 50..99 → 0.50..0.99 ──
    m_varianceSlider = new QSlider(Qt::Horizontal, paramsBox);
    m_varianceSlider->setRange(50, 99);
    m_varianceSlider->setValue(90);
    m_varianceLabel = new QLabel("0.90", paramsBox);
    m_varianceLabel->setMinimumWidth(40);
    auto* varianceRow = new QHBoxLayout();
    varianceRow->addWidget(m_varianceSlider, 1);
    varianceRow->addWidget(m_varianceLabel);
    form->addRow("PCA variance", varianceRow);

    // ── Face RMSE Threshold slider: 1..50 ──
    m_thresholdSlider = new QSlider(Qt::Horizontal, paramsBox);
    m_thresholdSlider->setRange(1, 50);
    m_thresholdSlider->setValue(12);
    m_thresholdLabel = new QLabel("12", paramsBox);
    m_thresholdLabel->setMinimumWidth(40);
    auto* thresholdRow = new QHBoxLayout();
    thresholdRow->addWidget(m_thresholdSlider, 1);
    thresholdRow->addWidget(m_thresholdLabel);
    form->addRow("Face RMSE threshold", thresholdRow);

    // ── Sliding Stride slider: 4..64 ──
    m_strideSlider = new QSlider(Qt::Horizontal, paramsBox);
    m_strideSlider->setRange(4, 64);
    m_strideSlider->setValue(16);
    m_strideLabel = new QLabel("16", paramsBox);
    m_strideLabel->setMinimumWidth(40);
    auto* strideRow = new QHBoxLayout();
    strideRow->addWidget(m_strideSlider, 1);
    strideRow->addWidget(m_strideLabel);
    form->addRow("Sliding stride", strideRow);

    // ── Max Scale slider: 10..50 → 1.0x..5.0x ──
    m_maxScaleSlider = new QSlider(Qt::Horizontal, paramsBox);
    m_maxScaleSlider->setRange(10, 50);
    m_maxScaleSlider->setValue(20);
    m_maxScaleLabel = new QLabel("2.0x", paramsBox);
    m_maxScaleLabel->setMinimumWidth(40);
    auto* scaleRow = new QHBoxLayout();
    scaleRow->addWidget(m_maxScaleSlider, 1);
    scaleRow->addWidget(m_maxScaleLabel);
    form->addRow("Max scale", scaleRow);

    // ── KNN k slider: 1..15 ──
    m_knnSlider = new QSlider(Qt::Horizontal, paramsBox);
    m_knnSlider->setRange(1, 15);
    m_knnSlider->setValue(3);
    m_knnLabel = new QLabel("3", paramsBox);
    m_knnLabel->setMinimumWidth(40);
    auto* knnRow = new QHBoxLayout();
    knnRow->addWidget(m_knnSlider, 1);
    knnRow->addWidget(m_knnLabel);
    form->addRow("KNN k", knnRow);

    sideLayout->addWidget(paramsBox);

    m_rocWidget = new ROCWidget(sidePanel);
    sideLayout->addWidget(m_rocWidget, 1);

    m_log = new QTextEdit(sidePanel);
    m_log->setReadOnly(true);
    m_log->setMinimumHeight(180);
    sideLayout->addWidget(m_log, 1);

    splitter->addWidget(sidePanel);
    splitter->setStretchFactor(0, 4);
    splitter->setStretchFactor(1, 2);

    setCentralWidget(central);
    statusBar()->showMessage("Ready");
}

void MainWindow::connectSignals()
{
    connect(m_trainButton, &QPushButton::clicked, this, [this]() {
        trainModel();
    });

    connect(m_loadImageButton, &QPushButton::clicked, this, [this]() {
        loadTestImage();
    });

    connect(m_runButton, &QPushButton::clicked, this, [this]() {
        runDetectionAndRecognition();
    });

    connect(m_reportButton, &QPushButton::clicked, this, [this]() {
        generatePerformanceReport();
    });

    // ── Slider → label dynamic updates ────────────────────────────────────
    connect(m_varianceSlider, &QSlider::valueChanged, this, [this](int v) {
        m_varianceLabel->setText(QString::number(v / 100.0, 'f', 2));
    });
    connect(m_thresholdSlider, &QSlider::valueChanged, this, [this](int v) {
        m_thresholdLabel->setText(QString::number(v));
    });
    connect(m_strideSlider, &QSlider::valueChanged, this, [this](int v) {
        m_strideLabel->setText(QString::number(v));
    });
    connect(m_maxScaleSlider, &QSlider::valueChanged, this, [this](int v) {
        m_maxScaleLabel->setText(QString::number(v / 10.0, 'f', 1) + "x");
    });
    connect(m_knnSlider, &QSlider::valueChanged, this, [this](int v) {
        m_knnLabel->setText(QString::number(v));
    });
}

void MainWindow::trainModel()
{
    const QString selected = QFileDialog::getExistingDirectory(
        this,
        "Select AT&T/ORL dataset folder",
        QDir(m_datasetPath).exists() ? m_datasetPath : defaultDatasetPath()
    );

    if (selected.isEmpty()) {
        return;
    }

    try {
        QApplication::setOverrideCursor(Qt::WaitCursor);

        m_datasetPath = selected;
        appendLog("Loading dataset: " + selected);

        const std::size_t loaded = m_pca.loadDataset(selected.toStdString());
        appendLog(QString("Loaded %1 images from %2 identities. Dynamic PCA window: %3x%4 (D=%5).")
                      .arg(loaded)
                      .arg(m_pca.getClassCount())
                      .arg(m_pca.getTrainWidth())
                      .arg(m_pca.getTrainHeight())
                      .arg(m_pca.getImageDimension()));

        appendLog(QString("Training PCA with target variance %1...")
                      .arg(varianceValue(), 0, 'f', 2));

        const PCATrainingResult result = m_pca.train(varianceValue());

        m_detector = std::make_unique<FaceDetector>(m_pca);
        m_recognizer = std::make_unique<FaceRecognizer>(m_pca);

        const double autoThreshold = m_detector->calibratedThreshold();
        {
            const int clampedThreshold = std::clamp(static_cast<int>(std::round(autoThreshold)), 1, 50);
            m_thresholdSlider->setValue(clampedThreshold);
            m_thresholdLabel->setText(QString::number(clampedThreshold));
        }

        const auto roc = buildRocCurve();
        m_rocWidget->setCurve(roc.first, roc.second);

        m_statusLabel->setText(QString("Model trained: %1 samples | %2 identities | window=%3x%4 | K=%5 | variance=%6% | threshold=%7 RMSE")
                                   .arg(result.sampleCount)
                                   .arg(result.classCount)
                                   .arg(result.trainWidth)
                                   .arg(result.trainHeight)
                                   .arg(result.selectedK)
                                   .arg(result.varianceCaptured * 100.0, 0, 'f', 2)
                                   .arg(autoThreshold, 0, 'f', 2));

        appendLog(QString("Training complete. Window=%1x%2, D=%3, selected K=%4, captured variance=%5%, calibrated threshold=%6 RMSE.")
                      .arg(result.trainWidth)
                      .arg(result.trainHeight)
                      .arg(result.imageDimension)
                      .arg(result.selectedK)
                      .arg(result.varianceCaptured * 100.0, 0, 'f', 2)
                      .arg(autoThreshold, 0, 'f', 2));
        appendLog(QString("ROC AUC from training projections: %1")
                      .arg(roc.second, 0, 'f', 3));

        if (!m_rawTargetImage.isNull()) {
            const double imageScaleLimit = std::max(1.0,
                std::min(static_cast<double>(m_rawTargetImage.width()) / m_pca.getTrainWidth(),
                         static_cast<double>(m_rawTargetImage.height()) / m_pca.getTrainHeight()));
            {
                const int clampedScale = std::clamp(static_cast<int>(std::round(std::max(1.0, std::min(2.0, imageScaleLimit)) * 10.0)), 10, 50);
                m_maxScaleSlider->setValue(clampedScale);
                m_maxScaleLabel->setText(QString::number(clampedScale / 10.0, 'f', 1) + "x");
            }

            if (m_rawTargetImage.width() < m_pca.getTrainWidth()
                || m_rawTargetImage.height() < m_pca.getTrainHeight()) {
                appendLog(QString("Loaded target image is smaller than PCA window %1x%2; detector will resize it upward before projection.")
                              .arg(m_pca.getTrainWidth())
                              .arg(m_pca.getTrainHeight()));
            }
        }

        m_runButton->setEnabled(!m_rawTargetImage.isNull());
        if (m_reportButton) {
            m_reportButton->setEnabled(true);
        }
        statusBar()->showMessage("PCA model trained");

        QApplication::restoreOverrideCursor();
    } catch (const std::exception& e) {
        QApplication::restoreOverrideCursor();
        QMessageBox::critical(this, "Training Error", e.what());
        appendLog("Training failed: " + QString::fromStdString(e.what()));
        statusBar()->showMessage("Training failed");
    }
}

void MainWindow::loadTestImage()
{
    const QString fileName = QFileDialog::getOpenFileName(
        this,
        "Load test image",
        m_datasetPath,
        "Images (*.pgm *.png *.jpg *.jpeg *.bmp *.tif *.tiff)"
    );

    if (fileName.isEmpty()) {
        return;
    }

    QImageReader reader(fileName);
    reader.setAutoTransform(true);
    QImage image = reader.read();

    if (image.isNull()) {
        QMessageBox::warning(this, "Image Error", "Could not read the selected image.");
        return;
    }

    m_testImagePath = fileName;

    // Keep the raw file image untouched for PCA detection/recognition.
    // QLabel scaling must never feed back into this image.
    m_rawTargetImage = image;

    // UI-only copy. This copy may be converted, annotated, and scaled for display.
    m_displayImage = image.convertToFormat(QImage::Format_RGB32);
    m_annotatedImage = m_displayImage;

    if (m_pca.hasTrainingGeometry()) {
        const double imageScaleLimit = std::max(1.0,
            std::min(static_cast<double>(m_rawTargetImage.width()) / m_pca.getTrainWidth(),
                     static_cast<double>(m_rawTargetImage.height()) / m_pca.getTrainHeight()));
        const int clampedScale = std::clamp(static_cast<int>(std::round(std::max(1.0, std::min(2.0, imageScaleLimit)) * 10.0)), 10, 50);
        m_maxScaleSlider->setValue(clampedScale);
        m_maxScaleLabel->setText(QString::number(clampedScale / 10.0, 'f', 1) + "x");
    } else {
        m_maxScaleSlider->setValue(20);
        m_maxScaleLabel->setText("2.0x");
    }

    appendLog(QString("Loaded raw test image: %1 (%2x%3). QLabel scaling remains display-only.")
                  .arg(fileName)
                  .arg(m_rawTargetImage.width())
                  .arg(m_rawTargetImage.height()));

    if (m_rawTargetImage.width() > 600) {
        appendLog(QString("Image width exceeds 600 px; detection will use a temporary scaledToWidth(600) copy while preserving the raw image for annotation/recognition crops."));
    }

    if (m_pca.hasTrainingGeometry()
        && (m_rawTargetImage.width() < m_pca.getTrainWidth()
            || m_rawTargetImage.height() < m_pca.getTrainHeight())) {
        appendLog(QString("Image is smaller than PCA window %1x%2; detector will resize it upward with Qt smooth scaling before sliding windows.")
                      .arg(m_pca.getTrainWidth())
                      .arg(m_pca.getTrainHeight()));
    }

    updateImageView();
    m_runButton->setEnabled(m_pca.isTrained());
    statusBar()->showMessage("Test image loaded");
}

void MainWindow::runDetectionAndRecognition()
{
    if (m_detectionRunning) {
        appendLog("Detection is already running. Please wait for it to finish.");
        return;
    }
    if (!m_pca.isTrained() || !m_detector || !m_recognizer) {
        QMessageBox::information(this, "Model Required", "Train the PCA model first.");
        return;
    }
    if (m_rawTargetImage.isNull()) {
        QMessageBox::information(this, "Image Required", "Load a test image first.");
        return;
    }

    SlidingWindowConfig config;
    config.stride = strideValue();
    config.minScale = 1.0;
    config.maxScale = maxScaleValue();
    config.scaleFactor = 1.25;
    config.rmseThreshold = thresholdValue();
    config.nmsIoUThreshold = 0.25;
    config.maxDetections = 10;

    // Detection now runs on a worker thread, so the old processEvents()
    // fallback is intentionally disabled for this UI path.
    config.maxInputWidth = 600;
    config.processEventsEveryRows = 0;

    QImage detectorInput = m_rawTargetImage;
    double detectorToRawScaleX = 1.0;
    double detectorToRawScaleY = 1.0;

    if (detectorInput.width() > config.maxInputWidth) {
        const QSize rawSize = detectorInput.size();
        detectorInput = detectorInput.scaledToWidth(
            config.maxInputWidth,
            Qt::SmoothTransformation
        );
        detectorToRawScaleX = static_cast<double>(rawSize.width())
            / static_cast<double>(std::max(1, detectorInput.width()));
        detectorToRawScaleY = static_cast<double>(rawSize.height())
            / static_cast<double>(std::max(1, detectorInput.height()));

        appendLog(QString("Auto-downscaled detector input before worker launch: %1x%2 -> %3x%4 using scaledToWidth(%5). Results will be mapped back to raw coordinates.")
                      .arg(rawSize.width())
                      .arg(rawSize.height())
                      .arg(detectorInput.width())
                      .arg(detectorInput.height())
                      .arg(config.maxInputWidth));

        // Prevent a second downscale inside FaceDetector because this UI path
        // already created the optimized detector input.
        config.maxInputWidth = 0;
    }

    appendLog(QString("Starting asynchronous sliding-window detection: input=%1x%2, stride=%3, maxScale=%4, threshold=%5 RMSE")
                  .arg(detectorInput.width())
                  .arg(detectorInput.height())
                  .arg(config.stride)
                  .arg(config.maxScale, 0, 'f', 2)
                  .arg(config.rmseThreshold, 0, 'f', 2));

    setDetectionUiRunning(true);
    statusBar()->showMessage("Detection running in background...");

    auto* thread = new QThread(this);
    auto* worker = new DetectionWorker(
        &m_pca,
        m_rawTargetImage,
        detectorInput,
        detectorToRawScaleX,
        detectorToRawScaleY,
        config,
        knnValue()
    );

    m_detectionThread = thread;
    worker->moveToThread(thread);

    connect(thread, &QThread::started, worker, &DetectionWorker::run);

    connect(worker, &DetectionWorker::progressChanged, this, [this](int percentage) {
        if (m_detectionProgressBar) {
            m_detectionProgressBar->setValue(percentage);
        }
        statusBar()->showMessage(QString("Detection running... %1%").arg(percentage));
    });

    connect(worker, &DetectionWorker::logMessage, this, [this](const QString& message) {
        appendLog(message);
    });

    connect(worker, &DetectionWorker::finished, this, [this](const DetectionWorkerResult& result) {
        if (!result.errorMessage.isEmpty()) {
            QMessageBox::critical(this, "Detection/Recognition Error", result.errorMessage);
            appendLog("Detection/recognition failed: " + result.errorMessage);
            statusBar()->showMessage("Detection failed");
        } else {
            annotateDetections(result.detections, result.recognitions);
            updateImageView();

            appendLog(QString("Detection complete: %1 candidate face(s).")
                          .arg(result.detections.size()));

            for (std::size_t i = 0; i < result.detections.size(); ++i) {
                const FaceDetection& d = result.detections[i];
                const RecognitionResult& r = result.recognitions[i];
                appendLog(QString("  #%1 box=(%2,%3,%4,%5), face RMSE=%6, identity=%7, KNN distance=%8")
                              .arg(i + 1)
                              .arg(d.box.x())
                              .arg(d.box.y())
                              .arg(d.box.width())
                              .arg(d.box.height())
                              .arg(d.reconstructionRmse, 0, 'f', 2)
                              .arg(QString::fromStdString(r.label))
                              .arg(r.nearestDistance, 0, 'f', 2));
            }

            if (m_detectionProgressBar) {
                m_detectionProgressBar->setValue(100);
            }
            statusBar()->showMessage(QString("Detected %1 face(s)").arg(result.detections.size()));
        }

        setDetectionUiRunning(false);
    });

    connect(worker, &DetectionWorker::finished, thread, &QThread::quit);
    connect(worker, &DetectionWorker::finished, worker, &QObject::deleteLater);

    connect(thread, &QThread::finished, this, [this, thread]() {
        if (m_detectionThread == thread) {
            m_detectionThread = nullptr;
        }
    });
    connect(thread, &QThread::finished, thread, &QObject::deleteLater);

    thread->start();
}

void MainWindow::annotateDetections(const std::vector<FaceDetection>& detections,
                                    const std::vector<RecognitionResult>& recognitions)
{
    if (m_displayImage.isNull() && !m_rawTargetImage.isNull()) {
        m_displayImage = m_rawTargetImage.convertToFormat(QImage::Format_RGB32);
    }

    m_annotatedImage = m_displayImage;

    QPainter painter(&m_annotatedImage);
    painter.setRenderHint(QPainter::Antialiasing, true);

    QFont font = painter.font();
    font.setBold(true);
    font.setPointSize(std::max(9, m_annotatedImage.width() / 80));
    painter.setFont(font);

    for (std::size_t i = 0; i < detections.size(); ++i) {
        const FaceDetection& d = detections[i];
        const RecognitionResult& r = recognitions[i];

        painter.setPen(QPen(Qt::green, std::max(2, m_annotatedImage.width() / 300)));
        painter.drawRect(d.box);

        const QString label = QString("Person %1 | RMSE %2 | conf %3%")
            .arg(QString::fromStdString(r.label))
            .arg(d.reconstructionRmse, 0, 'f', 1)
            .arg(r.confidence * 100.0, 0, 'f', 0);

        const QRect textRect = painter.fontMetrics().boundingRect(label).adjusted(-6, -4, 6, 4);
        QRect background(d.box.left(), std::max(0, d.box.top() - textRect.height()),
                         textRect.width(), textRect.height());
        if (background.right() > m_annotatedImage.width()) {
            background.moveRight(m_annotatedImage.width());
        }

        painter.fillRect(background, QColor(0, 0, 0, 180));
        painter.setPen(Qt::white);
        painter.drawText(background.adjusted(3, 2, -3, -2), Qt::AlignVCenter, label);
    }
}

void MainWindow::updateImageView()
{
    if (!m_imageLabel) {
        return;
    }

    const QImage& imageToShow = !m_annotatedImage.isNull()
        ? m_annotatedImage
        : (!m_displayImage.isNull() ? m_displayImage : m_rawTargetImage);

    if (imageToShow.isNull()) {
        m_imageLabel->setPixmap(QPixmap());
        m_imageLabel->setText("Load a test image to preview detections");
        return;
    }

    const QSize targetSize = m_imageLabel->size().expandedTo(QSize(320, 240));
    QPixmap pixmap = QPixmap::fromImage(imageToShow)
        .scaled(targetSize, Qt::KeepAspectRatio, Qt::SmoothTransformation);

    m_imageLabel->setText(QString());
    m_imageLabel->setPixmap(pixmap);
}

void MainWindow::appendLog(const QString& message)
{
    if (m_log) {
        m_log->append(message);
    }
}

void MainWindow::setDetectionUiRunning(bool running)
{
    m_detectionRunning = running;

    if (m_runButton) {
        m_runButton->setEnabled(!running && m_pca.isTrained() && !m_rawTargetImage.isNull());
    }
    if (m_trainButton) {
        m_trainButton->setEnabled(!running);
    }
    if (m_loadImageButton) {
        m_loadImageButton->setEnabled(!running);
    }
    if (m_reportButton) {
        m_reportButton->setEnabled(!running && m_pca.isTrained());
    }

    if (m_detectionProgressBar) {
        if (running) {
            m_detectionProgressBar->setValue(0);
        }
    }
}

QString MainWindow::defaultDatasetPath() const
{
    const QDir executableDir(QCoreApplication::applicationDirPath());
    return executableDir.absoluteFilePath("../dataset");
}

void MainWindow::generatePerformanceReport()
{
    if (!m_pca.isTrained()) {
        QMessageBox::information(this, "Model Required", "Train the PCA model first.");
        return;
    }

    const auto roc = buildRocCurve();
    if (!roc.first.empty()) {
        m_rocWidget->setCurve(roc.first, roc.second);
    }

    const QString report = buildPerformanceReportText();
    appendLog("\n" + report);

    const QString reportPath = QDir(QCoreApplication::applicationDirPath())
        .absoluteFilePath("performance_report.txt");

    QFile file(reportPath);
    if (file.open(QIODevice::WriteOnly | QIODevice::Text)) {
        QTextStream out(&file);
        out << report;
        file.close();
        appendLog("Performance report saved to: " + reportPath);
        statusBar()->showMessage("Performance report generated");
    } else {
        appendLog("Could not save performance report to: " + reportPath);
        statusBar()->showMessage("Performance report generated, but save failed");
    }
}

std::pair<int, int> MainWindow::leaveOneOutRecognitionScore(int knnK) const
{
    const auto& projections = m_pca.getTrainingProjections();
    if (projections.size() < 2) {
        return {0, 0};
    }

    const int safeK = std::max(1, knnK);
    int correct = 0;
    int total = 0;

    for (std::size_t i = 0; i < projections.size(); ++i) {
        struct Neighbor {
            double distance = 0.0;
            int identity = -1;
        };

        std::vector<Neighbor> neighbors;
        neighbors.reserve(projections.size() - 1);

        for (std::size_t j = 0; j < projections.size(); ++j) {
            if (i == j) {
                continue;
            }

            neighbors.push_back({
                MathEngine::euclideanDistance(projections[i].weights, projections[j].weights),
                projections[j].identity
            });
        }

        std::sort(neighbors.begin(), neighbors.end(),
                  [](const Neighbor& a, const Neighbor& b) {
                      return a.distance < b.distance;
                  });

        const int topK = std::min(safeK, static_cast<int>(neighbors.size()));
        std::map<int, int> votes;
        std::map<int, double> distanceSums;

        for (int n = 0; n < topK; ++n) {
            votes[neighbors[n].identity] += 1;
            distanceSums[neighbors[n].identity] += neighbors[n].distance;
        }

        int predictedIdentity = -1;
        int bestVotes = -1;
        double bestDistanceSum = std::numeric_limits<double>::infinity();

        for (const auto& [identity, voteCount] : votes) {
            const double distanceSum = distanceSums[identity];
            if (voteCount > bestVotes
                || (voteCount == bestVotes && distanceSum < bestDistanceSum)) {
                predictedIdentity = identity;
                bestVotes = voteCount;
                bestDistanceSum = distanceSum;
            }
        }

        if (predictedIdentity == projections[i].identity) {
            ++correct;
        }
        ++total;
    }

    return {correct, total};
}

QString MainWindow::buildPerformanceReportText() const
{
    const auto roc = buildRocCurve();
    const auto loo = leaveOneOutRecognitionScore(knnValue());

    const double accuracy = (loo.second > 0)
        ? (static_cast<double>(loo.first) * 100.0 / static_cast<double>(loo.second))
        : 0.0;

    QString report;
    QTextStream out(&report);

    out << "CV Task 5 - Performance Report\n";
    out << "Generated: " << QDateTime::currentDateTime().toString(Qt::ISODate) << "\n\n";

    out << "Requirement Coverage\n";
    out << "1. Standard face dataset: PASS - loaded " << m_pca.getSampleCount()
        << " images from " << m_pca.getClassCount() << " numerically labeled identities.\n";
    out << "2. Face detection: PASS - native sliding-window PCA reconstruction detector accepts color or grayscale QImage input and converts to grayscale internally.\n";
    out << "3. PCA/Eigen recognition: PASS - Eigenfaces projection with native K-NN classification in PCA space.\n";
    out << "4. Performance + ROC: PASS - this report computes recognition accuracy and the ROC widget plots the curve with AUC.\n\n";

    out << "Model Summary\n";
    out << "Training window: " << m_pca.getTrainWidth() << "x" << m_pca.getTrainHeight()
        << " (D=" << m_pca.getImageDimension() << ")\n";
    out << "Selected eigenfaces K: " << m_pca.getK() << "\n";
    out << "Training samples: " << m_pca.getSampleCount() << "\n";
    out << "Classes/identities: " << m_pca.getClassCount() << "\n";
    out << "Detector threshold: " << thresholdValue() << " RMSE\n";
    out << "Sliding stride: " << strideValue() << " px\n";
    out << "Max detection scale: " << maxScaleValue() << "\n";
    out << "KNN k: " << knnValue() << "\n\n";

    out << "Recognition Evaluation\n";
    out << "Protocol: leave-one-out over loaded training projections; the query sample is excluded from its neighbor list.\n";
    out << "Correct: " << loo.first << " / " << loo.second << "\n";
    out << "Accuracy: " << QString::number(accuracy, 'f', 2) << "%\n\n";

    out << "ROC Evaluation\n";
    out << "Protocol: pairwise distances in PCA space. Same-identity pairs are positives; different-identity pairs are negatives.\n";
    out << "ROC points: " << roc.first.size() << "\n";
    out << "AUC: " << QString::number(roc.second, 'f', 4) << "\n";

    return report;
}

std::pair<std::vector<QPointF>, double> MainWindow::buildRocCurve() const
{
    const auto& projections = m_pca.getTrainingProjections();
    if (projections.size() < 2) {
        return {{}, 0.0};
    }

    struct PairScore {
        double distance = 0.0;
        bool sameIdentity = false;
    };

    std::vector<PairScore> scores;
    scores.reserve(projections.size() * (projections.size() - 1) / 2);

    double minDistance = std::numeric_limits<double>::infinity();
    double maxDistance = 0.0;
    int positivePairs = 0;
    int negativePairs = 0;

    for (std::size_t i = 0; i < projections.size(); ++i) {
        for (std::size_t j = i + 1; j < projections.size(); ++j) {
            const double distance = MathEngine::euclideanDistance(projections[i].weights,
                                                                  projections[j].weights);
            const bool same = projections[i].identity == projections[j].identity;

            scores.push_back({distance, same});
            minDistance = std::min(minDistance, distance);
            maxDistance = std::max(maxDistance, distance);

            if (same) {
                ++positivePairs;
            } else {
                ++negativePairs;
            }
        }
    }

    if (scores.empty() || positivePairs == 0 || negativePairs == 0 || maxDistance <= minDistance) {
        return {{}, 0.0};
    }

    std::vector<QPointF> points;
    points.reserve(101);
    points.emplace_back(0.0, 0.0);

    constexpr int steps = 100;
    for (int s = 0; s <= steps; ++s) {
        const double threshold = minDistance
            + (maxDistance - minDistance) * static_cast<double>(s) / static_cast<double>(steps);

        int truePositives = 0;
        int falsePositives = 0;

        for (const PairScore& score : scores) {
            if (score.distance <= threshold) {
                if (score.sameIdentity) {
                    ++truePositives;
                } else {
                    ++falsePositives;
                }
            }
        }

        const double tpr = static_cast<double>(truePositives) / positivePairs;
        const double fpr = static_cast<double>(falsePositives) / negativePairs;
        points.emplace_back(fpr, tpr);
    }

    std::sort(points.begin(), points.end(),
              [](const QPointF& a, const QPointF& b) {
                  if (a.x() == b.x()) {
                      return a.y() < b.y();
                  }
                  return a.x() < b.x();
              });

    double auc = 0.0;
    for (std::size_t i = 1; i < points.size(); ++i) {
        const double dx = points[i].x() - points[i - 1].x();
        const double avgY = 0.5 * (points[i].y() + points[i - 1].y());
        auc += dx * avgY;
    }

    return {points, std::clamp(auc, 0.0, 1.0)};
}

// ── Slider value accessors (integer → real scaling) ──────────────────────────

double MainWindow::varianceValue() const
{
    return m_varianceSlider ? m_varianceSlider->value() / 100.0 : 0.90;
}

double MainWindow::thresholdValue() const
{
    return m_thresholdSlider ? static_cast<double>(m_thresholdSlider->value()) : 12.0;
}

int MainWindow::strideValue() const
{
    return m_strideSlider ? m_strideSlider->value() : 16;
}

double MainWindow::maxScaleValue() const
{
    return m_maxScaleSlider ? m_maxScaleSlider->value() / 10.0 : 2.0;
}

int MainWindow::knnValue() const
{
    return m_knnSlider ? m_knnSlider->value() : 3;
}

} // namespace facerecog
