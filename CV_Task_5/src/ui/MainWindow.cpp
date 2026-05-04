#include "MainWindow.h"

#include "../core/MathEngine.h"

#include <QApplication>
#include <QColor>
#include <QCoreApplication>
#include <QDir>
#include <QDoubleSpinBox>
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
#include <QPen>
#include <QPixmap>
#include <QPushButton>
#include <QResizeEvent>
#include <QScrollArea>
#include <QSpinBox>
#include <QSplitter>
#include <QStatusBar>
#include <QTextEdit>
#include <QVBoxLayout>

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace facerecog {

MainWindow::MainWindow(QWidget* parent)
    : QMainWindow(parent)
{
    buildUi();
    connectSignals();

    setWindowTitle("CV Task 5 - PCA Face Detection & Recognition");
    resize(1180, 760);

    m_datasetPath = defaultDatasetPath();
    appendLog("Ready. Train the PCA model first, then load a test image.");
    appendLog("Default dataset path: " + m_datasetPath);
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
    m_runButton->setEnabled(false);

    actionsLayout->addWidget(m_trainButton);
    actionsLayout->addWidget(m_loadImageButton);
    actionsLayout->addWidget(m_runButton);
    sideLayout->addWidget(actionsBox);

    auto* paramsBox = new QGroupBox("Parameters", sidePanel);
    auto* form = new QFormLayout(paramsBox);

    m_varianceSpin = new QDoubleSpinBox(paramsBox);
    m_varianceSpin->setRange(0.50, 1.00);
    m_varianceSpin->setSingleStep(0.01);
    m_varianceSpin->setDecimals(2);
    m_varianceSpin->setValue(0.95);

    m_thresholdSpin = new QDoubleSpinBox(paramsBox);
    m_thresholdSpin->setRange(0.0, 255.0);
    m_thresholdSpin->setSingleStep(1.0);
    m_thresholdSpin->setDecimals(2);
    m_thresholdSpin->setValue(0.0);
    m_thresholdSpin->setToolTip("0 means auto-calibrate from training reconstruction errors.");

    m_strideSpin = new QSpinBox(paramsBox);
    m_strideSpin->setRange(1, 64);
    m_strideSpin->setValue(8);

    m_maxScaleSpin = new QDoubleSpinBox(paramsBox);
    m_maxScaleSpin->setRange(1.0, 5.0);
    m_maxScaleSpin->setSingleStep(0.25);
    m_maxScaleSpin->setDecimals(2);
    m_maxScaleSpin->setValue(1.0);

    m_knnSpin = new QSpinBox(paramsBox);
    m_knnSpin->setRange(1, 15);
    m_knnSpin->setValue(3);

    form->addRow("PCA variance", m_varianceSpin);
    form->addRow("Face RMSE threshold", m_thresholdSpin);
    form->addRow("Sliding stride", m_strideSpin);
    form->addRow("Max scale", m_maxScaleSpin);
    form->addRow("KNN k", m_knnSpin);
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
                      .arg(m_varianceSpin->value(), 0, 'f', 2));

        const PCATrainingResult result = m_pca.train(m_varianceSpin->value());

        m_detector = std::make_unique<FaceDetector>(m_pca);
        m_recognizer = std::make_unique<FaceRecognizer>(m_pca);

        const double autoThreshold = m_detector->calibratedThreshold();
        m_thresholdSpin->setValue(autoThreshold);

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
            m_maxScaleSpin->setValue(std::min(5.0, imageScaleLimit));

            if (m_rawTargetImage.width() < m_pca.getTrainWidth()
                || m_rawTargetImage.height() < m_pca.getTrainHeight()) {
                appendLog(QString("Loaded target image is smaller than PCA window %1x%2; detector will resize it upward before projection.")
                              .arg(m_pca.getTrainWidth())
                              .arg(m_pca.getTrainHeight()));
            }
        }

        m_runButton->setEnabled(!m_rawTargetImage.isNull());
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
        m_maxScaleSpin->setValue(std::min(5.0, imageScaleLimit));
    } else {
        m_maxScaleSpin->setValue(1.0);
    }

    appendLog(QString("Loaded raw test image: %1 (%2x%3). Raw image will be passed directly to detector.")
                  .arg(fileName)
                  .arg(m_rawTargetImage.width())
                  .arg(m_rawTargetImage.height()));

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
    if (!m_pca.isTrained() || !m_detector || !m_recognizer) {
        QMessageBox::information(this, "Model Required", "Train the PCA model first.");
        return;
    }
    if (m_rawTargetImage.isNull()) {
        QMessageBox::information(this, "Image Required", "Load a test image first.");
        return;
    }

    try {
        QApplication::setOverrideCursor(Qt::WaitCursor);

        SlidingWindowConfig config;
        config.stride = m_strideSpin->value();
        config.minScale = 1.0;
        config.maxScale = m_maxScaleSpin->value();
        config.scaleFactor = 1.25;
        config.rmseThreshold = m_thresholdSpin->value();
        config.nmsIoUThreshold = 0.25;
        config.maxDetections = 10;

        appendLog(QString("Running sliding-window detector: stride=%1, maxScale=%2, threshold=%3 RMSE")
                      .arg(config.stride)
                      .arg(config.maxScale, 0, 'f', 2)
                      .arg(config.rmseThreshold, 0, 'f', 2));

        std::vector<FaceDetection> detections = m_detector->detect(m_rawTargetImage, config);
        std::vector<RecognitionResult> recognitions;
        recognitions.reserve(detections.size());

        for (const FaceDetection& detection : detections) {
            const QImage crop = m_rawTargetImage.copy(detection.box);
            recognitions.push_back(m_recognizer->recognize(crop, m_knnSpin->value()));
        }

        annotateDetections(detections, recognitions);
        updateImageView();

        appendLog(QString("Detection complete: %1 candidate face(s).")
                      .arg(detections.size()));

        for (std::size_t i = 0; i < detections.size(); ++i) {
            const FaceDetection& d = detections[i];
            const RecognitionResult& r = recognitions[i];
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

        statusBar()->showMessage(QString("Detected %1 face(s)").arg(detections.size()));

        QApplication::restoreOverrideCursor();
    } catch (const std::exception& e) {
        QApplication::restoreOverrideCursor();
        QMessageBox::critical(this, "Detection/Recognition Error", e.what());
        appendLog("Detection/recognition failed: " + QString::fromStdString(e.what()));
        statusBar()->showMessage("Detection failed");
    }
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

QString MainWindow::defaultDatasetPath() const
{
    const QDir executableDir(QCoreApplication::applicationDirPath());
    return executableDir.absoluteFilePath("../dataset");
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

} // namespace facerecog
