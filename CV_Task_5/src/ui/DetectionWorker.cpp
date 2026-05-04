#include "DetectionWorker.h"

#include <QRect>
#include <QtGlobal>

#include <algorithm>
#include <cmath>
#include <exception>
#include <stdexcept>
#include <utility>

namespace facerecog {

DetectionWorker::DetectionWorker(const PCAPipeline* pca,
                                 QImage rawImage,
                                 QImage detectorInput,
                                 double detectorToRawScaleX,
                                 double detectorToRawScaleY,
                                 SlidingWindowConfig config,
                                 int knnK,
                                 QObject* parent)
    : QObject(parent)
    , m_pca(pca)
    , m_rawImage(std::move(rawImage))
    , m_detectorInput(std::move(detectorInput))
    , m_detectorToRawScaleX(detectorToRawScaleX)
    , m_detectorToRawScaleY(detectorToRawScaleY)
    , m_config(std::move(config))
    , m_knnK(knnK)
{
}

void DetectionWorker::run()
{
    DetectionWorkerResult result;

    try {
        if (m_pca == nullptr) {
            throw std::runtime_error("DetectionWorker::run - PCA model pointer is null.");
        }
        if (!m_pca->isTrained()) {
            throw std::runtime_error("DetectionWorker::run - PCA model is not trained.");
        }
        if (m_rawImage.isNull() || m_detectorInput.isNull()) {
            throw std::runtime_error("DetectionWorker::run - target image is null.");
        }

        FaceDetector detector(*m_pca);
        FaceRecognizer recognizer(*m_pca);

        emit progressChanged(0);

        result.detections = detector.detect(
            m_detectorInput,
            m_config,
            [this](int detectorPercentage) {
                // Detection is the expensive phase. Reserve the last 10% for
                // KNN recognition and annotation preparation.
                emit progressChanged(std::clamp((detectorPercentage * 90) / 100, 0, 90));
            }
        );

        mapDetectionsBackToRaw(result.detections);

        result.recognitions.reserve(result.detections.size());

        if (result.detections.empty()) {
            emit progressChanged(100);
        } else {
            for (std::size_t i = 0; i < result.detections.size(); ++i) {
                const FaceDetection& detection = result.detections[i];
                const QImage crop = m_rawImage.copy(detection.box);
                result.recognitions.push_back(recognizer.recognize(crop, m_knnK));

                const int recognitionProgress = 90 + static_cast<int>(
                    std::floor((static_cast<double>(i + 1) * 10.0)
                               / static_cast<double>(result.detections.size()))
                );
                emit progressChanged(std::clamp(recognitionProgress, 90, 100));
            }
        }
    } catch (const std::exception& e) {
        result.errorMessage = QString::fromStdString(e.what());
    } catch (...) {
        result.errorMessage = "Unknown detection worker error.";
    }

    emit finished(result);
}

void DetectionWorker::mapDetectionsBackToRaw(std::vector<FaceDetection>& detections) const
{
    if (m_detectorToRawScaleX == 1.0 && m_detectorToRawScaleY == 1.0) {
        return;
    }

    const QRect rawBounds(0, 0, m_rawImage.width(), m_rawImage.height());

    for (FaceDetection& detection : detections) {
        const QRect& box = detection.box;
        const int x1 = static_cast<int>(std::floor(box.left() * m_detectorToRawScaleX));
        const int y1 = static_cast<int>(std::floor(box.top() * m_detectorToRawScaleY));
        const int x2 = static_cast<int>(std::ceil((box.right() + 1) * m_detectorToRawScaleX));
        const int y2 = static_cast<int>(std::ceil((box.bottom() + 1) * m_detectorToRawScaleY));

        detection.box = QRect(x1, y1,
                              std::max(1, x2 - x1),
                              std::max(1, y2 - y1))
                            .intersected(rawBounds);
    }
}

} // namespace facerecog
