#ifndef DETECTIONWORKER_H
#define DETECTIONWORKER_H

/**
 * @file DetectionWorker.h
 * @brief Background worker for asynchronous PCA sliding-window detection and KNN recognition.
 */

#include "../pipeline/FaceDetector.h"
#include "../pipeline/FaceRecognizer.h"
#include "../pipeline/PCAPipeline.h"

#include <QObject>
#include <QImage>
#include <QString>

#include <vector>

namespace facerecog {

struct DetectionWorkerResult {
    std::vector<FaceDetection> detections;
    std::vector<RecognitionResult> recognitions;
    QString errorMessage;
};

class DetectionWorker : public QObject {
    Q_OBJECT

public:
    DetectionWorker(const PCAPipeline* pca,
                    QImage rawImage,
                    QImage detectorInput,
                    double detectorToRawScaleX,
                    double detectorToRawScaleY,
                    SlidingWindowConfig config,
                    int knnK,
                    QObject* parent = nullptr);

public slots:
    void run();

signals:
    void progressChanged(int percentage);
    void logMessage(const QString& message);
    void finished(const facerecog::DetectionWorkerResult& result);

private:
    void mapDetectionsBackToRaw(std::vector<FaceDetection>& detections) const;

private:
    const PCAPipeline* m_pca = nullptr;
    QImage m_rawImage;
    QImage m_detectorInput;
    double m_detectorToRawScaleX = 1.0;
    double m_detectorToRawScaleY = 1.0;
    SlidingWindowConfig m_config;
    int m_knnK = 3;
};

} // namespace facerecog

Q_DECLARE_METATYPE(facerecog::DetectionWorkerResult)

#endif // DETECTIONWORKER_H
