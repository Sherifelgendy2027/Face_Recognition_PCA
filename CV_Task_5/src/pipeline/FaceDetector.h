#ifndef FACEDETECTOR_H
#define FACEDETECTOR_H

/**
 * @file FaceDetector.h
 * @brief Step 3: Sliding-window face detector using PCA reconstruction error.
 *
 * The detector uses the dynamic PCA training window stored inside PCAPipeline.
 * If a target image is smaller than that window, it is resized upward with
 * Qt::SmoothTransformation instead of being black-padded.
 */

#include "PCAPipeline.h"

#include <QRect>
#include <QImage>
#include <QSize>

#include <cstddef>
#include <vector>

namespace facerecog {

struct FaceDetection {
    QRect box;              ///< Bounding box in the original target image
    double reconstructionRmse = 0.0;
    double confidence = 0.0;
};

struct SlidingWindowConfig {
    int stride = 8;
    double minScale = 1.0;
    double maxScale = 1.0;
    double scaleFactor = 1.25;
    double rmseThreshold = -1.0;   ///< If <= 0, detector uses calibrated threshold
    double nmsIoUThreshold = 0.25;
    std::size_t maxDetections = 10;
};

class FaceDetector {
public:
    explicit FaceDetector(const PCAPipeline& pca);

    /**
     * @brief Estimate a useful reconstruction RMSE threshold from training data.
     * Uses mean + 3 * std of training-face reconstruction RMSE values.
     */
    double calibratedThreshold() const;

    /**
     * @brief Run PCA-space sliding-window detection on a target image.
     */
    std::vector<FaceDetection> detect(const QImage& image,
                                      const SlidingWindowConfig& config = {}) const;

private:
    struct PreparedDetectionImage {
        QImage image;           ///< Grayscale working image used by the detector
        QSize originalSize;     ///< Original raw target size
        double scaleX = 1.0;    ///< working x coordinate = original x * scaleX
        double scaleY = 1.0;    ///< working y coordinate = original y * scaleY
        bool resized = false;
    };

    /**
     * @brief Convert to grayscale and scale up if smaller than the PCA window.
     */
    PreparedDetectionImage prepareDetectionImage(const QImage& image) const;

    QRect mapBoxToOriginalImage(const QRect& workingBox,
                                const PreparedDetectionImage& prepared) const;

    static double intersectionOverUnion(const QRect& a, const QRect& b);
    static std::vector<FaceDetection> nonMaximumSuppression(std::vector<FaceDetection> detections,
                                                            double iouThreshold,
                                                            std::size_t maxDetections);

private:
    const PCAPipeline& m_pca;
};

} // namespace facerecog

#endif // FACEDETECTOR_H
