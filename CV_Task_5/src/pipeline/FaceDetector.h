#ifndef FACEDETECTOR_H
#define FACEDETECTOR_H

/**
 * @file FaceDetector.h
 * @brief Step 3: Sliding-window face detector using PCA reconstruction error.
 *
 * The detector uses the dynamic PCA training window stored inside PCAPipeline.
 * Large images can be downscaled before scanning, and smaller images are
 * resized upward using Qt smooth scaling rather than padded with black pixels.
 */

#include "PCAPipeline.h"

#include <QImage>
#include <QRect>
#include <QSize>

#include <cstddef>
#include <functional>
#include <vector>

namespace facerecog {

struct FaceDetection {
    QRect box;              ///< Bounding box in the original detector input image
    double reconstructionRmse = 0.0;
    double confidence = 0.0;
};

struct SlidingWindowConfig {
    int stride = 8;                   ///< Pixel step between candidate windows
    int maxInputWidth = 600;          ///< Downscale detector input if width exceeds this value; <= 0 disables
    int processEventsEveryRows = 0;   ///< Legacy synchronous mode helper; <= 0 disables
    double minScale = 1.0;
    double maxScale = 2.0;
    double scaleFactor = 1.25;
    double rmseThreshold = -1.0;      ///< If <= 0, detector uses calibrated threshold
    double nmsIoUThreshold = 0.25;
    std::size_t maxDetections = 10;
};

class FaceDetector {
public:
    using ProgressCallback = std::function<void(int percentage)>;

    explicit FaceDetector(const PCAPipeline& pca);

    /**
     * @brief Estimate a useful reconstruction RMSE threshold from training data.
     * Uses mean + 3 * std of training-face reconstruction RMSE values.
     */
    double calibratedThreshold() const;

    /**
     * @brief Run PCA-space sliding-window detection on a target image.
     *
     * @param image Input image. This can be the raw image, or a UI/downscaled
     * image if the caller wants to limit the search space.
     * @param config Sliding-window configuration.
     * @param progressCallback Optional callback receiving 0..100 progress based
     * on total scheduled sliding-window evaluations across all scales.
     */
    std::vector<FaceDetection> detect(const QImage& image,
                                      const SlidingWindowConfig& config = {},
                                      const ProgressCallback& progressCallback = {}) const;

private:
    struct PreparedDetectionImage {
        QImage image;           ///< Grayscale working image used by the detector
        QSize originalSize;     ///< Original detector input size
        double scaleX = 1.0;    ///< working x coordinate = original x * scaleX
        double scaleY = 1.0;    ///< working y coordinate = original y * scaleY
        bool resized = false;
    };

    struct ScaleWorkItem {
        double scale = 1.0;
        int windowW = 0;
        int windowH = 0;
        int rows = 0;
        int cols = 0;
        std::size_t windowCount = 0;
    };

    /**
     * @brief Convert to grayscale, optionally downscale very large inputs, and
     * scale up if smaller than the PCA window.
     */
    PreparedDetectionImage prepareDetectionImage(const QImage& image,
                                                 const SlidingWindowConfig& config) const;

    std::vector<ScaleWorkItem> buildScaleWorkItems(const QImage& gray,
                                                   const SlidingWindowConfig& config) const;

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
