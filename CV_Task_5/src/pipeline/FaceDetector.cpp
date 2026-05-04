#include "FaceDetector.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>

#include <QSize>

namespace facerecog {

FaceDetector::FaceDetector(const PCAPipeline& pca)
    : m_pca(pca)
{
}

double FaceDetector::calibratedThreshold() const
{
    if (!m_pca.isTrained()) {
        throw std::runtime_error("FaceDetector::calibratedThreshold - PCA model is not trained.");
    }

    const auto& samples = m_pca.getSamples();
    if (samples.empty()) {
        return 45.0;
    }

    std::vector<double> rmses;
    rmses.reserve(samples.size());

    const double dimensionScale = std::sqrt(static_cast<double>(m_pca.getImageDimension()));

    for (const FaceSample& sample : samples) {
        const double error = m_pca.reconstructionError(sample.pixels);
        rmses.push_back(error / dimensionScale);
    }

    const double mean = std::accumulate(rmses.begin(), rmses.end(), 0.0)
        / static_cast<double>(rmses.size());

    double variance = 0.0;
    for (double rmse : rmses) {
        const double diff = rmse - mean;
        variance += diff * diff;
    }
    variance /= static_cast<double>(rmses.size());

    const double stddev = std::sqrt(variance);

    // Keep the threshold permissive enough for unseen faces but not so high that
    // arbitrary flat/non-face patches pass easily.
    return std::max(20.0, mean + 3.0 * stddev);
}

std::vector<FaceDetection> FaceDetector::detect(const QImage& image,
                                                const SlidingWindowConfig& config) const
{
    if (!m_pca.isTrained()) {
        throw std::runtime_error("FaceDetector::detect - PCA model is not trained.");
    }
    if (!m_pca.hasTrainingGeometry()) {
        throw std::runtime_error("FaceDetector::detect - PCA training geometry is not initialized.");
    }
    if (image.isNull()) {
        throw std::invalid_argument("FaceDetector::detect - input image is null.");
    }

    SlidingWindowConfig cfg = config;
    if (cfg.stride <= 0) {
        cfg.stride = 8;
    }
    if (cfg.scaleFactor <= 1.0) {
        cfg.scaleFactor = 1.25;
    }
    if (cfg.minScale <= 0.0) {
        cfg.minScale = 1.0;
    }
    if (cfg.maxScale < cfg.minScale) {
        cfg.maxScale = cfg.minScale;
    }

    const double threshold = (cfg.rmseThreshold > 0.0)
        ? cfg.rmseThreshold
        : calibratedThreshold();

    const PreparedDetectionImage prepared = prepareDetectionImage(image);
    const QImage& gray = prepared.image;
    const int trainW = m_pca.getTrainWidth();
    const int trainH = m_pca.getTrainHeight();
    const double dimensionScale = std::sqrt(static_cast<double>(m_pca.getImageDimension()));

    std::vector<FaceDetection> detections;

    for (double scale = cfg.minScale;
         scale <= cfg.maxScale + 1e-9;
         scale *= cfg.scaleFactor) {

        const int windowW = std::max(trainW,
                                     static_cast<int>(std::round(trainW * scale)));
        const int windowH = std::max(trainH,
                                     static_cast<int>(std::round(trainH * scale)));

        if (windowW > gray.width() || windowH > gray.height()) {
            continue;
        }

        for (int y = 0; y <= gray.height() - windowH; y += cfg.stride) {
            for (int x = 0; x <= gray.width() - windowW; x += cfg.stride) {
                const QRect workingBox(x, y, windowW, windowH);
                const QImage crop = gray.copy(workingBox);
                const Vector vector = m_pca.imageToVector(crop);
                const double rmse = m_pca.reconstructionError(vector) / dimensionScale;

                if (rmse <= threshold) {
                    FaceDetection detection;
                    detection.box = mapBoxToOriginalImage(workingBox, prepared);
                    detection.reconstructionRmse = rmse;
                    detection.confidence = std::max(0.0, 1.0 - (rmse / threshold));

                    if (!detection.box.isEmpty()) {
                        detections.push_back(detection);
                    }
                }
            }
        }
    }

    return nonMaximumSuppression(std::move(detections),
                                 cfg.nmsIoUThreshold,
                                 cfg.maxDetections);
}

FaceDetector::PreparedDetectionImage FaceDetector::prepareDetectionImage(const QImage& image) const
{
    PreparedDetectionImage prepared;
    prepared.originalSize = image.size();
    prepared.image = image.convertToFormat(QImage::Format_Grayscale8);

    const int trainW = m_pca.getTrainWidth();
    const int trainH = m_pca.getTrainHeight();

    if (prepared.image.width() < trainW || prepared.image.height() < trainH) {
        int newWidth = trainW;
        int newHeight = trainH;
        Qt::AspectRatioMode aspectMode = Qt::IgnoreAspectRatio;

        // If only one side is smaller, scale the whole image up without
        // shrinking the larger side. If both sides are smaller, use an exact
        // trainW x trainH smooth resize so at least one full PCA window maps
        // back to the entire original image.
        if (!(prepared.image.width() < trainW && prepared.image.height() < trainH)) {
            const double scaleX = static_cast<double>(trainW) /
                                  static_cast<double>(std::max(1, prepared.image.width()));
            const double scaleY = static_cast<double>(trainH) /
                                  static_cast<double>(std::max(1, prepared.image.height()));
            const double resizeScale = std::max(scaleX, scaleY);

            newWidth = std::max(trainW,
                static_cast<int>(std::ceil(prepared.image.width() * resizeScale)));
            newHeight = std::max(trainH,
                static_cast<int>(std::ceil(prepared.image.height() * resizeScale)));
            aspectMode = Qt::KeepAspectRatio;
        }

        std::cout
            << "[FaceDetector] Target image "
            << prepared.image.width() << "x" << prepared.image.height()
            << " is smaller than PCA window " << trainW << "x" << trainH
            << "; resizing to " << newWidth << "x" << newHeight
            << " using Qt::SmoothTransformation.\n";

        prepared.image = prepared.image.scaled(
            newWidth,
            newHeight,
            aspectMode,
            Qt::SmoothTransformation
        );

        // Defensive guard: Qt keeps aspect ratio and may round dimensions. The
        // detector must still have at least one trainW x trainH window. If a
        // rounding edge case leaves one side short, use a final smooth resize
        // to the exact PCA window instead of falling back to black padding.
        if (prepared.image.width() < trainW || prepared.image.height() < trainH) {
            prepared.image = prepared.image.scaled(
                trainW,
                trainH,
                Qt::IgnoreAspectRatio,
                Qt::SmoothTransformation
            );
        }

        prepared.resized = true;
    }

    prepared.scaleX = static_cast<double>(prepared.image.width()) /
                      static_cast<double>(std::max(1, prepared.originalSize.width()));
    prepared.scaleY = static_cast<double>(prepared.image.height()) /
                      static_cast<double>(std::max(1, prepared.originalSize.height()));

    return prepared;
}

QRect FaceDetector::mapBoxToOriginalImage(const QRect& workingBox,
                                          const PreparedDetectionImage& prepared) const
{
    const int x1 = static_cast<int>(std::floor(workingBox.left() / prepared.scaleX));
    const int y1 = static_cast<int>(std::floor(workingBox.top() / prepared.scaleY));
    const int x2 = static_cast<int>(std::ceil((workingBox.right() + 1) / prepared.scaleX));
    const int y2 = static_cast<int>(std::ceil((workingBox.bottom() + 1) / prepared.scaleY));

    QRect mapped(x1, y1, std::max(1, x2 - x1), std::max(1, y2 - y1));
    const QRect originalBounds(0, 0,
                               prepared.originalSize.width(),
                               prepared.originalSize.height());
    mapped = mapped.intersected(originalBounds);

    // If the whole target was upscaled solely to let the detector run one
    // meaningful window, make sure a full-window detection maps back cleanly
    // to the whole original image rather than losing edge pixels to rounding.
    if (prepared.resized && mapped.isEmpty()) {
        return originalBounds;
    }

    return mapped;
}

double FaceDetector::intersectionOverUnion(const QRect& a, const QRect& b)
{
    const QRect intersection = a.intersected(b);
    if (intersection.isEmpty()) {
        return 0.0;
    }

    const double intersectionArea = static_cast<double>(intersection.width() * intersection.height());
    const double unionArea = static_cast<double>(a.width() * a.height()
                                                + b.width() * b.height())
        - intersectionArea;

    if (unionArea <= 0.0) {
        return 0.0;
    }

    return intersectionArea / unionArea;
}

std::vector<FaceDetection> FaceDetector::nonMaximumSuppression(std::vector<FaceDetection> detections,
                                                               double iouThreshold,
                                                               std::size_t maxDetections)
{
    std::sort(detections.begin(), detections.end(),
              [](const FaceDetection& a, const FaceDetection& b) {
                  return a.reconstructionRmse < b.reconstructionRmse;
              });

    std::vector<FaceDetection> kept;
    kept.reserve(std::min(maxDetections, detections.size()));

    for (const FaceDetection& candidate : detections) {
        bool overlaps = false;

        for (const FaceDetection& selected : kept) {
            if (intersectionOverUnion(candidate.box, selected.box) > iouThreshold) {
                overlaps = true;
                break;
            }
        }

        if (!overlaps) {
            kept.push_back(candidate);
            if (kept.size() >= maxDetections) {
                break;
            }
        }
    }

    return kept;
}

} // namespace facerecog
