#ifndef PCAPIPELINE_H
#define PCAPIPELINE_H

/**
 * @file PCAPipeline.h
 * @brief Dynamic-size PCA/Eigenfaces training and projection pipeline.
 *
 * The PCA window size is no longer hardcoded. The loader reads the dimensions
 * of the first valid training image and uses that resolution for all flattening,
 * training, projection, reconstruction, detection, and recognition paths.
 */

#include "../core/MathEngine.h"

#include <cstddef>
#include <string>
#include <vector>

class QImage;

namespace facerecog {

struct FaceSample {
    Vector pixels;          ///< Flattened grayscale image, size trainWidth * trainHeight
    int identity = -1;      ///< Numeric folder identity, e.g. 1, 2, 3, ...
    std::string label;      ///< Folder label as text
    std::string filePath;   ///< Original image path
};

struct ProjectedFaceSample {
    Vector weights;         ///< K-dimensional PCA projection
    int identity = -1;
    std::string label;
    std::string filePath;
};

struct PCATrainingResult {
    std::size_t sampleCount = 0;
    std::size_t classCount = 0;
    std::size_t imageDimension = 0;
    std::size_t selectedK = 0;
    int trainWidth = 0;
    int trainHeight = 0;
    double varianceCaptured = 0.0;
};

class PCAPipeline {
public:
    PCAPipeline();

    // Dataset loading
    std::size_t loadDataset(const std::string& datasetRoot = "../dataset");
    void clear();

    // PCA training
    PCATrainingResult train(double targetVariance = 0.95);

    // Projection / reconstruction
    Vector project(const Vector& faceVector) const;
    Vector reconstruct(const Vector& weights) const;
    double reconstructionError(const Vector& faceVector) const;

    // Image conversion helpers using this trained/loaded model geometry
    Vector imageToVector(const QImage& image) const;
    QImage vectorToImage(const Vector& pixels) const;

    // Image conversion helpers for explicit dimensions
    static Vector imageToVector(const QImage& image, int targetWidth, int targetHeight);
    static QImage vectorToImage(const Vector& pixels, int width, int height);

    // Accessors
    bool isTrained() const;
    bool hasTrainingGeometry() const;
    std::size_t getK() const;
    std::size_t getSampleCount() const;
    std::size_t getClassCount() const;
    std::size_t getImageDimension() const;
    int getTrainWidth() const;
    int getTrainHeight() const;

    const std::vector<FaceSample>& getSamples() const;
    const std::vector<ProjectedFaceSample>& getTrainingProjections() const;

    const Vector& getMeanFace() const;
    const Vector& getEigenvalues() const;
    const Matrix& getEigenfaces() const;

private:
    void resetModelOnly();
    void resetTrainingGeometry();
    void setTrainingGeometryFromImage(const QImage& image, const std::string& filePath);

    void validateFaceVector(const Vector& faceVector, const std::string& caller) const;
    static void validateFaceVectorSize(const Vector& faceVector,
                                       std::size_t expectedDimension,
                                       const std::string& caller);
    static bool isNumericFolderName(const std::string& name);
    static int folderNameToIdentity(const std::string& name);

private:
    std::vector<FaceSample> m_samples;

    bool m_trained = false;

    int m_trainWidth = 0;
    int m_trainHeight = 0;
    std::size_t m_imageDimension = 0;

    Vector m_meanFace;          ///< D-dimensional mean face
    Vector m_eigenvalues;       ///< Eigenvalues sorted descending
    Matrix m_eigenfaces;        ///< K x D, each row is one eigenface
    std::size_t m_K = 0;

    std::vector<ProjectedFaceSample> m_trainingProjections;
};

} // namespace facerecog

#endif // PCAPIPELINE_H
