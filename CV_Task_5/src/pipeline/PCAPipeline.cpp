#include "PCAPipeline.h"

#include <QCollator>
#include <QDir>
#include <QDirIterator>
#include <QFileInfo>
#include <QImage>
#include <QImageReader>
#include <QString>
#include <QStringList>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <set>
#include <stdexcept>
#include <utility>

namespace facerecog {

namespace {

QStringList supportedImageFilters()
{
    return {
        "*.pgm",
        "*.png",
        "*.jpg",
        "*.jpeg",
        "*.bmp",
        "*.tif",
        "*.tiff"
    };
}

double computeCapturedVarianceRatio(const Vector& eigenvalues, std::size_t k)
{
    double total = 0.0;
    double captured = 0.0;

    for (double value : eigenvalues) {
        if (value > 0.0) {
            total += value;
        }
    }

    for (std::size_t i = 0; i < k && i < eigenvalues.size(); ++i) {
        if (eigenvalues[i] > 0.0) {
            captured += eigenvalues[i];
        }
    }

    if (total <= 1e-12) {
        return 0.0;
    }

    return captured / total;
}

std::size_t countPositiveEigenvalues(const Vector& eigenvalues)
{
    std::size_t count = 0;

    for (double value : eigenvalues) {
        if (value > 1e-8) {
            ++count;
        }
    }

    return count;
}

} // anonymous namespace

PCAPipeline::PCAPipeline() = default;

void PCAPipeline::clear()
{
    m_samples.clear();
    resetModelOnly();
    resetTrainingGeometry();
}

void PCAPipeline::resetModelOnly()
{
    m_trained = false;
    m_meanFace.clear();
    m_eigenvalues.clear();
    m_eigenfaces.clear();
    m_trainingProjections.clear();
    m_K = 0;
}

void PCAPipeline::resetTrainingGeometry()
{
    m_trainWidth = 0;
    m_trainHeight = 0;
    m_imageDimension = 0;
}

void PCAPipeline::setTrainingGeometryFromImage(const QImage& image, const std::string& filePath)
{
    if (image.isNull()) {
        throw std::invalid_argument("PCAPipeline::setTrainingGeometryFromImage - input image is null.");
    }

    m_trainWidth = image.width();
    m_trainHeight = image.height();
    m_imageDimension = static_cast<std::size_t>(m_trainWidth) *
                       static_cast<std::size_t>(m_trainHeight);

    if (m_trainWidth <= 0 || m_trainHeight <= 0 || m_imageDimension == 0) {
        throw std::runtime_error(
            "PCAPipeline::setTrainingGeometryFromImage - invalid training image size from: "
            + filePath
        );
    }

    std::cout
        << "[PCAPipeline] Dynamic PCA training window detected from first image: "
        << m_trainWidth << "x" << m_trainHeight
        << " (D=" << m_imageDimension << ")"
        << " | source=" << filePath
        << '\n';
}

std::size_t PCAPipeline::loadDataset(const std::string& datasetRoot)
{
    clear();

    QDir rootDir(QString::fromStdString(datasetRoot));

    if (!rootDir.exists()) {
        throw std::runtime_error(
            "PCAPipeline::loadDataset - dataset directory does not exist: " + datasetRoot
        );
    }

    std::cout << "[PCAPipeline] Loading dataset from: " << datasetRoot << '\n';

    std::vector<QString> personDirectories;

    QDirIterator dirIterator(
        rootDir.absolutePath(),
        QDir::Dirs | QDir::NoDotAndDotDot,
        QDirIterator::NoIteratorFlags
    );

    while (dirIterator.hasNext()) {
        const QString personPath = dirIterator.next();
        const QFileInfo info(personPath);
        const std::string folderName = info.fileName().toStdString();

        if (isNumericFolderName(folderName)) {
            personDirectories.push_back(personPath);
        }
    }

    std::sort(
        personDirectories.begin(),
        personDirectories.end(),
        [](const QString& a, const QString& b) {
            const int left = QFileInfo(a).fileName().toInt();
            const int right = QFileInfo(b).fileName().toInt();
            return left < right;
        }
    );

    if (personDirectories.empty()) {
        throw std::runtime_error(
            "PCAPipeline::loadDataset - no numerically named identity folders found in: "
            + datasetRoot
        );
    }

    QCollator fileNameSorter;
    fileNameSorter.setNumericMode(true);
    fileNameSorter.setCaseSensitivity(Qt::CaseInsensitive);

    const QStringList imageFilters = supportedImageFilters();
    std::size_t resizedTrainingImages = 0;

    for (const QString& personPath : personDirectories) {
        const QFileInfo personInfo(personPath);
        const std::string label = personInfo.fileName().toStdString();
        const int identity = folderNameToIdentity(label);

        std::vector<QString> imagePaths;

        QDirIterator imageIterator(
            personPath,
            imageFilters,
            QDir::Files,
            QDirIterator::NoIteratorFlags
        );

        while (imageIterator.hasNext()) {
            imagePaths.push_back(imageIterator.next());
        }

        std::sort(
            imagePaths.begin(),
            imagePaths.end(),
            [&fileNameSorter](const QString& a, const QString& b) {
                return fileNameSorter.compare(
                    QFileInfo(a).fileName(),
                    QFileInfo(b).fileName()
                ) < 0;
            }
        );

        for (const QString& imagePath : imagePaths) {
            QImageReader reader(imagePath);
            reader.setAutoTransform(true);

            QImage image = reader.read();

            if (image.isNull()) {
                std::cerr
                    << "[PCAPipeline] Skipping unreadable image: "
                    << imagePath.toStdString()
                    << '\n';
                continue;
            }

            image = image.convertToFormat(QImage::Format_Grayscale8);

            if (!hasTrainingGeometry()) {
                setTrainingGeometryFromImage(image, imagePath.toStdString());
            } else if (image.width() != m_trainWidth || image.height() != m_trainHeight) {
                ++resizedTrainingImages;
                if (resizedTrainingImages <= 5) {
                    std::cout
                        << "[PCAPipeline] Training image size mismatch; resizing "
                        << imagePath.toStdString()
                        << " from " << image.width() << "x" << image.height()
                        << " to " << m_trainWidth << "x" << m_trainHeight
                        << '\n';
                }
            }

            FaceSample sample;
            sample.pixels = imageToVector(image);
            sample.identity = identity;
            sample.label = label;
            sample.filePath = imagePath.toStdString();

            m_samples.push_back(std::move(sample));
        }
    }

    if (m_samples.empty()) {
        throw std::runtime_error(
            "PCAPipeline::loadDataset - no readable images found in: " + datasetRoot
        );
    }

    if (resizedTrainingImages > 5) {
        std::cout
            << "[PCAPipeline] ... plus " << (resizedTrainingImages - 5)
            << " more training image(s) resized to the dynamic PCA window.\n";
    }

    std::cout
        << "[PCAPipeline] Loaded "
        << m_samples.size()
        << " images from "
        << getClassCount()
        << " identities. PCA window="
        << m_trainWidth << "x" << m_trainHeight
        << " (D=" << m_imageDimension << ").\n";

    return m_samples.size();
}

PCATrainingResult PCAPipeline::train(double targetVariance)
{
    if (m_samples.empty()) {
        throw std::runtime_error("PCAPipeline::train - dataset is empty. Call loadDataset() first.");
    }

    if (!hasTrainingGeometry()) {
        throw std::runtime_error("PCAPipeline::train - training geometry is not initialized.");
    }

    if (targetVariance <= 0.0 || targetVariance > 1.0) {
        throw std::invalid_argument("PCAPipeline::train - targetVariance must be in the range (0, 1].");
    }

    resetModelOnly();

    const std::size_t N = m_samples.size();
    const std::size_t D = m_imageDimension;

    std::cout
        << "[PCAPipeline] Starting PCA training. "
        << "Samples: " << N
        << ", Window: " << m_trainWidth << "x" << m_trainHeight
        << ", Dimension: " << D
        << ", Target variance: " << targetVariance * 100.0
        << "%\n";

    // 1. Build master data matrix X: N x D.
    Matrix X(N, Vector(D, 0.0));

    for (std::size_t i = 0; i < N; ++i) {
        validateFaceVector(m_samples[i].pixels, "PCAPipeline::train");
        X[i] = m_samples[i].pixels;
    }

    // 2. Compute mean face.
    m_meanFace = MathEngine::meanVector(X);

    // 3. Center data: A = X - mean.
    Matrix A = MathEngine::subtractVectorFromRows(X, m_meanFace);

    X.clear();
    X.shrink_to_fit();

    // 4. Turk-Pentland optimization.
    // A is N x D, so we solve the smaller N x N matrix A * A^T.
    Matrix smallCovariance = MathEngine::smallCovarianceMatrix(A);

    // 5. Eigendecompose the small covariance matrix.
    MathEngine::EigenResult eigenResult =
        MathEngine::eigenDecomposition(smallCovariance);

    smallCovariance.clear();
    smallCovariance.shrink_to_fit();

    m_eigenvalues = eigenResult.eigenvalues;

    const std::size_t positiveEigenCount = countPositiveEigenvalues(m_eigenvalues);

    if (positiveEigenCount == 0) {
        throw std::runtime_error("PCAPipeline::train - no positive eigenvalues found.");
    }

    // 6. Select K dynamically to capture target variance.
    m_K = MathEngine::selectK(m_eigenvalues, targetVariance);
    m_K = std::min(m_K, positiveEigenCount);

    if (m_K == 0) {
        m_K = 1;
    }

    const double varianceRatio = computeCapturedVarianceRatio(m_eigenvalues, m_K);

    std::cout
        << "[PCAPipeline] Selected K = "
        << m_K
        << " eigenfaces. Captured variance = "
        << varianceRatio * 100.0
        << "%\n";

    // 7. Recover full eigenfaces.
    // small eigenvector v_i has size N.
    // full eigenface u_i = A^T * v_i, then normalize to unit length.
    m_eigenfaces.assign(m_K, Vector(D, 0.0));

    for (std::size_t k = 0; k < m_K; ++k) {
        const Vector& smallEigenvector = eigenResult.eigenvectors[k];

        Vector eigenface(D, 0.0);

        for (std::size_t d = 0; d < D; ++d) {
            double value = 0.0;

            for (std::size_t n = 0; n < N; ++n) {
                value += A[n][d] * smallEigenvector[n];
            }

            eigenface[d] = value;
        }

        m_eigenfaces[k] = MathEngine::normalize(eigenface);
    }

    A.clear();
    A.shrink_to_fit();

    // 8. Project all training samples into K-dimensional PCA space.
    m_trained = true;

    m_trainingProjections.clear();
    m_trainingProjections.reserve(N);

    for (const FaceSample& sample : m_samples) {
        ProjectedFaceSample projected;
        projected.weights = project(sample.pixels);
        projected.identity = sample.identity;
        projected.label = sample.label;
        projected.filePath = sample.filePath;

        m_trainingProjections.push_back(std::move(projected));
    }

    PCATrainingResult result;
    result.sampleCount = N;
    result.classCount = getClassCount();
    result.imageDimension = D;
    result.selectedK = m_K;
    result.trainWidth = m_trainWidth;
    result.trainHeight = m_trainHeight;
    result.varianceCaptured = varianceRatio;

    std::cout << "[PCAPipeline] PCA training complete.\n";

    return result;
}

Vector PCAPipeline::project(const Vector& faceVector) const
{
    if (!m_trained) {
        throw std::runtime_error("PCAPipeline::project - PCA model has not been trained.");
    }

    validateFaceVector(faceVector, "PCAPipeline::project");

    Vector centered(m_imageDimension, 0.0);

    for (std::size_t i = 0; i < m_imageDimension; ++i) {
        centered[i] = faceVector[i] - m_meanFace[i];
    }

    Vector weights(m_K, 0.0);

    for (std::size_t k = 0; k < m_K; ++k) {
        weights[k] = MathEngine::dot(centered, m_eigenfaces[k]);
    }

    return weights;
}

Vector PCAPipeline::reconstruct(const Vector& weights) const
{
    if (!m_trained) {
        throw std::runtime_error("PCAPipeline::reconstruct - PCA model has not been trained.");
    }

    if (weights.size() != m_K) {
        throw std::invalid_argument(
            "PCAPipeline::reconstruct - expected weight vector of size "
            + std::to_string(m_K)
            + ", got "
            + std::to_string(weights.size())
        );
    }

    Vector reconstructed = m_meanFace;

    for (std::size_t k = 0; k < m_K; ++k) {
        for (std::size_t d = 0; d < m_imageDimension; ++d) {
            reconstructed[d] += weights[k] * m_eigenfaces[k][d];
        }
    }

    return reconstructed;
}

double PCAPipeline::reconstructionError(const Vector& faceVector) const
{
    validateFaceVector(faceVector, "PCAPipeline::reconstructionError");

    const Vector weights = project(faceVector);
    const Vector reconstructed = reconstruct(weights);

    return MathEngine::euclideanDistance(faceVector, reconstructed);
}

Vector PCAPipeline::imageToVector(const QImage& image) const
{
    if (!hasTrainingGeometry()) {
        throw std::runtime_error("PCAPipeline::imageToVector - training geometry is not initialized.");
    }

    return imageToVector(image, m_trainWidth, m_trainHeight);
}

QImage PCAPipeline::vectorToImage(const Vector& pixels) const
{
    if (!hasTrainingGeometry()) {
        throw std::runtime_error("PCAPipeline::vectorToImage - training geometry is not initialized.");
    }

    return vectorToImage(pixels, m_trainWidth, m_trainHeight);
}

Vector PCAPipeline::imageToVector(const QImage& image, int targetWidth, int targetHeight)
{
    if (image.isNull()) {
        throw std::invalid_argument("PCAPipeline::imageToVector - input image is null.");
    }

    if (targetWidth <= 0 || targetHeight <= 0) {
        throw std::invalid_argument("PCAPipeline::imageToVector - target dimensions must be positive.");
    }

    QImage gray = image.convertToFormat(QImage::Format_Grayscale8);

    if (gray.width() != targetWidth || gray.height() != targetHeight) {
        gray = gray.scaled(
            targetWidth,
            targetHeight,
            Qt::IgnoreAspectRatio,
            Qt::SmoothTransformation
        );
    }

    const std::size_t dimension = static_cast<std::size_t>(targetWidth) *
                                  static_cast<std::size_t>(targetHeight);
    Vector pixels(dimension, 0.0);

    for (int y = 0; y < targetHeight; ++y) {
        const uchar* row = gray.constScanLine(y);

        for (int x = 0; x < targetWidth; ++x) {
            pixels[static_cast<std::size_t>(y * targetWidth + x)] =
                static_cast<double>(row[x]);
        }
    }

    return pixels;
}

QImage PCAPipeline::vectorToImage(const Vector& pixels, int width, int height)
{
    if (width <= 0 || height <= 0) {
        throw std::invalid_argument("PCAPipeline::vectorToImage - image dimensions must be positive.");
    }

    const std::size_t expectedDimension = static_cast<std::size_t>(width) *
                                          static_cast<std::size_t>(height);
    validateFaceVectorSize(pixels, expectedDimension, "PCAPipeline::vectorToImage");

    QImage image(width, height, QImage::Format_Grayscale8);

    for (int y = 0; y < height; ++y) {
        uchar* row = image.scanLine(y);

        for (int x = 0; x < width; ++x) {
            double value = pixels[static_cast<std::size_t>(y * width + x)];
            value = std::max(0.0, std::min(255.0, value));
            row[x] = static_cast<uchar>(std::round(value));
        }
    }

    return image;
}

bool PCAPipeline::isTrained() const
{
    return m_trained;
}

bool PCAPipeline::hasTrainingGeometry() const
{
    return m_trainWidth > 0 && m_trainHeight > 0 && m_imageDimension > 0;
}

std::size_t PCAPipeline::getK() const
{
    return m_K;
}

std::size_t PCAPipeline::getSampleCount() const
{
    return m_samples.size();
}

std::size_t PCAPipeline::getClassCount() const
{
    std::set<int> identities;

    for (const FaceSample& sample : m_samples) {
        identities.insert(sample.identity);
    }

    return identities.size();
}

std::size_t PCAPipeline::getImageDimension() const
{
    return m_imageDimension;
}

int PCAPipeline::getTrainWidth() const
{
    return m_trainWidth;
}

int PCAPipeline::getTrainHeight() const
{
    return m_trainHeight;
}

const std::vector<FaceSample>& PCAPipeline::getSamples() const
{
    return m_samples;
}

const std::vector<ProjectedFaceSample>& PCAPipeline::getTrainingProjections() const
{
    return m_trainingProjections;
}

const Vector& PCAPipeline::getMeanFace() const
{
    return m_meanFace;
}

const Vector& PCAPipeline::getEigenvalues() const
{
    return m_eigenvalues;
}

const Matrix& PCAPipeline::getEigenfaces() const
{
    return m_eigenfaces;
}

void PCAPipeline::validateFaceVector(const Vector& faceVector, const std::string& caller) const
{
    validateFaceVectorSize(faceVector, m_imageDimension, caller);
}

void PCAPipeline::validateFaceVectorSize(const Vector& faceVector,
                                         std::size_t expectedDimension,
                                         const std::string& caller)
{
    if (faceVector.size() != expectedDimension) {
        throw std::invalid_argument(
            caller
            + " - expected vector size "
            + std::to_string(expectedDimension)
            + ", got "
            + std::to_string(faceVector.size())
        );
    }
}

bool PCAPipeline::isNumericFolderName(const std::string& name)
{
    if (name.empty()) {
        return false;
    }

    for (char ch : name) {
        if (ch < '0' || ch > '9') {
            return false;
        }
    }

    return true;
}

int PCAPipeline::folderNameToIdentity(const std::string& name)
{
    if (!isNumericFolderName(name)) {
        throw std::invalid_argument(
            "PCAPipeline::folderNameToIdentity - non-numeric identity folder: " + name
        );
    }

    return std::stoi(name);
}

} // namespace facerecog
