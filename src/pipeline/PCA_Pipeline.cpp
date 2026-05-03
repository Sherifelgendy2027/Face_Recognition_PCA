#include "PCA_Pipeline.h"
#include "core/MathEngine.h"

#include <QImage>
#include <QDir>
#include <QDirIterator>
#include <QFileInfo>

#include <algorithm>
#include <iostream>
#include <numeric>
#include <set>
#include <stdexcept>

namespace facerecog {

// ════════════════════════════════════════════════════════════════════════════
// Construction
// ════════════════════════════════════════════════════════════════════════════

PCA_Pipeline::PCA_Pipeline()
    : m_trained(false)
    , m_K(0)
{
}

// ════════════════════════════════════════════════════════════════════════════
// Dataset Loading
// ════════════════════════════════════════════════════════════════════════════

size_t PCA_Pipeline::loadDataset(const std::string& rootPath)
{
    m_samples.clear();
    m_trained = false;

    QDir rootDir(QString::fromStdString(rootPath));
    if (!rootDir.exists()) {
        throw std::runtime_error(
            "PCA_Pipeline::loadDataset — directory does not exist: " + rootPath);
    }

    // Iterate over subdirectories (each is a person: "1", "2", ..., "41")
    QStringList subdirs = rootDir.entryList(QDir::Dirs | QDir::NoDotAndDotDot,
                                            QDir::Name);

    // Sort numerically so "2" comes before "10"
    std::sort(subdirs.begin(), subdirs.end(),
              [](const QString& a, const QString& b) {
                  bool okA, okB;
                  int na = a.toInt(&okA);
                  int nb = b.toInt(&okB);
                  if (okA && okB) return na < nb;
                  return a < b;
              });

    // Supported image formats
    QStringList imageFilters;
    imageFilters << "*.pgm" << "*.png" << "*.jpg" << "*.jpeg"
                 << "*.bmp" << "*.tif" << "*.tiff";

    for (const QString& subdir : subdirs) {
        QString personPath = rootDir.absoluteFilePath(subdir);
        QDir personDir(personPath);
        if (!personDir.exists()) continue;

        std::string label = subdir.toStdString();

        // List image files in this person's folder
        QStringList images = personDir.entryList(imageFilters,
                                                  QDir::Files,
                                                  QDir::Name);

        for (const QString& imgFile : images) {
            QString fullPath = personDir.absoluteFilePath(imgFile);
            QImage image(fullPath);

            if (image.isNull()) {
                std::cerr << "  ⚠ Skipping unreadable image: "
                          << fullPath.toStdString() << std::endl;
                continue;
            }

            FaceSample sample;
            sample.pixels   = imageToVector(image);
            sample.label    = label;
            sample.filePath = fullPath.toStdString();
            m_samples.push_back(std::move(sample));
        }
    }

    if (m_samples.empty()) {
        throw std::runtime_error(
            "PCA_Pipeline::loadDataset — no valid images found in: " + rootPath);
    }

    std::cout << "  Loaded " << m_samples.size() << " images from "
              << getNumClasses() << " identities" << std::endl;

    return m_samples.size();
}

size_t PCA_Pipeline::getNumClasses() const
{
    std::set<std::string> labels;
    for (const auto& s : m_samples)
        labels.insert(s.label);
    return labels.size();
}

// ════════════════════════════════════════════════════════════════════════════
// Training
// ════════════════════════════════════════════════════════════════════════════

TrainResult PCA_Pipeline::train(double targetVariance)
{
    if (m_samples.empty()) {
        throw std::runtime_error("PCA_Pipeline::train — no samples loaded");
    }

    size_t N = m_samples.size();       // number of images
    size_t D = FACE_DIM;               // pixel dimension (92 * 112 = 10304)

    std::cout << "  Training PCA on " << N << " samples, D=" << D << std::endl;

    // ── Step 1: Build data matrix X (N × D) ────────────────────────────
    Matrix X(N, Vector(D));
    for (size_t i = 0; i < N; ++i) {
        X[i] = m_samples[i].pixels;
    }

    // ── Step 2: Compute mean face μ (D) ────────────────────────────────
    std::cout << "  Computing mean face..." << std::endl;
    m_meanFace = MathEngine::meanVector(X);

    // ── Step 3: Center data A = X − μ (N × D) ─────────────────────────
    std::cout << "  Centering data..." << std::endl;
    Matrix A = MathEngine::subtractVectorFromRows(X, m_meanFace);

    // Free X — no longer needed, A holds centered data
    X.clear();
    X.shrink_to_fit();

    // ── Step 4: Small covariance matrix (Turk-Pentland trick) ──────────
    // Instead of D×D (10304×10304), compute N×N (e.g. 400×400)
    // C_small = (1/(N-1)) * A · Aᵀ
    std::cout << "  Computing small covariance matrix (" << N << "x" << N
              << ") via Turk-Pentland trick..." << std::endl;
    Matrix C_small = MathEngine::smallCovarianceMatrix(A);

    // ── Step 5: Eigendecompose C_small → eigenvalues + small eigenvectors
    std::cout << "  Running eigendecomposition..." << std::endl;
    auto eigenResult = MathEngine::eigenDecomposition(C_small);

    // Free C_small
    C_small.clear();
    C_small.shrink_to_fit();

    // Store eigenvalues (they are the same for both A·Aᵀ and Aᵀ·A
    // up to the non-zero ones)
    m_eigenvalues = eigenResult.eigenvalues;

    // ── Step 6: Recover full eigenvectors of the D×D covariance ────────
    // uᵢ = Aᵀ · vᵢ    (then normalize to unit length)
    // eigenResult.eigenvectors[i] is the i-th small eigenvector (length N)
    std::cout << "  Recovering full eigenfaces via Aᵀ·v trick..." << std::endl;

    // Aᵀ is D × N
    Matrix At = MathEngine::transpose(A);

    // Determine how many eigenvalues are positive (meaningful components)
    size_t maxComponents = 0;
    for (size_t i = 0; i < m_eigenvalues.size(); ++i) {
        if (m_eigenvalues[i] > 1e-6) {
            ++maxComponents;
        } else {
            break; // sorted descending, so stop at first non-positive
        }
    }

    // Auto-select K based on target variance
    m_K = MathEngine::selectK(m_eigenvalues, targetVariance);
    if (m_K > maxComponents) m_K = maxComponents;
    if (m_K == 0) m_K = 1;

    // Compute actual variance captured
    double totalVar = 0.0;
    for (size_t i = 0; i < maxComponents; ++i)
        totalVar += m_eigenvalues[i];

    double capturedVar = 0.0;
    for (size_t i = 0; i < m_K; ++i)
        capturedVar += m_eigenvalues[i];

    double varianceRatio = (totalVar > 0) ? capturedVar / totalVar : 0.0;

    std::cout << "  Selected K=" << m_K << " eigenfaces capturing "
              << (varianceRatio * 100.0) << "% of variance" << std::endl;

    // Recover top K full eigenfaces: each is D-dimensional
    m_eigenfaces.resize(m_K, Vector(D));
    for (size_t i = 0; i < m_K; ++i) {
        // vᵢ is the i-th small eigenvector (length N)
        const Vector& vi = eigenResult.eigenvectors[i];

        // uᵢ = Aᵀ · vᵢ  → Aᵀ is D×N, vᵢ is N×1 → result is D×1
        // Manual matrix-vector multiply for efficiency
        Vector ui(D, 0.0);
        for (size_t d = 0; d < D; ++d) {
            for (size_t n = 0; n < N; ++n) {
                ui[d] += At[d][n] * vi[n];
            }
        }

        // Normalize to unit length
        m_eigenfaces[i] = MathEngine::normalize(ui);
    }

    // Free At
    At.clear();
    At.shrink_to_fit();

    // ── Step 7: Project all training faces ─────────────────────────────
    std::cout << "  Projecting " << N << " training faces into K="
              << m_K << " subspace..." << std::endl;
    m_trainingProjections.clear();
    m_trainingProjections.reserve(N);

    for (size_t i = 0; i < N; ++i) {
        Vector weights = project(m_samples[i].pixels);
        m_trainingProjections.emplace_back(std::move(weights), m_samples[i].label);
    }

    m_trained = true;

    std::cout << "  ✅ Training complete!" << std::endl;

    TrainResult result;
    result.numSamples       = N;
    result.numClasses        = getNumClasses();
    result.dimension         = D;
    result.selectedK         = m_K;
    result.varianceCaptured  = varianceRatio;
    return result;
}

// ════════════════════════════════════════════════════════════════════════════
// Projection & Reconstruction
// ════════════════════════════════════════════════════════════════════════════

Vector PCA_Pipeline::project(const Vector& face) const
{
    if (!m_trained && m_eigenfaces.empty()) {
        throw std::runtime_error("PCA_Pipeline::project — model not trained");
    }

    // Center: x_centered = face − μ
    Vector centered(face.size());
    for (size_t i = 0; i < face.size(); ++i) {
        centered[i] = face[i] - m_meanFace[i];
    }

    // w = Eᵀ · centered  →  w[k] = eigenfaces[k] · centered
    Vector weights(m_K);
    for (size_t k = 0; k < m_K; ++k) {
        weights[k] = MathEngine::dot(m_eigenfaces[k], centered);
    }

    return weights;
}

Vector PCA_Pipeline::reconstruct(const Vector& weights) const
{
    if (!m_trained) {
        throw std::runtime_error("PCA_Pipeline::reconstruct — model not trained");
    }

    size_t D = m_meanFace.size();
    Vector face(D);

    // x̂ = μ + Σ wₖ · eₖ
    for (size_t d = 0; d < D; ++d) {
        face[d] = m_meanFace[d];
        for (size_t k = 0; k < m_K; ++k) {
            face[d] += weights[k] * m_eigenfaces[k][d];
        }
    }

    return face;
}

double PCA_Pipeline::reconstructionError(const Vector& face) const
{
    Vector weights = project(face);
    Vector reconstructed = reconstruct(weights);

    double error = 0.0;
    for (size_t i = 0; i < face.size(); ++i) {
        double diff = face[i] - reconstructed[i];
        error += diff * diff;
    }
    return error;
}

// ════════════════════════════════════════════════════════════════════════════
// Image Utilities
// ════════════════════════════════════════════════════════════════════════════

Vector PCA_Pipeline::imageToVector(const QImage& image)
{
    // Convert to grayscale if needed
    QImage gray = image.convertToFormat(QImage::Format_Grayscale8);

    // Resize to standard dimensions, preserving aspect ratio isn't needed
    // since AT&T dataset is already 92×112. But handle arbitrary input.
    if (gray.width() != FACE_WIDTH || gray.height() != FACE_HEIGHT) {
        gray = gray.scaled(FACE_WIDTH, FACE_HEIGHT,
                           Qt::IgnoreAspectRatio,
                           Qt::SmoothTransformation);
    }

    // Flatten to 1D vector: row-major, pixel values as doubles [0, 255]
    Vector pixels(FACE_DIM);
    for (int y = 0; y < FACE_HEIGHT; ++y) {
        const uchar* scanline = gray.constScanLine(y);
        for (int x = 0; x < FACE_WIDTH; ++x) {
            pixels[y * FACE_WIDTH + x] = static_cast<double>(scanline[x]);
        }
    }

    return pixels;
}

QImage PCA_Pipeline::vectorToImage(const Vector& pixels)
{
    if (static_cast<int>(pixels.size()) != FACE_DIM) {
        throw std::invalid_argument(
            "PCA_Pipeline::vectorToImage — expected " +
            std::to_string(FACE_DIM) + " pixels, got " +
            std::to_string(pixels.size()));
    }

    QImage image(FACE_WIDTH, FACE_HEIGHT, QImage::Format_Grayscale8);

    for (int y = 0; y < FACE_HEIGHT; ++y) {
        uchar* scanline = image.scanLine(y);
        for (int x = 0; x < FACE_WIDTH; ++x) {
            double val = pixels[y * FACE_WIDTH + x];
            // Clamp to [0, 255]
            val = std::max(0.0, std::min(255.0, val));
            scanline[x] = static_cast<uchar>(val + 0.5); // round
        }
    }

    return image;
}

} // namespace facerecog
