#include "MathEngine.h"

#include <cmath>
#include <numeric>
#include <stdexcept>
#include <iostream>
#include <iomanip>
#include <algorithm>

#include <Eigen/Dense>

namespace facerecog {

// ── Matrix Construction ────────────────────────────────────────────────────

Matrix MathEngine::zeros(size_t rows, size_t cols)
{
    return Matrix(rows, Vector(cols, 0.0));
}

Matrix MathEngine::identity(size_t n)
{
    Matrix I = zeros(n, n);
    for (size_t i = 0; i < n; ++i)
        I[i][i] = 1.0;
    return I;
}

// ── Matrix Arithmetic ──────────────────────────────────────────────────────

Matrix MathEngine::transpose(const Matrix& A)
{
    if (A.empty()) return {};
    size_t m = A.size(), n = A[0].size();
    Matrix T(n, Vector(m));
    for (size_t i = 0; i < m; ++i)
        for (size_t j = 0; j < n; ++j)
            T[j][i] = A[i][j];
    return T;
}

Matrix MathEngine::multiply(const Matrix& A, const Matrix& B)
{
    if (A.empty() || B.empty()) return {};
    size_t m = A.size(), k = A[0].size(), n = B[0].size();
    if (k != B.size())
        throw std::invalid_argument("MathEngine::multiply — inner dimension mismatch");

    Matrix C = zeros(m, n);
    // i-k-j loop order for cache-friendly row-major access
    for (size_t i = 0; i < m; ++i)
        for (size_t p = 0; p < k; ++p) {
            double a_ip = A[i][p];
            for (size_t j = 0; j < n; ++j)
                C[i][j] += a_ip * B[p][j];
        }
    return C;
}

Matrix MathEngine::multiplyScalar(const Matrix& A, double scalar)
{
    Matrix C = A;
    for (auto& row : C)
        for (auto& val : row)
            val *= scalar;
    return C;
}

Matrix MathEngine::add(const Matrix& A, const Matrix& B)
{
    if (A.size() != B.size() || (!A.empty() && A[0].size() != B[0].size()))
        throw std::invalid_argument("MathEngine::add — dimension mismatch");
    Matrix C = A;
    for (size_t i = 0; i < A.size(); ++i)
        for (size_t j = 0; j < A[0].size(); ++j)
            C[i][j] += B[i][j];
    return C;
}

Matrix MathEngine::subtract(const Matrix& A, const Matrix& B)
{
    if (A.size() != B.size() || (!A.empty() && A[0].size() != B[0].size()))
        throw std::invalid_argument("MathEngine::subtract — dimension mismatch");
    Matrix C = A;
    for (size_t i = 0; i < A.size(); ++i)
        for (size_t j = 0; j < A[0].size(); ++j)
            C[i][j] -= B[i][j];
    return C;
}

// ── Vector Operations ──────────────────────────────────────────────────────

double MathEngine::dot(const Vector& a, const Vector& b)
{
    if (a.size() != b.size())
        throw std::invalid_argument("MathEngine::dot — size mismatch");
    double result = 0.0;
    for (size_t i = 0; i < a.size(); ++i)
        result += a[i] * b[i];
    return result;
}

double MathEngine::norm(const Vector& a) { return std::sqrt(dot(a, a)); }

double MathEngine::euclideanDistance(const Vector& a, const Vector& b)
{
    if (a.size() != b.size())
        throw std::invalid_argument("MathEngine::euclideanDistance — size mismatch");
    double sum = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        double d = a[i] - b[i];
        sum += d * d;
    }
    return std::sqrt(sum);
}

Vector MathEngine::normalize(const Vector& a)
{
    double n = norm(a);
    if (n < 1e-12) return Vector(a.size(), 0.0);
    Vector r(a.size());
    for (size_t i = 0; i < a.size(); ++i) r[i] = a[i] / n;
    return r;
}

// ── Statistical Operations ─────────────────────────────────────────────────

Vector MathEngine::meanVector(const std::vector<Vector>& vectors)
{
    if (vectors.empty())
        throw std::invalid_argument("MathEngine::meanVector — empty input");
    size_t D = vectors[0].size(), N = vectors.size();
    Vector mean(D, 0.0);
    for (const auto& v : vectors) {
        if (v.size() != D)
            throw std::invalid_argument("MathEngine::meanVector — inconsistent dims");
        for (size_t j = 0; j < D; ++j) mean[j] += v[j];
    }
    for (size_t j = 0; j < D; ++j) mean[j] /= static_cast<double>(N);
    return mean;
}

Matrix MathEngine::subtractVectorFromRows(const Matrix& data, const Vector& mean)
{
    if (data.empty()) return {};
    if (data[0].size() != mean.size())
        throw std::invalid_argument("MathEngine::subtractVectorFromRows — dim mismatch");
    Matrix centered = data;
    for (auto& row : centered)
        for (size_t j = 0; j < row.size(); ++j)
            row[j] -= mean[j];
    return centered;
}

Matrix MathEngine::covarianceMatrix(const Matrix& centeredData)
{
    if (centeredData.empty()) return {};
    size_t N = centeredData.size();
    Matrix At = transpose(centeredData);
    Matrix AtA = multiply(At, centeredData);
    double scale = (N > 1) ? 1.0 / static_cast<double>(N - 1) : 1.0;
    return multiplyScalar(AtA, scale);
}

Matrix MathEngine::smallCovarianceMatrix(const Matrix& centeredData)
{
    if (centeredData.empty()) return {};
    size_t N = centeredData.size();
    // Turk-Pentland: N×N matrix A·Aᵀ instead of D×D Aᵀ·A
    Matrix At = transpose(centeredData);
    Matrix AAt = multiply(centeredData, At);
    double scale = (N > 1) ? 1.0 / static_cast<double>(N - 1) : 1.0;
    return multiplyScalar(AAt, scale);
}

// ── Eigendecomposition ─────────────────────────────────────────────────────

MathEngine::EigenResult MathEngine::eigenDecomposition(const Matrix& sym)
{
    if (sym.empty()) return {{}, {}};
    size_t n = sym.size();
    if (sym[0].size() != n)
        throw std::invalid_argument("MathEngine::eigenDecomposition — not square");

    // Convert to Eigen matrix
    Eigen::MatrixXd em(n, n);
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < n; ++j)
            em(i, j) = sym[i][j];

    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(em);
    if (solver.info() != Eigen::Success)
        throw std::runtime_error("Eigen solver failed");

    Eigen::VectorXd vals = solver.eigenvalues();
    Eigen::MatrixXd vecs = solver.eigenvectors();

    // Sort indices descending by eigenvalue
    std::vector<size_t> idx(n);
    std::iota(idx.begin(), idx.end(), 0);
    std::sort(idx.begin(), idx.end(),
              [&vals](size_t a, size_t b) { return vals(a) > vals(b); });

    EigenResult result;
    result.eigenvalues.resize(n);
    result.eigenvectors.resize(n, Vector(n));
    for (size_t i = 0; i < n; ++i) {
        size_t si = idx[i];
        result.eigenvalues[i] = vals(si);
        for (size_t j = 0; j < n; ++j)
            result.eigenvectors[i][j] = vecs(j, si);
    }
    return result;
}

size_t MathEngine::selectK(const Vector& eigenvalues, double targetVariance)
{
    if (eigenvalues.empty()) return 0;
    double total = 0.0;
    for (double v : eigenvalues)
        if (v > 0.0) total += v;
    if (total < 1e-12) return 1;

    double cumulative = 0.0;
    for (size_t k = 0; k < eigenvalues.size(); ++k) {
        if (eigenvalues[k] > 0.0) cumulative += eigenvalues[k];
        if (cumulative / total >= targetVariance) return k + 1;
    }
    return eigenvalues.size();
}

// ── Utility ────────────────────────────────────────────────────────────────

Vector MathEngine::flatten(const Matrix& mat)
{
    Vector r;
    for (const auto& row : mat)
        r.insert(r.end(), row.begin(), row.end());
    return r;
}

Matrix MathEngine::reshape(const Vector& vec, size_t rows, size_t cols)
{
    if (vec.size() != rows * cols)
        throw std::invalid_argument("MathEngine::reshape — size mismatch");
    Matrix mat(rows, Vector(cols));
    for (size_t i = 0; i < rows; ++i)
        for (size_t j = 0; j < cols; ++j)
            mat[i][j] = vec[i * cols + j];
    return mat;
}

void MathEngine::printMatrix(const Matrix& A, const std::string& name)
{
    std::cout << name << " [" << A.size() << "x"
              << (A.empty() ? 0 : A[0].size()) << "]:\n";
    for (const auto& row : A) {
        std::cout << "  [";
        for (size_t j = 0; j < row.size(); ++j) {
            std::cout << std::setw(10) << std::setprecision(4)
                      << std::fixed << row[j];
            if (j + 1 < row.size()) std::cout << ", ";
        }
        std::cout << "]\n";
    }
    std::cout << std::endl;
}

void MathEngine::printVector(const Vector& v, const std::string& name)
{
    std::cout << name << " [" << v.size() << "]: [";
    for (size_t i = 0; i < v.size(); ++i) {
        std::cout << std::setw(10) << std::setprecision(4) << std::fixed << v[i];
        if (i + 1 < v.size()) std::cout << ", ";
    }
    std::cout << "]\n";
}

} // namespace facerecog
