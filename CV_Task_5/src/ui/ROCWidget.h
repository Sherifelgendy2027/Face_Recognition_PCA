#ifndef ROCWIDGET_H
#define ROCWIDGET_H

/**
 * @file ROCWidget.h
 * @brief Step 5: Native Qt ROC plotting widget using QPainter only.
 */

#include <QPointF>
#include <QWidget>

#include <vector>

namespace facerecog {

class ROCWidget : public QWidget {
public:
    explicit ROCWidget(QWidget* parent = nullptr);

    void setCurve(const std::vector<QPointF>& points, double auc);
    void clearCurve();

protected:
    void paintEvent(QPaintEvent* event) override;

private:
    std::vector<QPointF> m_points;
    double m_auc = 0.0;
};

} // namespace facerecog

#endif // ROCWIDGET_H
