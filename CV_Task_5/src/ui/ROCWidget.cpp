#include "ROCWidget.h"

#include <QPainter>
#include <QPainterPath>
#include <QPen>
#include <QString>

#include <algorithm>

namespace facerecog {

ROCWidget::ROCWidget(QWidget* parent)
    : QWidget(parent)
{
    setMinimumSize(320, 240);
}

void ROCWidget::setCurve(const std::vector<QPointF>& points, double auc)
{
    m_points = points;
    m_auc = auc;
    update();
}

void ROCWidget::clearCurve()
{
    m_points.clear();
    m_auc = 0.0;
    update();
}

void ROCWidget::paintEvent(QPaintEvent* event)
{
    Q_UNUSED(event);

    QPainter painter(this);
    painter.setRenderHint(QPainter::Antialiasing, true);

    painter.fillRect(rect(), palette().window());

    const int left = 48;
    const int right = 18;
    const int top = 24;
    const int bottom = 44;

    const QRect plotRect(left,
                         top,
                         std::max(1, width() - left - right),
                         std::max(1, height() - top - bottom));

    painter.setPen(QPen(palette().mid().color(), 1));
    painter.drawRect(plotRect);

    auto mapPoint = [&plotRect](const QPointF& p) {
        const double x = std::clamp(p.x(), 0.0, 1.0);
        const double y = std::clamp(p.y(), 0.0, 1.0);
        return QPointF(plotRect.left() + x * plotRect.width(),
                       plotRect.bottom() - y * plotRect.height());
    };

    // Grid lines
    painter.setPen(QPen(palette().midlight().color(), 1, Qt::DashLine));
    for (int i = 1; i < 5; ++i) {
        const int gx = plotRect.left() + i * plotRect.width() / 5;
        const int gy = plotRect.top() + i * plotRect.height() / 5;
        painter.drawLine(gx, plotRect.top(), gx, plotRect.bottom());
        painter.drawLine(plotRect.left(), gy, plotRect.right(), gy);
    }

    // Random classifier diagonal
    painter.setPen(QPen(palette().mid().color(), 1, Qt::DashLine));
    painter.drawLine(mapPoint(QPointF(0, 0)), mapPoint(QPointF(1, 1)));

    // ROC curve
    if (!m_points.empty()) {
        QPainterPath path;
        path.moveTo(mapPoint(m_points.front()));
        for (std::size_t i = 1; i < m_points.size(); ++i) {
            path.lineTo(mapPoint(m_points[i]));
        }
        painter.setPen(QPen(palette().highlight().color(), 3));
        painter.drawPath(path);
    }

    painter.setPen(palette().text().color());
    painter.drawText(QRect(0, 0, width(), top), Qt::AlignCenter,
                     QString("ROC Curve  |  AUC = %1").arg(m_auc, 0, 'f', 3));

    painter.drawText(QRect(plotRect.left(), plotRect.bottom() + 8,
                           plotRect.width(), 28),
                     Qt::AlignCenter, "False Positive Rate");

    painter.save();
    painter.translate(14, plotRect.center().y());
    painter.rotate(-90);
    painter.drawText(QRect(-plotRect.height() / 2, -12,
                           plotRect.height(), 24),
                     Qt::AlignCenter, "True Positive Rate");
    painter.restore();

    painter.drawText(plotRect.left() - 8, plotRect.bottom() + 18, "0");
    painter.drawText(plotRect.right() - 8, plotRect.bottom() + 18, "1");
    painter.drawText(plotRect.left() - 22, plotRect.top() + 8, "1");
}

} // namespace facerecog
