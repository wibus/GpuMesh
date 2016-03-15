#ifndef GPUMESH_SMOOTHINGREPORT
#define GPUMESH_SMOOTHINGREPORT

#include <QImage>
#include <QTextEdit>
#include <QTextDocument>

#include "DataStructures/OptimizationPlot.h"


class SmoothingReport
{
public:
    SmoothingReport();
    virtual ~SmoothingReport();

    virtual void setPreSmoothingShot(const QImage& snapshot);
    virtual void setPostSmoothingShot(const QImage& snapshot);
    virtual void setOptimizationPlot(const OptimizationPlot& plot);

    virtual bool save(const std::string& fileName) const;
    virtual void display(QTextEdit& textEdit) const;

protected:
    virtual void print(QTextDocument& document, bool paged) const;
    virtual void printMinimumQualityPlot(QPixmap& pixmap) const;
    virtual void printHistogramPlot(QPixmap& pixmap) const;

    QImage _preSmoothingShot;
    QImage _postSmoothingShot;
    OptimizationPlot _plot;
};

#endif // GPUMESH_SMOOTHINGREPORT
