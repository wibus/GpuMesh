#include "SmoothingReport.h"

#include <QPainter>
#include <QPdfWriter>

#include <QMargins>
#include <QTextTable>
#include <QTextCursor>
#include <QTextDocument>


SmoothingReport::SmoothingReport()
{

}

SmoothingReport::~SmoothingReport()
{

}

void SmoothingReport::setPreSmoothingShot(const QImage& snapshot)
{
    _preSmoothingShot = snapshot;
}

void SmoothingReport::setPostSmoothingShot(const QImage& snapshot)
{
    _postSmoothingShot = snapshot;
}

void SmoothingReport::setOptimizationPlot(const OptimizationPlot& plot)
{
    _plot = plot;
}

bool SmoothingReport::save(const std::string& fileName) const
{
    QPdfWriter writer(fileName.c_str());
    writer.setPageSize(QPagedPaintDevice::Letter);
    writer.setPageMargins(QMargins(10, 10, 10, 10),
                          QPageLayout::Millimeter);


    QTextDocument document;
    float dHdW = float(_preSmoothingShot.height()) /
                 float(_preSmoothingShot.width());
    float pageWidth = writer.pageLayout().paintRectPoints().width();
    document.addResource(QTextDocument::ImageResource,
                         QUrl("snapshots://preSmoothing.png"),
                         QVariant(_preSmoothingShot));
    document.addResource(QTextDocument::ImageResource,
                         QUrl("snapshots://postSmoothing.png"),
                         QVariant(_postSmoothingShot));

    QTextCursor cursor(&document);
    QTextBlockFormat blockFormat;

    cursor.insertBlock(blockFormat);
    cursor.insertHtml(("<h1>" + _plot.smoothingMethodName() +
                       ": " + _plot.meshModelName() + "</h1>").c_str());

    blockFormat.setTopMargin(20);

    QTextImageFormat imageFormat;
    imageFormat.setWidth(pageWidth);
    imageFormat.setHeight(pageWidth * dHdW);

    cursor.insertBlock(blockFormat);
    imageFormat.setName("snapshots://preSmoothing.png");
    cursor.insertImage(imageFormat);

    cursor.insertBlock(blockFormat);
    imageFormat.setName("snapshots://postSmoothing.png");
    cursor.insertImage(imageFormat);

    // Page break;
    blockFormat.setPageBreakPolicy(QTextFormat::PageBreak_AlwaysBefore);
    cursor.insertBlock(blockFormat);
    blockFormat.setPageBreakPolicy(QTextFormat::PageBreak_Auto);

    cursor.insertHtml("<h2>Mesh</h2>");

    size_t meshPropCount = _plot.meshProperties().size();
    QTextTable* meshTable = cursor.insertTable(meshPropCount, 2);
    int rowIdex = 0;
    for(const auto& keyVal : _plot.meshProperties())
    {
        const std::string& name = keyVal.first;
        QTextTableCell nameCell = meshTable->cellAt(rowIdex, 0);
        QTextCursor nameCursor = nameCell.firstCursorPosition();
        nameCursor.insertText(name.c_str());

        const std::string& value = keyVal.second;
        QTextTableCell valueCell = meshTable->cellAt(rowIdex, 1);
        QTextCursor valueCursor = valueCell.firstCursorPosition();
        valueCursor.insertText(value.c_str());

        ++rowIdex;
    }

    document.print(&writer);
}
