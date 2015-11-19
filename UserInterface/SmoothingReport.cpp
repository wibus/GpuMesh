#include "SmoothingReport.h"

#include <GLM/glm.hpp>

#include <QPainter>
#include <QPdfWriter>
#include <QGraphicsScene>
#include <QGraphicsTextItem>

#include <QMargins>
#include <QTextTable>
#include <QTextCursor>
#include <QTextDocument>

using namespace std;


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
    writer.setPageMargins(QMargins(10, 10, 10, 10),
                          QPageLayout::Millimeter);
    writer.setPageSize(QPagedPaintDevice::Letter);
    writer.setTitle((_plot.smoothingMethodName() +
                     ": " + _plot.meshModelName()).c_str());

    qreal textWidth = writer.pageLayout().paintRectPoints().width();

    QTextDocument document;
    document.setTextWidth(textWidth);
    print(document, true);
    document.print(&writer);
	return true;
}

void SmoothingReport::display(QTextEdit& textEdit) const
{
    print(*textEdit.document(), false);
}

void SmoothingReport::print(QTextDocument& document, bool paged) const
{
    document.clear();

    qreal textWidth = qreal(800);
    qreal docWidth = document.textWidth();
    if(docWidth > 0)
        textWidth = docWidth;

    float dHdW = float(_preSmoothingShot.height()) /
                 float(_preSmoothingShot.width());

    QPixmap plotPixmap = QPixmap(QSize(800, 800 * dHdW));
    plotPixmap.fill(Qt::transparent);
    printOptimizationPlot(plotPixmap);

    document.addResource(QTextDocument::ImageResource,
                         QUrl("snapshots://preSmoothing.png"),
                         QVariant(_preSmoothingShot));
    document.addResource(QTextDocument::ImageResource,
                         QUrl("snapshots://postSmoothing.png"),
                         QVariant(_postSmoothingShot));
    document.addResource(QTextDocument::ImageResource,
                         QUrl("snapshots://graphicPlot.png"),
                         QVariant(plotPixmap.toImage()));

    QTextCursor cursor(&document);
    QTextBlockFormat blockFormat;

    // First page (Title + Screenshots)
    blockFormat.setAlignment(Qt::AlignHCenter);
    cursor.insertBlock(blockFormat);
    blockFormat.setAlignment(Qt::AlignJustify);
    cursor.insertHtml(("<h1>" + _plot.smoothingMethodName() +
                       ": " + _plot.meshModelName() + "</h1>").c_str());

    blockFormat.setTopMargin(20);

    QTextImageFormat imageFormat;
    imageFormat.setWidth(textWidth);
    imageFormat.setHeight(textWidth * dHdW);

    cursor.insertBlock(blockFormat);
    imageFormat.setName("snapshots://preSmoothing.png");
    cursor.insertImage(imageFormat);

    cursor.insertBlock(blockFormat);
    imageFormat.setName("snapshots://postSmoothing.png");
    cursor.insertImage(imageFormat);

    // Second page (Plot + mesh properties)
    blockFormat.setPageBreakPolicy(QTextFormat::PageBreak_AlwaysBefore);
    cursor.insertBlock(blockFormat);
    blockFormat.setPageBreakPolicy(QTextFormat::PageBreak_Auto);


    imageFormat.setName("snapshots://graphicPlot.png");
    cursor.insertImage(imageFormat);



    QTextCharFormat tableHeaderCharFormat;
    tableHeaderCharFormat.setFontItalic(true);

    QTextCharFormat tableTotalCharFormat;
    tableTotalCharFormat.setFontWeight(QFont::Bold);

    QTextTableFormat tableFormat;
    tableFormat.setWidth(textWidth * 0.45);
    tableFormat.setBorderStyle(QTextFrameFormat::BorderStyle_Solid);

    QTextTableCell tableCell;
    QTextCursor tableCursor;


    cursor.insertBlock(blockFormat);
    cursor.insertHtml("<h2>Properties</h2>");

    cursor.insertBlock(blockFormat);
    cursor.insertHtml("<h3>Mesh</h3>");

    size_t meshPropCount = _plot.meshProperties().size();
    QTextTable* meshTable = cursor.insertTable(meshPropCount+1, 2, tableFormat);
    tableCell = meshTable->cellAt(0, 0);
    tableCursor = tableCell.firstCursorPosition();
    tableCursor.insertText("Property", tableHeaderCharFormat);
    tableCell = meshTable->cellAt(0, 1);
    tableCursor = tableCell.firstCursorPosition();
    tableCursor.insertText("Value", tableHeaderCharFormat);

    for(size_t p=0; p < meshPropCount; ++p)
    {
        const auto& property = _plot.meshProperties()[p];

        const std::string& name = property.first;
        tableCell = meshTable->cellAt(1+p, 0);
        tableCursor = tableCell.firstCursorPosition();
        tableCursor.insertText(name.c_str());

        const std::string& value = property.second;
        QTextTableCell valueCell = meshTable->cellAt(1+p, 1);
        QTextCursor valueCursor = valueCell.firstCursorPosition();
        valueCursor.insertText(value.c_str());
    }
    cursor.movePosition(QTextCursor::End);


    cursor.insertBlock(blockFormat);
    cursor.insertHtml("<h3>Smoothing Method</h3>");

    size_t smoothPropCount = _plot.smoothingProperties().size();
    QTextTable* smoothTable = cursor.insertTable(smoothPropCount+1, 2, tableFormat);
    tableCell = smoothTable->cellAt(0, 0);
    tableCursor = tableCell.firstCursorPosition();
    tableCursor.insertText("Property", tableHeaderCharFormat);
    tableCell = smoothTable->cellAt(0, 1);
    tableCursor = tableCell.firstCursorPosition();
    tableCursor.insertText("Value", tableHeaderCharFormat);

    for(size_t p=0; p < smoothPropCount; ++p)
    {
        const auto& property = _plot.smoothingProperties()[p];

        const std::string& name = property.first;
        tableCell = smoothTable->cellAt(1+p, 0);
        tableCursor = tableCell.firstCursorPosition();
        tableCursor.insertText(name.c_str());

        const std::string& value = property.second;
        QTextTableCell valueCell = smoothTable->cellAt(1+p, 1);
        QTextCursor valueCursor = valueCell.firstCursorPosition();
        valueCursor.insertText(value.c_str());
    }
    cursor.movePosition(QTextCursor::End);


    blockFormat.setAlignment(Qt::AlignLeft);
    blockFormat.setPageBreakPolicy(QTextFormat::PageBreak_AlwaysBefore);
    cursor.insertBlock(blockFormat);
    blockFormat.setPageBreakPolicy(QTextFormat::PageBreak_Auto);

    cursor.insertHtml("<h2>Implementations</h2>");

    size_t maxPassCount = 0;
    double minMeanGain = INFINITY;
    double minImplTime = INFINITY;
    size_t implCount = _plot.implementations().size();
    for(size_t i=0; i < implCount; ++i)
    {
        const OptimizationImpl& impl = _plot.implementations()[i];

        maxPassCount = std::max(maxPassCount,
                impl.passes.size());

        minMeanGain = std::min(minMeanGain,
                impl.passes.back().qualityMean -
                impl.passes.front().qualityMean);

        minImplTime = std::min(minImplTime,
                impl.passes.back().timeStamp);
    }


    cursor.insertBlock(blockFormat);
    cursor.insertHtml("<h3>Mean Quality</h3>");
    QTextTable* implMeanTable = cursor.insertTable(
        maxPassCount + 4,
        implCount + 1,
        tableFormat);

    tableCell = implMeanTable->cellAt(0, 0);
    tableCursor = tableCell.firstCursorPosition();
    tableCursor.insertText("Passes", tableHeaderCharFormat);
    tableCell = implMeanTable->cellAt(maxPassCount+2, 0);
    tableCursor = tableCell.firstCursorPosition();
    tableCursor.insertText("Gain", tableTotalCharFormat);
    tableCell = implMeanTable->cellAt(maxPassCount+3, 0);
    tableCursor = tableCell.firstCursorPosition();
    tableCursor.insertText("Ratio", tableTotalCharFormat);
    for(size_t p=0; p < maxPassCount; ++p)
    {
        tableCell = implMeanTable->cellAt(1+p, 0);
        tableCursor = tableCell.firstCursorPosition();
        tableCursor.insertText(QString::number(p));
    }

    for(size_t i=0; i < implCount; ++i)
    {
        const OptimizationImpl& impl = _plot.implementations()[i];

        tableCell = implMeanTable->cellAt(0, i+1);
        tableCursor = tableCell.firstCursorPosition();
        tableCursor.insertText(impl.name.c_str(), tableHeaderCharFormat);

        size_t passCount = impl.passes.size();
        for(size_t p=0; p < passCount; ++p)
        {
            const OptimizationPass& pass = impl.passes[p];

            tableCell = implMeanTable->cellAt(1+p, i+1);
            tableCursor = tableCell.firstCursorPosition();
            tableCursor.insertText(QString::number(pass.qualityMean, 'f'));
        }

        double meanGain = impl.passes.back().qualityMean -
                          impl.passes.front().qualityMean;

        tableCell = implMeanTable->cellAt(maxPassCount+2, i+1);
        tableCursor = tableCell.firstCursorPosition();
        tableCursor.insertText(QString::number(meanGain, 'f'));
        tableCell = implMeanTable->cellAt(maxPassCount+3, i+1);
        tableCursor = tableCell.firstCursorPosition();
        tableCursor.insertText(QString::number(meanGain / minMeanGain, 'f'));
    }
    cursor.movePosition(QTextCursor::End);


    cursor.insertBlock(blockFormat);
    cursor.insertHtml("<h3>Times</h3>");
    QTextTable* implTimesTable = cursor.insertTable(
        maxPassCount + 3,
        implCount + 1,
        tableFormat);

    tableCell = implTimesTable->cellAt(0, 0);
    tableCursor = tableCell.firstCursorPosition();
    tableCursor.insertText("Passes", tableHeaderCharFormat);
    tableCell = implTimesTable->cellAt(maxPassCount+2, 0);
    tableCursor = tableCell.firstCursorPosition();
    tableCursor.insertText("Ratio", tableTotalCharFormat);
    for(size_t p=0; p < maxPassCount; ++p)
    {
        tableCell = implTimesTable->cellAt(1+p, 0);
        tableCursor = tableCell.firstCursorPosition();
        tableCursor.insertText(QString::number(p));
    }

    for(size_t i=0; i < implCount; ++i)
    {
        const OptimizationImpl& impl = _plot.implementations()[i];

        tableCell = implTimesTable->cellAt(0, i+1);
        tableCursor = tableCell.firstCursorPosition();
        tableCursor.insertText(impl.name.c_str(), tableHeaderCharFormat);

        size_t passCount = impl.passes.size();
        for(size_t p=0; p < passCount; ++p)
        {
            const OptimizationPass& pass = impl.passes[p];

            tableCell = implTimesTable->cellAt(1+p, i+1);
            tableCursor = tableCell.firstCursorPosition();
            tableCursor.insertText(QString::number(pass.timeStamp, 'f'));
        }

        tableCell = implTimesTable->cellAt(maxPassCount+2, i+1);
        tableCursor = tableCell.firstCursorPosition();
        tableCursor.insertText(QString::number(
            impl.passes.back().timeStamp / minImplTime, 'f'));
    }
    cursor.movePosition(QTextCursor::End);
}

void SmoothingReport::printOptimizationPlot(QPixmap& pixmap) const
{
    QGraphicsScene scene;

    double sceneWidth = pixmap.width();
    double sceneHeight = pixmap.height();

    double maxT = 0.0;
    double minQ = 1.0;
    double maxQ = 0.0;
    double begQ = 0.0;
    map<string, QPen> pens;
    for(const OptimizationImpl& impl : _plot.implementations())
    {
        const string& label = impl.name;

        pens.insert(make_pair(
            label,
            QPen(QColor(
                ((uchar) label[0]) % 3 * 75,
                ((uchar) label[1]) % 3 * 75,
                ((uchar) label[2]) % 3 * 75))
        ));

        for(const auto& pass : impl.passes)
        {
            maxT = glm::max(maxT, pass.timeStamp);
            minQ = glm::min(minQ, pass.qualityMean);
            maxQ = glm::max(maxQ, pass.qualityMean);
        }

        begQ = impl.passes.front().qualityMean;
    }
    double marginQ = (maxQ - minQ) * 0.05;
    double bottomQ = minQ - marginQ;
    double topQ = maxQ + marginQ;
    double scaleQ = 1.0 / (topQ - bottomQ);

    // Min timestamp and min qualityMean
    QGraphicsTextItem* minTimeText = scene.addText("0s");
    double minTimeTextLen = minTimeText->document()->size().width();
    minTimeText->setPos(-minTimeTextLen/2.0, sceneHeight);

    // Algo final quality mean
    QGraphicsTextItem* begQualityText = scene.addText(QString::number(minQ));
    double begQualityTextWidth = begQualityText->document()->size().width();
    double begQualityTextHeight = begQualityText->document()->size().height();
    begQualityText->setPos(-begQualityTextWidth, sceneHeight * (topQ-begQ)*scaleQ - begQualityTextHeight/2.0);


    double labelHeight = 10.0;
    double legendLeft = sceneWidth - 200;
    double legendTop = sceneHeight - 20.0 * (_plot.implementations().size() + 2);
    for(const OptimizationImpl& impl : _plot.implementations())
    {
        const string& label = impl.name;
        const std::vector<OptimizationPass>& samples = impl.passes;

        // Asymptotes
        double totalTime = samples.back().timeStamp;
        double xAsymptote = sceneWidth * (totalTime / maxT);
        scene.addLine(xAsymptote, 0, xAsymptote, sceneHeight, QPen(Qt::lightGray));
        double yAsymptote = sceneHeight * (topQ - samples.back().qualityMean) * scaleQ;
        scene.addLine(0, yAsymptote, sceneWidth, yAsymptote, QPen(Qt::lightGray));

        // Algo total times
        QGraphicsTextItem* timeText = scene.addText(QString::number(totalTime) + "s");
        double timeTextLen = timeText->document()->size().width();
        timeText->setPos(xAsymptote - timeTextLen/2.0, sceneHeight);
        timeText->setDefaultTextColor(pens[label].color());

        // Algo final quality mean
        double finalQuality = samples.back().qualityMean;
        QGraphicsTextItem* qualityText = scene.addText(QString::number(finalQuality));
        double qualityTextHeight = qualityText->document()->size().height();
        qualityText->setPos(sceneWidth, sceneHeight * (topQ - finalQuality) * scaleQ - qualityTextHeight/2.0);
        qualityText->setDefaultTextColor(pens[label].color());

        // Legend
        double gainValue = samples.back().qualityMean - samples.front().qualityMean;
        QString gainText = " (" + ((gainValue < 0.0 ? "-" : "+") + QString::number(gainValue)) + ")";
        QGraphicsTextItem* text = scene.addText(label.c_str() + gainText);
        text->setPos(legendLeft + 10.0, legendTop + labelHeight);
        text->setDefaultTextColor(pens[label].color());
        labelHeight += 20.0;
    }
    scene.addRect(
        legendLeft, legendTop,
        180.0, 20.0 * (_plot.implementations().size()+1.3));


    // Optimization curves
    for(const OptimizationImpl& impl : _plot.implementations())
    {
        const string& label = impl.name;
        const std::vector<OptimizationPass>& samples = impl.passes;
        for(size_t i=1; i < samples.size(); ++i)
        {
            const OptimizationPass& prevPass = samples[i-1];
            const OptimizationPass& currPass = samples[i];
            scene.addLine(
                sceneWidth * prevPass.timeStamp / maxT,
                sceneHeight * (topQ - prevPass.qualityMean) * scaleQ,
                sceneWidth * currPass.timeStamp / maxT,
                sceneHeight * (topQ - currPass.qualityMean) * scaleQ,
                pens[label]);
        }
    }

    // Graphics borders
    scene.addRect(0, 0, sceneWidth, sceneHeight);


    QFont titleFont;
    titleFont.setPointSize(20);
    QGraphicsTextItem* titleText = scene.addText(
        (_plot.smoothingMethodName() + ": " + _plot.meshModelName()).c_str(), titleFont);
    titleText->setPos((sceneWidth - titleText->document()->size().width())/2.0, -50.0);

    QPainter painter(&pixmap);
    scene.render(&painter);
}
