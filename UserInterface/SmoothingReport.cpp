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
    writer.setPageOrientation(QPageLayout::Landscape);
    writer.setTitle((_plot.meshModelName()).c_str());

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

    qreal textWidth = qreal(1200);
    qreal docWidth = document.textWidth();
    if(docWidth > 0)
        textWidth = docWidth;

    float dHdW = float(_preSmoothingShot.height()) /
                 float(_preSmoothingShot.width());

    QPixmap plotPixmap = QPixmap(QSize(textWidth, textWidth * dHdW));
    plotPixmap.fill(Qt::transparent);
    printHistogramPlot(plotPixmap);

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
    cursor.insertHtml(("<h1>" + _plot.meshModelName() + "</h1>").c_str());

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

    QTextTableFormat propertyTableFormat;
    propertyTableFormat.setWidth(textWidth * 0.60);
    propertyTableFormat.setBorderStyle(QTextFrameFormat::BorderStyle_Solid);

    QTextTableCell tableCell;
    QTextCursor tableCursor;


    cursor.insertBlock(blockFormat);
    cursor.insertHtml("<h2>Properties</h2>");

    cursor.insertBlock(blockFormat);
    cursor.insertHtml("<h3>Mesh</h3>");

    size_t meshPropCount = _plot.meshProperties().size();
    QTextTable* meshTable = cursor.insertTable(meshPropCount+1, 2, propertyTableFormat);
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
    cursor.insertHtml("<h3>Node Groups</h3>");

    QTextTableFormat groupsTableFormat;
    groupsTableFormat.setWidth(textWidth);
    groupsTableFormat.setBorderStyle(QTextFrameFormat::BorderStyle_Solid);
    size_t groupCount = _plot.nodeGroups().parallelGroups().size();

    QTextTable* groupsTable = cursor.insertTable(groupCount+1, 3+1, groupsTableFormat);
    tableCell = groupsTable->cellAt(0, 0);
    tableCursor = tableCell.firstCursorPosition();
    tableCursor.insertText("Type", tableHeaderCharFormat);
    tableCell = groupsTable->cellAt(0, 1);
    tableCursor = tableCell.firstCursorPosition();
    tableCursor.insertText("Boundary", tableHeaderCharFormat);
    tableCell = groupsTable->cellAt(0, 2);
    tableCursor = tableCell.firstCursorPosition();
    tableCursor.insertText("Subsurface", tableHeaderCharFormat);
    tableCell = groupsTable->cellAt(0, 3);
    tableCursor = tableCell.firstCursorPosition();
    tableCursor.insertText("Interior", tableHeaderCharFormat);

    for(size_t g=0; g < groupCount; ++g)
    {
        const NodeGroups::ParallelGroup& group =
                _plot.nodeGroups().parallelGroups()[g];

        tableCell = groupsTable->cellAt(1+g, 0);
        tableCursor = tableCell.firstCursorPosition();
        tableCursor.insertText("Group " + QString().number(g));

        tableCell = groupsTable->cellAt(1+g, 1);
        tableCursor = tableCell.firstCursorPosition();
        tableCursor.insertText(QString::number(group.boundaryRange.size()));

        tableCell = groupsTable->cellAt(1+g, 2);
        tableCursor = tableCell.firstCursorPosition();
        tableCursor.insertText(QString::number(group.subsurfaceRange.size()));

        tableCell = groupsTable->cellAt(1+g, 3);
        tableCursor = tableCell.firstCursorPosition();
        tableCursor.insertText(QString::number(group.interiorRange.size()));
    }

    cursor.movePosition(QTextCursor::End);


    for(const OptimizationImpl& impl : _plot.implementations())
    {
        cursor.insertBlock(blockFormat);
        cursor.insertHtml(("<h3>" + impl.name + "</h3>").c_str());

        size_t smoothPropCount = impl.smoothingProperties.size();
        QTextTable* smoothTable = cursor.insertTable(smoothPropCount+1, 2, propertyTableFormat);
        tableCell = smoothTable->cellAt(0, 0);
        tableCursor = tableCell.firstCursorPosition();
        tableCursor.insertText("Property", tableHeaderCharFormat);
        tableCell = smoothTable->cellAt(0, 1);
        tableCursor = tableCell.firstCursorPosition();
        tableCursor.insertText("Value", tableHeaderCharFormat);

        for(size_t p=0; p < smoothPropCount; ++p)
        {
            const auto& property = impl.smoothingProperties[p];

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
    }


    blockFormat.setAlignment(Qt::AlignLeft);
    blockFormat.setPageBreakPolicy(QTextFormat::PageBreak_AlwaysBefore);
    cursor.insertBlock(blockFormat);
    blockFormat.setPageBreakPolicy(QTextFormat::PageBreak_Auto);

    cursor.insertHtml("<h2>Implementations</h2>");

    size_t maxPassCount = 0;
    double minImplTime = INFINITY;
    double minMeanImplGain = INFINITY;
    double minMinImplGain = INFINITY;
    size_t implCount = _plot.implementations().size();
    for(size_t i=0; i < implCount; ++i)
    {
        const OptimizationImpl& impl = _plot.implementations()[i];

        maxPassCount = std::max(maxPassCount,
                impl.passes.size());

        minMinImplGain = std::min(minMinImplGain,
                impl.finalHistogram.minimumQuality() -
                _plot.initialHistogram().minimumQuality());

        minMeanImplGain = std::min(minMeanImplGain,
                impl.finalHistogram.harmonicMean() -
                _plot.initialHistogram().harmonicMean());

        minImplTime = std::min(minImplTime,
                impl.passes.back().timeStamp);
    }

    QTextTableFormat statsTableFormat;
    statsTableFormat.setWidth(textWidth * (implCount+1) / 7.0f);
    statsTableFormat.setBorderStyle(QTextFrameFormat::BorderStyle_Solid);

    cursor.insertBlock(blockFormat);
    cursor.insertHtml("<h3>Minimum Quality</h3>");
    QTextTable* implMinQualTable = cursor.insertTable(
        maxPassCount + 5,
        implCount + 1,
        statsTableFormat);

    tableCell = implMinQualTable->cellAt(0, 0);
    tableCursor = tableCell.firstCursorPosition();
    tableCursor.insertText("Passes", tableHeaderCharFormat);
    tableCell = implMinQualTable->cellAt(maxPassCount+2, 0);
    tableCursor = tableCell.firstCursorPosition();
    tableCursor.insertText("Final", tableTotalCharFormat);
    tableCell = implMinQualTable->cellAt(maxPassCount+3, 0);
    tableCursor = tableCell.firstCursorPosition();
    tableCursor.insertText("Gain", tableTotalCharFormat);
    tableCell = implMinQualTable->cellAt(maxPassCount+4, 0);
    tableCursor = tableCell.firstCursorPosition();
    tableCursor.insertText("Ratio", tableTotalCharFormat);
    for(size_t p=0; p < maxPassCount; ++p)
    {
        tableCell = implMinQualTable->cellAt(1+p, 0);
        tableCursor = tableCell.firstCursorPosition();
        tableCursor.insertText(QString::number(p));
    }

    for(size_t i=0; i < implCount; ++i)
    {
        const OptimizationImpl& impl = _plot.implementations()[i];

        tableCell = implMinQualTable->cellAt(0, i+1);
        tableCursor = tableCell.firstCursorPosition();
        tableCursor.insertText(impl.name.c_str(), tableHeaderCharFormat);

        size_t passCount = impl.passes.size();
        for(size_t p=0; p < passCount; ++p)
        {
            const OptimizationPass& pass = impl.passes[p];

            tableCell = implMinQualTable->cellAt(1+p, i+1);
            tableCursor = tableCell.firstCursorPosition();
            tableCursor.insertText(QString::number(pass.histogram.minimumQuality(), 'f'));
        }

        double finalMin = impl.finalHistogram.minimumQuality();
        double minGain = impl.finalHistogram.minimumQuality() -
                         _plot.initialHistogram().minimumQuality();

        tableCell = implMinQualTable->cellAt(maxPassCount+2, i+1);
        tableCursor = tableCell.firstCursorPosition();
        tableCursor.insertText(QString::number(finalMin, 'f'));
        tableCell = implMinQualTable->cellAt(maxPassCount+3, i+1);
        tableCursor = tableCell.firstCursorPosition();
        tableCursor.insertText(QString::number(minGain, 'f'));
        tableCell = implMinQualTable->cellAt(maxPassCount+4, i+1);
        tableCursor = tableCell.firstCursorPosition();
        tableCursor.insertText(QString::number(minGain / minMinImplGain, 'f'));
    }
    cursor.movePosition(QTextCursor::End);


    cursor.insertBlock(blockFormat);
    cursor.insertHtml("<h3>Mean Quality</h3>");
    QTextTable* implMeanQualTable = cursor.insertTable(
        maxPassCount + 5,
        implCount + 1,
        statsTableFormat);

    tableCell = implMeanQualTable->cellAt(0, 0);
    tableCursor = tableCell.firstCursorPosition();
    tableCursor.insertText("Passes", tableHeaderCharFormat);
    tableCell = implMeanQualTable->cellAt(maxPassCount+2, 0);
    tableCursor = tableCell.firstCursorPosition();
    tableCursor.insertText("Final", tableTotalCharFormat);
    tableCell = implMeanQualTable->cellAt(maxPassCount+3, 0);
    tableCursor = tableCell.firstCursorPosition();
    tableCursor.insertText("Gain", tableTotalCharFormat);
    tableCell = implMeanQualTable->cellAt(maxPassCount+4, 0);
    tableCursor = tableCell.firstCursorPosition();
    tableCursor.insertText("Ratio", tableTotalCharFormat);
    for(size_t p=0; p < maxPassCount; ++p)
    {
        tableCell = implMeanQualTable->cellAt(1+p, 0);
        tableCursor = tableCell.firstCursorPosition();
        tableCursor.insertText(QString::number(p));
    }

    for(size_t i=0; i < implCount; ++i)
    {
        const OptimizationImpl& impl = _plot.implementations()[i];

        tableCell = implMeanQualTable->cellAt(0, i+1);
        tableCursor = tableCell.firstCursorPosition();
        tableCursor.insertText(impl.name.c_str(), tableHeaderCharFormat);

        size_t passCount = impl.passes.size();
        for(size_t p=0; p < passCount; ++p)
        {
            const OptimizationPass& pass = impl.passes[p];

            tableCell = implMeanQualTable->cellAt(1+p, i+1);
            tableCursor = tableCell.firstCursorPosition();
            tableCursor.insertText(QString::number(pass.histogram.harmonicMean(), 'f'));
        }

        double finalMean = impl.finalHistogram.harmonicMean();
        double meanGain = impl.finalHistogram.harmonicMean() -
                         _plot.initialHistogram().harmonicMean();

        tableCell = implMeanQualTable->cellAt(maxPassCount+2, i+1);
        tableCursor = tableCell.firstCursorPosition();
        tableCursor.insertText(QString::number(finalMean, 'f'));
        tableCell = implMeanQualTable->cellAt(maxPassCount+3, i+1);
        tableCursor = tableCell.firstCursorPosition();
        tableCursor.insertText(QString::number(meanGain, 'f'));
        tableCell = implMeanQualTable->cellAt(maxPassCount+4, i+1);
        tableCursor = tableCell.firstCursorPosition();
        tableCursor.insertText(QString::number(meanGain / minMeanImplGain, 'f'));
    }
    cursor.movePosition(QTextCursor::End);


    cursor.insertBlock(blockFormat);
    cursor.insertHtml("<h3>Times (seconds)</h3>");
    QTextTable* implTimesTable = cursor.insertTable(
        maxPassCount + 3,
        implCount + 1,
        statsTableFormat);

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

void SmoothingReport::printMinimumQualityPlot(QPixmap& pixmap) const
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
            minQ = glm::min(minQ, pass.histogram.minimumQuality());
            maxQ = glm::max(maxQ, pass.histogram.minimumQuality());
        }

        begQ = impl.passes.front().histogram.minimumQuality();
    }
    double marginQ = (maxQ - minQ) * 0.05;
    double bottomQ = minQ - marginQ;
    double topQ = maxQ + marginQ;
    double scaleQ = 1.0 / (topQ - bottomQ);

    // Min timestamp and min histogram.minimumQuality()
    QGraphicsTextItem* minTimeText = scene.addText("0s");
    double minTimeTextLen = minTimeText->document()->size().width();
    minTimeText->setPos(-minTimeTextLen/2.0, sceneHeight);

    // Algo final minimum quality
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
        double yAsymptote = sceneHeight * (topQ - samples.back().histogram.minimumQuality()) * scaleQ;
        scene.addLine(0, yAsymptote, sceneWidth, yAsymptote, QPen(Qt::lightGray));

        // Algo total times
        QGraphicsTextItem* timeText = scene.addText(QString::number(totalTime) + "s");
        double timeTextLen = timeText->document()->size().width();
        timeText->setPos(xAsymptote - timeTextLen/2.0, sceneHeight);
        timeText->setDefaultTextColor(pens[label].color());

        // Algo final minimum quality
        double finalQuality = samples.back().histogram.minimumQuality();
        QGraphicsTextItem* qualityText = scene.addText(QString::number(finalQuality));
        double qualityTextHeight = qualityText->document()->size().height();
        qualityText->setPos(sceneWidth, sceneHeight * (topQ - finalQuality) * scaleQ - qualityTextHeight/2.0);
        qualityText->setDefaultTextColor(pens[label].color());

        // Legend
        double gainValue = samples.back().histogram.minimumQuality() - samples.front().histogram.minimumQuality();
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
                sceneHeight * (topQ - prevPass.histogram.minimumQuality()) * scaleQ,
                sceneWidth * currPass.timeStamp / maxT,
                sceneHeight * (topQ - currPass.histogram.minimumQuality()) * scaleQ,
                pens[label]);
        }
    }

    // Graphics borders
    scene.addRect(0, 0, sceneWidth, sceneHeight);


    QFont titleFont;
    titleFont.setPointSize(20);
    QGraphicsTextItem* titleText = scene.addText(
        (_plot.meshModelName()).c_str(), titleFont);
    titleText->setPos((sceneWidth - titleText->document()->size().width())/2.0, -50.0);

    QPainter painter(&pixmap);
    scene.render(&painter);
}

void SmoothingReport::printHistogramPlot(QPixmap& pixmap) const
{
    QGraphicsScene scene;

    double sceneWidth = pixmap.width();
    double sceneHeight = pixmap.height();

    QBrush brushes[] = {
        QBrush(Qt::black),
        QBrush(Qt::blue),
        QBrush(Qt::green),
        QBrush(Qt::darkRed),
        QBrush(Qt::cyan),
        QBrush(Qt::magenta),
        QBrush(Qt::yellow),
        QBrush(Qt::gray),
        QBrush(QColor("olive")),
        QBrush(QColor("darksalmon"))
    };

    double maxElemRatio = 0.0;
    const QualityHistogram initHist = _plot.initialHistogram();
    for(int bucket : initHist.buckets())
        maxElemRatio = glm::max(maxElemRatio, bucket / double(initHist.sampleCount()));
    for(const OptimizationImpl& impl : _plot.implementations())
    {
        const QualityHistogram& histogram = impl.passes.back().histogram;
        for(int bucket : histogram.buckets())
            maxElemRatio = glm::max(maxElemRatio, bucket / double(histogram.sampleCount()));
    }
    double scaleY = 0.80 / maxElemRatio;

    std::vector<const QualityHistogram*> hists;
    hists.push_back(&_plot.initialHistogram());
    for(const OptimizationImpl& impl : _plot.implementations())
        hists.push_back(&impl.finalHistogram);

    // Histogram (Bands)
    double offset = 0.0;
    size_t bucketCount = initHist.bucketCount();
    double bandWidth = sceneWidth / (bucketCount * (hists.size()+1));
    for(size_t h = 0; h < hists.size(); ++h)
    {
        for(size_t i=0; i < bucketCount; ++i)
        {
            double ratio = hists[h]->buckets()[i] /
                    double(hists[h]->sampleCount());

            scene.addRect(
                sceneWidth * double(i) / (bucketCount) + offset,
                sceneHeight,
                bandWidth,
                -sceneHeight * (ratio * scaleY),
                QPen(brushes[h],0), brushes[h]);
        }

        offset += bandWidth;
    }

    // Graphics borders
    scene.addRect(0, 0, sceneWidth, sceneHeight);


    QFont titleFont;
    titleFont.setPointSize(20);
    QGraphicsTextItem* titleText = scene.addText(
        (_plot.meshModelName()).c_str(), titleFont);
    titleText->setPos((sceneWidth - titleText->document()->size().width())/2.0, 30.0);

    QPainter painter(&pixmap);
    scene.render(&painter);
}
