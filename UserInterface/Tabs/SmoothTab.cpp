#include "SmoothTab.h"

#include <QCheckBox>
#include <QGraphicsView>
#include <QGraphicsScene>
#include <QGraphicsTextItem>
#include <QWheelEvent>
#include <QTextDocument>
#include <QFont>

#include "GpuMeshCharacter.h"
#include "ui_MainWindow.h"

using namespace std;


SmoothTab::SmoothTab(Ui::MainWindow* ui,
                     const std::shared_ptr<GpuMeshCharacter>& character) :
    _ui(ui),
    _character(character),
    _currentView(nullptr),
    _currentScene(nullptr)
{
    _activeImpls = {
      {"Serial", false},
      {"Thread", true},
      {"GLSL", true}
    };

    deployTechniques();
    connect(_ui->smoothingTechniqueMenu,
            static_cast<void(QComboBox::*)(const QString&)>(&QComboBox::currentIndexChanged),
            this, &SmoothTab::techniqueChanged);

    deployImplementations();
    connect(_ui->smoothingImplementationMenu,
            static_cast<void(QComboBox::*)(const QString&)>(&QComboBox::currentIndexChanged),
            this, &SmoothTab::ImplementationChanged);

    connect(_ui->smoothMeshButton,
            static_cast<void(QPushButton::*)(bool)>(&QPushButton::clicked),
            this, &SmoothTab::smoothMesh);

    connect(_ui->smoothBenchmarkImplButton,
            static_cast<void(QPushButton::*)(bool)>(&QPushButton::clicked),
            this, &SmoothTab::benchmarkImplementations);
}

SmoothTab::~SmoothTab()
{

}

void SmoothTab::techniqueChanged(const QString&)
{
    deployImplementations();
}

void SmoothTab::ImplementationChanged(const QString& implName)
{
    _lastImpl = implName.toStdString();
}

void SmoothTab::smoothMesh()
{
    _character->smoothMesh(
        _ui->smoothingTechniqueMenu->currentText().toStdString(),
        _ui->shapeMeasureTypeMenu->currentText().toStdString(),
        _ui->smoothingImplementationMenu->currentText().toStdString(),
        _ui->smoothMinIterationSpin->value(),
        _ui->smoothMoveFactorSpin->value(),
        _ui->smoothGainThresholdSpin->value());
}

void SmoothTab::benchmarkImplementations()
{
    displayOptimizationPlot(
        _character->benchmarkSmoother(
            _ui->smoothingTechniqueMenu->currentText().toStdString(),
            _ui->shapeMeasureTypeMenu->currentText().toStdString(),
            _activeImpls,
            _ui->smoothMinIterationSpin->value(),
            _ui->smoothMoveFactorSpin->value(),
            _ui->smoothGainThresholdSpin->value())
   );
}

void SmoothTab::deployTechniques()
{
    OptionMapDetails techniques = _character->availableSmoothers();

    _ui->smoothingTechniqueMenu->clear();
    for(const auto& name : techniques.options)
        _ui->smoothingTechniqueMenu->addItem(QString(name.c_str()));
    _ui->smoothingTechniqueMenu->setCurrentText(techniques.defaultOption.c_str());
}

void SmoothTab::deployImplementations()
{
    OptionMapDetails implementations = _character->availableSmootherImplementations(
        _ui->smoothingTechniqueMenu->currentText().toStdString());


    // Fill implementation combo box
    bool lastImplFound = false;
    string lastImplCopy = _lastImpl;
    _ui->smoothingImplementationMenu->clear();
    for(const auto& name : implementations.options)
    {
        _ui->smoothingImplementationMenu->addItem(QString(name.c_str()));

        if(name == _lastImpl)
            lastImplFound = true;
    }

    if(lastImplFound)
    {
        _ui->smoothingImplementationMenu->setCurrentText(
                    lastImplCopy.c_str());
    }
    else
    {
        _ui->smoothingImplementationMenu->setCurrentText(
                    implementations.defaultOption.c_str());
    }


    // Define active implementations for benchmarking
    if(_ui->smoothActiveImplLayout->layout())
        QWidget().setLayout(_ui->smoothActiveImplLayout->layout());

    map<string, bool> newActiveImpls;
    QFormLayout* layout = new QFormLayout();
    for(const auto& name : implementations.options)
    {
        QCheckBox* check = new QCheckBox();

        int isActive = true;
        auto lastStateIt = _activeImpls.find(name);
        if(lastStateIt != _activeImpls.end())
            isActive = lastStateIt->second;

        check->setChecked(isActive);
        newActiveImpls.insert(make_pair(name, isActive));

        connect(check, &QCheckBox::stateChanged,
                [=](int state) { _activeImpls[name] = state; });

        layout->addRow(QString(name.c_str()), check);
    }
    _ui->smoothActiveImplLayout->setLayout(layout);
    _activeImpls = newActiveImpls;
}

class OptimizationPlotView : public QGraphicsView
{
public:
    OptimizationPlotView() : QGraphicsView() {}

protected:
    virtual void wheelEvent(QWheelEvent* e)
    {
        double factor = glm::clamp(1.0 - e->delta() / (8.0 * 360.0), 0.5, 1.5);
        scale(factor, factor);

        if(transform().m11() < 1.0)
        {
            resetTransform();
        }
    }
};

void SmoothTab::displayOptimizationPlot(const OptimizationPlot& plot)
{
    delete _currentView;
    delete _currentScene;

    _currentView = new OptimizationPlotView();
    _currentScene = new QGraphicsScene();

    double sceneWidth = 800;
    double sceneHeight = 600;

    double maxT = 0.0;
    double minQ = 1.0;
    double maxQ = 0.0;
    map<string, QPen> pens;
    for(const auto& keyVal : plot.curves())
    {
        const string& label = keyVal.first;

        pens.insert(make_pair(
            label,
            QPen(QColor(
                ((uchar) label[0]) % 3 * 75,
                ((uchar) label[1]) % 3 * 75,
                ((uchar) label[2]) % 3 * 75))
        ));

        for(const auto& pass : keyVal.second)
        {
            maxT = glm::max(maxT, pass.timeStamp);
            minQ = glm::min(minQ, pass.qualityMean);
            maxQ = glm::max(maxQ, pass.qualityMean);
        }
    }
    double marginQ = (maxQ - minQ) * 0.05;
    double bottomQ = minQ - marginQ;
    double topQ = maxQ + marginQ;
    double scaleQ = 1.0 / (topQ - bottomQ);

    // Min timestamp and min qualityMean
    QGraphicsTextItem* minTimeText = _currentScene->addText("0s");
    double minTimeTextLen = minTimeText->document()->size().width();
    minTimeText->setPos(-minTimeTextLen/2.0, sceneHeight);

    // Algo final quality mean
    QGraphicsTextItem* minQualityText = _currentScene->addText(QString::number(minQ));
    double minQualityTextWidth = minQualityText->document()->size().width();
    double minQualityTextHeight = minQualityText->document()->size().height();
    minQualityText->setPos(-minQualityTextWidth, sceneHeight * (1.0-marginQ*scaleQ) - minQualityTextHeight/2.0);


    double labelHeight = 10.0;
    double legendLeft = sceneWidth - 200;
    double legendTop = sceneHeight - 20.0 * (plot.curves().size() + 2);
    for(const auto& keyVal : plot.curves())
    {
        const string& label = keyVal.first;
        const OptimizationPassVect& samples = keyVal.second;

        // Asymptotes
        double totalTime = samples.back().timeStamp;
        double xAsymptote = sceneWidth * (totalTime / maxT);
        _currentScene->addLine(xAsymptote, 0, xAsymptote, sceneHeight, QPen(Qt::lightGray));
        double yAsymptote = sceneHeight * (topQ - samples.back().qualityMean) * scaleQ;
        _currentScene->addLine(0, yAsymptote, sceneWidth, yAsymptote, QPen(Qt::lightGray));

        // Algo total times
        QGraphicsTextItem* timeText = _currentScene->addText(QString::number(totalTime) + "s");
        double timeTextLen = timeText->document()->size().width();
        timeText->setPos(xAsymptote - timeTextLen/2.0, sceneHeight);
        timeText->setDefaultTextColor(pens[label].color());

        // Algo final quality mean
        double finalQuality = samples.back().qualityMean;
        QGraphicsTextItem* qualityText = _currentScene->addText(QString::number(finalQuality));
        double qualityTextHeight = qualityText->document()->size().height();
        qualityText->setPos(sceneWidth, sceneHeight * (topQ - finalQuality) * scaleQ - qualityTextHeight/2.0);
        qualityText->setDefaultTextColor(pens[label].color());

        // Legend
        double gainValue = samples.back().qualityMean - samples.front().qualityMean;
        QString gainText = " (" + ((gainValue < 0.0 ? "-" : "+") + QString::number(gainValue)) + ")";
        QGraphicsTextItem* text = _currentScene->addText(label.c_str() + gainText);
        text->setPos(legendLeft + 10.0, legendTop + labelHeight);
        text->setDefaultTextColor(pens[label].color());
        labelHeight += 20.0;
    }
    _currentScene->addRect(
        legendLeft, legendTop,
        180.0, 20.0 * (plot.curves().size()+1.3));


    // Optimization curves
    for(const auto& keyVal : plot.curves())
    {
        const string& label = keyVal.first;
        const OptimizationPassVect& samples = keyVal.second;
        for(size_t i=1; i < samples.size(); ++i)
        {
            const OptimizationPass& prevPass = samples[i-1];
            const OptimizationPass& currPass = samples[i];
            _currentScene->addLine(
                sceneWidth * prevPass.timeStamp / maxT,
                sceneHeight * (topQ - prevPass.qualityMean) * scaleQ,
                sceneWidth * currPass.timeStamp / maxT,
                sceneHeight * (topQ - currPass.qualityMean) * scaleQ,
                pens[label]);
        }
    }

    // Graphics borders
    _currentScene->addRect(0, 0, sceneWidth, sceneHeight);


    QFont titleFont;
    titleFont.setPointSize(20);
    QGraphicsTextItem* titleText = _currentScene->addText(plot.title().c_str(), titleFont);
    titleText->setPos((sceneWidth - titleText->document()->size().width())/2.0, -50.0);

    _currentView->setScene(_currentScene);
    _currentView->setDragMode(QGraphicsView::ScrollHandDrag);
    _currentView->setRenderHints(
        QPainter::Antialiasing |
        QPainter::SmoothPixmapTransform);
    _currentView->resize(
        sceneWidth + 180.0,
        sceneHeight + 100.0);
    _currentView->show();
}
