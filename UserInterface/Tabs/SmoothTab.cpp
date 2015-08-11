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

    double maxX = 0.0;
    for(const auto& keyVal : plot.curves())
    {
        for(const auto& pass : keyVal.second)
        {
            maxX = glm::max(maxX, pass.timeStamp);
        }
    }

    double legendLeft = sceneWidth - 200;
    double legendTop = sceneHeight - 20.0 * (plot.curves().size() + 2);
    double labelHeight = 10.0;
    for(const auto& keyVal : plot.curves())
    {
        const string& label = keyVal.first;
        const OptimizationPassVect& samples = keyVal.second;

        QPen pen(QColor(
            ((uchar) label[0]) % 3 * 75,
            ((uchar) label[1]) % 3 * 75,
            ((uchar) label[2]) % 3 * 75));

        // Asymptotes
        double totalTime = samples.back().timeStamp;
        double xAsymptote = sceneWidth * (totalTime / maxX);
        _currentScene->addLine(xAsymptote, 0, xAsymptote, sceneHeight, QPen(Qt::lightGray));
        double yAsymptote = sceneHeight * (1.0 - samples.back().qualityMean);
        _currentScene->addLine(0, yAsymptote, sceneWidth, yAsymptote, QPen(Qt::lightGray));

        // Algo total times
        QGraphicsTextItem* timeText = _currentScene->addText(QString::number(totalTime) + "s");
        timeText->adjustSize();
        timeText->setPos(xAsymptote - timeText->textWidth()/2.0, sceneHeight);
        timeText->setDefaultTextColor(pen.color());

        // Legend
        double gainValue = samples.back().qualityMean - samples.front().qualityMean;
        QString gainText = " (" + ((gainValue < 0.0 ? "-" : "+") + QString::number(gainValue)) + ")";
        QGraphicsTextItem* text = _currentScene->addText(label.c_str() + gainText);
        text->setPos(legendLeft + 10.0, legendTop + labelHeight);
        text->setDefaultTextColor(pen.color());
        labelHeight += 20.0;
    }
    _currentScene->addRect(
        legendLeft, legendTop,
        180.0, 20.0 * (plot.curves().size()+1.3));

    for(const auto& keyVal : plot.curves())
    {
        const string& label = keyVal.first;
        const OptimizationPassVect& samples = keyVal.second;

        QPen pen(QColor(
            ((uchar) label[0]) % 3 * 75,
            ((uchar) label[1]) % 3 * 75,
            ((uchar) label[2]) % 3 * 75));

        for(size_t i=1; i < samples.size(); ++i)
        {
            const OptimizationPass& prevPass = samples[i-1];
            const OptimizationPass& currPass = samples[i];
            _currentScene->addLine(
                sceneWidth * prevPass.timeStamp / maxX,
                sceneHeight * (1.0 - prevPass.qualityMean),
                sceneWidth * currPass.timeStamp / maxX,
                sceneHeight * (1.0 - currPass.qualityMean),
                pen);
        }
    }
    _currentScene->addRect(0, 0, sceneWidth, sceneHeight);


    QGraphicsTextItem* titleText = _currentScene->addText(plot.title().c_str());
    titleText->setPos((sceneWidth - titleText->document()->size().width())/2.0, -30.0);

    _currentView->setScene(_currentScene);
    _currentView->setDragMode(QGraphicsView::ScrollHandDrag);
    _currentView->setRenderHints(
        QPainter::Antialiasing |
        QPainter::SmoothPixmapTransform);
    _currentView->resize(
        sceneWidth + 50.0,
        sceneHeight + 60.0);
    _currentView->show();
}
