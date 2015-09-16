#include "SmoothTab.h"

#include <QCheckBox>
#include <QGraphicsView>
#include <QGraphicsScene>
#include <QGraphicsTextItem>
#include <QWheelEvent>
#include <QTextDocument>
#include <QFont>

#include <CellarWorkbench/Image/Image.h>
#include <CellarWorkbench/GL/GlToolkit.h>

#include "../SmoothingReport.h"
#include "GpuMeshCharacter.h"
#include "ui_MainWindow.h"

using namespace std;
using namespace cellar;


SmoothTab::SmoothTab(Ui::MainWindow* ui,
                     const std::shared_ptr<GpuMeshCharacter>& character) :
    _ui(ui),
    _character(character),
    _reportWidget(nullptr)
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
            this, &SmoothTab::implementationChanged);

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

void SmoothTab::implementationChanged(const QString& implName)
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
    const string REPORT_PATH = "Reports/Report.pdf";
    const QString preShootPath = "Reports/PreSmoothingShot.png";
    const QString postShootPath = "Reports/PostSmoothingShot.png";

    Image preSmoothingShot;
    GlToolkit::takeFramebufferShot(preSmoothingShot);
    preSmoothingShot.save(preShootPath.toStdString());

    OptimizationPlot plot =
        _character->benchmarkSmoother(
            _ui->smoothingTechniqueMenu->currentText().toStdString(),
            _ui->shapeMeasureTypeMenu->currentText().toStdString(),
            _activeImpls,
            _ui->smoothMinIterationSpin->value(),
            _ui->smoothMoveFactorSpin->value(),
            _ui->smoothGainThresholdSpin->value());

    QApplication::processEvents();

    Image postSmoothingShot;
    GlToolkit::takeFramebufferShot(postSmoothingShot);
    postSmoothingShot.save(postShootPath.toStdString());

    SmoothingReport report;
    report.setPreSmoothingShot(QImage(preShootPath));
    report.setPostSmoothingShot(QImage(postShootPath));
    report.setOptimizationPlot(plot);
    report.save(REPORT_PATH);

    delete _reportWidget;
    _reportWidget = new QTextEdit();
    _reportWidget->resize(1000, 800);
    report.display(*_reportWidget);
    _reportWidget->show();
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
