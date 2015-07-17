#include "SmoothingTab.h"

#include "GpuMeshCharacter.h"
#include "ui_MainWindow.h"

using namespace std;


SmoothingTab::SmoothingTab(Ui::MainWindow* ui,
                                 const std::shared_ptr<GpuMeshCharacter>& character) :
    _ui(ui),
    _character(character)
{
    deployTechniques();
    connect(_ui->smoothingTechniqueMenu,
            static_cast<void(QComboBox::*)(const QString&)>(&QComboBox::currentIndexChanged),
            this, &SmoothingTab::techniqueChanged);

    deployShapeMeasures();

    deployImplementations();

    connect(_ui->smoothMeshButton,
            static_cast<void(QPushButton::*)(bool)>(&QPushButton::clicked),
            this, &SmoothingTab::smoothMesh);
}

SmoothingTab::~SmoothingTab()
{

}

void SmoothingTab::techniqueChanged(const QString&)
{
    deployImplementations();
}

void SmoothingTab::smoothMesh()
{
    _character->smoothMesh(
                _ui->smoothingTechniqueMenu->currentText().toStdString(),
                _ui->smoothShapeMeasureMenu->currentText().toStdString(),
                _ui->smoothingImplementationMenu->currentText().toStdString(),
                _ui->minIterationSpin->value(),
                _ui->moveFactorSpin->value(),
                _ui->gainThresholdSpin->value());
}


void SmoothingTab::deployTechniques()
{
    OptionMapDetails techniques = _character->availableSmoothers();

    _ui->smoothingTechniqueMenu->clear();
    for(const auto& name : techniques.options)
        _ui->smoothingTechniqueMenu->addItem(QString(name.c_str()));
    _ui->smoothingTechniqueMenu->setCurrentText(techniques.defaultOption.c_str());
}

void SmoothingTab::deployShapeMeasures()
{
    OptionMapDetails evaluators = _character->availableEvaluators();

    _ui->smoothShapeMeasureMenu->clear();
    for(const auto& name : evaluators.options)
        _ui->smoothShapeMeasureMenu->addItem(QString(name.c_str()));
    _ui->smoothShapeMeasureMenu->setCurrentText(evaluators.defaultOption.c_str());
}

void SmoothingTab::deployImplementations()
{
    OptionMapDetails implementations = _character->availableImplementations(
        _ui->smoothingTechniqueMenu->currentText().toStdString());

    _ui->smoothingImplementationMenu->clear();
    for(const auto& name : implementations.options)
        _ui->smoothingImplementationMenu->addItem(QString(name.c_str()));
    _ui->smoothingImplementationMenu->setCurrentText(implementations.defaultOption.c_str());
}
