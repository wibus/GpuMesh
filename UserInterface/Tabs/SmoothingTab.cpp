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
    vector<string> techniqueNames = _character->availableSmoothers();

    _ui->smoothingTechniqueMenu->clear();
    for(const auto& name : techniqueNames)
        _ui->smoothingTechniqueMenu->addItem(QString(name.c_str()));
}

void SmoothingTab::deployShapeMeasures()
{
    vector<string> evaluatorNames = _character->availableEvaluators();

    _ui->smoothShapeMeasureMenu->clear();
    for(const auto& name : evaluatorNames)
        _ui->smoothShapeMeasureMenu->addItem(QString(name.c_str()));
}

void SmoothingTab::deployImplementations()
{
    string smoother = _ui->smoothingTechniqueMenu->currentText().toStdString();
    vector<string> implementationNames = _character->availableImplementations(smoother);

    _ui->smoothingImplementationMenu->clear();
    for(const auto& name : implementationNames)
        _ui->smoothingImplementationMenu->addItem(QString(name.c_str()));
}
