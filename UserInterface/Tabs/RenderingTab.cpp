#include "RenderingTab.h"

#include "GpuMeshCharacter.h"
#include "ui_MainWindow.h"

using namespace std;


RenderingTab::RenderingTab(Ui::MainWindow* ui,
                           const std::shared_ptr<GpuMeshCharacter>& character) :
    _ui(ui),
    _character(character)
{
    // Defining renderers
    deployRenderTypes();
    connect(_ui->renderingTypeMenu,
            static_cast<void(QComboBox::*)(const QString&)>(&QComboBox::currentIndexChanged),
            this, &RenderingTab::renderTypeChanged);


    // Defining shadings
    deployShadings();
    connect(_ui->shadingMenu,
            static_cast<void(QComboBox::*)(const QString&)>(&QComboBox::currentIndexChanged),
            this, &RenderingTab::shadingChanged);
    _ui->shadingMenu->setCurrentText("Wireframe");


    // Defining visual quality measures
    deployShapeMeasures();
    connect(_ui->visualShapeMeasureMenu,
            static_cast<void(QComboBox::*)(const QString&)>(&QComboBox::currentIndexChanged),
            this, &RenderingTab::shapeMeasureChanged);


    // Define virtual cut plane usage
    _character->useVirtualCutPlane(_ui->virtualCutPlaneCheck->isChecked());
    connect(_ui->virtualCutPlaneCheck, &QCheckBox::stateChanged,
            this, &RenderingTab::useVirtualCutPlane);
}

RenderingTab::~RenderingTab()
{

}

void RenderingTab::deployRenderTypes()
{
    OptionMapDetails renderers = _character->availableRenderers();

    _ui->renderingTypeMenu->clear();
    for(const auto& name : renderers.options)
        _ui->renderingTypeMenu->addItem(QString(name.c_str()));
    _ui->renderingTypeMenu->setCurrentText(renderers.defaultOption.c_str());

    // Renderer must be set here, or no renderer will be installed
    _character->useRenderer(renderers.defaultOption);
}

void RenderingTab::deployShadings()
{
    OptionMapDetails shadings = _character->availableShadings();

    _ui->shadingMenu->clear();
    for(const auto& name : shadings.options)
        _ui->shadingMenu->addItem(QString(name.c_str()));
    _ui->shadingMenu->setCurrentText(shadings.defaultOption.c_str());

    // Shading must be set here, or no shading will be installed
    _character->useShading(shadings.defaultOption);
}

void RenderingTab::deployShapeMeasures()
{
    OptionMapDetails evaluators = _character->availableEvaluators();

    _ui->visualShapeMeasureMenu->clear();
    for(const auto& name : evaluators.options)
        _ui->visualShapeMeasureMenu->addItem(QString(name.c_str()));
    _ui->visualShapeMeasureMenu->setCurrentText(evaluators.defaultOption.c_str());

    // Shading must be set here, or no shading will be installed
    _character->displayQuality(evaluators.defaultOption);
}

void RenderingTab::renderTypeChanged(QString text)
{
    _character->useRenderer(text.toStdString());

    deployShadings();
    _character->useVirtualCutPlane(
        _ui->virtualCutPlaneCheck->isChecked());
}

void RenderingTab::shadingChanged(QString text)
{
    if(text.length() != 0)
    {
        _character->useShading(text.toStdString());
    }
}

void RenderingTab::shapeMeasureChanged(QString text)
{
    _character->displayQuality(text.toStdString());
}

void RenderingTab::useVirtualCutPlane(bool checked)
{
    _character->useVirtualCutPlane(checked);
}
