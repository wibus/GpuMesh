#include "EvaluateTab.h"

#include "GpuMeshCharacter.h"
#include "ui_MainWindow.h"

using namespace std;


EvaluateTab::EvaluateTab(Ui::MainWindow* ui,
                         const std::shared_ptr<GpuMeshCharacter>& character) :
    _ui(ui),
    _character(character)
{
    deployShapeMeasures();
    connect(_ui->shapeMeasureTypeMenu,
            static_cast<void(QComboBox::*)(const QString&)>(&QComboBox::currentIndexChanged),
            this, &EvaluateTab::shapeMeasureTypeChanged);

    deployImplementations();
    connect(_ui->shapeMeasureImplMenu,
            static_cast<void(QComboBox::*)(const QString&)>(&QComboBox::currentIndexChanged),
            this, &EvaluateTab::ImplementationChanged);

    connect(_ui->evaluateMesh,
            static_cast<void(QPushButton::*)(bool)>(&QPushButton::clicked),
            this, &EvaluateTab::evaluateMesh);

    connect(_ui->evaluateBenchmarkImplButton,
            static_cast<void(QPushButton::*)(bool)>(&QPushButton::clicked),
            this, &EvaluateTab::benchmarkImplementations);


    connect(_ui->enableAnisotropyCheck, &QCheckBox::toggled,
            this, &EvaluateTab::enableAnisotropy);

    deployDiscretizations();
    connect(_ui->discretizationTypeMenu,
            static_cast<void(QComboBox::*)(const QString&)>(&QComboBox::currentIndexChanged),
            this, &EvaluateTab::discretizationTypeChanged);

    _character->useDiscretizationDensity(
        _ui->discretizetionDensitySpin->value());
    connect(_ui->discretizetionDensitySpin,
            static_cast<void(QSpinBox::*)(int)>(&QSpinBox::valueChanged),
            this, &EvaluateTab::discretizationDensityChanged);

    _character->displayDiscretizationMesh(
        _ui->discretizationDisplayCheck->isChecked());
    connect(_ui->discretizationDisplayCheck, &QCheckBox::toggled,
            this, &EvaluateTab::displayDicretizationToggled);

    enableAnisotropy(_ui->enableAnisotropyCheck->isChecked());
}

EvaluateTab::~EvaluateTab()
{

}

void EvaluateTab::shapeMeasureTypeChanged(const QString& type)
{
    deployImplementations();
    _character->useEvaluator(type.toStdString());
}

void EvaluateTab::ImplementationChanged(const QString& implName)
{
    _lastImpl = implName.toStdString();
}

void EvaluateTab::evaluateMesh()
{
    _character->evaluateMesh(
        _ui->shapeMeasureTypeMenu->currentText().toStdString(),
        _ui->shapeMeasureImplMenu->currentText().toStdString());
}

void EvaluateTab::benchmarkImplementations()
{
    _character->benchmarkEvaluator(
        _ui->shapeMeasureTypeMenu->currentText().toStdString(),
        _cycleCounts);
}

void EvaluateTab::enableAnisotropy(bool enabled)
{
    _ui->discretizationGroup->setEnabled(enabled);
    if(enabled)
    {
        _character->useDiscretizer(
            _ui->discretizationTypeMenu->currentText().toStdString());
        _character->displayDiscretizationMesh(
            _ui->discretizationDisplayCheck->isChecked());
    }
    else
    {
        _character->disableAnisotropy();
    }
}

void EvaluateTab::discretizationTypeChanged(const QString& type)
{
    _character->useDiscretizer(type.toStdString());
}

void EvaluateTab::discretizationDensityChanged(int unused)
{
    _character->useDiscretizationDensity(
        _ui->discretizetionDensitySpin->value());
}

void EvaluateTab::displayDicretizationToggled(bool display)
{
    _character->displayDiscretizationMesh(display);
}

void EvaluateTab::deployShapeMeasures()
{
    OptionMapDetails evaluators = _character->availableEvaluators();

    _ui->shapeMeasureTypeMenu->clear();
    for(const auto& name : evaluators.options)
        _ui->shapeMeasureTypeMenu->addItem(QString(name.c_str()));
    _ui->shapeMeasureTypeMenu->setCurrentText(evaluators.defaultOption.c_str());

    _character->useEvaluator(evaluators.defaultOption);
}

void EvaluateTab::deployImplementations()
{
    OptionMapDetails implementations = _character->availableEvaluatorImplementations(
        _ui->shapeMeasureTypeMenu->currentText().toStdString());


    // Fill implementation combo box
    bool lastImplFound = false;
    string lastImplCopy = _lastImpl;
    _ui->shapeMeasureImplMenu->clear();
    for(const auto& name : implementations.options)
    {
        _ui->shapeMeasureImplMenu->addItem(QString(name.c_str()));

        if(name == _lastImpl)
            lastImplFound = true;
    }

    if(lastImplFound)
    {
        _ui->shapeMeasureImplMenu->setCurrentText(
                    lastImplCopy.c_str());
    }
    else
    {
        _ui->shapeMeasureImplMenu->setCurrentText(
                    implementations.defaultOption.c_str());
    }


    // Define cycle counts for each implementations
    if(_ui->evaluateImplCycleCountsLayout->layout())
        QWidget().setLayout(_ui->evaluateImplCycleCountsLayout->layout());

    map<string, int> newCycleCounts;
    QFormLayout* layout = new QFormLayout();
    for(const auto& name : implementations.options)
    {
        QSpinBox* spin = new QSpinBox();

        int cycleCount = 5;
        auto lastCountIt = _cycleCounts.find(name);
        if(lastCountIt != _cycleCounts.end())
            cycleCount = lastCountIt->second;

        spin->setMinimum(0);
        spin->setMaximum(100);
        spin->setValue(cycleCount);
        newCycleCounts.insert(make_pair(name, cycleCount));

        connect(spin, static_cast<void(QSpinBox::*)(int)>(&QSpinBox::valueChanged),
                [=](int count) { _cycleCounts[name] = count; });

        layout->addRow(QString(name.c_str()), spin);
    }
    _ui->evaluateImplCycleCountsLayout->setLayout(layout);
    _cycleCounts = newCycleCounts;
}

void EvaluateTab::deployDiscretizations()
{
    OptionMapDetails discretizers = _character->availableDiscretizers();

    _ui->discretizationTypeMenu->clear();
    for(const auto& name : discretizers.options)
        _ui->discretizationTypeMenu->addItem(QString(name.c_str()));
    _ui->discretizationTypeMenu->setCurrentText(discretizers.defaultOption.c_str());

    _character->useDiscretizer(discretizers.defaultOption);
}
