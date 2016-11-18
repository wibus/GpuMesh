#include "ConfigComparator.h"
#include "ui_ConfigComparator.h"

#include "GpuMeshCharacter.h"
#include "DataStructures/OptimizationPlot.h"

const QString ALL_OPTIONS = "[ALL]";

ConfigComparator::ConfigComparator(
        const std::shared_ptr<GpuMeshCharacter>& character) :
    _ui(new Ui::ConfigComparator),
    _character(character)
{
    _ui->setupUi(this);


    _ui->samplerList->addItem(new QListWidgetItem(ALL_OPTIONS));
    for(auto opt : _character->availableSamplers().options)
    {
        _allSamplers.push_back(QString(opt.c_str()));
        _ui->samplerList->addItem(new QListWidgetItem(opt.c_str()));
    }

    _ui->smootherList->addItem(new QListWidgetItem(ALL_OPTIONS));
    for(auto opt : _character->availableSmoothers().options)
    {
        _allSmoothers.push_back(QString(opt.c_str()));
        _ui->smootherList->addItem(new QListWidgetItem(opt.c_str()));
    }

    _allImplementations.push_back("Serial");
    _allImplementations.push_back("Thread");
    _allImplementations.push_back("GLSL");
    _allImplementations.push_back("CUDA");

    _ui->implementationList->addItem(new QListWidgetItem(ALL_OPTIONS));
    for(auto opt : _allImplementations)
    {
        _ui->implementationList->addItem(new QListWidgetItem(opt));
    }


    _ui->implementationTable->setColumnCount(3);
    _ui->implementationTable->setHorizontalHeaderLabels(
        {"Sampler", "Smoother", "Implementation"});


    connect(_ui->addToListButton, &QPushButton::clicked,
            this, &ConfigComparator::addToList);

    connect(_ui->resetButton, &QPushButton::clicked,
            this, &ConfigComparator::clearList);

    connect(_ui->defaultButton, &QPushButton::clicked,
            this, &ConfigComparator::defaultList);

    connect(this, &ConfigComparator::accepted,
            this, &ConfigComparator::buildConfigurations);
}

ConfigComparator::~ConfigComparator()
{
    delete _ui;
}

const std::vector<Configuration> ConfigComparator::configurations() const
{
    return _configurations;
}

void ConfigComparator::addToList()
{
    if(_ui->samplerList->selectedItems().empty() ||
       _ui->smootherList->selectedItems().empty() ||
       _ui->implementationList->selectedItems().empty())
    {
        return;
    }

    std::vector<QString> samplerNames;
    auto samplerList = _ui->samplerList->selectedItems();
    for(QListWidgetItem* sampler : samplerList)
    {
        if(sampler->text() == ALL_OPTIONS)
        {
            samplerNames = _allSamplers;
            break;
        }
        else
        {
            samplerNames.push_back(sampler->text());
        }
    }

    std::vector<QString> smootherNames;
    auto smootherList = _ui->smootherList->selectedItems();
    for(QListWidgetItem* smoother : smootherList)
    {
        if(smoother->text() == ALL_OPTIONS)
        {
            smootherNames = _allSmoothers;
            break;
        }
        else
        {
            smootherNames.push_back(smoother->text());
        }
    }

    std::vector<QString> implementationNames;
    auto implementationList = _ui->implementationList->selectedItems();
    for(QListWidgetItem* implementation : implementationList)
    {
        if(implementation->text() == ALL_OPTIONS)
        {
            implementationNames = _allImplementations;
            break;
        }
        else
        {
            implementationNames.push_back(implementation->text());
        }
    }

    for(const QString& sampler : samplerNames)
    {
        for(const QString& smoother : smootherNames)
        {
            for(const QString& implementation : implementationNames)
            {
                int row = _ui->implementationTable->rowCount();
                _ui->implementationTable->setRowCount(row+1);

                _ui->implementationTable->setItem(row, 0,
                    new QTableWidgetItem(sampler));

                _ui->implementationTable->setItem(row, 1,
                    new QTableWidgetItem(smoother));

                _ui->implementationTable->setItem(row, 2,
                    new QTableWidgetItem(implementation));
            }
        }
    }
}

void ConfigComparator::clearList()
{
    _ui->implementationTable->setRowCount(0);
}

void ConfigComparator::defaultList()
{
    _ui->implementationTable->setRowCount(4);

    _ui->implementationTable->setItem(0, 0, new QTableWidgetItem("Local"));
    _ui->implementationTable->setItem(0, 1, new QTableWidgetItem("Gradient Descent"));
    _ui->implementationTable->setItem(0, 2, new QTableWidgetItem("Thread"));

    _ui->implementationTable->setItem(1, 0, new QTableWidgetItem("Local"));
    _ui->implementationTable->setItem(1, 1, new QTableWidgetItem("Nelder-Mead"));
    _ui->implementationTable->setItem(1, 2, new QTableWidgetItem("Thread"));

    _ui->implementationTable->setItem(2, 0, new QTableWidgetItem("Texture"));
    _ui->implementationTable->setItem(2, 1, new QTableWidgetItem("Spawn Search"));
    _ui->implementationTable->setItem(2, 2, new QTableWidgetItem("GLSL"));

    _ui->implementationTable->setItem(3, 0, new QTableWidgetItem("Texture"));
    _ui->implementationTable->setItem(3, 1, new QTableWidgetItem("Spawn Search"));
    _ui->implementationTable->setItem(3, 2, new QTableWidgetItem("CUDA"));
}

void ConfigComparator::buildConfigurations()
{
    int rowCount = _ui->implementationTable->rowCount();

    _configurations.clear();
    for(int i=0; i < rowCount; ++i)
    {
        QTableWidgetItem* sampler = _ui->implementationTable->item(i, 0);
        QTableWidgetItem* smoother = _ui->implementationTable->item(i, 1);
        QTableWidgetItem* implementation = _ui->implementationTable->item(i, 2);

        _configurations.push_back(Configuration{
              sampler->text().toStdString(),
              smoother->text().toStdString(),
              implementation->text().toStdString()
        });
    }
}
