#include "MastersTestsDialog.h"
#include "ui_MastersTestsDialog.h"

#include <QVBoxLayout>
#include <QCheckBox>

#include "GpuMeshCharacter.h"
#include "DataStructures/OptionMap.h"

using namespace std;


MastersTestsDialog::MastersTestsDialog(
        const shared_ptr<GpuMeshCharacter> &character) :
    _ui(new Ui::MastersTestsDialog)
{
    _ui->setupUi(this);

    const auto& options = character->availableMastersTests().options;

    QVBoxLayout* layout = new QVBoxLayout();

    for(const auto& t : options)
    {
        QCheckBox* check = new QCheckBox(t.c_str());
        _testChecks.push_back(check);
        layout->addWidget(check);
        check->setChecked(false);
    }

    layout->addStretch();

    _ui->mastersTestsGroup->setLayout(layout);


    connect(_ui->checkAllButton,
            static_cast<void(QPushButton::*)(bool)>(&QPushButton::clicked),
            this, &MastersTestsDialog::checkAll);
}

MastersTestsDialog::~MastersTestsDialog()
{
    delete _ui;
}

void MastersTestsDialog::checkAll()
{
    for(QCheckBox* c : _testChecks)
    {
        c->setChecked(true);
    }
}

vector<string> MastersTestsDialog::tests() const
{
    vector<string> names;

    for(QCheckBox* c : _testChecks)
    {
        if(c->isChecked())
        {
            names.push_back(c->text().toStdString());
        }
    }

    return names;
}
