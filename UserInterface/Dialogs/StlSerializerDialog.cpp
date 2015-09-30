#include "StlSerializerDialog.h"
#include "ui_StlSerializerDialog.h"


StlSerializerDialog::StlSerializerDialog(QWidget *parent) :
    QDialog(parent),
    _ui(new Ui::StlSerializerDialog)
{
    _ui->setupUi(this);

    connect(_ui->asciiRadioBtn, &QRadioButton::toggled,
            this, &StlSerializerDialog::asciiChecked);

    connect(_ui->binaryRadioBtn, &QRadioButton::toggled,
            this, &StlSerializerDialog::binaryChecked);
}

StlSerializerDialog::~StlSerializerDialog()
{
    delete _ui;
}

bool StlSerializerDialog::binaryFormat() const
{
    return _ui->binaryRadioBtn->isChecked();
}

bool StlSerializerDialog::embedQuality() const
{
    return _ui->embedQualityCheck->isChecked();
}

void StlSerializerDialog::asciiChecked(bool isChecked)
{
    if(isChecked)
    {
        _ui->embedQualityCheck->setEnabled(false);
        _ui->embedQualityCheck->setChecked(false);
    }
}

void StlSerializerDialog::binaryChecked(bool isChecked)
{
    if(isChecked)
    {
        _ui->embedQualityCheck->setEnabled(true);
    }
}
