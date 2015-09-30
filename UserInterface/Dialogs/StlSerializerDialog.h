#ifndef GPUMESH_STLSERIALIZERDIALOG
#define GPUMESH_STLSERIALIZERDIALOG

#include <QDialog>

namespace Ui
{
    class StlSerializerDialog;
}


class StlSerializerDialog : public QDialog
{
    Q_OBJECT

public:
    explicit StlSerializerDialog(QWidget *parent = 0);
    virtual ~StlSerializerDialog();

    bool binaryFormat() const;
    bool embedQuality() const;


protected slots:
    virtual void asciiChecked(bool isChecked);
    virtual void binaryChecked(bool isChecked);

private:
    Ui::StlSerializerDialog *_ui;
};

#endif // GPUMESH_STLSERIALIZERDIALOG
