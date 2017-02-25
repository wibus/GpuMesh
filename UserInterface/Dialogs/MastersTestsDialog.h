#ifndef GPUMESH_MASTERSTESTSDIALOG
#define GPUMESH_MASTERSTESTSDIALOG

#include <memory>

#include <QDialog>
#include <QCheckBox>

namespace Ui {
class MastersTestsDialog;
}

class GpuMeshCharacter;


class MastersTestsDialog : public QDialog
{
    Q_OBJECT

public:
    explicit MastersTestsDialog(
        const std::shared_ptr<GpuMeshCharacter>& character);
    ~MastersTestsDialog();

    std::vector<std::string> tests() const;

protected slots:
    virtual void checkAll();

private:
    Ui::MastersTestsDialog *_ui;
    std::vector<QCheckBox*> _testChecks;
};

#endif // GPUMESH_MASTERSTESTSDIALOG
