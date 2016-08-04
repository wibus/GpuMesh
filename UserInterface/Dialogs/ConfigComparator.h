#ifndef GPUMESH_CONFIGCOMPARATOR
#define GPUMESH_CONFIGCOMPARATOR

#include <memory>

#include <QDialog>

struct Configuration;
class GpuMeshCharacter;

namespace Ui
{
    class ConfigComparator;
}

class ConfigComparator : public QDialog
{
    Q_OBJECT

public:
    explicit ConfigComparator(
        const std::shared_ptr<GpuMeshCharacter>& character);
    ~ConfigComparator();

    const std::vector<Configuration> configurations() const;

protected slots:
    virtual void addToList();
    virtual void clearList();
    virtual void defaultList();

    virtual void buildConfigurations();

private:
    Ui::ConfigComparator* _ui;
    std::shared_ptr<GpuMeshCharacter> _character;

    std::vector<QString> _allSamplers;
    std::vector<QString> _allSmoothers;
    std::vector<QString> _allImplementations;

    std::vector<Configuration> _configurations;
};

#endif // GPUMESH_CONFIGCOMPARATOR
