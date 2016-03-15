#ifndef GPUMESH_OPTIMIZATIONPLOT
#define GPUMESH_OPTIMIZATIONPLOT

#include <map>
#include <vector>
#include <string>

#include "QualityHistogram.h"

typedef std::vector<std::pair<std::string, std::string>> Properties;

struct OptimizationPass
{
    double timeStamp;
    QualityHistogram histogram;
};

struct OptimizationImpl
{
    std::string name;
    std::vector<OptimizationPass> passes;
};


class OptimizationPlot
{
public:
    explicit OptimizationPlot();
    virtual ~OptimizationPlot();

    void setMeshModelName(const std::string& name);

    void setSmoothingMethodName(const std::string& name);

    void addMeshProperty(const std::string& name, const std::string& value);

    void addSmoothingProperty(const std::string& name, const std::string& value);

    void setInitialHistogram(const QualityHistogram& histogram);

    void addImplementation(const OptimizationImpl& impl);

    const std::string& meshModelName() const;
    const std::string& smoothingMethodName() const;
    const Properties& meshProperties() const;
    const Properties& smoothingProperties() const;
    const QualityHistogram& initialHistogram() const;
    const std::vector<OptimizationImpl>& implementations() const;


private:
    std::string _meshModelName;
    std::string _smoothingMethodName;
    Properties _meshProperties;
    Properties _smoothingProperties;
    QualityHistogram _initialHistogram;
    std::vector<OptimizationImpl> _implementations;
};

#endif // GPUMESH_OPTIMIZATIONPLOT
