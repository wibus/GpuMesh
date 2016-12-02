#ifndef GPUMESH_OPTIMIZATIONPLOT
#define GPUMESH_OPTIMIZATIONPLOT

#include <map>
#include <vector>
#include <string>

#include "NodeGroups.h"
#include "QualityHistogram.h"

typedef std::vector<std::pair<std::string, std::string>> Properties;


struct Configuration
{
    std::string samplerName;
    std::string smootherName;
    std::string implementationName;
};

struct OptimizationPass
{
    double timeStamp;
    QualityHistogram histogram;
};

struct OptimizationImpl
{
    std::string name;
    Configuration configuration;
    bool isTopologicalOperationOn;
    Properties smoothingProperties;
    std::vector<OptimizationPass> passes;

    void addSmoothingProperty(const std::string& name, const std::string& value);
};


class OptimizationPlot
{
public:
    explicit OptimizationPlot();
    virtual ~OptimizationPlot();

    void setMeshModelName(const std::string& name);

    void setNodeGroups(const NodeGroups& groups);

    void addMeshProperty(const std::string& name, const std::string& value);

    void setInitialHistogram(const QualityHistogram& histogram);

    void addImplementation(const OptimizationImpl& impl);

    const NodeGroups& nodeGroups() const;
    const std::string& meshModelName() const;
    const Properties& meshProperties() const;
    const QualityHistogram& initialHistogram() const;
    const std::vector<OptimizationImpl>& implementations() const;


private:
    NodeGroups _nodeGroups;
    Properties _meshProperties;
    std::string _meshModelName;
    QualityHistogram _initialHistogram;
    std::vector<OptimizationImpl> _implementations;
};

#endif // GPUMESH_OPTIMIZATIONPLOT
