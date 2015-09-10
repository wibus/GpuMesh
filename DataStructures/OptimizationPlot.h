#ifndef GPUMESH_OPTIMIZATIONPLOT
#define GPUMESH_OPTIMIZATIONPLOT

#include <map>
#include <vector>
#include <string>


struct OptimizationPass
{
    double timeStamp;
    double minQuality;
    double qualityMean;
};

struct OptimizationImpl
{
    std::string name;
    std::map<std::string, std::string> parameters;
    std::vector<OptimizationPass> passes;
};


class OptimizationPlot
{
public:
    explicit OptimizationPlot();
    virtual ~OptimizationPlot();

    void setMeshModelName(const std::string& name);

    void setSmoothingMethodName(const std::string& name);

    void addImplementation(const OptimizationImpl& impl);

    void addMeshProperty(const std::string& name, const std::string& value);

    const std::string& meshModelName() const;
    const std::string& smoothingMethodName() const;
    const std::vector<OptimizationImpl>& implementations() const;
    const std::map<std::string, std::string>& meshProperties() const;


private:
    std::string _meshModelName;
    std::string _smoothingMethodName;
    std::vector<OptimizationImpl> _implementations;
    std::map<std::string, std::string> _meshProperties;
};

#endif // GPUMESH_OPTIMIZATIONPLOT
