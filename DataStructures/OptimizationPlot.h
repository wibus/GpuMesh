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

typedef std::vector<OptimizationPass> OptimizationPassVect;
typedef std::map<std::string, OptimizationPassVect> OptimizationCurvesMap;


class OptimizationPlot
{
public:
    explicit OptimizationPlot(const std::string& title);
    virtual ~OptimizationPlot();

    void clear();

    void addCurve(const std::string label, const OptimizationPassVect& passes);

    const std::string& title() const;
    const OptimizationCurvesMap& curves() const;


private:
    std::string _title;
    OptimizationCurvesMap _curves;
};

#endif // GPUMESH_OPTIMIZATIONPLOT
