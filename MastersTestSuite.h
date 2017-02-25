#ifndef GPUMESH_MASTERSTESTSUITE
#define GPUMESH_MASTERSTESTSUITE

#include <map>
#include <memory>
#include <vector>
#include <functional>

#include <QString>

#include "DataStructures/OptionMap.h"

class GpuMeshCharacter;


class MastersTestSuite
{
public:
    explicit MastersTestSuite(
        GpuMeshCharacter& character);

    virtual ~MastersTestSuite();


    OptionMapDetails availableTests() const;

    std::string runTests(const std::vector<std::string>& tests);


protected:
    std::string metricCost(const std::string& mesher, const std::string& model);
    std::string metricCostSphere();
    std::string metricCostHexGrid();

    std::string metricPrecision();

    std::string nodeOrder();


private:
    GpuMeshCharacter& _character;

    typedef std::function<std::string()> MastersTestFunc;
    OptionMap<MastersTestFunc> _availableMastersTests;

    std::map<std::string, QString> _translateSampling;
    std::map<std::string, QString> _translateImplementations;
};

#endif // GPUMESH_MASTERSTESTSUITE
