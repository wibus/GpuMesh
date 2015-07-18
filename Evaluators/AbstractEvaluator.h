#ifndef GPUMESH_ABSTRACTEVALUATOR
#define GPUMESH_ABSTRACTEVALUATOR

#include <CellarWorkbench/GL/GlProgram.h>

#include "DataStructures/Mesh.h"


class AbstractEvaluator
{
public:
    AbstractEvaluator(const std::string& shapeMeasuresShader);
    virtual ~AbstractEvaluator();

    virtual double tetQuality(const Mesh& mesh, const MeshTet& tet) const;
    virtual double tetQuality(const glm::dvec3 verts[]) const = 0;

    virtual double priQuality(const Mesh& mesh, const MeshPri& pri) const;
    virtual double priQuality(const glm::dvec3 verts[]) const = 0;

    virtual double hexQuality(const Mesh& mesh, const MeshHex& hex) const;
    virtual double hexQuality(const glm::dvec3 verts[]) const = 0;


    virtual void evaluateCpuMeshQuality(
            const Mesh& mesh,
            double& minQuality,
            double& qualityMean);

    virtual void evaluateGpuMeshQuality(
            const Mesh& mesh,
            double& minQuality,
            double& qualityMean);

    virtual std::string shapeMeasureShader() const;


    virtual bool assessMeasureValidy();

    virtual void gpuSpin(Mesh& mesh, size_t cycleCount);

    virtual void cpuSpin(Mesh& mesh, size_t cycleCount);


protected:

    virtual void initializeProgram(const Mesh& mesh);

    bool _initialized;
    bool _computeSimultaneously;
    std::string _shapeMeasuresShader;
    cellar::GlProgram _simultaneousProgram;
    cellar::GlProgram _tetProgram;
    cellar::GlProgram _priProgram;
    cellar::GlProgram _hexProgram;
    GLuint _qualSsbo;

    static const size_t WORKGROUP_SIZE;
    static const size_t POLYHEDRON_TYPE_COUNT;
    static const size_t MAX_GROUP_PARTICIPANTS;

    static const double VALIDITY_EPSILON;
    static const double MAX_INTEGER_VALUE;
    static const double MIN_QUALITY_PRECISION_DENOM;
    static const double MAX_QUALITY_VALUE;
};

#endif // GPUMESH_ABSTRACTEVALUATOR
