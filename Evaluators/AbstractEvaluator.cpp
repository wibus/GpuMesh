#include "AbstractEvaluator.h"

#include <chrono>
#include <sstream>

#include <CellarWorkbench/Misc/Log.h>

#include "DataStructures/GpuMesh.h"

using namespace cellar;
using namespace std;



struct GpuQual
{
    GLuint min;
    GLuint mean;

    inline GpuQual() : min(UINT32_MAX), mean(0) {}
};


const size_t AbstractEvaluator::WORKGROUP_SIZE = 256;
const size_t AbstractEvaluator::POLYHEDRON_TYPE_COUNT = 3;
const size_t AbstractEvaluator::MAX_GROUP_PARTICIPANTS =
        AbstractEvaluator::WORKGROUP_SIZE *
        AbstractEvaluator::POLYHEDRON_TYPE_COUNT;

const double AbstractEvaluator::VALIDITY_EPSILON = 1e-6;
const double AbstractEvaluator::MAX_INTEGER_VALUE = 4294967296.0;
const double AbstractEvaluator::MIN_QUALITY_PRECISION_DENOM = 4096.0;

const double AbstractEvaluator::MAX_QUALITY_VALUE =
        AbstractEvaluator::MAX_INTEGER_VALUE /
        (double) AbstractEvaluator::MAX_GROUP_PARTICIPANTS;

AbstractEvaluator::AbstractEvaluator(const std::string& shapeMeasuresShader) :
    _initialized(false),
    _computeSimultaneously(true),
    _shapeMeasuresShader(shapeMeasuresShader),
    _qualSsbo(0)
{
    static_assert(AbstractEvaluator::MAX_QUALITY_VALUE >=
                  AbstractEvaluator::MIN_QUALITY_PRECISION_DENOM,
                  "Shape measure on GPU may not be suffciently precise \
                   given this workgroup size.");
}

AbstractEvaluator::~AbstractEvaluator()
{
    glDeleteBuffers(1, &_qualSsbo);
}

double AbstractEvaluator::tetQuality(const Mesh& mesh, const MeshTet& tet) const
{
    const glm::dvec3 verts[] = {
        mesh.vert[tet.v[0]],
        mesh.vert[tet.v[1]],
        mesh.vert[tet.v[2]],
        mesh.vert[tet.v[3]],
    };

    return tetQuality(verts);
}

double AbstractEvaluator::priQuality(const Mesh& mesh, const MeshPri& pri) const
{
    const glm::dvec3 verts[] = {
        mesh.vert[pri.v[0]],
        mesh.vert[pri.v[1]],
        mesh.vert[pri.v[2]],
        mesh.vert[pri.v[3]],
        mesh.vert[pri.v[4]],
        mesh.vert[pri.v[5]]
    };

    return priQuality(verts);
}

double AbstractEvaluator::hexQuality(const Mesh& mesh, const MeshHex& hex) const
{
    const glm::dvec3 verts[] = {
        mesh.vert[hex.v[0]],
        mesh.vert[hex.v[1]],
        mesh.vert[hex.v[2]],
        mesh.vert[hex.v[3]],
        mesh.vert[hex.v[4]],
        mesh.vert[hex.v[5]],
        mesh.vert[hex.v[6]],
        mesh.vert[hex.v[7]]
    };

    return hexQuality(verts);
}

bool AbstractEvaluator::assessMeasureValidy()
{
    Mesh mesh;
    mesh.vert.push_back(glm::dvec3(0, 0, 0));
    mesh.vert.push_back(glm::dvec3(1, 0, 0));
    mesh.vert.push_back(glm::dvec3(0.5, sqrt(3.0)/6, sqrt(2.0/3)));
    mesh.vert.push_back(glm::dvec3(0.5, sqrt(3.0)/2, 0));

    mesh.vert.push_back(glm::dvec3(0, 0, 0));
    mesh.vert.push_back(glm::dvec3(1, 0, 0));
    mesh.vert.push_back(glm::dvec3(0, 1, 0));
    mesh.vert.push_back(glm::dvec3(1, 1, 0));
    mesh.vert.push_back(glm::dvec3(0, 0.5, sqrt(3.0)/2));
    mesh.vert.push_back(glm::dvec3(1, 0.5, sqrt(3.0)/2));

    mesh.vert.push_back(glm::dvec3(0, 0, 0));
    mesh.vert.push_back(glm::dvec3(1, 0, 0));
    mesh.vert.push_back(glm::dvec3(0, 1, 0));
    mesh.vert.push_back(glm::dvec3(1, 1, 0));
    mesh.vert.push_back(glm::dvec3(0, 0, 1));
    mesh.vert.push_back(glm::dvec3(1, 0, 1));
    mesh.vert.push_back(glm::dvec3(0, 1, 1));
    mesh.vert.push_back(glm::dvec3(1, 1, 1));

    const MeshTet tet = MeshTet(0, 1, 2, 3);
    const MeshPri pri = MeshPri(4, 5, 6, 7, 8, 9);
    const MeshHex hex = MeshHex(10, 11, 12, 13, 14, 15, 16, 17);
    double regularTet = tetQuality(mesh, tet);
    double regularPri = priQuality(mesh, pri);
    double regularHex = hexQuality(mesh, hex);

    if(glm::abs(regularTet - 1.0) < VALIDITY_EPSILON &&
       glm::abs(regularPri - 1.0) < VALIDITY_EPSILON &&
       glm::abs(regularHex - 1.0) < VALIDITY_EPSILON)
    {
        getLog().postMessage(new Message('I', false,
            "Quality evaluator's measure is valid.", "AbstractEvaluator"));
        return true;
    }
    else
    {
        stringstream log;
        log.precision(20);
        log << "Quality evaluator's measure is invalid." << endl;
        log << "Regular tetrahedron quality: " << regularTet << endl;
        log << "Regular prism quality: " << regularPri << endl;
        log << "Regular hexahedron quality: " << regularHex << endl;
        getLog().postMessage(new Message('E', true, log.str(), "AbstractEvaluator"));

        assert(glm::abs(regularTet - 1.0) < VALIDITY_EPSILON);
        assert(glm::abs(regularPri - 1.0) < VALIDITY_EPSILON);
        assert(glm::abs(regularHex - 1.0) < VALIDITY_EPSILON);
        return false;
    }
}

void AbstractEvaluator::evaluateMeshQualityCpp(
        const Mesh& mesh,
        double& minQuality,
        double& qualityMean)
{
    int tetCount = mesh.tetra.size();
    int priCount = mesh.prism.size();
    int hexCount = mesh.hexa.size();

    int elemCount = tetCount + priCount + hexCount;
    std::vector<double> qualities(elemCount);
    int idx = 0;

    for(int i=0; i < tetCount; ++i, ++idx)
        qualities[idx] = tetQuality(mesh, mesh.tetra[i]);

    for(int i=0; i < priCount; ++i, ++idx)
        qualities[idx] = priQuality(mesh, mesh.prism[i]);

    for(int i=0; i < hexCount; ++i, ++idx)
        qualities[idx] = hexQuality(mesh, mesh.hexa[i]);


    minQuality = 1.0;
    qualityMean = 0.0;
    for(int i=0; i < elemCount; ++i)
    {
        double qual = qualities[i];

        if(qual < minQuality)
            minQuality = qual;

        qualityMean = (qualityMean * i + qual) / (i + 1);
    }
}

void AbstractEvaluator::evaluateMeshQualityGlsl(
        const Mesh& mesh,
        double& minQuality,
        double& qualityMean)
{
    initializeProgram(mesh);


    size_t tetCount = mesh.tetra.size();
    size_t priCount = mesh.prism.size();
    size_t hexCount = mesh.hexa.size();
    size_t maxSize = glm::max(glm::max(tetCount, priCount), hexCount);
    size_t workgroupCount = ceil(maxSize / (double)WORKGROUP_SIZE);


    std::vector<GpuQual> qualBuff(workgroupCount);
    size_t qualSize = sizeof(decltype(qualBuff.front())) * qualBuff.size();
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, _qualSsbo);
    glBufferData(GL_SHADER_STORAGE_BUFFER, qualSize, qualBuff.data(), GL_STATIC_DRAW);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);

    mesh.bindShaderStorageBuffers();
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER,
                     mesh.firstFreeBufferBinding(), _qualSsbo);


    // Simulataneous and specialized series gives about the same performance
    // Specialized series gives a tiny, not stable speed boost.
    // (tested on a parametric pri/hex mesh)
    if(_computeSimultaneously)
    {
        _simultaneousProgram.pushProgram();
        glDispatchCompute(workgroupCount, 1, 1);
        _simultaneousProgram.popProgram();
    }
    else
    {
        if(tetCount > 0)
        {
            _tetProgram.pushProgram();
            glDispatchCompute(ceil(tetCount / (double)WORKGROUP_SIZE), 1, 1);
            _tetProgram.popProgram();
        }

        if(priCount > 0)
        {
            _priProgram.pushProgram();
            glDispatchCompute(ceil(priCount / (double)WORKGROUP_SIZE), 1, 1);
            _priProgram.popProgram();
        }

        if(hexCount > 0)
        {
            _hexProgram.pushProgram();
            glDispatchCompute(ceil(hexCount / (double)WORKGROUP_SIZE), 1, 1);
            _hexProgram.popProgram();
        }
    }

    // Fetch workgroup's statistics from GPU
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, _qualSsbo);
    glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, qualSize, qualBuff.data());
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);


    // Combine workgroup's statistics
    minQuality = 1.0;
    qualityMean = 0.0;
    double meanWeight = 0.0;
    for(size_t i=0, polyId=0; i < workgroupCount; ++i, polyId+=WORKGROUP_SIZE)
    {
        minQuality = glm::min(minQuality, qualBuff[i].min / MAX_QUALITY_VALUE);

        size_t groupTet = (tetCount > polyId) ? glm::min(tetCount - polyId, WORKGROUP_SIZE) : 0;
        size_t groupPri = (priCount > polyId) ? glm::min(priCount - polyId, WORKGROUP_SIZE) : 0;
        size_t groupHex = (hexCount > polyId) ? glm::min(hexCount - polyId, WORKGROUP_SIZE) : 0;
        double groupParticipants = (groupTet + groupPri + groupHex);
        double groupWeight = groupParticipants / MAX_GROUP_PARTICIPANTS;
        double groupMean = (qualBuff[i].mean / MAX_INTEGER_VALUE);

        // 'groupMean' is implicitly multiplied by 'groupWeight'
        qualityMean = (qualityMean*meanWeight + groupMean) /
                      (meanWeight + groupWeight);
        meanWeight += groupWeight;
    }
}

void AbstractEvaluator::initializeProgram(const Mesh& mesh)
{
    if(_initialized)
        return;


    getLog().postMessage(new Message('I', false,
        "Initializing evaluator compute shader", "AbstractEvaluator"));

    std::vector<std::string> qualityInterface = {
        mesh.meshGeometryShaderName(),
        ":/shaders/compute/Quality/QualityInterface.glsl"
    };

    std::vector<std::string> shapeMeasure = {
        mesh.meshGeometryShaderName(),
        _shapeMeasuresShader
    };


    // Simultenous evaluation shader
    _simultaneousProgram.addShader(GL_COMPUTE_SHADER, qualityInterface);
    _simultaneousProgram.addShader(GL_COMPUTE_SHADER, shapeMeasure);
    _simultaneousProgram.addShader(GL_COMPUTE_SHADER, {
        mesh.meshGeometryShaderName(),
        ":/shaders/compute/Measuring/SimultaneousEvaluation.glsl"});
    _simultaneousProgram.link();
    _simultaneousProgram.pushProgram();
    _simultaneousProgram.setFloat("MaxQuality", MAX_QUALITY_VALUE);
    _simultaneousProgram.popProgram();
    mesh.uploadGeometry(_simultaneousProgram);


    // Specialized evaluation shader series
    _tetProgram.addShader(GL_COMPUTE_SHADER, qualityInterface);
    _tetProgram.addShader(GL_COMPUTE_SHADER, shapeMeasure);
    _tetProgram.addShader(GL_COMPUTE_SHADER, {
        mesh.meshGeometryShaderName(),
        ":/shaders/compute/Measuring/TetrahedraEvaluation.glsl"});
    _tetProgram.link();
    _tetProgram.pushProgram();
    _tetProgram.setFloat("MaxQuality", MAX_QUALITY_VALUE);
    _tetProgram.popProgram();
    mesh.uploadGeometry(_tetProgram);

    _priProgram.addShader(GL_COMPUTE_SHADER, qualityInterface);
    _priProgram.addShader(GL_COMPUTE_SHADER, shapeMeasure);
    _priProgram.addShader(GL_COMPUTE_SHADER, {
        mesh.meshGeometryShaderName(),
        ":/shaders/compute/Measuring/PrismsEvaluation.glsl"});
    _priProgram.link();
    _priProgram.pushProgram();
    _priProgram.setFloat("MaxQuality", MAX_QUALITY_VALUE);
    _priProgram.popProgram();
    mesh.uploadGeometry(_priProgram);

    _hexProgram.addShader(GL_COMPUTE_SHADER, qualityInterface);
    _hexProgram.addShader(GL_COMPUTE_SHADER, shapeMeasure);
    _hexProgram.addShader(GL_COMPUTE_SHADER, {
        mesh.meshGeometryShaderName(),
        ":/shaders/compute/Measuring/HexahedraEvaluation.glsl"});
    _hexProgram.link();
    _hexProgram.pushProgram();
    _hexProgram.setFloat("MaxQuality", MAX_QUALITY_VALUE);
    _hexProgram.popProgram();
    mesh.uploadGeometry(_hexProgram);

    // Shader storage quality blocks
    glGenBuffers(1, &_qualSsbo);


    _initialized = true;
}

void AbstractEvaluator::benchmark(const Mesh& mesh, uint cycleCount)
{
    initializeProgram(mesh);

    double minQual, qualMean;
    const uint MARK_COUNT = 50;
    size_t markSize = cycleCount / glm::min(MARK_COUNT, cycleCount);

    using std::chrono::high_resolution_clock;
    high_resolution_clock::time_point tStart;
    high_resolution_clock::time_point tEnd;


    getLog().postMessage(new Message('I', false,
       "Benchmarking C++ implementation",
       "AbstractEvaluator"));

    high_resolution_clock::duration cppTime(0);
    for(size_t i=0, m=markSize; i < cycleCount; ++i)
    {
        minQual = 0;
        qualMean = 0;

        tStart = high_resolution_clock::now();
        evaluateMeshQualityCpp(mesh, minQual, qualMean);
        tEnd = high_resolution_clock::now();

        cppTime += (tEnd - tStart);

        if(i == m)
        {
            float progress = 50.0f * i / (float) cycleCount;
            getLog().postMessage(new Message('I', false,
               "Benchmark progress : " + to_string(progress) + "%" +
               "(min=" + to_string(minQual) + ", mean=" + to_string(qualMean) + ")",
               "AbstractEvaluator"));
            m += markSize;
        }
    }


    getLog().postMessage(new Message('I', false,
       "Benchmarking GLSL implementation",
       "AbstractEvaluator"));

    high_resolution_clock::duration glslTime(0);
    for(size_t i=0, m=0; i < cycleCount; ++i)
    {
        minQual = 0;
        qualMean = 0;

        tStart = high_resolution_clock::now();
        evaluateMeshQualityGlsl(mesh, minQual, qualMean);
        tEnd = high_resolution_clock::now();

        glslTime += (tEnd - tStart);

        if(i == m)
        {
            float progress = 50.0f + 50.0f * i / (float) cycleCount;
            getLog().postMessage(new Message('I', false,
               "Benchmark progress : " + to_string(progress) + "%"  +
               "(min=" + to_string(minQual) + ", mean=" + to_string(qualMean) + ")",
               "AbstractEvaluator"));
            m += markSize;
        }
    }

    auto cppNano = cppTime.count();
    auto glslNano = glslTime.count();
    double minNano = glm::min(cppNano, glslNano);

    double cppRatio = cppNano / minNano;
    double glslRatio = glslNano / minNano;

    getLog().postMessage(new Message('I', false,
       "Shape measure time ratio (ms) : \tC++:GLSL \t= " +
        to_string(cppNano / 1000000.0) + ":" + to_string(glslNano / 1000000.0) + " \t= " +
        to_string(cppRatio) + ":" + to_string(glslRatio),
       "AbstractEvaluator"));
}

string AbstractEvaluator::shapeMeasureShader() const
{
    return _shapeMeasuresShader;
}
