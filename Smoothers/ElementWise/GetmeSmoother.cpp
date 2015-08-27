#include "GetmeSmoother.h"

#include <mutex>

#include "../SmoothingHelper.h"
#include "Evaluators/AbstractEvaluator.h"

using namespace std;


class NotThreadSafeVertexAccum : public IVertexAccum
{
public:
    NotThreadSafeVertexAccum() : _posAccum(), _weightAccum(0.0) {}
    virtual ~NotThreadSafeVertexAccum() {}

    virtual void add(const glm::dvec3 pos, double weight) override
    {
        _posAccum += pos * weight;
        _weightAccum += weight;
    }

    virtual bool assignAverage(glm::dvec3& pos) const override
    {
        if(_weightAccum != 0.0)
        {
            pos = _posAccum / _weightAccum;
            return true;
        }
        return false;
    }

private:
    glm::dvec3 _posAccum;
    double _weightAccum;
};

class ThreadSafeVertexAccum : public IVertexAccum
{
public:
    ThreadSafeVertexAccum() : _posAccum(), _weightAccum(0.0) {}
    virtual ~ThreadSafeVertexAccum() {}

    virtual void add(const glm::dvec3 pos, double weight) override
    {
        _mutex.lock();
        _posAccum += pos * weight;
        _weightAccum += weight;
        _mutex.unlock();
    }

    virtual bool assignAverage(glm::dvec3& pos) const override
    {
        _mutex.lock();
        if(_weightAccum != 0.0)
        {
            pos = _posAccum / _weightAccum;
            _mutex.unlock();
            return true;
        }
        _mutex.unlock();
        return false;
    }

private:
    glm::dvec3 _posAccum;
    double _weightAccum;
    mutable mutex _mutex;
};

GetmeSmoother::GetmeSmoother() :
    AbstractElementWiseSmoother({":/shader/compute/Smoothing/GETMe.glsl"}),
    _lambda(0.78)
{

}

GetmeSmoother::~GetmeSmoother()
{

}


void GetmeSmoother::smoothMeshSerial(
        Mesh& mesh,
        AbstractEvaluator& evaluator)
{
    // Allocate vertex accumulators
    size_t vertCount = mesh.verts.size();
    _vertexAccums = new IVertexAccum*[vertCount];
    for(size_t i=0; i < vertCount; ++i)
        _vertexAccums[i] = new NotThreadSafeVertexAccum();

    AbstractElementWiseSmoother::smoothMeshSerial(mesh, evaluator);

    // Deallocate vertex accumulators
    for(size_t i=0; i < vertCount; ++i)
        delete _vertexAccums[i];
    delete[] _vertexAccums;
}

void GetmeSmoother::smoothMeshThread(
        Mesh& mesh,
        AbstractEvaluator& evaluator)
{
    // Allocate vertex accumulators
    size_t vertCount = mesh.verts.size();
    _vertexAccums = new IVertexAccum*[vertCount];
    for(size_t i=0; i < vertCount; ++i)
        _vertexAccums[i] = new ThreadSafeVertexAccum();

    AbstractElementWiseSmoother::smoothMeshThread(mesh, evaluator);

    // Deallocate vertex accumulators
    for(size_t i=0; i < vertCount; ++i)
        delete _vertexAccums[i];
    delete[] _vertexAccums;
}

void GetmeSmoother::updateVertexPositions(
        Mesh& mesh,
        AbstractEvaluator& evaluator)
{
    vector<MeshVert>& verts = mesh.verts;
    const vector<MeshTopo>& topos = mesh.topos;

    size_t vertCount = mesh.verts.size();
    for(size_t v=0; v < vertCount; ++v)
    {
        glm::dvec3 pos = verts[v].p;
        glm::dvec3 posPrim = pos;
        if(_vertexAccums[v]->assignAverage(posPrim))
        {
            const MeshTopo& topo = topos[v];
            if(topo.isBoundary)
                posPrim = (*topo.snapToBoundary)(posPrim);

            double patchQuality =
                SmoothingHelper::computePatchQuality(
                    mesh, evaluator, v);

            verts[v].p = posPrim;

            double patchQualityPrime =
                SmoothingHelper::computePatchQuality(
                    mesh, evaluator, v);

            if(patchQualityPrime < patchQuality)
                verts[v].p = pos;
        }
    }
}

void GetmeSmoother::smoothTets(
        Mesh& mesh,
        AbstractEvaluator& evaluator,
        size_t first,
        size_t last,
        bool synchronize)
{
    const vector<MeshVert>& verts = mesh.verts;
    const vector<MeshTopo>& topos = mesh.topos;
    const vector<MeshTet>& tets = mesh.tets;

    for(int e = first; e < last; ++e)
    {
        const MeshTet& tet = tets[e];

        uint vi[] = {
            tet.v[0],
            tet.v[1],
            tet.v[2],
            tet.v[3],
        };

        glm::dvec3 vp[] = {
            verts[vi[0]],
            verts[vi[1]],
            verts[vi[2]],
            verts[vi[3]],
        };

        glm::dvec3 center = 0.25 * (
            vp[0] + vp[1] + vp[2] + vp[3]);

        double volume =
            glm::determinant(
                glm::dmat3(
                    vp[0] - vp[3],
                    vp[1] - vp[3],
                    vp[2] - vp[3]));

        double quality = evaluator.tetQuality(vp);

        glm::dvec3 n[] = {
            glm::cross(vp[3]-vp[1], vp[1]-vp[2]),
            glm::cross(vp[3]-vp[2], vp[2]-vp[0]),
            glm::cross(vp[1]-vp[3], vp[3]-vp[0]),
            glm::cross(vp[1]-vp[0], vp[0]-vp[2]),
        };

        vp[0] = vp[0] + _lambda * n[0] / glm::sqrt(glm::length(n[0]));
        vp[1] = vp[1] + _lambda * n[1] / glm::sqrt(glm::length(n[1]));
        vp[2] = vp[2] + _lambda * n[2] / glm::sqrt(glm::length(n[2]));
        vp[3] = vp[3] + _lambda * n[3] / glm::sqrt(glm::length(n[3]));

        double volumePrime =
            glm::determinant(
                glm::dmat3(
                    vp[0] - vp[3],
                    vp[1] - vp[3],
                    vp[2] - vp[3]));

        double volumeVar = glm::pow(volume / volumePrime, 1.0/3.0);

        vp[0] = center + volumeVar * (vp[0] - center);
        vp[1] = center + volumeVar * (vp[1] - center);
        vp[2] = center + volumeVar * (vp[2] - center);
        vp[3] = center + volumeVar * (vp[3] - center);

        if(topos[vi[0]].isBoundary) vp[0] = (*topos[vi[0]].snapToBoundary)(vp[0]);
        if(topos[vi[1]].isBoundary) vp[1] = (*topos[vi[1]].snapToBoundary)(vp[1]);
        if(topos[vi[2]].isBoundary) vp[2] = (*topos[vi[2]].snapToBoundary)(vp[2]);
        if(topos[vi[3]].isBoundary) vp[3] = (*topos[vi[3]].snapToBoundary)(vp[3]);

        double qualityPrime = evaluator.tetQuality(vp);
        double weight = qualityPrime / quality;

        _vertexAccums[tet[0]]->add(vp[0], weight);
        _vertexAccums[tet[1]]->add(vp[1], weight);
        _vertexAccums[tet[2]]->add(vp[2], weight);
        _vertexAccums[tet[3]]->add(vp[3], weight);
    }
}

void GetmeSmoother::smoothPris(
        Mesh& mesh,
        AbstractEvaluator& evaluator,
        size_t first,
        size_t last,
        bool synchronize)
{

}

void GetmeSmoother::smoothHexs(
        Mesh& mesh,
        AbstractEvaluator& evaluator,
        size_t first,
        size_t last,
        bool synchronize)
{

}
