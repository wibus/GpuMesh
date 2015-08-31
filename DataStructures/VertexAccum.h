#ifndef GPUMESH_VERTEXACCUM
#define GPUMESH_VERTEXACCUM

#include <mutex>

#include <GLM/glm.hpp>


class IVertexAccum
{
protected:
    IVertexAccum();

public:
    virtual ~IVertexAccum();
    virtual void addPosition(const glm::dvec3 pos, double weight) = 0;
    virtual bool assignAverage(glm::dvec3& pos) const = 0;
    virtual void reinit() = 0;
};


class NotThreadSafeVertexAccum : public IVertexAccum
{
public:
    NotThreadSafeVertexAccum();
    virtual ~NotThreadSafeVertexAccum();

    virtual void addPosition(const glm::dvec3 pos, double weight) override;
    virtual bool assignAverage(glm::dvec3& pos) const override;
    virtual void reinit() override;

private:
    glm::dvec3 _posAccum;
    double _weightAccum;
};


class ThreadSafeVertexAccum : public IVertexAccum
{
public:
    ThreadSafeVertexAccum();
    virtual ~ThreadSafeVertexAccum();

    virtual void addPosition(const glm::dvec3 pos, double weight) override;
    virtual bool assignAverage(glm::dvec3& pos) const override;
    virtual void reinit() override;

private:
    glm::dvec3 _posAccum;
    double _weightAccum;
    mutable std::mutex _mutex;
};

#endif // GPUMESH_VERTEXACCUM
