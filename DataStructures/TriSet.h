#ifndef GPUMESH_TRISET
#define GPUMESH_TRISET

#include <vector>

#include "Triangle.h"


struct TriSetNode
{
    TriSetNode(const Triangle& tri) :
        tri(tri),
        next(nullptr)
    {
    }

    Triangle tri;
    TriSetNode* next;
};

struct TriSet
{
    void clear()
    {
        _tris.clear();
    }

    void reset(std::size_t bucketCount)
    {
        gather();
        _buckets.resize(bucketCount, nullptr);
        clear();
    }

    inline void xOrTri(const Triangle& tri)
    {
        std::size_t hash = tri.hash();
        hash %= _buckets.size();

        TriSetNode* parent = nullptr;
        TriSetNode* node = _buckets[hash];
        while(node != nullptr)
        {
            if(node->tri == tri)
            {
                if(parent == nullptr)
                    _buckets[hash] = node->next;
                else
                    parent->next = node->next;

                _disposeNode(node);
                return;
            }

            parent = node;
            node = node->next;
        }

        node = _acquireNode(tri);
        node->next = _buckets[hash];
        _buckets[hash] = node;
    }

    inline const std::vector<Triangle>& gather()
    {
        std::size_t bucketCount = _buckets.size();
        for(int i=0; i< bucketCount; ++i)
        {
            TriSetNode* node = _buckets[i];
            _buckets[i] = nullptr;

            while(node != nullptr)
            {
                _tris.push_back(node->tri);
                _disposeNode(node);
                node = node->next;
            }
        }

        return _tris;
    }

    void releaseMemoryPool()
    {
        gather();

        _tris.clear();
        _tris.shrink_to_fit();

        _buckets.clear();
        _buckets.shrink_to_fit();


        int nodeCount = _nodePool.size();
        for(int i=0; i < nodeCount; ++i)
            delete _nodePool[i];

        _nodePool.clear();
        _nodePool.shrink_to_fit();
    }

private:
    inline static TriSetNode* _acquireNode(const Triangle& tri)
    {
        if(_nodePool.empty())
        {
            return new TriSetNode(tri);
        }
        else
        {
            TriSetNode* node = _nodePool.back();
            _nodePool.pop_back();
            node->tri = tri;
            return node;
        }
    }

    inline static void _disposeNode(TriSetNode* node)
    {
        _nodePool.push_back(node);
    }

    std::vector<Triangle> _tris;
    std::vector<TriSetNode*> _buckets;
    static std::vector<TriSetNode*> _nodePool;
};

#endif // GPUMESH_TRISET
