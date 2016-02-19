#ifndef GPUMESH_TRISET
#define GPUMESH_TRISET

#include <vector>

#include <GLM/glm.hpp>

#include "Triangle.h"


struct TriSetNode
{
    TriSetNode(const Triangle& tri, uint owner, uint side) :
        tri(tri),
        next(nullptr),
        owner(owner),
        side(side)
    {
    }

    Triangle tri;
    TriSetNode* next;
    uint owner;
    uint side;
};


struct TriSet
{
    static const uint NO_OWNER;

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

    inline glm::uvec2 xOrTri(
            const Triangle& tri,
            uint owner = NO_OWNER,
            uint side = 0)
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
                return glm::uvec2(node->owner, node->side);
            }

            parent = node;
            node = node->next;
        }

        node = _acquireNode(tri, owner, side);
        node->next = _buckets[hash];
        _buckets[hash] = node;

        return glm::uvec2(NO_OWNER, 0);
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
    inline static TriSetNode* _acquireNode(const Triangle& tri, uint owner, uint side)
    {
        if(_nodePool.empty())
        {
            return new TriSetNode(tri, owner, side);
        }
        else
        {
            TriSetNode* node = _nodePool.back();
            _nodePool.pop_back();
            node->owner = owner;
            node->side = side;
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
