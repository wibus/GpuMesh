#ifndef GPUMESH_TETLIST
#define GPUMESH_TETLIST

#include <cassert>

#include <vector>

struct Tetrahedron;


struct TetListNode
{
    Tetrahedron* tet;
    TetListNode* next;

    TetListNode() :
        tet(nullptr),
        next(nullptr)
    {
    }
};

struct TetList
{
    TetListNode* head;

    TetList() :
        head(nullptr)
    {
    }

    inline void addTet(Tetrahedron* tet)
    {
        TetListNode* node = _acquireNode();
        node->next = head;
        node->tet = tet;
        head = node;
    }

    inline void delTet(Tetrahedron* tet)
    {
        TetListNode* node = head;
        TetListNode* parent = nullptr;
        while(node != nullptr)
        {
            if(node->tet == tet)
            {
                if(parent != nullptr)
                    parent->next = node->next;
                else
                    head = node->next;
                _disposeNode(node);
                return;
            }

            parent = node;
            node = node->next;
        }

        bool tetDeleted = false;
        assert(tetDeleted);
    }

    inline void clrTet()
    {
        TetListNode* node = head;
        while(node != nullptr)
        {
            TetListNode* next = node->next;
            delete node;
            node = next;
        }
        head = nullptr;
    }

    static void releaseMemoryPool()
    {
        int nodeCount = _nodePool.size();
        for(int i=0; i < nodeCount; ++i)
            delete _nodePool[i];

        _nodePool.clear();
        _nodePool.shrink_to_fit();
    }

private:
    inline static TetListNode* _acquireNode()
    {
        if(_nodePool.empty())
        {
            return new TetListNode();
        }
        else
        {
            TetListNode* node = _nodePool.back();
            _nodePool.pop_back();
            return node;
        }
    }

    inline static void _disposeNode(TetListNode* node)
    {
        _nodePool.push_back(node);
    }

    static std::vector<TetListNode*> _nodePool;
};

#endif // GPUMESH_TETLIST
