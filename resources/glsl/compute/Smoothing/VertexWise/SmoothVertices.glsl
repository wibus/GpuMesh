const uint NODE_COUNT = 256;


// Algorithm entry point
void smoothVert(uint vId);

// Smoothing Helper
uint getInvocationVertexId();
bool isSmoothableVertex(uint vId);


void main()
{
    uint vId = getInvocationVertexId();

    if(isSmoothableVertex(vId))
    {
        smoothVert(vId);
    }
}
