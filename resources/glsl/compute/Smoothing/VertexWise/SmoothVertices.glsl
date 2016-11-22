const uint NODE_COUNT = 256;
layout (local_size_x = NODE_COUNT, local_size_y = 1, local_size_z = 1) in;


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
