layout (local_size_x = 256, local_size_y = 1, local_size_z = 1) in;


// Algorithm entry point
void smoothVertex(uint vId);

// Smoothing Helper
uint getInvocationVertexId();
bool isSmoothableVertex(uint vId);


void main()
{
    uint vId = getInvocationVertexId();

    if(isSmoothableVertex(vId))
    {
        smoothVertex(vId);
    }
}
