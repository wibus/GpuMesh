layout (local_size_x = 256, local_size_y = 1, local_size_z = 1) in;


uniform float MoveCoeff;


// Boundaries
vec3 snapToBoundary(int boundaryID, vec3 pos);

// Optimization helper functions
bool isSmoothable(uint vId);
float computePatchQuality(in uint vId);


void main()
{
    uint vId = gl_GlobalInvocationID.x;

    if(!isSmoothable(vId))
        return;


}
