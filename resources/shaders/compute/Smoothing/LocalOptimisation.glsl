layout (local_size_x = 256, local_size_y = 1, local_size_z = 1) in;


uniform float MoveCoeff;


vec3 snapToBoundary(int boundaryID, vec3 pos);


void main()
{
    uint vId = gl_GlobalInvocationID.x;

    if(vId < verts.length())
    {

    }
}
