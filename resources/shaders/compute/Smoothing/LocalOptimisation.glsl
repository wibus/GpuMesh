layout (local_size_x = 256, local_size_y = 1, local_size_z = 1) in;


uniform float MoveCoeff;


vec3 snapToBoundary(int boundaryID, vec3 pos);


void main()
{
    uint uid = gl_GlobalInvocationID.x;

    if(uid < verts.length())
    {

    }
}
