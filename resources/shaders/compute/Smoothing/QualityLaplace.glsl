layout (local_size_x = 256, local_size_y = 1, local_size_z = 1) in;


uniform int VertCount;
uniform float MoveCoeff;


vec3 snapToBoundary(int boundaryID, vec3 pos);

void main()
{
    uint uid = gl_GlobalInvocationID.x;

    if(uid < VertCount)
    {

        // Read
        vec3 pos = vec3(verts[uid].p);


        // Modification
        int type = topos[uid].type;
        int count = topos[uid].neigCount;
        if(type >= 0 && count > 0)
        {
            float weightSum = 0.0;
            vec3 barycenter = vec3(0.0);

            int n = topos[uid].neigBase;
            for(int i=0; i<count; ++i, ++n)
            {
                vec3 npos = vec3(verts[neigs[n].v].p);

                vec3 dist = npos - pos;
                float weight = dot(dist, dist) + 0.0001;

                barycenter += npos * weight;
                weightSum += weight;
            }

            barycenter /= weightSum;
            pos = mix(pos, barycenter, MoveCoeff);

            if(type > 0)
            {
                pos = snapToBoundary(type, pos);
            }
        }


        // Write
        verts[uid].p = vec4(pos, 0.0);
    }
}
