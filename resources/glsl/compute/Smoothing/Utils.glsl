// Independent group range
uniform int GroupBase;
uniform int GroupSize;


uint getInvocationVertexId()
{
    // Default value is invalid
    // See isSmoothableVertex()
    uint vId = verts.length();

    // Assign real index only if this
    // invocation does not overflow
    if(gl_GlobalInvocationID.x < GroupSize)
        vId = groupMembers[GroupBase + gl_GlobalInvocationID.x];

    return vId;
}

uint getInvocationTetId()
{
    return gl_GlobalInvocationID.x;
}

uint getInvocationPriId()
{
    return gl_GlobalInvocationID.x;
}

uint getInvocationHexId()
{
    return gl_GlobalInvocationID.x;
}


bool isSmoothableVertex(uint vId)
{
    if(vId >= verts.length())
        return false;

    return true;
}

bool isSmoothableTet(uint eId)
{
    if(eId >= tets.length())
        return false;

    return true;
}

bool isSmoothablePri(uint eId)
{
    if(eId >= pris.length())
        return false;

    return true;
}

bool isSmoothableHex(uint eId)
{
    if(eId >= hexs.length())
        return false;

    return true;
}


void swap(inout vec4 v1, inout vec4 v2)
{
    vec4 tmp = v1;
    v1 = v2;
    v2 = tmp;
}
