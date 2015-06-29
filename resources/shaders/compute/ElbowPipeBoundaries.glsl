#version 440

const int PIPE_SURFACE_ID = 1;
const int PIPE_EXTREMITY_FACE_ID = 2;
const int PIPE_EXTREMITY_EDGE_ID = 3;


const float PIPE_RADIUS = 0.3;
const vec3 EXT_NORMAL = vec3(1.0, 0, 0);
const vec3 EXT_CENTER = vec3(-1, 0.5, 0.0);

vec3 snapToPipeSurface(vec3 pos)
{
    vec3 center;

    if(pos.x < 0.5) // Straights
    {
        center = vec3(pos.x, (pos.y < 0.0 ? -0.5 : 0.5), 0.0);
    }
    else // Arc
    {
        center = pos - vec3(0.5, 0.0, pos.z);
        center = normalize(center) * 0.5;
        center = vec3(0.5, 0, 0) + center;
    }

    vec3 dist = pos - center;
    vec3 extProj = normalize(dist) * PIPE_RADIUS;
    return center + extProj;
}

vec3 snapToPipeExtremityFace(vec3 pos)
{
    vec3 center = EXT_CENTER;
    center.y *= sign(pos.y);

    float offset = dot(pos - center, EXT_NORMAL);
    return pos - EXT_NORMAL * offset;
}

vec3 snapToPipeExtremityEdge(vec3 pos)
{
    vec3 center = EXT_CENTER;
    center.y *= sign(pos.y);

    vec3 dist = pos - center;
    float offset = dot(dist, EXT_NORMAL);
    vec3 extProj = dist - EXT_NORMAL * offset;
    return center + normalize(extProj) * PIPE_RADIUS;
}

vec3 snapToBoundary(int boundaryID, vec3 pos)
{
    switch(boundaryID)
    {
    case PIPE_SURFACE_ID :
        return snapToPipeSurface(pos);
    case PIPE_EXTREMITY_FACE_ID :
        return snapToPipeExtremityFace(pos);
    case PIPE_EXTREMITY_EDGE_ID :
        return snapToPipeExtremityEdge(pos);
    }

    return pos;
}
