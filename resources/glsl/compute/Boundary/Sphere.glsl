vec3 snapToBoundary(int boundaryID, vec3 pos)
{
    vec3 npos = normalize(pos);
    if(boundaryID == 1)
        return npos * 0.5;
    else
        return npos;
}
