uniform sampler3D TopLineTex;
uniform sampler3D SideTriTex;
uniform mat4 TexTransform;


subroutine mat3 metricAtSub(in vec3 position, inout uint cachedRefTet);
layout(location=METRIC_AT_SUBROUTINE_LOC)
subroutine uniform metricAtSub metricAtUni;

mat3 metricAt(in vec3 position, inout uint cachedRefTet)
{
    return metricAtUni(position, cachedRefTet);
}


layout(index=METRIC_AT_SUBROUTINE_IDX) subroutine(metricAtSub)
mat3 metricAtImpl(in vec3 position, inout uint cachedRefTet)
{
    vec3 coor = vec3(TexTransform * vec4(position, 1.0));

    vec3 topLine = texture(TopLineTex, coor).xyz;
    vec3 sideTri = texture(SideTriTex, coor).xyz;

    mat3 metric = mat3(topLine.x, topLine.y, topLine.z,
                       topLine.y, sideTri.x, sideTri.y,
                       topLine.z, sideTri.y, sideTri.z);

    return metric;
}
