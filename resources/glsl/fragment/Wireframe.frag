#version 440

in float qual;
in float dist;


layout(location = 0) out vec4 FragColor;


vec3 qualityLut(in float q);


void main()
{
    if(dist > 0.0)
        discard;

    FragColor = vec4(qualityLut(qual), 1);
}
