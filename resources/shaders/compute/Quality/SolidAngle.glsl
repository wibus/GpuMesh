
float solidAngle(in vec3 a, in vec3 b, in vec3 c)
{
    float al = length(a);
    float bl = length(b);
    float cl = length(c);

    return abs(determinant(mat3(a, b, c))) /
            sqrt( 2.0 * (al*bl + dot(a, b)) *
                        (bl*cl + dot(b, c)) *
                        (cl*al + dot(c, a)));
}

float tetQuality(in vec3 vp[4])
{
    float q0 = solidAngle(vp[1] - vp[0], vp[2] - vp[0], vp[3] - vp[0]);
    float q1 = solidAngle(vp[0] - vp[1], vp[2] - vp[1], vp[3] - vp[1]);
    float q2 = solidAngle(vp[0] - vp[2], vp[1] - vp[2], vp[3] - vp[2]);
    float q3 = solidAngle(vp[0] - vp[3], vp[1] - vp[3], vp[2] - vp[3]);

    float minQ = min(min(q0, q1),
                     min(q2, q3));

    return minQ * 3.67423461417; // 9 / sqrt(6)
}

float priQuality(in vec3 vp[6])
{
    float q0 = solidAngle(vp[1] - vp[0], vp[2] - vp[0], vp[4] - vp[0]);
    float q1 = solidAngle(vp[0] - vp[1], vp[3] - vp[1], vp[5] - vp[1]);
    float q2 = solidAngle(vp[0] - vp[2], vp[3] - vp[2], vp[4] - vp[2]);
    float q3 = solidAngle(vp[1] - vp[3], vp[2] - vp[3], vp[5] - vp[3]);
    float q4 = solidAngle(vp[0] - vp[4], vp[2] - vp[4], vp[5] - vp[4]);
    float q5 = solidAngle(vp[1] - vp[5], vp[3] - vp[5], vp[4] - vp[5]);

    float minQ = min(min(q0, q1),
                     min(min(q2, q3),
                         min(q4, q5)));

    return minQ * 2.0; // 1.0 / <max val for regular prism>
}

float hexQuality(in vec3 vp[8])
{
    float q0 = solidAngle(vp[1] - vp[0], vp[2] - vp[0], vp[4] - vp[0]);
    float q1 = solidAngle(vp[0] - vp[1], vp[3] - vp[1], vp[5] - vp[1]);
    float q2 = solidAngle(vp[0] - vp[2], vp[3] - vp[2], vp[6] - vp[2]);
    float q3 = solidAngle(vp[1] - vp[3], vp[2] - vp[3], vp[7] - vp[3]);
    float q4 = solidAngle(vp[0] - vp[4], vp[5] - vp[4], vp[6] - vp[4]);
    float q5 = solidAngle(vp[1] - vp[5], vp[4] - vp[5], vp[7] - vp[5]);
    float q6 = solidAngle(vp[2] - vp[6], vp[4] - vp[6], vp[7] - vp[6]);
    float q7 = solidAngle(vp[3] - vp[7], vp[5] - vp[7], vp[6] - vp[7]);

    float minQ = min(min(min(q0, q1), min(q2, q3)),
                     min(min(q4, q5), min(q6, q7)));

    return minQ * 1.41421356237; // sqrt(2)
}
