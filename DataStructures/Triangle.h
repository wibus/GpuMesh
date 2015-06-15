#ifndef GPUMESH_TRIANGLE
#define GPUMESH_TRIANGLE


struct Triangle
{
    Triangle(int v0, int v1, int v2)
    {
        if(v0 < v1)
        {
            if(v0 < v2)
            {
                if(v1 < v2)
                {
                    v[0] = v0;
                    v[1] = v1;
                    v[2] = v2;
                }
                else // v2 < v1
                {
                    v[0] = v0;
                    v[1] = v2;
                    v[2] = v1;
                }
            }
            else // v2 < v0
            {
                v[0] = v2;
                v[1] = v0;
                v[2] = v1;
            }
        }
        else // v1 < v0
        {
            if(v1 < v2)
            {
                if(v0 < v2)
                {
                    v[0] = v1;
                    v[1] = v0;
                    v[2] = v2;
                }
                else // v2 < v0
                {
                    v[0] = v1;
                    v[1] = v2;
                    v[2] = v0;
                }
            }
            else // v2 < v1
            {
                v[0] = v2;
                v[1] = v1;
                v[2] = v0;
            }
        }
    }

    bool operator==(const Triangle& t) const
    {
        return v[0] == t.v[0] &&
               v[1] == t.v[1] &&
               v[2] == t.v[2];
    }

    int hash() const
    {
        return v[2] * (1000001 + v[1] * (3 + v[0] * 1234567));
    }

    int v[3];
};


#endif // GPUMESH_TRIANGLE
