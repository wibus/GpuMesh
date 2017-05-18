#version 440


/** Taken from Earl F. Glynn's web page:
* <a href="http://www.efg2.com/Lab/ScienceAndEngineering/Spectra.htm">Spectra Lab Report</a>
* */
vec3 waveLengthToRGB(float Wavelength)
{

    float Gamma = 0.80;

    float factor;
    float Red,Green,Blue;

    if((Wavelength >= 380) && (Wavelength<440))
    {
        Red = -(Wavelength - 440) / (440 - 380);
        Green = 0.0;
        Blue = 1.0;
    }
    else if((Wavelength >= 440) && (Wavelength<490))
    {
        Red = 0.0;
        Green = (Wavelength - 440) / (490 - 440);
        Blue = 1.0;
    }
    else if((Wavelength >= 490) && (Wavelength<510))
    {
        Red = 0.0;
        Green = 1.0;
        Blue = -(Wavelength - 510) / (510 - 490);
    }
    else if((Wavelength >= 510) && (Wavelength<580))
    {
        Red = (Wavelength - 510) / (580 - 510);
        Green = 1.0;
        Blue = 0.0;
    }
    else if((Wavelength >= 580) && (Wavelength<645))
    {
        Red = 1.0;
        Green = -(Wavelength - 645) / (645 - 580);
        Blue = 0.0;
    }
    else if((Wavelength >= 645) && (Wavelength<781))
    {
        Red = 1.0;
        Green = 0.0;
        Blue = 0.0;
    }
    else
    {
        Red = 0.0;
        Green = 0.0;
        Blue = 0.0;
    };

    return pow(vec3(Red, Green, Blue), vec3(Gamma));
}

vec3 qualityLut(in float q)
{
    if(q <= 0.0)
    {
        return vec3(1);
    }

    float minLength = 420;
    float maxFreq = 1.0 / minLength;
    float maxLength = 700;
    float minFreq = 1.0 / maxLength;

    float wavelength = 1.0 / ( q * (maxFreq - minFreq) + minFreq);
    return waveLengthToRGB(wavelength);
}
