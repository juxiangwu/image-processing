__kernel void rgb2lms(__global const float* rgb, __global float* result, __global const int* image_shape)
{
    float r, g, b, x, y, z;
    int i = get_global_id(0);
    int j = get_global_id(1);
    int h = image_shape[1];
    int c = image_shape[2];
    int idx = i * h * c  + j * c;

    r = rgb[idx]/255.0f;
    g = rgb[idx+1]/255.0f;
    b = rgb[idx+2]/255.0f;

    r = r > 0.04045f ? pow(((r + 0.055f) / 1.055f), 2.4f) : r / 12.92f;
    g = g > 0.04045f ? pow(((g + 0.055f) / 1.055f), 2.4f) : g / 12.92f;
    b = b > 0.04045f ? pow(((b + 0.055f) / 1.055f), 2.4f) : b / 12.92f;

    r *= 100.0f;
    g *= 100.0f;
    b *= 100.0f;

    //         sRGB, Illuminant = D65
    x = (r * 0.4124564f) + (g * 0.3575761f) + (b * 0.1804375f);
    y = (r * 0.2126729f) + (g * 0.7151522f) + (b * 0.0721750f);
    z = (r * 0.0193339f) + (g * 0.1191920f) + (b * 0.9503041f);

    result[idx] = 0.7328f * x + 0.4296f * y - 0.1624f * z;
    result[idx+1] = -0.7036f * x + 1.6975f * y + 0.0061f * z;
    result[idx+2] = 0.0030f * x + 0.0136f * y + 0.9834f * z;
}

__kernel void lms2rgb(__global const float* lms, __global float* result, __global const int* image_shape)
{
    float r, g, b, x, y, z, l, m, s;
    int i = get_global_id(0);
    int j = get_global_id(1);
    int h = image_shape[1];
    int c = image_shape[2];
    int idx = i * h * c  + j * c;

    l = lms[idx];
    m = lms[idx+1];
    s = lms[idx+2];

    x = 1.096124f * l - 0.278869f * m + 0.182745f * s;
    y = 0.454369f * l + 0.473533f * m + 0.072098f * s;
    z = -0.009628f * l - 0.005698f * m + 1.015326f * s;

    x = x / 100.0f;
    y = y / 100.0f;
    z = z / 100.0f;

    //         sRGB, Illuminant = D65
    r = (x * 3.2404542f) + (y * -1.5371385f) + (z * -0.4985314f);
    g = (x * -0.9692660f) + (y * 1.8760108f) + (z * 0.0415560f);
    b = (x * 0.0556434f) + (y * -0.2040259f) + (z * 1.0572252f);

    r = r > 0.0031308f ? 1.055f * (pow(r, (1.0f / 2.4f))) - 0.055f : 12.92f * r;
    g = g > 0.0031308f ? 1.055f * (pow(g, (1.0f / 2.4f))) - 0.055f : 12.92f * g;
    b = b > 0.0031308f ? 1.055f * (pow(b, (1.0f / 2.4f))) - 0.055f : 12.92f * b;

    r = r > 1.0f ? 1.0f : r;
    r = r < 0.0f ? 0.0f : r;

    g = g > 1.0f ? 1.0f : g;
    g = g < 0.0f ? 0.0f : g;

    b = b > 1.0f ? 1.0f : b;
    b = b < 0.0f ? 0.0f : b;

    result[idx] = r*255.0f;
    result[idx+1] = g*255.0f;
    result[idx+2] = b*255.0f;
}




float value(float hue_phase, float intensity, float saturation) {
	float pure = 0.5f * (1.0f + cos(hue_phase * 3.14159265359f / 3.0f));
	return (intensity * (1.0f - saturation * (1.0f - pure)));
}

__kernel void rgb2hsi(__global const float* rgb, __global float* result, __global const int* image_shape)
{
    float r, g, b, x, y, z;
    int i = get_global_id(0);
    int j = get_global_id(1);
    int h = image_shape[1];
    int c = image_shape[2];
    int idx = i * h * c  + j * c;

    r = rgb[idx]/255.0f;
    g = rgb[idx+1]/255.0f;
    b = rgb[idx+2]/255.0f;

    float mid = (r + g + b) / 3.0f;
    float mr = r - mid;
    float mg = g - mid;
    float mb = b - mid;

    float intensity = mid + sqrt(2.0f * (mr * mr + mg * mg + mb * mb) / 3.0f);

    float cos_hue = (2.0f * mr - mg - mb) / sqrt((mr * mr + mg * mg + mb * mb) * 6.0f);

    float saturation = 2.0f * (intensity - mid) / intensity;

    cos_hue = cos_hue > 1.0f ? 1.0f : cos_hue;
    cos_hue = cos_hue < -1.0f ? -1.0f : cos_hue;

    float hue = acos(cos_hue) * 3.0f / 3.14159265359f;

	if (r == g && r == b) {
		hue = 0.0f;
	}


	if (b > g) {
		hue = 6.0f - hue;
	}

	hue = saturation == 0.0f ? -1.0f : hue;

    result[idx] = hue;
	result[idx+1] = saturation;
	result[idx+2] = intensity;
}

__kernel void hsi2rgb(__global const float* hsi, __global float* result, __global const int* image_shape)
{
	int i = get_global_id(0);
    int j = get_global_id(1);
    int h = image_shape[1];
    int c = image_shape[2];
    int idx = i * h * c  + j * c;

    float hue = hsi[idx];
	float saturation = hsi[idx+1];
	float intensity = hsi[idx+2];


	if (saturation == 0.0f || hue == -1.0f) {
		intensity = intensity > 1.0f ? 255.0f : intensity*255.0f;
		result[idx] = intensity;
		result[idx+1] = intensity;
		result[idx+2] = intensity;
	}
	else {

	float red = value(hue + 0.0f, intensity, saturation);
	float green = value(hue + 4.0f, intensity, saturation);
	float blue = value(hue + 2.0f, intensity, saturation);

   red = red < 0.0f ? 0.0f : red;
   red = red > 1.0f ? 1.0f : red;
   green = green < 0.0f ? 0.0f : green;
   green = green > 1.0f ? 1.0f : green;
   blue = blue < 0.0f ? 0.0f : blue;
   blue = blue > 1.0f ? 1.0f : blue;

   result[idx] = red*255.0f;
   result[idx+1] = green*255.0f;
   result[idx+2] = blue*255.0f;
   }
}



