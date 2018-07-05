/* Please Write the OpenCL Kernel(s) code here*/

__kernel void rgb2hsi_filter(__read_only image2d_t input,
                               __write_only image2d_t output){
    const sampler_t sampler = CLK_FILTER_NEAREST |
                              CLK_NORMALIZED_COORDS_FALSE |
                              CLK_ADDRESS_CLAMP_TO_EDGE;

    const int2 size = get_image_dim(input);

    int2 coord = (int2)(get_global_id(0),get_global_id(1));

    float4 color = (float4)(0,0,0,1);

    float r,g,b,num,den,dim,minVal,theta,H,S,I;

    float4 srcColor = read_imagef(input,sampler,coord);

    b = srcColor.x;
    g = srcColor.y;
    r = srcColor.z;

    num = 0.5 * ((r - g) + (r - b));
    den = sqrt((r - g) * (r - g) + (r - b) * (g - b));

    if(den == 0) {
        H = 0;
    }else{
        theta = acos(num / den);
        if(b > g){
            H = (2 * 3.14159265358979323846f - theta) / (2 * 3.14159265358979323846f);
        }else{
            H = theta / (2 * 3.14159265358979323846f);
        }
    }

    minVal = (b > g) ? g : b;
    minVal = (minVal > r) ? r : minVal;

    den = r + g + b;
    if(den == 0){
        S = 0;
    }else{
        S = 1 - 3 * minVal / den;
    }

    I = (r + g + b) / 3;

    color.x = H;
    color.y = S;
    color.z = I;

    write_imagef(output,coord,color);
}
