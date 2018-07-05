/* Please Write the OpenCL Kernel(s) code here*/
#define PI_F 3.14159265358979323846f
__kernel void sphere_filter(__read_only image2d_t input,__write_only image2d_t output){

    const sampler_t sampler = CLK_FILTER_NEAREST |
                               CLK_NORMALIZED_COORDS_FALSE |
                               CLK_ADDRESS_CLAMP_TO_EDGE;
                               
    const int2 size = get_image_dim(input);
    int2 coord = (int2)(get_global_id(0),get_global_id(1));
    
    int r = size.x > size.y ? size.x / 2 : size.y / 2;
    
    int x1 = (int)(r * sin(coord.x * PI_F / (2 * r)));
    int y1 = (int)(r * sin(coord.y * PI_F / (2 * r)));
    
   
    
    float4 color = read_imagef(input,sampler,(int2)(x1,y1));
    
    write_imagef(output,coord,color);
}