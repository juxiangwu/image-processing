/* Please Write the OpenCL Kernel(s) code here*/
#define PI_F 3.14159265358979323846f
__kernel void rotate_filter(__read_only image2d_t input,
                              __write_only image2d_t output){
    const sampler_t sampler = CLK_FILTER_NEAREST |
                              CLK_NORMALIZED_COORDS_FALSE |
                              CLK_ADDRESS_CLAMP;

    const int2 size = get_image_dim(input);

    int2 coord = (int2)(get_global_id(0),get_global_id(1));
    
    int width = size.x;
    int height = size.y;
    
    int xc = width / 2;
    int yc = height / 2;
    
    float angle = 45.0f;
    float theta = angle * PI_F / 180.0f;
    
    float xpos = (coord.x - xc) * cos(theta) - (coord.y - yc) * sin(theta) + xc;
    float ypos = (coord.x - xc) * sin(theta) + (coord.y - yc) * cos(theta) + yc;
    
    int2 pos = convert_int2((float2)(xpos,ypos));
    if(pos.x >= 0 && pos.x < width && pos.y >= 0 && pos.y < height){
        float4 color = read_imagef(input,sampler,pos);
    
        write_imagef(output,coord,color);
    }
}