/* Please Write the OpenCL Kernel(s) code here*/

__kernel void add_filter(__read_only image2d_t input,__read_only image2d_t input2,
                              __write_only image2d_t output){
    const sampler_t sampler = CLK_FILTER_NEAREST |
                              CLK_NORMALIZED_COORDS_FALSE |
                              CLK_ADDRESS_CLAMP;

    const int2 size = get_image_dim(input);

    int2 coord = (int2)(get_global_id(0),get_global_id(1));
    if(coord.x >= 0 && coord.x < size.x && coord.y >= 0 && coord.y < size.y){
        float4 color = read_imagef(input,sampler,coord) + read_imagef(input2,sampler,coord);
        color.w = 1.0f;

        write_imagef(output,coord,color);
    }
}