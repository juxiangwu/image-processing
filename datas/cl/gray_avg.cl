/* Please Write the OpenCL Kernel(s) code here*/

__kernel void gray_avg_filter(__read_only image2d_t input, __write_only image2d_t output){

    const sampler_t sampler = CLK_FILTER_NEAREST |
                          CLK_NORMALIZED_COORDS_FALSE |
                          CLK_ADDRESS_CLAMP_TO_EDGE;

    int2 size = get_image_dim(input);
    int2 coord = (int2)(get_global_id(0),get_global_id(1));
    float4 pixel = read_imagef(input,sampler,coord);
    pixel.x = pixel.y = pixel.z = (pixel.x + pixel.y + pixel.z) / 3.0f;

    write_imagef(output,coord,pixel);
}