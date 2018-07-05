/* Please Write the OpenCL Kernel(s) code here*/

__kernel void sketch_filter(__read_only image2d_t input,
                              __write_only image2d_t output){
   
   const sampler_t sampler = CLK_FILTER_NEAREST |
                             CLK_NORMALIZED_COORDS_FALSE |
                             CLK_ADDRESS_CLAMP_TO_EDGE;

   const int2 size = get_image_dim(input);

   int2 coord = (int2)(get_global_id(0),get_global_id(1));
   
   float4 rgb = read_imagef(input,sampler,coord);
   
    float topLeft = luminance(read_imagef(input,sampler,(int2)(coord.x - 1,coord.y - 1)));
    // top
    float top = luminance(read_imagef(input,sampler,(int2)(coord.x,coord.y - 1)));
    // top right
    float topRight = luminance(read_imagef(input,sampler,(int2)(coord.x + 1,coord.y - 1)));
    // left
    float left = luminance(read_imagef(input,sampler,(int2)(coord.x - 1,coord.y)));
    // center
    float center = luminance(read_imagef(input,sampler,(int2)(coord.x,coord.y)));
    // right
    float right = luminance(read_imagef(input,sampler,(int2)(coord.x + 1,coord.y)));
    // bottom left
    float bottomLeft = luminance(read_imagef(input,sampler,(int2)(coord.x - 1,coord.y + 1)));
    // bottom
    float bottom = luminance(read_imagef(input,sampler,(int2)(coord.x,coord.y + 1)));
    // bottom right
    float bottomRight = luminance(read_imagef(input,sampler,(int2)(coord.x + 1,coord.y + 1)));
    
    
 
    float h = -topLeft-2.0f*top-topRight+bottomLeft+2.0f*bottom+bottomRight;
	float v = -bottomLeft-2.0f*left-topLeft+bottomRight+2.0f*right+topRight;

	float mag = 1.0f - length((float2)(h, v));

    rgb.x = rgb.y = rgb.z = mag;
    rgb.w = 1.0f;
    
    write_imagef(output,coord,rgb);

}