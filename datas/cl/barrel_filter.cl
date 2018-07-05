/* Please Write the OpenCL Kernel(s) code here*/
float2 barrel(float2 coord,float distortion,float2 dim){
    float2 cc = coord - dim / 2;
    float d = dot(cc,cc);
    
    return coord + cc * (d + distortion * d * d) * distortion;
}

__kernel void barrel_filter(__read_only image2d_t input,
                              __write_only image2d_t output){
   
   const sampler_t sampler = CLK_FILTER_NEAREST |
                             CLK_NORMALIZED_COORDS_FALSE |
                             CLK_ADDRESS_CLAMP_TO_EDGE;

   const int2 dim = get_image_dim(input);
   
   float2 coord = (float2)(get_global_id(0),get_global_id(1));
   float2 texture_size = (float2)(dim.x,dim.y);
   float distortion = 0.00005;
   float2 xy = barrel(convert_float2(coord * texture_size / convert_float2(dim) * convert_float2(dim) / texture_size),distortion,convert_float2(dim));
   
   float4 color = read_imagef(input,sampler,xy);
   
   write_imagef(output,convert_int2(coord),color);
   
}