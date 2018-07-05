/* Please Write the OpenCL Kernel(s) code here*/

void color_matrix_4x5_internal(__read_only image2d_t input,__write_only image2d_t output,float * mask){
    const sampler_t sampler = CLK_FILTER_NEAREST |
                             CLK_NORMALIZED_COORDS_FALSE|
                             CLK_ADDRESS_CLAMP_TO_EDGE;

   const int2 dim = get_image_dim(input);
   
   float2 coord = (float2)(get_global_id(0),get_global_id(1));
   
   float4 color = read_imagef(input,sampler,coord) * 255.0f;
   
   float4 rgba;
  
   rgba.x = mask[0] * color.x + mask[1] * color.y + mask[2] * color.z + mask[3] * color.w + mask[4];
   rgba.y = mask[0 + 5] * color.x + mask[1 + 5] * color.y + mask[2 + 5] * color.z + mask[3 + 5] * color.w + mask[4 + 5];
   rgba.z = mask[0 + 5 * 2] * color.x + mask[1 + 5 * 2] * color.y + mask[2 + 5 * 2] * color.z + mask[3 + 5 * 2] * color.w + mask[4 + 5 * 2];
   rgba.w = mask[0 + 5 * 3] * color.x + mask[1 + 5 * 3] * color.y + mask[2 + 5 * 3] * color.z + mask[4 + 5 * 3] * color.w + mask[4 + 5 * 3];
   
   rgba = clamp(rgba,0.0f,255.0f);
   
   rgba /= 255.0f;
   
   write_imagef(output,convert_int2(coord),rgba);
}
__kernel void brownie_filter(__read_only image2d_t input,
                              __write_only image2d_t output){
    
    float color_matrix[] = {
        0.5997023498159715f,0.34553243048391263f,-0.2708298674538042f,0,47.43192855600873f,
	   -0.037703249837783157f,0.8609577587992641f,0.15059552388459913f,0,-36.96841498319127f,
		0.24113635128153335f,-0.07441037908422492f,0.44972182064877153f,0,-7.562075277591283f,
	   0.0f,0.0f,0.0f,1.0f,0.0f

    }; 

     color_matrix_4x5_internal(input,output,color_matrix);
}