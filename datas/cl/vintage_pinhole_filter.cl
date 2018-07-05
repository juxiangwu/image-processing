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
__kernel void vintage_pinhole_filter(__read_only image2d_t input,
                              __write_only image2d_t output){
    
    float color_matrix[] = {
            0.6279345635605994f,0.3202183420819367f,-0.03965408211312453f,0.0f,9.3651285835294123f,
			0.02578397704808868f,0.6441188644374771f,0.03259127616149294f,0.0f,7.462829176470591f,
			0.0466055556782719f,-0.0851232987247891f,0.5241648018700465f,0.0f,5.159190588235296f,
			0.0f,0.0f,0.0f,1.0f,0.0f
    }; 

    color_matrix_4x5_internal(input,output,color_matrix);

}