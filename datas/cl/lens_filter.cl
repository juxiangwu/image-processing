/* Please Write the OpenCL Kernel(s) code here*/
#define PI_F 3.14159265358979323846f
__kernel void lens_filter(__read_only image2d_t input,
                              __write_only image2d_t output){
   
   const sampler_t sampler = CLK_FILTER_NEAREST |
                             CLK_NORMALIZED_COORDS_FALSE |
                             CLK_ADDRESS_CLAMP_TO_EDGE;

   const int2 dim = get_image_dim(input);
   
   int2 coord = (int2)(get_global_id(0),get_global_id(1));
   
   float CX = coord.x - dim.x / 2;
   float CY = coord.y - dim.y / 2;
   
   float theta = 0.0f * PI_F / 180.0f;
   
   float dx = CX * cos(theta) - CY * sin(theta);
   float dy = CX * sin(theta) + CY * cos(theta);
   
   float r = sqrt(dx * dx + dy * dy);
   
   float A = 0.0000000005f;
   float B = 0.0000000005f;
   float C = 0.0000000005f;
   
   float corr = 1.0f - A * r * r - B * r * r * r - C * r * r * r * r;
   
   float xu = dx * corr + dim.x / 2;
   float yu = dy * corr + dim.y / 2;
   
   float4 color = read_imagef(input,sampler,(float2)(xu,yu));
   
  // if(length(color.xyz) < 0.6f){
       //color.xyz = (0.0f,0.0f,0.0f);
 //  }else{
       //color.xyz = (1,1,1);
 //  }
   
   write_imagef(output,convert_int2(coord),color);
   
}