/* Please Write the OpenCL Kernel(s) code here*/

float4 process_color(__read_only image2d_t input,
                   __write_only image2d_t output,int2 coord,int k00,int k01,int k02,
                   int k20,int k21,int k22,int invert,float thresh){
    const sampler_t sampler = CLK_FILTER_NEAREST |
                             CLK_NORMALIZED_COORDS_FALSE |
                             CLK_ADDRESS_CLAMP_TO_EDGE;
                             
    float4 color1 = read_imagef(input,sampler,coord + (int2)(-1,-1))* 255.0f;
    float4 color2 = read_imagef(input,sampler,coord + (int2)(0,-1))* 255.0f;
    float4 color3 = read_imagef(input,sampler,coord + (int2)(1,-1))* 255.0f;
    
    float4 color4 = read_imagef(input,sampler,coord + (int2)(-1,0))* 255.0f;
    float4 color5 = read_imagef(input,sampler,coord + (int2)(0,0))* 255.0f;
    float4 color6 = read_imagef(input,sampler,coord + (int2)(-1,1))* 255.0f;
    
    float4 color7 = read_imagef(input,sampler,coord + (int2)(0,1))* 255.0f;
    float4 color8 = read_imagef(input,sampler,coord + (int2)(1,1))* 255.0f;
    
    float3 colorSum1 = color1.xyz * k00 + 
                        color3.xyz * k02 +
                        color2.xyz * k01 + 
                        color6.xyz * k20 +
                        color7.xyz * k21 +
                        color8.xyz * k22;
                        
   float3 colorSum2 = color1.xyz * k00 + 
                      color3.xyz * k20 +
                      color4.xyz * k01 +
                      color6.xyz * k02 +
                      color5.xyz * k21 +
                      color8.xyz * k22;
   
   float3 dst_color = colorSum1 * colorSum1 + colorSum2 * colorSum2;
   
   if(invert){
       dst_color = 1.0f - dst_color;
   }
   
   if(dst_color.x >= thresh){
       dst_color.x = 1.0f;
   }else{
       dst_color.x = 0.0f;
   }
   
   if(dst_color.y >= thresh){
       dst_color.y = 1.0f;
   }else{
       dst_color.y = 0.0f;
   }
   
   if(dst_color.z >= thresh){
       dst_color.z = 1.0f;
   }else{
       dst_color.z = 0.0f;
   }
   
   return (float4)(dst_color,1.0f);
   
}

float4 process_gray(__read_only image2d_t input,
                   int2 coord,int k00,int k01,int k02,
                   int k20,int k21,int k22,int invert,float thresh){
    const sampler_t sampler = CLK_FILTER_NEAREST |
                             CLK_NORMALIZED_COORDS_FALSE |
                             CLK_ADDRESS_CLAMP_TO_EDGE;
                             
    float4 color1 = read_imagef(input,sampler,coord + (int2)(-1,-1)) * 255.0f;
    float4 color2 = read_imagef(input,sampler,coord + (int2)(0,-1))* 255.0f;
    float4 color3 = read_imagef(input,sampler,coord + (int2)(1,-1))* 255.0f;
    
    float4 color4 = read_imagef(input,sampler,coord + (int2)(-1,0))* 255.0f;
    float4 color5 = read_imagef(input,sampler,coord + (int2)(0,0))* 255.0f;
    float4 color6 = read_imagef(input,sampler,coord + (int2)(-1,1))* 255.0f;
    
    float4 color7 = read_imagef(input,sampler,coord + (int2)(0,1))* 255.0f;
    float4 color8 = read_imagef(input,sampler,coord + (int2)(1,1))* 255.0f;
    
    float4 colorSum1 = (color1 * k00 + color2 * k01 + color3 * k02 +
                        color6 * k20 + color7 * k21 + color3 * k22);
    
    float4 colorSum2 = (color1 * k00 + color2 * k01 + color3 * k20 +
                        color4 * k21 + color6 * k02 + color8 * k22);
                        
   float3 dst_color = colorSum1.xyz * colorSum1.xyz + colorSum2.xyz * colorSum2.xyz;
   
   if(invert){
       dst_color = 255.0f - dst_color;
   }
   
   if(dst_color.x >= thresh){
       dst_color.x = 255.0f;
   }else{
       dst_color.x = 0.0f;
   }
   
   if(dst_color.y >= thresh){
       dst_color.y = 1.0f;
   }else{
       dst_color.y = 0.0f;
   }
   
   if(dst_color.z >= thresh){
       dst_color.z = 1.0f;
   }else{
       dst_color.z = 0.0f;
   }
   
   return (float4)(dst_color,1.0f);
    
}

__kernel void param_edge_detect_filter(__read_only image2d_t input,
                              __write_only image2d_t output){
   
   const sampler_t sampler = CLK_FILTER_NEAREST |
                             CLK_NORMALIZED_COORDS_FALSE |
                             CLK_ADDRESS_CLAMP_TO_EDGE;

   const int2 dim = get_image_dim(input);
   
   int2 coord = (int2)(get_global_id(0),get_global_id(1));
   float k00 = 1.0f;
   float k01 = 2.0f;
   float k02 = 1.0f;
   
   int dogray = 1;
   int doinvert = 1;
   
   k00 = k00 * 255.0f;
   k01 = k01 * 255.0f;
   k02 = k02 * 255.0f;
   
   float threshold = 0.25f;
   float threshFactor = threshold * 255.0f * 2.0f;
   float threshSq = threshFactor * threshFactor;
   
   
   float4 color = process_gray(input,coord,k00,k01,k02,-k00,-k01,-k02,doinvert,threshSq);
   
   color = color / 255.0f;
   
   write_imagef(output,coord,color);
}