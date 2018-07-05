/* Please Write the OpenCL Kernel(s) code here*/

__kernel void shrink_filter(__read_only image2d_t input,
                              __write_only image2d_t output){
   
   const sampler_t sampler = CLK_FILTER_NEAREST |
                             CLK_NORMALIZED_COORDS_FALSE|
                             CLK_ADDRESS_CLAMP_TO_EDGE;

   const int2 dim = get_image_dim(input);
   
   float2 coord = (float2)(get_global_id(0),get_global_id(1));
   //float4 color = read_imagef(input,sampler,coord);
   
   float x1 = 2.0f * coord.x / dim.x - 1.0f;
   float y1 = 2.0f * coord.y / dim.y - 1.0f;
   
   float radius = sqrt(x1 * x1 + y1 * y1);
   float phase = atan2(y1,x1);
   
   float param1 = 1.8f;
   float param2 = 0.8f;
   
   radius = pow(radius,1.0f / param1) * param2;
   
   
   float newX = radius * cos(phase);
   float newY = radius * sin(phase);
   
   float centerX = (newX + 1.0f) / 2.0f * dim.x;
   float centerY = (newY + 1.0f) / 2.0f * dim.y;
   
   float baseX = floor(centerX);
   float baseY = floor(centerY);
   
   float ratioR = centerX - baseX;
   float ratioL = 1.0f - ratioR;
   float ratioB = centerY - baseY;
   float ratioT = 1.0f - ratioB;
   
   if(baseX >= 0 && baseY >= 0 && baseX < dim.x && baseY < dim.y){
       float pstl = (baseX + baseY * dim.x) * 4;
       float pstr = pstl + 4;
       float psbl = pstl + dim.x * 4;
       float psbr = psbl + 4;
       
       float4 rgba1 = read_imagef(input,sampler,(float2)(baseX,baseY));
       float4 rgba2 = read_imagef(input,sampler,(float2)(baseX + 1,baseY + 1));
       float4 rgba3 = read_imagef(input,sampler,(float2)(baseX + 2,baseY + 2));
       float4 rgba4 = read_imagef(input,sampler,(float2)(baseX + 3,baseY + 3));
       
       float4 tc = rgba1 * ratioL + rgba2 * ratioR;
       float4 bc = rgba3 * ratioL + rgba3 * ratioR;
       
       float4 rgba = tc * ratioT + bc * ratioB;
       rgba.w = 1.0f;
       write_imagef(output,convert_int2(coord),rgba);
   }else{
       write_imagef(output,convert_int2(coord),(float4)(0.0f,0.0f,0.0f,1.0f));
   }
}