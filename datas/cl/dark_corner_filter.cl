/* Please Write the OpenCL Kernel(s) code here*/

float bezier_curve_pow3(float x, float p0, float p1, float p2, float p3){

   //基于三次贝塞尔曲线 
   return p0 * pow((1 - x), 3) + 3 * p1 * x * pow((1 - x), 2) + 3 * p2 * x * x * (1 - x) + p3 * pow(x, 3);
}

float3 calDark(float x, float y, float3 p,float middleX,float middleY,float startDistance,float maxDistance,float lastLevel){
  //计算距中心点距离
  float dist = length((float2)(x - middleX,y - middleY));//distance([x, y], [middleX, middleY]);
  float currBilv = (dist - startDistance) / (maxDistance - startDistance);
  if(currBilv < 0) currBilv = 0;
   //应该增加暗度
    return  bezier_curve_pow3(currBilv, 0, 0.02, 0.3, 1) * p * lastLevel / 255;
  }

__kernel void dark_corner_filter(__read_only image2d_t input,
                              __write_only image2d_t output){
   
   const sampler_t sampler = CLK_FILTER_NEAREST |
                             CLK_NORMALIZED_COORDS_FALSE |
                             CLK_ADDRESS_CLAMP_TO_EDGE;

   const int2 size = get_image_dim(input);

   int2 coord = (int2)(get_global_id(0),get_global_id(1));
   
   int R = 3;//from 1~10
   int lastLevel = 20;//0 ~ 255
   
   int width = size.x;
   int height = size.y;
   
   int lenght = R * 2 + 1;
   
   float middleX = width * 2 / 3;
   float middleY = height * 1/ 2;
   
   float maxDistance = distance(middleX ,middleY);
                //开始产生暗角的距离
   float startDistance = maxDistance * (1 - R / 10);
   
   float4 color = read_imagef(input,sampler,coord);
   
   float3 darkness = calDark(coord.x,coord.y,color.xyz,middleX,middleY,startDistance,maxDistance,lastLevel);
   
   color.xyz -= darkness;
   
   write_imagef(output,coord,color);
   
}