#coding:utf-8
import numpy as np
import pyopencl as cl
import cv2
from PIL import Image

def RoundUp(groupSize, globalSize):  
    r = globalSize % groupSize;  
    if r == 0:  
        return globalSize
    else:  
        return globalSize + groupSize - r

# 创建Context
# 如果有多个设备，则会提示选择
ctx = cl.create_some_context()
# 创建CommandQueue
queue = cl.CommandQueue(ctx)

mf = cl.mem_flags

# 通过字符串内容编译OpenCL的Program
prg = cl.Program(ctx, """
void filter2d_internal(__read_only image2d_t input,
                       __write_only image2d_t output,
                       const int maskWidth,
                       const int maskHeight,
                        float * mask,int compute_aver){

   const sampler_t sampler = CLK_FILTER_NEAREST |
                             CLK_NORMALIZED_COORDS_FALSE |
                             CLK_ADDRESS_CLAMP_TO_EDGE;

   const int2 size = get_image_dim(input);

   int2 coord = (int2)(get_global_id(0),get_global_id(1));

   const int maskrows = maskWidth / 2;
   const int maskcols = maskHeight / 2;

   float4 color = (float4)(0,0,0,1.0f);
   int idx = 0;

   for(int y = -maskrows;y <= maskrows;++y){
      for(int x = -maskcols; x <= maskcols;++x){
        float4 srcColor = read_imagef(input,sampler,(int2)(x + coord.x,y + coord.y));
          color.xyz += srcColor.xyz * mask[idx];
        idx++;
      }
   }
   if(compute_aver){
     color.xyz = color.xyz / (maskWidth * maskHeight);
   }
  write_imagef(output,coord,color);
}

__kernel void edge_enhance_filter(__read_only image2d_t input,
                              __write_only image2d_t output){
                              
    float color_mask [9] = {0,0,0,-20,20,0,0,0,0};
    filter2d_internal(input,output,3,3,color_mask,1);
}
""").build()

# 打开图片文件
src1 = Image.open('../datas/f2.jpg')
print(src1.size)
dist = Image.new('RGB',(640,480),(255,255,255))

# OpenCL处理的图片文件格式RGBA,unit8
imageFormat = cl.ImageFormat(cl.channel_order.RGB,cl.channel_type.UNSIGNED_INT8)

# 将图片从Host复制到Device
img1 = cl.Image(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR,imageFormat,src1.size,None,src1.tobytes())
output = cl.Image(context=ctx,flags=mf.WRITE_ONLY,format=imageFormat,shape=src1.size)

# 根据图片大小定义WorkSize
localWorkSize = ( 8, 8 )  
globalWorkSize = ( RoundUp(localWorkSize[0], src1.size[0]),  
                    RoundUp(localWorkSize[1], src1.size[1]))
# 执行Kernel
prg.edge_enhance_filter(queue,globalWorkSize,localWorkSize,img1,output)

buffer = np.zeros(src1.size[0] * src1.size[1] * 3, np.uint8)  
origin = ( 0, 0, 0 )  
region = ( src1.size[0], src1.size[1], 1 )  
# 将处理好的图片从设备复制到HOST 
cl.enqueue_read_image(queue, output,
                        origin, region, buffer).wait()
# 保存图片
dist = Image.frombytes("RGB",src1.size, buffer.tobytes())
dist.save('../temp/cl-output.jpg')