#coding:utf-8
import pyopencl as cl  
import sys  
from PIL import Image  
import numpy  
import time

def RoundUp(groupSize, globalSize):  
    r = globalSize % groupSize;  
    if r == 0:  
        return globalSize;  
    else:  
        return globalSize + groupSize - r;  
  
def main():  
      
    imageObjects = [ 0, 0 ]  
              
    #if len(sys.argv) != 3:  
    #    print "USAGE: " + sys.argv[0] + " <inputImageFile> <outputImageFile>"  
    #    return 1  
      
    # create context and command queue  
    ctx = cl.create_some_context()  
    queue = cl.CommandQueue(ctx)  
      
    # load image  
    im = Image.open('../datas/f2.jpg')  
    if im.mode != "RGBA":  
        im = im.convert("RGBA")  
    imgSize = im.size  
    buffer = im.tobytes()  
  
      
    # Create ouput image object  
    clImageFormat = cl.ImageFormat(cl.channel_order.RGBA,   
                                cl.channel_type.UNSIGNED_INT8)  
    imageObjects[0] = cl.Image(ctx,  
                                cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,  
                                clImageFormat,  
                                imgSize,  
                                None,  
                                buffer)  
    imageObjects[1] = cl.Image(ctx,  
                            cl.mem_flags.WRITE_ONLY,  
                            clImageFormat,  
                            imgSize)  
  
    # load the kernel source code  
    #kernelFile = open("grayscale.cl", "r")  
    kernelSrc =    """
    const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE |   
                          CLK_ADDRESS_CLAMP_TO_EDGE |   
                          CLK_FILTER_NEAREST;  
  
__kernel void rgbaToGrayscale(__read_only image2d_t srcImg,  
                              __write_only image2d_t dstImg)  
{  
    // Converts RGBA image to gray scale intensity using the following formula:   
    // I = 0.2126 * R + 0.7152 * G + 0.0722 * B   
  
    int2 coord = (int2) (get_global_id(0), get_global_id(1));  
    int width = get_image_width(srcImg);  
    int height = get_image_height(srcImg);  
  
    if (coord.x < width && coord.y < height)  
    {  
        uint4 color = read_imageui(srcImg, sampler, coord);  
        float luminance = 0.2126f * color.x + 0.7152f * color.y + 0.0722f * color.z;  
        color.x = color.y = color.z = (uint)luminance;  
          
        // Write the output value to image  
        write_imageui(dstImg, coord, color);  
    }  
}"""
  
    # Create OpenCL program  
    program = cl.Program(ctx, kernelSrc).build()  
      
    # Call the kernel directly  
    localWorkSize = ( 16, 16 )  
    globalWorkSize = ( RoundUp(localWorkSize[0], imgSize[0]),  
                    RoundUp(localWorkSize[1], imgSize[1]) )  
    
    gr = time.time()
  
    program.rgbaToGrayscale(queue,  
                            globalWorkSize,  
                            localWorkSize,  
                            imageObjects[0],  
                            imageObjects[1])  
          
    # Read the output buffer back to the Host  
    buffer = numpy.zeros(imgSize[0] * imgSize[1] * 4, numpy.uint8)  
    origin = ( 0, 0, 0 )  
    region = ( imgSize[0], imgSize[1], 1 )  
      
    cl.enqueue_read_image(queue, imageObjects[1],  
                        origin, region, buffer).wait()  
    print (time.time()-gr)
    
    print ("Executed program succesfully." ) 
      
    # Save the image to disk  
    gsim = Image.frombytes("RGBA", imgSize, buffer.tobytes())  
    gsim.save('../temp/cl-out.png')  
      
main()