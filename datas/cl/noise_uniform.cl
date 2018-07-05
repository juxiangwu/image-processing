/* Please Write the OpenCL Kernel(s) code here*/
#define LOG2_F 1.442695f
#define IA 16807    			// a
#define IM 2147483647 			// m
#define AM (1.0f/IM) 			// 1/m - To calculate floating point result
#define IQ 127773 
#define IR 2836
#define NTAB 16
#define NDIV (1 + (IM - 1)/ NTAB)
#define EPS 1.2e-7
#define RMAX (1.0f - EPS)
#define GROUP_SIZE 64

float ran1(int idum, __local int *iv)
{
    int j;
    int k;
    int iy = 0;
    int tid = get_local_id(0) + get_local_id(1) * get_local_size(0);

    for(j = NTAB; j >=0; j--)			//Load the shuffle
    {
        k = idum / IQ;
        idum = IA * (idum - k * IQ) - IR * k;

        if(idum < 0)
            idum += IM;

        if(j < NTAB)
            iv[NTAB* tid + j] = idum;
    }
    iy = iv[NTAB* tid];

    k = idum / IQ;
    idum = IA * (idum - k * IQ) - IR * k;

    if(idum < 0)
        idum += IM;

    j = iy / NDIV;
    iy = iv[NTAB * tid + j];
    return (AM * iy);	//AM *iy will be between 0.0 and 1.0
}

__kernel void noise_uniform(__read_only image2d_t input,__write_only image2d_t output){
    
    const sampler_t sampler = CLK_FILTER_NEAREST |
                               CLK_NORMALIZED_COORDS_FALSE |
                               CLK_ADDRESS_CLAMP_TO_EDGE;
                               
    const int2 size = get_image_dim(input);
    int2 coord = (int2)(get_global_id(0),get_global_id(1));
    
    int factor = 1.75;
    
	int pos = get_global_id(0) + get_global_id(1) * get_global_size(0);

    float4 temp = read_imagef(input,sampler,coord);

	//float4 temp = convert_float4(inputImage[pos]);

	/* compute average value of a pixel from its compoments */
	float avg = (temp.x + temp.y + temp.z + temp.y) / 4;

	/* Each thread has NTAB private values */
	/* Local memory is used as indexed arrays use global memory instead of registers */
	__local int iv[NTAB * GROUP_SIZE];  

	/* Calculate deviation from the avg value of a pixel */
	float dev = ran1(-avg, iv);
	dev = (dev - 0.55f) * factor;

	/* Saturate(clamp) the values */
	//outputImage[pos] = convert_uchar4_sat(temp + (float4)(dev));
    write_imagef(output,coord,clamp(temp + dev,0,1.0f));
	
}