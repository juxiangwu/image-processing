    #include "opencv2/highgui/highgui.hpp"
    #include "opencv2/imgproc/imgproc.hpp"
    #include "opencv2/highgui/highgui.hpp"
    #include <iostream>
    #include<cuda.h>
    #include "cuPrintf.cuh"
    #include "cuPrintf.cu" 
    using namespace std;
    using namespace cv;
    int noofclick=0;
    int startx,starty,endx,endy;
    int startx1,starty1;
    float alpha=100,beta=0;
    void CallBackFunc_1(int event, int x, int y, int flags, void* userdata)
    {
        if  ( event == EVENT_LBUTTONDOWN )
        {
            noofclick++;
            cout << "Left button of the mouse is clicked - position (" << x << ", " << y << ")" << endl;
            if(noofclick==2)
            {
                endx=x;
                endy=y;	
                cvDestroyWindow("Image 1");
            }
            else
            {
                startx=x;
                starty=y;
            }
        }
    }
    void CallBackFunc_2(int event, int x, int y, int flags, void* userdata)
    {
        if  ( event == EVENT_LBUTTONDOWN )
        {
            cout <<  "1Left button of the mouse is clicked - position (" << x << ", " << y << ")" << endl;
            startx1=x;
            starty1=y;
            cvDestroyWindow("Image 2");
        }
    }

    __global__ void pyrup_kernel(unsigned char *d_in,unsigned char *d_out,int colorWidthStep,int aabhas,int height,int width)
    {
        const int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
        const int yIndex = blockIdx.y * blockDim.y + threadIdx.y;
        const int color_tid = (xIndex)* aabhas + (3 * (yIndex));
        const int color_tid1= (xIndex/2)* colorWidthStep + (3 * (yIndex/2));
        if(yIndex >=width || xIndex>=height)
        {
            //		printf("return %d %d\n",xIndex,yIndex);
            return;
        }
        //	printf("a=%d c=%d\n",aabhas,colorWidthStep);
        //	printf("%d %d %d a=%d c=%d\n",xIndex,yIndex,d_in[color_tid1],aabhas,colorWidthStep);
        //cout<<xIndex<<" "<<yIndex<<endl;
        if(yIndex%2==0 &&xIndex%2==0)
        {	
            d_out[color_tid]=d_in[color_tid1];
            d_out[color_tid+1]=d_in[color_tid1+1];
            d_out[color_tid+2]=d_in[color_tid1+2];
        }
        else
        {
            d_out[color_tid]=0;
            d_out[color_tid+1]=0;//d_in[color_tid1+1];
            d_out[color_tid+2]=0;//d_in[color_tid1+2];

        }
    }
    //	printf("%d %d %d\n",xIndex,yIndex,d_out[color_tid]);
    //int no=1;
    //gaussian blur TODO
    /*	         float blur[5][5] ={
             0.0000 ,0.0000 , 0.0002   ,0.0000   ,0.0000,
             0.0000 ,0.0113 , 0.0837   ,0.0113   ,0.0000,
             0.0002 ,0.0837 , 0.6187   ,0.0837   ,0.0002,
             0.0000 ,0.0113 , 0.0837   ,0.0113    ,0.0000,
             0.0000 ,0.0000 , 0.0002   ,0.0000    ,0.0000
             };*/
    //printf("Aabhas\n");
    //	__syncthreads();
    //printf("Tu\n");

    __global__ void blur_image(unsigned char *d_in,unsigned char *d_out,int aabhas,int height,int width)
    {
        const int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
        const int yIndex = blockIdx.y * blockDim.y + threadIdx.y;
        const int color_tid = (xIndex)* aabhas + (3 * (yIndex));

        float blur[5][5] ={
            {0.0025,  0.0125,  0.02  ,  0.0125,  0.0025},
            {0.0125,  0.0625,  0.1   ,  0.0625,  0.0125},
            { 0.02  ,  0.1   ,  0.16  ,  0.1   ,  0.02  },
            { 0.0125,  0.0625,  0.1   ,  0.0625,  0.0125},
            { 0.0025,  0.0125,  0.02  ,  0.0125,  0.0025}};
        int i,j;
        float output1,output2,output3;
        int loc;
        output1=0.0;
        output2=0.0;
        output3=0.0;
        //191 228 
        for(i=-2;i<=2;i++)
        {
            for(j=-2;j<=2;j++)
            {
                if(xIndex+i<height && yIndex+j<width)
                {
                    if( (xIndex+i)>=0 && (yIndex)+j >=0)
                    {
                        loc=  ( (xIndex)+i )*aabhas + (3*( (yIndex)+j));
                        //	output1+=blur[i+2][j+2]*(unsigned char)(d_in[loc]);
                        //	output2+=blur[i+2][j+2]*(unsigned char)(d_in[loc+1]);
                        //	output3+=blur[i+2][j+2]*(unsigned char)(d_in[loc+2]);
                        output1= output1+blur[i+2][j+2]*(float)(d_in[loc]);
                        output2=output2+blur[i+2][j+2]*(float)(d_in[loc+1]);
                        output3=output3+blur[i+2][j+2]*(float)(d_in[loc+2]);
                        //					 if(xIndex==191 && yIndex==228)
                        //						 printf("ap=%d %d %d %d %d\n",d_in[loc],i,j,loc,color_tid);

                    }
                }
                //old blure
                /*		 if( (xIndex/2 )+i<height/2 && (yIndex/2)+j <width/2)
                         if( (xIndex/2+i)>=0 && (yIndex/2)+j >=0)
                         {
                //const int color_tid1= (2*xIndex)* colorWidthStep + (3 * (2*yIndex));
                loc=  ( (xIndex/2)+i )*colorWidthStep + (3*( (yIndex/2)+j));
                output1+=blur[i+2][j+2]*d_in[loc];
                output2+=blur[i+2][j+2]*d_in[loc+1];
                output3+=blur[i+2][j+2]*d_in[loc+2];
                }*/
            }
        }
        d_out[color_tid]=static_cast<unsigned char>(4*output1);
        d_out[color_tid+1]=static_cast<unsigned char>(4*output2);
        d_out[color_tid+2]=static_cast<unsigned char>(4*output3);
        //	if(int(4*output1)-d_in[color_tid]<-50 && output1<10 )
        //	printf("%d %d %f %d %d\n",xIndex,yIndex,4*output1,d_in[color_tid],int(4*output1)-d_in[color_tid]);	
        //	d_out[color_tid]=d_in[color_tid1];
        //	d_out[color_tid+1]=d_in[color_tid1+1];
        //	d_out[color_tid+2]=d_in[color_tid1+2];
    }

    __global__ void GAUSSGPU(unsigned char*Input,unsigned char*Output,int rows,int cols,int Instep,int Outstep)
    {
        int x=blockIdx.x*blockDim.x+threadIdx.x;
        int y=blockIdx.y*blockDim.y+threadIdx.y;


        if(x>rows||y>cols)
            return;
    /*	float Gauss[5][5]={
            0.0030  ,  0.0133  ,  0.0219  ,  0.0133 ,   0.0030,
            0.0133  ,  0.0596  ,  0.0983  ,  0.0596 ,   0.0133,
            0.0219  ,  0.0983  ,  0.1621  ,  0.0983 ,   0.0219,
            0.0133  ,  0.0596  ,  0.0983  ,  0.0596 ,   0.0133,
            0.0030  ,  0.0133  ,  0.0219  ,  0.0133 ,   0.0030,
        };*/
        float Gauss[5][5] ={
            {0.0025,  0.0125,  0.02  ,  0.0125,  0.0025},
            {0.0125,  0.0625,  0.1   ,  0.0625,  0.0125},
            { 0.02  ,  0.1   ,  0.16  ,  0.1   ,  0.02  },
            { 0.0125,  0.0625,  0.1   ,  0.0625,  0.0125},
            { 0.0025,  0.0125,  0.02  ,  0.0125,  0.0025}};

        int i,j,x1,y1;

        int In=x*Instep+3*y;
        int Out=x*Outstep+3*y;

        float r=0,g=0,b=0;

        for(i=-2;i<=2;i++)
        {
            for(j=-2;j<=2;j++)
            {
                x1=x+i;
                y1=y+j;
                if(x1>=0&&y1>=0)
                {
                    if(x1<rows&&y1<cols)
                    {
                        In=x1*Instep+3*y1;
                        b=b+float(Input[In])*Gauss[i+2][j+2];
                        g=g+float(Input[In+1])*Gauss[i+2][j+2];
                        r=r+float(Input[In+2])*Gauss[i+2][j+2];
                    }
                }
            }
        }

        Output[Out] = 4*static_cast<unsigned char>(b);
        Output[Out+1] = 4*static_cast<unsigned char>(g);
        Output[Out+2] = 4*static_cast<unsigned char>(r);

    }

    __global__ void GAUSSGPU1(unsigned char*Input,unsigned char*Output,int rows,int cols,int Instep,int Outstep)
    {
        int x=blockIdx.x*blockDim.x+threadIdx.x;
        int y=blockIdx.y*blockDim.y+threadIdx.y;


        if(x>rows||y>cols)
            return;
    /*	float Gauss[5][5]={
            0.0030  ,  0.0133  ,  0.0219  ,  0.0133 ,   0.0030,
            0.0133  ,  0.0596  ,  0.0983  ,  0.0596 ,   0.0133,
            0.0219  ,  0.0983  ,  0.1621  ,  0.0983 ,   0.0219,
            0.0133  ,  0.0596  ,  0.0983  ,  0.0596 ,   0.0133,
            0.0030  ,  0.0133  ,  0.0219  ,  0.0133 ,   0.0030,
        };*/
        float Gauss[5][5] ={
            {0.0025,  0.0125,  0.02  ,  0.0125,  0.0025},
            {0.0125,  0.0625,  0.1   ,  0.0625,  0.0125},
            { 0.02  ,  0.1   ,  0.16  ,  0.1   ,  0.02  },
            { 0.0125,  0.0625,  0.1   ,  0.0625,  0.0125},
            { 0.0025,  0.0125,  0.02  ,  0.0125,  0.0025}};

        int i,j,x1,y1;

        int In=x*Instep+3*y;
        int Out=x*Outstep+3*y;

        float r=0,g=0,b=0;

        for(i=-2;i<=2;i++)
        {
            for(j=-2;j<=2;j++)
            {
                x1=x+i;
                y1=y+j;
                if(x1>=0&&y1>=0)
                {
                    if(x1<rows&&y1<cols)
                    {
                        In=x1*Instep+3*y1;
                        b=b+float(Input[In])*Gauss[i+2][j+2];
                        g=g+float(Input[In+1])*Gauss[i+2][j+2];
                        r=r+float(Input[In+2])*Gauss[i+2][j+2];
                    }
                }
            }
        }

        Output[Out] = static_cast<unsigned char>(b);
        Output[Out+1] = static_cast<unsigned char>(g);
        Output[Out+2] = static_cast<unsigned char>(r);

    }




    void pyrup(Mat &input,Mat& output_1)
    {
        int row=input.rows;
        int col=input.cols;
        int newrow=row*2;
        int newcol=col*2;
    //	cout<<newrow<<" "<<newcol<<endl;
        const int insize=input.step*row;
        Mat output(newrow,newcol,CV_8UC3);
        unsigned char *d_input,*d_output,*d_output1;// *d_output;
        cudaMalloc<unsigned char>(&d_input,insize);
        cudaMalloc<unsigned char>(&d_output,output.step*output.rows);
        cudaMalloc<unsigned char>(&d_output1,output.step*output.rows);
        cudaMemcpy(d_input,input.ptr(),insize,cudaMemcpyHostToDevice);
        const dim3 block(16,16);
        const dim3 grid( (newrow+block.x)/block.x , (newcol+block.y)/block.y );
        pyrup_kernel<<<grid,block>>>(d_input,d_output,input.step,output.step,newrow,newcol);
        cudaDeviceSynchronize();
    //	blur_image<<<grid,block>>>(d_output,d_output1,output.step,newrow,newcol);
        GAUSSGPU<<<grid,block>>>(d_output,d_output1,output.rows,output.cols,output.step,output.step);

        cudaDeviceSynchronize();
    //	cout<<"\n\n\n\n\nIMAGE FINISHED\n\n\n\n\n";
        cudaMemcpy(output.ptr(),d_output1,output.step*output.rows,cudaMemcpyDeviceToHost);
        output_1=output;
    }

    __global__ void submat_kernel(unsigned char *d_in1,unsigned char *d_in2,int colorWidthStep,int aabhas,int height,int width)
    {
        const int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
        const int yIndex = blockIdx.y * blockDim.y + threadIdx.y;
        if(yIndex >=width || xIndex>=height)
        {
            return;
        }
        const int color_tid2 = (xIndex)* aabhas + (3 * (yIndex));
        const int color_tid1= (xIndex)* colorWidthStep + (3 * (yIndex));
        //printf("%d %d %d %d\n",xIndex,yIndex,d_in1[color_tid1],d_in2[color_tid2]);
        int s=1;
        d_in2[color_tid2]=  s*(d_in1[color_tid1]-d_in2[color_tid2]);
        d_in2[color_tid2+1]=s*(d_in1[color_tid1+1]-d_in2[color_tid2+1]);
        d_in2[color_tid2+2]=s*(d_in1[color_tid1+2]-d_in2[color_tid2+2]);
    }
    void submat(Mat &input1,Mat& input2,Mat& output)
    {
        int row=min(input1.rows,input2.rows);
        int col=min(input1.cols,input2.cols);
        Mat out(row,col,CV_8UC3);
        unsigned char *d_input1,*d_input2,*d_output;
        const int insize1=input1.step*input1.rows;
        const int insize2=input2.step*input2.rows;
        //	cout<<"aabhas="<<insize1<<" "<<insize2;
        //	cout<<"aabhas1="<<input1.step<<" "<<input2.step;
        cudaMalloc<unsigned char>(&d_input1,insize1);
        cudaMalloc<unsigned char>(&d_input2,insize2);
        cudaMalloc<unsigned char>(&d_output,out.step*out.rows);
        cudaMemcpy(d_input1,input1.ptr(),insize1,cudaMemcpyHostToDevice);
        cudaMemcpy(d_input2,input2.ptr(),insize2,cudaMemcpyHostToDevice);
        const dim3 block(16,16);
        const dim3 grid( (row+block.x)/block.x , (col+block.y)/block.y);
        submat_kernel<<<grid,block>>>(d_input1,d_input2,input1.step,input2.step,row,col);
        cudaDeviceSynchronize();
        cudaMemcpy(out.ptr(),d_input2,out.step*out.rows,cudaMemcpyDeviceToHost);
        output=out;
    }



    __global__ void add2mat_kernel(unsigned char *d_in1,unsigned char *d_in2,int colorWidthStep,int aabhas,int height,int width)
    {
        const int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
        const int yIndex = blockIdx.y * blockDim.y + threadIdx.y;
        if(yIndex >=width || xIndex>=height)
        {   
            return;
        }   
        const int color_tid2 = (xIndex)* aabhas + (3 * (yIndex));
        //const int color_tid1= (xIndex/2)* colorWidthStep + (3 * (yIndex/2));
        const int color_tid1= (xIndex)* colorWidthStep + (3 * (yIndex));
        //printf("%d %d %d %d\n",xIndex,yIndex,d_in1[color_tid1],d_in2[color_tid2]);
        if(d_in1[color_tid1]+d_in2[color_tid2]>255)
        {	
    //		d_in2[color_tid2]=255;
    //		printf("YES %d\n ",d_in1[color_tid1]+d_in2[color_tid2]);
        }
    //	else
        d_in2[color_tid2]=(d_in1[color_tid1]+d_in2[color_tid2]);
        d_in2[color_tid2+1]=(d_in1[color_tid1+1]+d_in2[color_tid2+1]);
        d_in2[color_tid2+2]=(d_in1[color_tid1+2]+d_in2[color_tid2+2]);
    }
    void add2mat(Mat &input1,Mat& input2,Mat& output)
    {
        pyrup(input1,input1);
        int row=max(input1.rows,input2.rows);
        int col=max(input1.cols,input2.cols);
        Mat out(row,col,CV_8UC3);
        unsigned char *d_input1,*d_input2,*d_output;
        const int insize1=input1.step*input1.rows;
        const int insize2=input2.step*input2.rows;
        //      cout<<"aabhas="<<insize1<<" "<<insize2;
        //      cout<<"aabhas1="<<input1.step<<" "<<input2.step;
        cudaMalloc<unsigned char>(&d_input1,insize1);
        cudaMalloc<unsigned char>(&d_input2,insize2);
        cudaMalloc<unsigned char>(&d_output,out.step*out.rows);
        cudaMemcpy(d_input1,input1.ptr(),insize1,cudaMemcpyHostToDevice);
        cudaMemcpy(d_input2,input2.ptr(),insize2,cudaMemcpyHostToDevice);
        const dim3 block(16,16);
        const dim3 grid( (row+block.x)/block.x , (col+block.y)/block.y);
        add2mat_kernel<<<grid,block>>>(d_input1,d_input2,input1.step,input2.step,row,col);
        cudaDeviceSynchronize();
        cudaMemcpy(out.ptr(),d_input2,out.step*out.rows,cudaMemcpyDeviceToHost);
        output=out;
    }


    __global__ void pyrdown_kernel(unsigned char *d_in,unsigned char *d_out,int colorWidthStep,int aabhas,int height,int width)
    {
        const int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
        const int yIndex = blockIdx.y * blockDim.y + threadIdx.y;
        const int color_tid = (xIndex)* aabhas + (3 * (yIndex));
        const int color_tid1= (2*xIndex)* colorWidthStep + (3 * (2*yIndex));
        if(yIndex >=width || xIndex>=height)
        {
            //		printf("return %d %d\n",xIndex,yIndex);
            return;
        }
        //	printf("a=%d c=%d\n",aabhas,colorWidthStep);
        //	printf("%d %d %d a=%d c=%d\n",xIndex,yIndex,d_in[color_tid1],aabhas,colorWidthStep);
        //cout<<xIndex<<" "<<yIndex<<endl;
        d_out[color_tid]=d_in[color_tid1];
        d_out[color_tid+1]=d_in[color_tid1+1];
        d_out[color_tid+2]=d_in[color_tid1+2];

        //gaussian blur TODO
        /*
           0.0000    0.0000    0.0002    0.0000    0.0000
           0.0000    0.0113    0.0837    0.0113    0.0000
           0.0002    0.0837    0.6187    0.0837    0.0002
           0.0000    0.0113    0.0837    0.0113    0.0000
           0.0000    0.0000    0.0002    0.0000    0.0000
         */
        /*	 float blur[5][5] ={ 
             0.0000 ,0.0000 , 0.0002   ,0.0000   ,0.0000,
             0.0000 ,0.0113 , 0.0837   ,0.0113   ,0.0000,
             0.0002 ,0.0837 , 0.6187   ,0.0837   ,0.0002,
             0.0000 ,0.0113 , 0.0837   ,0.0113    ,0.0000,
             0.0000 ,0.0000 , 0.0002   ,0.0000    ,0.0000
             };	*/
    /*	float blur[5][5] ={
            {0.0025,  0.0125,  0.02  ,  0.0125,  0.0025},
            {0.0125,  0.0625,  0.1   ,  0.0625,  0.0125},
            { 0.02  ,  0.1   ,  0.16  ,  0.1   ,  0.02  },
            { 0.0125,  0.0625,  0.1   ,  0.0625,  0.0125},
            { 0.0025,  0.0125,  0.02  ,  0.0125,  0.0025}};
        int i,j;
        float output1,output2,output3;
        int loc;
        output1=0.0;
        output2=0.0;
        output3=0.0;
        for(i=-2;i<=2;i++)
        {
            for(j=-2;j<=2;j++)
            {
                if(2*xIndex+i<2*height && 2*yIndex+j <2*width)
                    if(2*xIndex+i>=0 && 2*yIndex+j >=0)
                    {
                        //const int color_tid1= (2*xIndex)* colorWidthStep + (3 * (2*yIndex));
                        loc=  (2*xIndex+i )*colorWidthStep + (3*(2*yIndex+j));
                        output1+=blur[i+2][j+2]*d_in[loc];
                        output2+=blur[i+2][j+2]*d_in[loc+1];
                        output3+=blur[i+2][j+2]*d_in[loc+2];
                    }
            }
        }
        d_out[color_tid]=output1;
        d_out[color_tid+1]=output2;
        d_out[color_tid+2]=output3;
        //	printf("%f %d %d\n",output1,d_in[color_tid1],int(output1)-d_in[color_tid1]);	

        //	d_out[color_tid]=d_in[color_tid1];
        //	d_out[color_tid+1]=d_in[color_tid1+1];
        //	d_out[color_tid+2]=d_in[color_tid1+2];
    */
    }

    void pyrdown(Mat &input,Mat& output_1,Mat &output_2,Mat& output_3,int mask=0)
    {
        int row=input.rows;
        int col=input.cols;
        int newrow=row/2;
        int newcol=col/2;
        const int insize=input.step*row;
        Mat output(newrow,newcol,CV_8UC3);
        unsigned char *d_input,*d_output,*d_output1,*d_output2,*d_temp;// *d_output;
        cudaMalloc<unsigned char>(&d_input,insize);
        cudaMalloc<unsigned char>(&d_temp,insize);
        //cout<<" insize"<<insize<<" d="<<newrow*newcol*sizeof(unsigned char)<<endl;
        cudaMalloc<unsigned char>(&d_output,output.step*output.rows);
        cudaMemcpy(d_input,input.ptr(),insize,cudaMemcpyHostToDevice);
        const dim3 block(16,16);
        const dim3 grid( (newrow+block.x)/block.x , (newcol+block.y)/block.y );	

        const dim3 grid_1((input.rows+block.x)/block.x , (input.cols+block.y)/block.y );
        GAUSSGPU1<<<grid_1,block>>>(d_input,d_temp,input.rows,input.cols,input.step,input.step);
        cudaDeviceSynchronize();
    //	Mat outputa1(input.rows,input.cols,CV_8UC3);
    //	cudaMemcpy(outputa1.ptr(),d_temp,input.step*input.rows,cudaMemcpyDeviceToHost);

    //	namedWindow("aabhas");
    //	imshow("aabhas",outputa1);
    //	waitKey(0);

        pyrdown_kernel<<<grid,block>>>(d_temp,d_output,input.step,output.step,newrow,newcol);
        cudaDeviceSynchronize();
        cudaMemcpy(output.ptr(),d_output,output.step*output.rows,cudaMemcpyDeviceToHost);
        cudaFree(d_temp);
        output_1=output;
        row=output.rows;
        col=output.cols;
        newrow=row/2;
        newcol=col/2;
        cv::Mat output1(newrow,newcol,CV_8UC3);
        const int insize1=output.step*row;
        cudaMalloc<unsigned char>(&d_temp,output.step*output.rows);
        cudaMalloc<unsigned char>(&d_output1,output.step*output.rows/4);
        const dim3 block1(16,16);
        const dim3 grid1( (newrow+block.x)/block.x , (newcol+block.y)/block.y );
        const dim3 grid_2( (row+block.x)/block.x , (col+block.y)/block.y );
        GAUSSGPU1<<<grid_2,block>>>(d_output,d_temp,output.rows,output.cols,output.step,output.step);
        cudaDeviceSynchronize();

        pyrdown_kernel<<<grid,block>>>(d_temp,d_output1,output.step,output1.step,newrow,newcol);
        cudaDeviceSynchronize();
        cudaMemcpy(output1.ptr(),d_output1,output1.step*output1.rows,cudaMemcpyDeviceToHost);
        cudaFree(d_temp);
        output_2=output1;	
        row=output1.rows;
        col=output1.cols;
        newrow=row/2;
        newcol=col/2;
        cv::Mat output2(newrow,newcol,CV_8UC3);
        cudaMalloc<unsigned char>(&d_temp,output1.step*output1.rows);
        const dim3 grid_3( (row+block.x)/block.x , (col+block.y)/block.y );

        GAUSSGPU1<<<grid_3,block>>>(d_output1,d_temp,output1.rows,output1.cols,output1.step,output1.step);

        cudaMalloc<unsigned char>(&d_output2,output1.step*output1.rows/4);
        const dim3 block2(16,16);
        const dim3 grid2( (newrow+block.x)/block.x , (newcol+block.y)/block.y );
        pyrdown_kernel<<<grid,block>>>(d_temp,d_output2,output1.step,output2.step,newrow,newcol);
        cudaDeviceSynchronize();
        cudaMemcpy(output2.ptr(),d_output2,output2.step*output2.rows,cudaMemcpyDeviceToHost);
        output_3=output2;

    }
    void pyrdownmask(Mat &input,Mat& output_1,Mat &output_2,Mat& output_3)
    {
        int row=input.rows;
        int col=input.cols;
        int newrow=row/2;
        int newcol=col/2;
        const int insize=input.step*row;
        Mat output(newrow,newcol,CV_8UC3);
        unsigned char *d_input,*d_output,*d_output1,*d_output2;// *d_output;
        cudaMalloc<unsigned char>(&d_input,insize);
        //cout<<" insize"<<insize<<" d="<<newrow*newcol*sizeof(unsigned char)<<endl;
        cudaMalloc<unsigned char>(&d_output,output.step*output.rows);
        cudaMemcpy(d_input,input.ptr(),insize,cudaMemcpyHostToDevice);
        const dim3 block(16,16);
        const dim3 grid( (newrow+block.x)/block.x , (newcol+block.y)/block.y );	

        const dim3 grid_1((input.rows+block.x)/block.x , (input.cols+block.y)/block.y );
    //	Mat outputa1(input.rows,input.cols,CV_8UC3);
    //	cudaMemcpy(outputa1.ptr(),d_temp,input.step*input.rows,cudaMemcpyDeviceToHost);

    //	namedWindow("aabhas");
    //	imshow("aabhas",outputa1);
    //	waitKey(0);

        pyrdown_kernel<<<grid,block>>>(d_input,d_output,input.step,output.step,newrow,newcol);
        cudaDeviceSynchronize();
        cudaMemcpy(output.ptr(),d_output,output.step*output.rows,cudaMemcpyDeviceToHost);
        output_1=output;
        row=output.rows;
        col=output.cols;
        newrow=row/2;
        newcol=col/2;
        cv::Mat output1(newrow,newcol,CV_8UC3);
        const int insize1=output.step*row;
        cudaMalloc<unsigned char>(&d_output1,output.step*output.rows/4);
        const dim3 block1(16,16);
        const dim3 grid1( (newrow+block.x)/block.x , (newcol+block.y)/block.y );
        const dim3 grid_2( (row+block.x)/block.x , (col+block.y)/block.y );

        pyrdown_kernel<<<grid,block>>>(d_output,d_output1,output.step,output1.step,newrow,newcol);
        cudaDeviceSynchronize();
        cudaMemcpy(output1.ptr(),d_output1,output1.step*output1.rows,cudaMemcpyDeviceToHost);
        output_2=output1;	
        row=output1.rows;
        col=output1.cols;
        newrow=row/2;
        newcol=col/2;
        cv::Mat output2(newrow,newcol,CV_8UC3);
        const dim3 grid_3( (row+block.x)/block.x , (col+block.y)/block.y );

        cudaMalloc<unsigned char>(&d_output2,output1.step*output1.rows/4);
        const dim3 block2(16,16);
        const dim3 grid2( (newrow+block.x)/block.x , (newcol+block.y)/block.y );
        pyrdown_kernel<<<grid,block>>>(d_output1,d_output2,output1.step,output2.step,newrow,newcol);
        cudaDeviceSynchronize();
        cudaMemcpy(output2.ptr(),d_output2,output2.step*output2.rows,cudaMemcpyDeviceToHost);
        output_3=output2;

    }
    __global__ void imageblend_kernel(unsigned char *d_input1,unsigned char *d_input2,int width,int height,int colorWidthStep,int aabhas,unsigned char *mask,int maskstep)
    {
        const int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
        const int yIndex = blockIdx.y * blockDim.y + threadIdx.y;
        if((xIndex>=width) || (yIndex>=height))
            return;
        const int color_tid1 = (yIndex)* aabhas + (3 * (xIndex));
        const int color_tid2 = (yIndex)* colorWidthStep + (3 * (xIndex));
        const int color_mask=(yIndex)*maskstep+(3*(xIndex));
        float m= ( unsigned char)mask[color_mask];
            if(mask[color_mask]!=255 && mask[color_mask]!=0)
                printf("c=%d %d %d %d\n",( unsigned char)mask[color_mask],xIndex,yIndex,height);
        float m1=m/255.0;
        float m2=1-m1;
    //	printf("%d %d %d\n",mask[color_mask],mask[color_mask+1],mask[color_mask+2]);
        int x=d_input2[color_tid2];
        d_input2[color_tid2]=static_cast<unsigned char> ((m1)* d_input2[color_tid2] +(m2)*d_input1[color_tid1]);
    //	if(m1<=0.58 && m1>=0.22)
    //		printf("%f %f %d %d %d\n %d %d %d\n",m1,m2,xIndex,yIndex, mask[color_mask],d_input2[color_tid2],x,d_input1[color_tid1]);
         m= ( unsigned char)mask[color_mask+1];
         m1=m/255.0;
         m2=1-m1;
    //	printf("2%f %f %d %d \n",m1,m2,xIndex,yIndex);
        d_input2[color_tid2+1]=static_cast<unsigned char>((m1)* d_input2[color_tid2+1] +(m2)*d_input1[color_tid1+1]);
         m= ( unsigned char)mask[color_mask+2];
         m1=m/255.0;
         m2=1-m1;
    //	printf("3%f %f %d %d \n",m1,m2,xIndex,yIndex);
        d_input2[color_tid2+2]=static_cast<unsigned char> ((m1)* d_input2[color_tid2+2] +(m2)*d_input1[color_tid1+2]);
            //	d_input2[color_tid2+1]=(beta/100.0)* d_input2[color_tid2+1] +(alpha/100.0)*d_input1[color_tid1+1];
            //	d_input2[color_tid2+2]=(beta/100.0)* d_input2[color_tid2+2] +(alpha/100.0)*d_input1[color_tid1+2] ;

    //	printf("%f\n",m);
    //	d_input2[color_tid]= d_input
    //	float alpha=100,beta=0;
        //if((startx1+xIndex<width) && (starty1+yIndex<height))
        {
        //	if((startx+xIndex<=endx) && (starty+yIndex<=endy))
            {
                //const int color_tid1 = (yIndex +starty)* aabhas + (3 * (xIndex+startx));
            //	const int color_tid2 = (yIndex +starty1)* colorWidthStep + (3 * (xIndex+startx1));
                //int a=d_input2[color_tid2];
            //	d_input2[color_tid2]=(beta/100.0)* d_input2[color_tid2] +(alpha/100.0)*d_input1[color_tid1];
            //	d_input2[color_tid2+1]=(beta/100.0)* d_input2[color_tid2+1] +(alpha/100.0)*d_input1[color_tid1+1];
            //	d_input2[color_tid2+2]=(beta/100.0)* d_input2[color_tid2+2] +(alpha/100.0)*d_input1[color_tid1+2] ;

            }
        }
    }

    void blendimage(Mat& input1 , Mat& input2 ,Mat& output1,int scale,Mat & mask)	
    {
        const int insize1=input1.step * input1.rows;
        const int insize2=input2.step * input2.rows;
        const int masksize=mask.step * mask.rows;
        unsigned char *d_input1,*d_input2,*d_mask;// *d_output;
        int x,y;
        Mat img=mask;
        for(x=0;x<img.cols;x++)
                 for(y=0;y<img.rows;y++)
                     if(img.at<cv::Vec3b>(y,x)[0]!=0 &&img.at<cv::Vec3b>(y,x)[0]!=255)
                     {
                         cout<<"\nmask fail\n"<<endl;
                     }

    //	cout<<insize1<<" "<<insize2<<" --- "<<mask.step*mask.rows<<endl;
    //	cout<<mask.cols<<" m "<<mask.rows<<" "<<mask.step<<endl;
    //	cout<<input1.cols<<" 1 "<<input1.rows<<" "<<input1.step<<endl;
    //	cout<<input2.cols<<" 2 "<<input2.rows<<" "<<input2.step<<endl;
        cudaMalloc<unsigned char>(&d_input1,insize1);
        cudaMalloc<unsigned char>(&d_input2,insize2);
        cudaMalloc<unsigned char>(&d_mask,masksize);
        cudaMemcpy(d_input1,input1.ptr(),insize1,cudaMemcpyHostToDevice);
        cudaMemcpy(d_input2,input2.ptr(),insize2,cudaMemcpyHostToDevice);
        cudaMemcpy(d_mask,mask.ptr(),masksize,cudaMemcpyHostToDevice);
        const dim3 block(16,16);
        Mat output(input2.rows,input2.cols,CV_8UC3);
    /*	startx=startx/scale;
        starty=starty/scale;
        startx1=startx1/scale;
        starty1=starty1/scale;
        endx=endx/scale;
        endy=endy/scale;*/
        const dim3 grid((input2.cols + block.x )/block.x, (input2.rows + block.y )/block.y);

        imageblend_kernel<<<grid,block>>>(d_input1,d_input2,input2.cols,input2.rows,input2.step,input1.step,d_mask,mask.step);
        cudaDeviceSynchronize();
        cudaMemcpy(output.ptr(),d_input2,insize2,cudaMemcpyDeviceToHost);
        startx=startx*scale;
        starty=starty*scale;
        startx1=startx1*scale;
        starty1=starty1*scale;
        endx=endx*scale;
        endy=endy*scale;
        output1=output;

    }
    void display(Mat &img)
    {
        namedWindow("debug",1);
        imshow("debug",img);
        waitKey(0);
    }
    int main(int argc, char** argv)
    {

        int debug=0;
        // Read image from file 
        //	cout<<"Enter the vale of alpha and beta\n";
        //	cin>>alpha>>beta;
        Mat img1 = imread("dataset/pepper.jpg");
        Mat mask_1= imread("dataset/mask.jpg");
        //if fail to read the image
        if ( img1.empty() )
        {
            cout << "Error loading the image 1" << endl;
            return -1;
        }
        //Create a window
        namedWindow("Mask", 1);
        imshow("Mask",mask_1);
        waitKey(0);

        namedWindow("Image 1", 1);
        //set the callback function for any mouse event
        setMouseCallback("Image 1", CallBackFunc_1, NULL);
        //show the image
        imshow("Image 1", img1);
        // Wait until user press some key
        waitKey(0);
        cout<<"position of first\n"<<startx<<" "<<starty<<" "<<endx<<" "<<endy<<endl;
        Mat img2=imread("dataset/snow.jpg");
        if ( img2.empty() )
        {
            cout << "Error loading the image 2" << endl;
            return -1;
        }
        namedWindow("Image 2",CV_WINDOW_AUTOSIZE);
        setMouseCallback("Image 2", CallBackFunc_2, NULL);
        //show the image
        imshow("Image 2", img2);
        waitKey(0);
        cout<<"position of second\n"<<startx1<<" "<<starty1<<endl;
        struct timespec t1, t2; 
                clock_gettime(CLOCK_MONOTONIC, &t1);
        int newrow=img1.rows/2;
        int newcol=img1.cols/2;
        //cv::Mat output(img1);//(newrow,newcol,CV_8UC3);;
        cv::Mat output(newrow,newcol,CV_8UC3);;
        Mat output_1,output_2,output_3;
        //	imshow("output",img1);
        //	waitKey(0);
        cv::Mat finaloutput;
        Mat mask_2,mask_3,mask_4;
    //	pyrdown(mask_1,mask_2,mask_3,mask_4);
        pyrdownmask(mask_1,mask_2,mask_3,mask_4);
    //	display(mask_3);
    //	cout<<"mat=\n"<<mask_1<<endl;
        pyrdown(img1,output_1,output_2,output_3);

        //	cv::pyrDown(img1,output);
        //	blendimage(img1,img2,output);
        if(debug==1)
        {
        namedWindow("output_0",1);
        imshow("output_0",img1);
        waitKey(0);
        namedWindow("output_1",1);
        imshow("output_1",output_1);
        waitKey(0);
        namedWindow("output_2",1);
        imshow("output_2",output_2);
        waitKey(0);
        namedWindow("output_3",1);
        imshow("output_3",output_3);
        waitKey(0);
        }
        //	cv::Mat output1(newrow*2,newcol*2,CV_8UC3);
        Mat pyoutput_1,pyoutput_2,pyoutput_3;
            pyrup(output_1,pyoutput_1);
            pyrup(output_2,pyoutput_2);
        pyrup(output_3,pyoutput_3);


        if(debug==1)
        {
        namedWindow("showall",1);
            imshow("showall",pyoutput_1);
            waitKey(0);
            imshow("showall",pyoutput_2);
            waitKey(0);
        imshow("showall",pyoutput_3);
        waitKey(0);
        }
        //cout<<"Mat="<<pyoutput_3<<endl;
        //	namedWindow("output",1);
        //	imshow("output",pyoutput_1-img1);
        //	waitKey(0);
        if(1==1)
        {
            Mat LA2,LA1,LA0;
            Mat LA3=output_3;
            submat(img1,pyoutput_1,LA0);
            submat(output_1,pyoutput_2,LA1);
            submat(output_2,pyoutput_3,LA2);
    //		cout<<LA0<<endl;
    //		display(LA0);
        if(debug==1)
        {
            namedWindow("submat1",1);
            imshow("submat1",LA0);
            waitKey(0);
            namedWindow("submat2",1);
            imshow("submat2",LA1);
            waitKey(0);
            namedWindow("submat3",1);
            imshow("submat3",LA2);
            waitKey(0);
            namedWindow("submat4",1);
            imshow("submat4",LA3);
            waitKey(0);
        }
            Mat output1_1,output1_2,output1_3;
            pyrdown(img2,output1_1,output1_2,output1_3);
            //	namedWindow("output_0",1);

        if(debug==1)
        {
            imshow("output_0",img2);
            waitKey(0);
            //	namedWindow("output_1",1);
            imshow("output_1",output1_1);
            waitKey(0);
            //	namedWindow("output_2",1);
            imshow("output_2",output1_2);
            waitKey(0);
            //	namedWindow("output_3",1);
            imshow("output_3",output1_3);
            waitKey(0);
        }
            Mat pyoutput1_1,pyoutput1_2,pyoutput1_3;
            pyrup(output1_1,pyoutput1_1);
            pyrup(output1_2,pyoutput1_2);
            pyrup(output1_3,pyoutput1_3);

            Mat LB2,LB1,LB0;
            Mat LB3=output1_3;
            submat(img2,pyoutput1_1,LB0);
            submat(output1_1,pyoutput1_2,LB1);
            submat(output1_2,pyoutput1_3,LB2);
            Mat LS3,LS2,LS1,LS0;
            //	namedWindow("submat1",1);
        if(debug==1)
        {
            imshow("submat1",LB0);
            waitKey(0);
            //	namedWindow("submat2",1);
            imshow("submat2",LB1);
            waitKey(0);
            //	namedWindow("submat3",1);
            imshow("submat3",LB2);
            waitKey(0);
            //	namedWindow("submat4",1);
            imshow("submat4",LB3);
            waitKey(0);
            //	cout<<LA0.rows<<" "<<LA0.cols<<endl;
            //	cout<<LB0.rows<<" "<<LB0.cols<<endl;
        }
        int gauss=0;
        if(gauss==1)
            GaussianBlur(mask_1,mask_1,Size( 7, 7), 0, 0);
            blendimage(LA0,LB0,LS0,1,mask_1);
        //	imwrite("debug/mask_1.jpg",mask_1);
    //	display(LS0);
        if(debug==1)
        {
            namedWindow("LS0",1);
            imshow("LS0",LS0);
            waitKey(0);
        }
        if(gauss==1)
            GaussianBlur(mask_2,mask_2,Size( 7, 7), 0, 0);
            blendimage(LA1,LB1,LS1,2,mask_2);
    //	cout<<mask_2<<endl;
    //		imwrite("debug/mask_2.jpg",mask_2);

            if(debug==1)
            {
            namedWindow("LS1",1);
            imshow("LS1",LS1);
            waitKey(0);
            }
        if(gauss==1)
            GaussianBlur(mask_3,mask_3,Size( 7, 7), 0, 0);
            blendimage(LA2,LB2,LS2,4,mask_3);
    //		imwrite("debug/mask_3.jpg",mask_3);
            if(debug==1)
            {
            namedWindow("LS2",1);
            imshow("LS2",LS2);
            waitKey(0);
            }
        if(gauss==1)
            GaussianBlur(mask_4,mask_4,Size( 7, 7), 0, 0);
            GaussianBlur(mask_4,mask_4,Size( 7, 7), 0, 0);
            blendimage(LA3,LB3,LS3,8,mask_4);
    //		cout<<mask_4;
    //		imwrite("debug/mask_4.jpg",mask_4);
            if(debug==1)
            {
            namedWindow("LS3",1);
            imshow("LS3",LS3);
            waitKey(0);
            }
            Mat final0,final1,final2,final3;
            add2mat(LS3,LS2,final3);
            if(debug==1)
            {
            namedWindow("final3",1);
            imshow("final3",final3);
            waitKey(0);
            }
            add2mat(final3,LS1,final2);
            if(debug==1)
            {
            namedWindow("final2",1);
            imshow("final2",final2);
            waitKey(0);
            }
               clock_gettime(CLOCK_MONOTONIC, &t2);
                       float time = ((t2.tv_sec - t1.tv_sec)*1000) + (((double)(t2.tv_nsec - t1.tv_nsec))/1000000.0);


                           printf("Time (in milliseconds): %f\n", time);

            add2mat(final2,LS0,final1);
            namedWindow("final1",1);
            imshow("final1",final1);
            waitKey(0);
            imwrite("final1.jpg",final1);
    //	Mat yo;
    //	medianBlur(final1, yo, 3 );
    //		GaussianBlur(final1,yo,Size( 7, 7), 0, 0);
    //	display(yo);
        }
        /*	add2mat(LS0,final1,final0);
            namedWindow("final0",1);
            imshow("final0",final0);
            waitKey(0);*/
        /*	pyrup(output_2,pyoutput_2);
            imshow("output",pyoutput_2-output_1);
            waitKey(0);
            pyrup(output_3,pyoutput_3);
            imshow("output",pyoutput_3-output_2);
            waitKey(0);*/
        //	pyrdown(img1,output);
        //	cout<<img1.rows<<" "<<img1.cols<<" "<<output.rows<<" "<<output.cols<<endl;
        return 0;
    }