#include <opencv2/opencv.hpp>
#include <math.h>
#include <time.h>
#include <iostream>
using namespace std;
using namespace cv;
#define IA 16807
#define IM 2147483647
#define AM (1.0/IM)
#define IQ 127773
#define IR 2836
#define MASK 123459876

class ParticleFilterTrackor
{
private:
	class SpaceState
	{  /* 状态空间变量 */
	public:
			int xt;               /* x坐标位置 */
			int yt;               /* x坐标位置 */
			float v_xt;           /* x方向运动速度 */
			float v_yt;           /* y方向运动速度 */
			int Hxt;              /* x方向半窗宽 */
			int Hyt;              /* y方向半窗宽 */
			float at_dot;         /* 尺度变换速度 */
	} ;

	SpaceState *states;
	float *weights;
	float *ModelHist;
	int NParticle;//number of particles
	int R_BIN,G_BIN,B_BIN;
	int nbin;//bin of hist
	long ran_seed;
	float DELTA_T ;    /* 帧频，可以为30，25，15，10等 */
	int POSITION_DISTURB ;      /* 位置扰动幅度   */
	float VELOCITY_DISTURB ;  /* 速度扰动幅值   */
	float SCALE_DISTURB ;      /* 窗宽高扰动幅度 */
	float SCALE_CHANGE_D;   /* 尺度变换速度扰动幅度 */
	float Pi_Thres; /* 权重阈值   */
	float Weight_Thres ;  /* 最大权重阈值，用来判断是否目标丢失 */

	/*some function for generate random*/
	long set_seed( long setvalue )
	{
		if ( setvalue != 0 ) /* 如果传入的参数setvalue!=0，设置该数为种子 */
			ran_seed = setvalue;
		else                 /* 否则，利用系统时间为种子数 */
		{
			ran_seed = time(NULL);
		}
		return( ran_seed );
	}

		/*
	采用Park and Miller方法产生[0,1]之间均匀分布的伪随机数
	算法详细描述见：
	[1] NUMERICAL RECIPES IN C: THE ART OF SCIENTIFIC COMPUTING.
	Cambridge University Press. 1992. pp.278-279.
	[2] Park, S.K., and Miller, K.W. 1988, Communications of the ACM, 
	vol. 31, pp. 1192–1201.
	*/

	float ran0(long *idum)
	{
		long k;
		float ans;

		/* *idum ^= MASK;*/      /* XORing with MASK allows use of zero and other */
		k=(*idum)/IQ;            /* simple bit patterns for idum.                 */
		*idum=IA*(*idum-k*IQ)-IR*k;  /* Compute idum=(IA*idum) % IM without over- */
		if (*idum < 0) *idum += IM;  /* flows by Schrage’s method.               */
		ans=AM*(*idum);          /* Convert idum to a floating result.            */
		/* *idum ^= MASK;*/      /* Unmask before return.                         */
		return ans;
	}
		/*
	获得一个[0,1]之间均匀分布的随机数
	*/
	float rand0_1()
	{
		return( ran0( &ran_seed ) );
	}



	/*
	获得一个x - N(u,sigma)Gaussian分布的随机数
	*/
	float randGaussian( float u, float sigma )
	{
		float x1, x2, v1, v2;
		float s = 100.0;
		float y;

		/*
		使用筛选法产生正态分布N(0,1)的随机数(Box-Mulles方法)
		1. 产生[0,1]上均匀随机变量X1,X2
		2. 计算V1=2*X1-1,V2=2*X2-1,s=V1^2+V2^2
		3. 若s<=1,转向步骤4，否则转1
		4. 计算A=(-2ln(s)/s)^(1/2),y1=V1*A, y2=V2*A
		y1,y2为N(0,1)随机变量
		*/
		while ( s > 1.0 )
		{
			x1 = rand0_1();
			x2 = rand0_1();
			v1 = 2 * x1 - 1;
			v2 = 2 * x2 - 1;
			s = v1*v1 + v2*v2;
		}
		y = (float)(sqrt( -2.0 * log(s)/s ) * v1);
		/*
		根据公式
		z = sigma * y + u
		将y变量转换成N(u,sigma)分布
		*/
		return( sigma * y + u );	
	}

	/*calculate color histogram of a region*/
	void CalcuColorHistogram( Rect toTrack, Mat img, float * ColorHist, int bins )
	{
		int x_begin, y_begin;  /* 指定图像区域的左上角坐标 */
		int y_end, x_end;
		int  index;
		int r, g, b;
		float k, r2, f;
		int a2;

		for (int i = 0; i < bins; i++ )     /* 直方图各个值赋0 */
			ColorHist[i] = 0.0;
		/* 考虑特殊情况：x0, y0在图像外面，或者，Wx<=0, Hy<=0 */
		/* 此时强制令彩色直方图为0 */
		Rect whole(0,0,img.cols,img.rows);
		toTrack &=whole;
		x_begin = toTrack.x;               /* 计算实际高宽和区域起始点 */
		y_begin = toTrack.y;
		x_end = x_begin + toTrack.width;
		y_end = y_begin + toTrack.height;
		int x0=(x_begin+x_end)/2;
		int y0=(y_begin+y_end)/2;
		a2 = (toTrack.width/2)*(toTrack.width/2)+(toTrack.height/2)*(toTrack.height/2);                /* 计算核函数半径平方a^2 */
		f = 0.0;   /* 归一化系数 */
		uchar* image = img.data; 
		int R_SHIFT= log(256/R_BIN)/log(2);
		int G_SHIFT= log(256/G_BIN)/log(2);   
		int B_SHIFT= log(256/B_BIN)/log(2);                    
		for (int y = y_begin; y < y_end; y++ )
		{
			for (int x = x_begin; x < x_end; x++ )
			{
				r = (int)(image[(y*img.cols+x)*3+2]) >> R_SHIFT;   /* 计算直方图 */
				g = (int)(image[(y*img.cols+x)*3+1])>> G_SHIFT; /*移位位数根据R、G、B条数 */
				b = (int)(image[(y*img.cols+x)*3]) >> B_SHIFT;
				index = r * G_BIN * B_BIN + g * B_BIN + b;
				// // cout<<"r   "<<image[(y*toTrack.width+x)*3+2]<<endl;
				// // cout<<"rshift  "<<r<<endl;
				// cout<<"index   "<<index<<endl;
				r2 = (float)(((y-y0)*(y-y0)+(x-x0)*(x-x0))*1.0/a2); /* 计算半径平方r^2 */
				k = 1 - r2;   /* 核函数k(r) = 1-r^2, |r| < 1; 其他值 k(r) = 0 */
				f = f + k;
				ColorHist[index] = ColorHist[index] + k;  /* 计算核密度加权彩色直方图 */
				//cout<<index<<endl;
				//circle(img,Point(x,y),2,Scalar(100,100,100));
		

			}
		}
		for (int i = 0; i < bins; i++ )     /* 归一化直方图 */
			ColorHist[i] = ColorHist[i]/f;
		return;
	}

	// void CalcuEdgeOrientationHistogram(Mat img,Rect toTrack)
	// {
	// 	Mat imgRect(img,toTrack);
	// }
	/*
	计算Bhattacharyya系数
	输入参数：
	float * p, * q：      两个彩色直方图密度估计
	int bins：            直方图条数
	返回值：
	Bhattacharyya系数
	*/
	float CalcuBhattacharyya( float * p, float * q, int bins )
	{
		int i;
		float rho;

		rho = 0.0;
		for ( i = 0; i < bins; i++ )
		{
			rho = (float)(rho + sqrt( p[i]*q[i] ));
			//cout<<"piqi  "<<sqrt( p[i]*q[i] )<<" "<<rho<< " ";
		}
		//cout<<rho<< " ";
		return( rho );
	}


	/*# define RECIP_SIGMA  3.98942280401  / * 1/(sqrt(2*pi)*sigma), 这里sigma = 0.1 * /*/
	# define SIGMA2       0.02           /* 2*sigma^2, 这里sigma = 0.1 */

	float CalcuWeightedPi( float rho )
	{
		float pi_n, d2;

		d2 = 1 - rho;
		//pi_n = (float)(RECIP_SIGMA * exp( - d2/SIGMA2 ));
		pi_n = (float)(exp( - d2/SIGMA2 ));

		return( pi_n );
	}
	/*
	计算归一化累计概率c'_i
	输入参数：
	float * weight：    为一个有N个权重（概率）的数组
	int N：             数组元素个数
	输出参数：
	float * cumulateWeight： 为一个有N+1个累计权重的数组，
	cumulateWeight[0] = 0;
	*/
	void NormalizeCumulatedWeight( float * weight, float * cumulateWeight, int N )
	{
		int i;

		for ( i = 0; i < N+1; i++ ) 
			cumulateWeight[i] = 0;
		for ( i = 0; i < N; i++ )
			cumulateWeight[i+1] = cumulateWeight[i] + weight[i];
		for ( i = 0; i < N+1; i++ )
			cumulateWeight[i] = cumulateWeight[i]/ cumulateWeight[N];

		return;
	}

	/*
	折半查找，在数组NCumuWeight[N]中寻找一个最小的j，使得
	NCumuWeight[j] <=v
	float v：              一个给定的随机数
	float * NCumuWeight：  权重数组
	int N：                数组维数
	返回值：
	数组下标序号
	*/
	int BinearySearch( float v, float * NCumuWeight, int N )
	{
		int l, r, m;

		l = 0; 	r = N-1;   /* extreme left and extreme right components' indexes */
		while ( r >= l)
		{
			m = (l+r)/2;
			if ( v >= NCumuWeight[m] && v < NCumuWeight[m+1] ) return( m );
			if ( v < NCumuWeight[m] ) r = m - 1;
			else l = m + 1;
		}
		return( 0 );
	}

	/*
	重新进行重要性采样
	输入参数：
	float * c：          对应样本权重数组pi(n)
	int N：              权重数组、重采样索引数组元素个数
	输出参数：
	int * ResampleIndex：重采样索引数组
	*/
	void ImportanceSampling( float * c, int * ResampleIndex, int N )
	{
		float rnum, * cumulateWeight;
		int i, j;

		cumulateWeight = new float [N+1]; /* 申请累计权重数组内存，大小为N+1 */
		NormalizeCumulatedWeight( c, cumulateWeight, N ); /* 计算累计权重 */
		for ( i = 0; i < N; i++ )
		{
			rnum = rand0_1();       /* 随机产生一个[0,1]间均匀分布的数 */ 
			j = BinearySearch( rnum, cumulateWeight, N+1 ); /* 搜索<=rnum的最小索引j */
			if ( j == N ) j--;
			ResampleIndex[i] = j;	/* 放入重采样索引数组 */		
		}

		delete cumulateWeight;

		return;	
	}
	/*
	样本选择，从N个输入样本中根据权重重新挑选出N个
	输入参数：
	SPACESTATE * state：     原始样本集合（共N个）
	float * weight：         N个原始样本对应的权重
	int N：                  样本个数
	输出参数：
	SPACESTATE * state：     更新过的样本集
	*/
	void ReSelect( SpaceState * state, float * weight, int N )
	{
		SpaceState * tmpState;
		int i, * rsIdx;

		tmpState = new SpaceState[N];
		rsIdx = new int[N];

		ImportanceSampling( weight, rsIdx, N ); /* 根据权重重新采样 */
		for ( i = 0; i < N; i++ )
			tmpState[i] = state[rsIdx[i]];//temState为临时变量,其中state[i]用state[rsIdx[i]]来代替
		for ( i = 0; i < N; i++ )
			state[i] = tmpState[i];

		delete[] tmpState;
		delete[] rsIdx;

		return;
	}

	/*
	传播：根据系统状态方程求取状态预测量
	状态方程为： S(t) = A S(t-1) + W(t-1)
	W(t-1)为高斯噪声
	输入参数：
	SPACESTATE * state：      待求的状态量数组
	int N：                   待求状态个数
	输出参数：
	SPACESTATE * state：      更新后的预测状态量数组
	*/
	void Propagate( SpaceState * state, int N)
	{
		int i;
		int j;
		float rn[7];

		/* 对每一个状态向量state[i](共N个)进行更新 */
		for ( i = 0; i < N; i++ )  /* 加入均值为0的随机高斯噪声 */
		{
			for ( j = 0; j < 7; j++ ) rn[j] = randGaussian( 0, (float)0.6 ); /* 产生7个随机高斯分布的数 */
			state[i].xt = (int)(state[i].xt + state[i].v_xt * DELTA_T + rn[0] * state[i].Hxt + 0.5);//加0.5应该是实现四舍五入
			state[i].yt = (int)(state[i].yt + state[i].v_yt * DELTA_T + rn[1] * state[i].Hyt + 0.5);
			state[i].v_xt = (float)(state[i].v_xt + rn[2] * VELOCITY_DISTURB);
			state[i].v_yt = (float)(state[i].v_yt + rn[3] * VELOCITY_DISTURB);
			state[i].Hxt = (int)(state[i].Hxt+state[i].Hxt*state[i].at_dot + rn[4] * SCALE_DISTURB + 0.5);
			state[i].Hyt = (int)(state[i].Hyt+state[i].Hyt*state[i].at_dot + rn[5] * SCALE_DISTURB + 0.5);
			state[i].at_dot = (float)(state[i].at_dot + rn[6] * SCALE_CHANGE_D);
			//Circle(pTrackImg,cvPoint(state[i].xt,state[i].yt),3, CV_RGB(0,255,0),-1);
		}
		return;
	}
	/*
	观测，根据状态集合St中的每一个采样，观测直方图，然后
	更新估计量，获得新的权重概率
	输入参数：
	SPACESTATE * state：      状态量数组
	int N：                   状态量数组维数
	unsigned char * image：   图像数据，按从左至右，从上至下的顺序扫描，
	颜色排列次序：RGB, RGB, ...						 
	int W, H：                图像的宽和高
	float * ObjectHist：      目标直方图
	int hbins：               目标直方图条数
	输出参数：
	float * weight：          更新后的权重
	*/
	void Observe( SpaceState * state, float * weight, int N,
				 Mat img,float * ObjectHist, int hbins )
	{
		int i;
		float * ColorHist;
		float rho;

		ColorHist = new float[hbins];

		for ( i = 0; i < N; i++ )
		{
			/* (1) 计算彩色直方图分布 */
			CalcuColorHistogram(Rect(state[i].xt-state[i].Hxt, state[i].yt-state[i].Hyt,2*state[i].Hxt, 2*state[i].Hyt),
				img,ColorHist, hbins );
			/* (2) Bhattacharyya系数 */
			rho = CalcuBhattacharyya( ColorHist, ObjectHist, hbins );
			/* (3) 根据计算得的Bhattacharyya系数计算各个权重值 */
			weight[i] = CalcuWeightedPi( rho );		
		}

		delete ColorHist;

		return;	
	}
	/*
	估计，根据权重，估计一个状态量作为跟踪输出
	输入参数：
	SPACESTATE * state：      状态量数组
	float * weight：          对应权重
	int N：                   状态量数组维数
	输出参数：
	SPACESTATE * EstState：   估计出的状态量
	*/
	void Estimation( SpaceState * state, float * weight, int N, 
					SpaceState & EstState )
	{
		int i;
		float at_dot, Hxt, Hyt, v_xt, v_yt, xt, yt;
		float weight_sum;

		at_dot = 0;
		Hxt = 0; 	Hyt = 0;
		v_xt = 0;	v_yt = 0;
		xt = 0;  	yt = 0;
		weight_sum = 0;
		for ( i = 0; i < N; i++ ) /* 求和 */
		{
			at_dot += state[i].at_dot * weight[i];
			Hxt += state[i].Hxt * weight[i];
			Hyt += state[i].Hyt * weight[i];
			v_xt += state[i].v_xt * weight[i];
			v_yt += state[i].v_yt * weight[i];
			xt += state[i].xt * weight[i];
			yt += state[i].yt * weight[i];
			weight_sum += weight[i];
		}
		/* 求平均 */
		if ( weight_sum <= 0 ) weight_sum = 1; /* 防止被0除，一般不会发生 */
		EstState.at_dot = at_dot/weight_sum;
		EstState.Hxt = (int)(Hxt/weight_sum + 0.5 );
		EstState.Hyt = (int)(Hyt/weight_sum + 0.5 );
		EstState.v_xt = v_xt/weight_sum;
		EstState.v_yt = v_yt/weight_sum;
		EstState.xt = (int)(xt/weight_sum + 0.5 );
		EstState.yt = (int)(yt/weight_sum + 0.5 );

		return;
	}
	/************************************************************
	模型更新
	输入参数：
	SPACESTATE EstState：   状态量的估计值
	float * TargetHist：    目标直方图
	int bins：              直方图条数
	float PiT：             阈值（权重阈值）
	unsigned char * img：   图像数据，RGB形式
	int W, H：              图像宽高 
	输出：
	float * TargetHist：    更新的目标直方图
	************************************************************/
	# define ALPHA_COEFFICIENT      0.2     /* 目标模型更新权重取0.1-0.3 */

	int ModelUpdate( SpaceState EstState, float * TargetHist, int bins, float PiT,Mat img)
	{
		float * EstHist, Bha, Pi_E;
		int i, rvalue = -1;

		EstHist = new float [bins];

		/* (1)在估计值处计算目标直方图 */
		CalcuColorHistogram( Rect(EstState.xt, EstState.yt, EstState.Hxt,EstState.Hyt), img,EstHist, bins );
		/* (2)计算Bhattacharyya系数 */
		Bha  = CalcuBhattacharyya( EstHist, TargetHist, bins );
		/* (3)计算概率权重 */
		Pi_E = CalcuWeightedPi( Bha );

		if ( Pi_E > PiT ) 
		{
			for ( i = 0; i < bins; i++ )
			{
				TargetHist[i] = (float)((1.0 - ALPHA_COEFFICIENT) * TargetHist[i]
				+ ALPHA_COEFFICIENT * EstHist[i]);
			}
			rvalue = 1;
		}

		delete EstHist;

		return( rvalue );
	}

public:
	ParticleFilterTrackor()
	{
		NParticle=75;//number of particles
		R_BIN=G_BIN=B_BIN=8;
		nbin=R_BIN*G_BIN*B_BIN;//bin of hist
		ran_seed=802163120;
		DELTA_T=0.05 ;    /* 帧频，可以为30，25，15，10等 */
		POSITION_DISTURB=15 ;      /* 位置扰动幅度   */
		VELOCITY_DISTURB=40 ;  /* 速度扰动幅值   */
		SCALE_DISTURB =0.0;      /* 窗宽高扰动幅度 */
		SCALE_CHANGE_D=0.001;   /* 尺度变换速度扰动幅度 */
		Pi_Thres=90; /* 权重阈值   */
		Weight_Thres =0.0001;  /* 最大权重阈值，用来判断是否目标丢失 */
	}

	int Initialize( Mat img, Rect toTrack )
	{
		int i, j;
		float rn[7];

		set_seed( 0 ); /* 使用系统时钟作为种子，这个函数在 */
		/* 系统初始化时候要调用一次,且仅调用1次 */
		//NParticle = 75; /* 采样粒子个数 */
		//Pi_Thres = (float)0.90; /* 设置权重阈值 */
		states = new SpaceState[NParticle]; /* 申请状态数组的空间 */
		if ( states == NULL ) return( -2 );
		weights = new float [NParticle];     /* 申请粒子权重数组的空间 */
		if ( weights == NULL ) return( -3 );	
		nbin = R_BIN * G_BIN * B_BIN; /* 确定直方图条数 */
		ModelHist = new float [nbin]; /* 申请直方图内存 */
		if ( ModelHist == NULL ) return( -1 );

		/* 计算目标模板直方图 */
		CalcuColorHistogram( toTrack,img, ModelHist, nbin );

		/* 初始化粒子状态(以(x0,y0,1,1,Wx,Hy,0.1)为中心呈N(0,0.4)正态分布) */
		states[0].xt = toTrack.x+toTrack.width/2;
		states[0].yt = toTrack.y+toTrack.height/2;
		states[0].v_xt = (float)0.0; // 1.0
		states[0].v_yt = (float)0.0; // 1.0
		states[0].Hxt = toTrack.width/2;
		states[0].Hyt = toTrack.height/2;
		states[0].at_dot = (float)0.0; // 0.1
		weights[0] = (float)(1.0/NParticle); /* 0.9; */
		for ( i = 1; i < NParticle; i++ )
		{
			for ( j = 0; j < 7; j++ ) rn[j] = randGaussian( 0, (float)0.6 ); /* 产生7个随机高斯分布的数 */
			states[i].xt = (int)( states[0].xt + rn[0] * toTrack.width/2 );
			states[i].yt = (int)( states[0].yt + rn[1] * toTrack.height/2);
			states[i].v_xt = (float)( states[0].v_xt + rn[2] * VELOCITY_DISTURB );
			states[i].v_yt = (float)( states[0].v_yt + rn[3] * VELOCITY_DISTURB );
			states[i].Hxt = (int)( states[0].Hxt + rn[4] * SCALE_DISTURB );
			states[i].Hyt = (int)( states[0].Hyt + rn[5] * SCALE_DISTURB );
			states[i].at_dot = (float)( states[0].at_dot + rn[6] * SCALE_CHANGE_D );
			/* 权重统一为1/N，让每个粒子有相等的机会 */
			weights[i] = (float)(1.0/NParticle);
			circle(img,Point(states[i].xt,states[i].yt),fabs(states[i].v_xt),Scalar(100,100,100));
		}

		return( 1 );
	}
	int ColorParticleTracking( Mat img, Rect &toTrack,float & max_weight)
	{
		SpaceState EState;
		int i;
		/* 选择：选择样本，并进行重采样 */
		ReSelect( states, weights, NParticle );
		/* 传播：采样状态方程，对状态变量进行预测 */
		Propagate( states, NParticle);
		/* 观测：对状态量进行更新 */
		Observe( states, weights, NParticle, img, ModelHist, nbin );
		/* 估计：对状态量进行估计，提取位置量 */
		Estimation( states, weights, NParticle, EState );
		int xc = EState.xt;
		int yc = EState.yt;
		int Wx_h = EState.Hxt;
		int Hy_h = EState.Hyt;
		toTrack=Rect(xc-Wx_h,yc-Hy_h,2*Wx_h,2*Hy_h);
		/* 模型更新 */
		ModelUpdate( EState, ModelHist, nbin, Pi_Thres,	img);

		/* 计算最大权重值 */
		max_weight = weights[0];
		for ( i = 1; i < NParticle; i++ )
			max_weight = max_weight < weights[i] ? weights[i] : max_weight;
		/* 进行合法性检验，不合法返回-1 */
		if ( xc < 0 || yc < 0 || xc >= img.cols || yc >= img.rows ||
			Wx_h <= 0 || Hy_h <= 0 ) 
			return( -1 );
		else 
			return( 1 );		
	}


};