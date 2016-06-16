#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "device_launch_parameters.h"
#include <cuda_runtime_api.h>
#include <cufft.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>
#include "time.h"
#include <helper_math.h>
#include "mt19937ar.h"

#define PI 3.14159265358979

#define ROW 256
#define COL 512

int **s;
int *VtoCNum, *CtoVNum;
int *VtoCSeq, *CtoVSeq;
int *VtoCIndex, *CtoVIndex;
int *VtoCCount, *CtoVCount;
int num_one, CtoVTableSize, VtoCTableSize;
int *CtoVTable, *CtoVTableIndex, *CtoVTableNum;
int *VtoCTable, *VtoCTableIndex, *VtoCTableNum;
int *num_one_Cpu;
int *VtoCInitTable;
float SNR, SNR_start, SNR_stop, SNR_precision;
double TER, BER;
int error_num;
int transerror;
double gausscount;

float *RecvSignal, *RecvInitSignal, *RecvTransSignal;
float *P;
float a0, a1, b0, b1, a11, a00;
float *a0_Cpu, *a1_Cpu, *b0_Cpu, *b1_Cpu, *a11_Cpu, *a00_Cpu;


float *RecvSigGpu;
float *RecvProGpu;
float *VtoCInfor;
float *CtoVInfor;
int *VtoCNumGpu;
int *CtoVNumGpu;
int *VtoCSeqGpu;
int *CtoVSeqGpu;
int *VtoCTableGpu;
int *CtoVTableGpu;
int *VtoCTableIndexGpu;
int *CtoVTableIndexGpu;
int *VtoCTableNumGpu;
int *CtoVTableNumGpu;
int *VtoCInitIndexGpu;
int *VtoCInitCountGpu;
float *VtoCInitInforGpu;
float *VtoCIterInforGpu;
int *VtoCInitTableGpu;
int *num_one_Gpu;
float *a0_Gpu, *a1_Gpu, *b0_Gpu, *b1_Gpu, *a11_Gpu, *a00_Gpu;

__device__ float precision_Gpu;
__device__ float scale_Gpu;
__device__ int decode_mode_Gpu;

/* Period parameters */  
#define N 624
#define M 397
#define MATRIX_A 0x9908b0dfUL   /* constant vector a */
#define UPPER_MASK 0x80000000UL /* most significant w-r bits */
#define LOWER_MASK 0x7fffffffUL /* least significant r bits */
static unsigned long mt[N]; /* the array for the state vector  */
static int mti=N+1; /* mti==N+1 means mt[N] is not initialized */

void OOK_init(float SNR)
{
	float e = (float)1.6 * 1e-19;;
	float Is = (float)2 * 1e-9;
	float Ts = (float)5 * 1e-10;
	float f = (float)0.7 / Ts;
	float k = (float)1.38 * 1e-23;
	float T = (float)300;
	float B = (float)PI / 2 * f;
	float Cp = (float)300 * 1e-15;
	float Rf = (float)1 / 2 / PI / f / Cp;
	float Pb = (float)3 * 1e-14;
	float R0 = (float)0.875;
	float A = (float)pow(10, SNR/10);
	float a = (float)0.10;

	float Ps_temp = (e * e + (2 * e * Is + 4 * k * T * B / Rf + Pb * Pb * R0 * R0) * (1 / A - a * a));
	float Ps = (e + pow(Ps_temp, float(0.5))) / (R0 * (1 / A - a * a));

	float m = Is * Ts / e;
	float sita2 = 2 * B * Ts * (Is * Ts / e + 2 * k * T * Ts / Rf / e / e);
	float n0 = R0 / e * Ts * (Pb + a * Ps);
	float n1 = R0 / e * Ts * (Pb + Ps);

	a1 = n1 + sita2;
	a0 = n0 + sita2;
	b1 = n1 + m;
	b0 = n0 + m;

	a11 = pow(a1, float(0.5));
	a00 = pow(a0, float(0.5));

	printf("a1: %f\n", a1);
	printf("a0: %f\n", a0);
	printf("b1: %f\n", b1);
	printf("b0: %f\n", b0);

	*a1_Cpu = a1;
	*a0_Cpu = a0;
	*b1_Cpu = b1;
	*b0_Cpu = b0;
	*a11_Cpu = a11;
	*a00_Cpu = a00;

	cudaMemcpy(a1_Gpu, a1_Cpu, sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(a0_Gpu, a0_Cpu, sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(b1_Gpu, b1_Cpu, sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(b0_Gpu, b0_Cpu, sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(a11_Gpu, a11_Cpu, sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(a00_Gpu, a00_Cpu, sizeof(float), cudaMemcpyHostToDevice);
}

/* initializes mt[N] with a seed */
void init_genrand(unsigned long s)
{
    mt[0]= s & 0xffffffffUL;
    for (mti=1; mti<N; mti++) 
	{
        mt[mti] = 
	    (1812433253UL * (mt[mti-1] ^ (mt[mti-1] >> 30)) + mti); 
        /* See Knuth TAOCP Vol2. 3rd Ed. P.106 for multiplier. */
        /* In the previous versions, MSBs of the seed affect   */
        /* only MSBs of the array mt[].                        */
        /* 2002/01/09 modified by Makoto Matsumoto             */
        mt[mti] &= 0xffffffffUL;
        /* for >32 bit machines */
    }
}

unsigned long genrand_int32(void)
{
    unsigned long y;
    static unsigned long mag01[2]={0x0UL, MATRIX_A};
    /* mag01[x] = x * MATRIX_A  for x=0,1 */

    if (mti >= N) { /* generate N words at one time */
        int kk;

        if (mti == N+1)   /* if init_genrand() has not been called, */
            init_genrand(5489UL); /* a default initial seed is used */

        for (kk=0;kk<N-M;kk++) 
		{
            y = (mt[kk]&UPPER_MASK)|(mt[kk+1]&LOWER_MASK);
            mt[kk] = mt[kk+M] ^ (y >> 1) ^ mag01[y & 0x1UL];
        }
        for (;kk<N-1;kk++) 
		{
            y = (mt[kk]&UPPER_MASK)|(mt[kk+1]&LOWER_MASK);
            mt[kk] = mt[kk+(M-N)] ^ (y >> 1) ^ mag01[y & 0x1UL];
        }
        y = (mt[N-1]&UPPER_MASK)|(mt[0]&LOWER_MASK);
        mt[N-1] = mt[M-1] ^ (y >> 1) ^ mag01[y & 0x1UL];

        mti = 0;
    }
  
    y = mt[mti++];

    /* Tempering */
    y ^= (y >> 11);
    y ^= (y << 7) & 0x9d2c5680UL;
    y ^= (y << 15) & 0xefc60000UL;
    y ^= (y >> 18);

    return y;
}


double Gaussrand()
{
    static double U, V;
    static int phase = 0;
    double Z;

    if(phase == 0)
    {
         U = (genrand_int32() + 1.) / (0xffffffff + 2.);
         V = genrand_int32() / (0xffffffff + 1.);
         Z = sqrt(-2 * log(U)) * sin(2 * PI * V);
    }
    else
    {
         Z = sqrt(-2 * log(U)) * cos(2 * PI * V);
    }

    phase = 1 - phase;
	gausscount++;

    return Z;
}

__global__ void CalculateInitCode_Kernel(float *sig, float *pro, 
											float *a1_device, float *a0_device,
											float *b1_device, float *b0_device,
											float *a11_device, float *a00_device)
{
	int myID = threadIdx.x + blockIdx.x * blockDim.x;  

	if(COL > myID)
	{
		if(-1 == sig[myID])pro[myID] = 0;
		else pro[myID] = log(*a11_device / *a00_device) 
							+ ((sig[myID] - *b1_device) * (sig[myID] - *b1_device) / *a1_device 
							- (sig[myID] - *b0_device) * (sig[myID] - *b0_device) / *a0_device) / 2;
	}
}

__global__ void VtoCInit_Kernel(float *infor, float *pro, int *vtocindex, int *numone)
{
	int myID = threadIdx.x + blockIdx.x * blockDim.x;  

	if(*(numone) > myID)
	{
		int index = vtocindex[myID];
		infor[myID] = pro[index];
		if((infor[myID] < scale_Gpu) && (infor[myID] > -scale_Gpu))
		{
			infor[myID] = round(infor[myID] / precision_Gpu) * precision_Gpu;
		}
		else
		{
			if(infor[myID] > scale_Gpu)infor[myID] = scale_Gpu;
			else if(infor[myID] < -scale_Gpu)infor[myID] = -scale_Gpu;
		}
	}
}

__global__ void VtoCCalculate_Kernel(float *vtocinfor, float *ctovinfor, int *table, int *tableindex, int *tablenum, float *initinfor, int *numone)
{
	int myID = threadIdx.x + blockIdx.x * blockDim.x;  

	int i;

	if(*(numone) > myID)
	{
		int index = tableindex[myID] - tablenum[myID];
		int num = tablenum[myID];

		float sum = 0;
		for(i = 0; i < num; i++)
		{
			sum = sum + ctovinfor[table[index + i]];
		}
		
		vtocinfor[myID] = initinfor[myID] + sum;
		if((vtocinfor[myID] < scale_Gpu) && (vtocinfor[myID] > -scale_Gpu))
		{
			vtocinfor[myID] = round(vtocinfor[myID] / precision_Gpu) * precision_Gpu;
		}
		else
		{
			if(vtocinfor[myID] > scale_Gpu)vtocinfor[myID] = scale_Gpu;
			else if(vtocinfor[myID] < -scale_Gpu)vtocinfor[myID] = -scale_Gpu;
		}
	}
}

__global__ void CtoVCalculate_Kernel(float *ctovinfor, float *vtocinfor, int *table, int *tableindex, int *tablenum, int *numone)
{
	int myID = threadIdx.x + blockIdx.x * blockDim.x;  

	int i;
	
	if(*(numone) > myID)
	{
		int index = tableindex[myID] - tablenum[myID];
		int num = tablenum[myID];

		int sgn = 1, sgn_ind = 1;
		float offset = 1;
		float temp_float = 0;

		/*
		for(i = 0; i < num; i++)
		{
			temp_float = vtocinfor[table[index + i]];
			if(0 > temp_float)sgn_ind = -1;
			if(0 == temp_float)sgn_ind = 0;
			if(0 < temp_float)sgn_ind = 1;
			sgn = sgn * sgn_ind;
		}

		for(i = 0; i < num; i++)
		{
			temp_float = abs(vtocinfor[table[index + i]]);
			temp_float = tanhf(temp_float/2);
			temp_float = log(temp_float);
			offset = offset + (-1) * temp_float;
		}
		offset = abs(offset);
		offset = tanhf(offset/2);
		offset = log(offset);
		offset = (-1) * offset;
		ctovinfor[myID] = sgn * offset;
		*/
		
		if(0 == decode_mode_Gpu)
		{
			for(i = 0; i < num; i++)
			{
				temp_float = vtocinfor[table[index + i]];
				temp_float = tanhf(temp_float/2);
				offset = offset * temp_float;
			}
			ctovinfor[myID] = 2 * atanhf(offset);
			if((ctovinfor[myID] < scale_Gpu) && (ctovinfor[myID] > -scale_Gpu))
			{
				ctovinfor[myID] = round(ctovinfor[myID] / precision_Gpu) * precision_Gpu;
			}
			else
			{
				if(ctovinfor[myID] > scale_Gpu)ctovinfor[myID] = scale_Gpu;
				else if(ctovinfor[myID] < -scale_Gpu)ctovinfor[myID] = -scale_Gpu;
			}
		}
		else if(1 == decode_mode_Gpu)
		{
			temp_float = abs(vtocinfor[table[index]]);
			for(i = 0; i < num; i++)
			{
				if(0 > vtocinfor[table[index + i]])sgn_ind = -1;
				else sgn_ind = 1;
				sgn = sgn * sgn_ind;
				if(temp_float >= abs(vtocinfor[table[index + i]]))
				{
					temp_float = abs(vtocinfor[table[index + i]]);
				}
			}
			ctovinfor[myID] = sgn * temp_float;
			if((ctovinfor[myID] < scale_Gpu) && (ctovinfor[myID] > -scale_Gpu))
			{
				ctovinfor[myID] = round(ctovinfor[myID] / precision_Gpu) * precision_Gpu;
			}
			else
			{
				if(ctovinfor[myID] > scale_Gpu)ctovinfor[myID] = scale_Gpu;
				else if(ctovinfor[myID] < -scale_Gpu)ctovinfor[myID] = -scale_Gpu;
			}
		}
	}
}

__global__ void ProbVarNode_Kernel(float *pro, float *ctovinfor, int *vtocnum, int *vtoccount, int *vtoctable, float *initinfor)
{
	int myID = threadIdx.x + blockIdx.x * blockDim.x;  

	int i;

	if(COL > myID)
	{
		int index = vtocnum[myID] - vtoccount[myID];
		int num = vtoccount[myID];

		float sum = 0;
		for(i = 0; i < num; i++)
		{
			sum = sum + ctovinfor[vtoctable[index + i]];
		}

		pro[myID] = initinfor[myID] + sum;
	}
}

void InitCodeWords()
{
	int i;
	
	for(i = 0; i < COL; i++)
	{
		if(0 == RecvInitSignal[i])
		{
			RecvSignal[i] = b0 + a00 * (float)Gaussrand();
		}
		else
		{
			RecvSignal[i] = b1 + a11 * (float)Gaussrand();
		}
		/*
		if(0 != RecvInitSignal[i])
		{
			if(0 > RecvSignal[i])
			{
				RecvSignal[i] = -1;
			}
			else
			{
				RecvSignal[i] = 1;
			}
		}
		else
		{
			RecvSignal[i] = 0;
		}
		if(RecvSignal[i] != RecvInitSignal[i])transerror++;
		*/
		if(0 >= RecvSignal[i])
		{
			RecvSignal[i] = 0;
		}
		if(-1 == RecvInitSignal[i])
		{
			RecvSignal[i] = -1;
		}
		if(((b1 / 2 > RecvSignal[i]) && (1 == RecvInitSignal[i]))
			|| ((b1 / 2 < RecvSignal[i]) && (0 == RecvInitSignal[i])))
			transerror++;
	}
	cudaMemcpy(RecvSigGpu, RecvSignal, sizeof(float) * COL, cudaMemcpyHostToDevice);
	CalculateInitCode_Kernel<<<8, 1024>>>(RecvSigGpu, RecvProGpu, a1_Gpu, a0_Gpu, b1_Gpu, b0_Gpu, a11_Gpu, a00_Gpu);
	/*
	FILE *frc;
	for(i = 0; i < COL; i++)
	{
		P[i] = 1;;
	}
	cudaMemcpy(P, RecvProGpu, sizeof(float) * COL, cudaMemcpyDeviceToHost);
	for(i = 0; i < COL; i++)
	{
		frc = fopen("Decode0.txt", "a+");
		fprintf(frc, "%f\n", P[i]);
    	fclose(frc);
	}
	frc = NULL;
	*/
	VtoCInit_Kernel<<<8, 1024>>>(VtoCInfor, RecvProGpu, VtoCInitIndexGpu, num_one_Gpu);
	cudaMemcpy(VtoCInitInforGpu, VtoCInfor, sizeof(float) * num_one, cudaMemcpyDeviceToDevice);
}

void VtoCUpdate()
{
	VtoCCalculate_Kernel<<<8, 1024>>>(VtoCInfor, CtoVInfor, VtoCTableGpu, VtoCTableIndexGpu, VtoCTableNumGpu, VtoCInitInforGpu, num_one_Gpu);
}

void CtoVUpdate()
{
	CtoVCalculate_Kernel<<<8, 1024>>>(CtoVInfor, VtoCInfor, CtoVTableGpu, CtoVTableIndexGpu, CtoVTableNumGpu, num_one_Gpu);
}

void ProVar()
{
	ProbVarNode_Kernel<<<8, 1024>>>(VtoCIterInforGpu, CtoVInfor, VtoCNumGpu, VtoCInitCountGpu, VtoCInitTableGpu, VtoCInitInforGpu);
}

void IterInit()
{
	checkCudaErrors(cudaMalloc((void**)&VtoCTableGpu, sizeof(int) * VtoCTableSize));
	checkCudaErrors(cudaMalloc((void**)&CtoVTableGpu, sizeof(int) * CtoVTableSize));
	checkCudaErrors(cudaMalloc((void**)&VtoCTableIndexGpu, sizeof(int) * num_one));
	checkCudaErrors(cudaMalloc((void**)&CtoVTableIndexGpu, sizeof(int) * num_one));
	checkCudaErrors(cudaMalloc((void**)&VtoCTableNumGpu, sizeof(int) * num_one));
	checkCudaErrors(cudaMalloc((void**)&CtoVTableNumGpu, sizeof(int) * num_one));
	checkCudaErrors(cudaMalloc((void**)&VtoCInitTableGpu, sizeof(int) * num_one));
}

void TableInit()
{
	int i;

	//printf("Init the Mapping Table.\n");
	VtoCTableSize = 0;
	for(i = 0; i < COL; i++)
	{
		VtoCTableSize = VtoCTableSize + VtoCCount[i] * (VtoCCount[i] - 1);
	}
	VtoCTable = (int*)malloc(sizeof(int) * VtoCTableSize);
	VtoCTableIndex = (int*)malloc(sizeof(int) * num_one);
	VtoCTableNum = (int*)malloc(sizeof(int) * num_one);

	CtoVTableSize = 0;
	for(i = 0; i < ROW; i++)
	{
		CtoVTableSize = CtoVTableSize + CtoVCount[i] * (CtoVCount[i] - 1);
	}
	CtoVTable = (int*)malloc(sizeof(int) * CtoVTableSize);
	CtoVTableIndex = (int*)malloc(sizeof(int) * num_one);
	CtoVTableNum = (int*)malloc(sizeof(int) * num_one);

	VtoCInitTable = (int*)malloc(sizeof(int) * num_one);
}

void VtoCInitTableBuild()
{
	int i, j, k;

	//printf("Init the V to C Original Table.\n");
	k = 0;
	for(i = 0; i < COL; i++)
	{
		for(j = 0; j < num_one; j++)
		{
			if(i == CtoVSeq[j])
			{
				VtoCInitTable[k] = j;
				k++;
			}
		}
	}
	//printf("------------------------------------------------------------------\n");
	cudaMemcpy(VtoCInitTableGpu, VtoCInitTable, sizeof(int) * num_one, cudaMemcpyHostToDevice);

	cudaMemcpy(VtoCInitIndexGpu, VtoCIndex, sizeof(int) * num_one, cudaMemcpyHostToDevice);
	cudaMemcpy(VtoCNumGpu, VtoCNum, sizeof(int) * COL, cudaMemcpyHostToDevice);
	cudaMemcpy(VtoCInitCountGpu, VtoCCount, sizeof(int) * COL, cudaMemcpyHostToDevice);
}

void VtoCTableBuild()
{
	int i, j, k;
	int ctovstart = 0, vtocstart = 0;
	int offset = 0;

	//printf("Init the V to C Table.\n");
	for(i = 0; i < VtoCTableSize; i++)
	{
		VtoCTable[i] = 0;
	}
	for(i = 0; i < num_one; i++)
	{
		VtoCTableIndex[i] = 0;
	}
	for(i = 0; i < num_one; i++)
	{
		VtoCTableNum[i] = 0;
	}

	for(i = 0; i < num_one; i++)
	{
		vtocstart = VtoCNum[VtoCIndex[i]] - VtoCCount[VtoCIndex[i]];
		for(j = 0; j < VtoCCount[VtoCIndex[i]]; j++)
		{
			if((vtocstart + j) != i)
			{
				ctovstart = CtoVNum[VtoCSeq[vtocstart + j]] - CtoVCount[VtoCSeq[vtocstart + j]];
				for(k = 0; k < CtoVCount[VtoCSeq[vtocstart + j]]; k++)
				{
					if(CtoVSeq[ctovstart + k] == VtoCIndex[vtocstart + j])
					{
						VtoCTable[offset] = ctovstart + k;
						offset++;
						k = CtoVCount[VtoCSeq[vtocstart + j]]; 
					}
				}
			}
		}
		VtoCTableIndex[i] = offset;
		VtoCTableNum[i] = VtoCCount[VtoCIndex[i]] - 1;
	}

	cudaMemcpy(VtoCTableGpu, VtoCTable, sizeof(int) * VtoCTableSize, cudaMemcpyHostToDevice);
	cudaMemcpy(VtoCTableIndexGpu, VtoCTableIndex, sizeof(int) * num_one, cudaMemcpyHostToDevice);
	cudaMemcpy(VtoCTableNumGpu, VtoCTableNum, sizeof(int) * num_one, cudaMemcpyHostToDevice);
}

void CtoVTableBuild()
{
	int i, j, k;
	int ctovstart = 0, vtocstart = 0;
	int offset = 0;

	//printf("Init the C to V Table.\n");
	for(i = 0; i < CtoVTableSize; i++)
	{
		CtoVTable[i] = 0;
	}
	for(i = 0; i < num_one; i++)
	{
		CtoVTableIndex[i] = 0;
	}
	for(i = 0; i < num_one; i++)
	{
		CtoVTableNum[i] = 0;
	}
	
	for(i = 0; i < num_one; i++)
	{
		ctovstart = CtoVNum[CtoVIndex[i]] - CtoVCount[CtoVIndex[i]];
		for(j = 0; j < CtoVCount[CtoVIndex[i]]; j++)
		{
			if((ctovstart + j) != i)
			{
				vtocstart = VtoCNum[CtoVSeq[ctovstart + j]] - VtoCCount[CtoVSeq[ctovstart + j]];
				for(k = 0; k < VtoCCount[CtoVSeq[ctovstart + j]]; k++)
				{
					if(VtoCSeq[vtocstart + k] == CtoVIndex[ctovstart + j])
					{
						CtoVTable[offset] = vtocstart + k;
						offset++;
						k = VtoCCount[CtoVSeq[ctovstart + j]]; 
					}
				}
			}
		}
		CtoVTableIndex[i] = offset;
		CtoVTableNum[i] = CtoVCount[CtoVIndex[i]] - 1;
	}

	cudaMemcpy(CtoVTableGpu, CtoVTable, sizeof(int) * CtoVTableSize, cudaMemcpyHostToDevice);
	cudaMemcpy(CtoVTableIndexGpu, CtoVTableIndex, sizeof(int) * num_one, cudaMemcpyHostToDevice);
	cudaMemcpy(CtoVTableNumGpu, CtoVTableNum, sizeof(int) * num_one, cudaMemcpyHostToDevice);
}

void StaticHMatrix()
{
	int i, j, k;

	cudaMemcpy(num_one_Gpu, num_one_Cpu, sizeof(int), cudaMemcpyHostToDevice);

	//printf("Calculate the Num.\n");
	for(i = 0; i < COL; i++)
	{
		if(0 != i)VtoCNum[i] = VtoCNum[i - 1];
		for(j = 0; j < ROW; j++)
		{
			if(1 == s[j][i])VtoCNum[i]++;
		}
	}
	for(i = 0; i < ROW; i++)
	{
		if(0 != i)CtoVNum[i] = CtoVNum[i - 1];
		for(j = 0; j < COL; j++)
		{
			if(1 == s[i][j])CtoVNum[i]++;
		}
	}

	//printf("Calculate the Index.\n");
	k = 0;
	for(i = 0; i < num_one; i++)
	{
		if(i >= VtoCNum[k])k++;
		VtoCIndex[i] = k;
	}
	k = 0;
	for(i = 0; i < num_one; i++)
	{
		if(i >= CtoVNum[k])k++;
		CtoVIndex[i] = k;
	}

	//printf("Calculate the Seq.\n");
	k = 0;
	for(i = 0; i < COL; i++)
	{
		for(j = 0; j < ROW; j++)
		{
			if(1 == s[j][i])
			{
				VtoCSeq[k] = j;
				k++;
			}
		}
	}
	k = 0;
	for(i = 0; i < ROW; i++)
	{
		for(j = 0; j < COL; j++)
		{
			if(1 == s[i][j])
			{
				CtoVSeq[k] = j;
				k++;
			}
		}
	}

	//printf("Calculate the Count.\n");
	for(i = 0; i < COL; i++)
	{
		if(0 == i)VtoCCount[i] = VtoCNum[i];
		else
			VtoCCount[i] = VtoCNum[i] - VtoCNum[i - 1];
	}
	for(i = 0; i < ROW; i++)
	{
		if(0 == i)CtoVCount[i] = CtoVNum[i];
		else
			CtoVCount[i] = CtoVNum[i] - CtoVNum[i - 1];
	}

	for (i = 0; i < ROW; i++)
	{
  		free(s[i]);
		s[i] = NULL;
  	}
	free(s);
	s = NULL;
}

void ShowLLRResult()
{
	int i;
	//FILE *frc;
	int temp_sig, error_count = 0;
	
	for(i = 0; i < COL; i++)
	{
		P[i] = 1;;
	}
	cudaMemcpy(P, VtoCIterInforGpu, sizeof(float) * COL, cudaMemcpyDeviceToHost);
	/*
	for(i = 0; i < COL; i++)
	{
		frc = fopen("Decode1.txt", "a+");
		
		if(0 > P[i])
		{
    		fprintf(frc, "1,");
		}
		else
		{
			fprintf(frc, "0,");
		}
		
		fprintf(frc, "%f\n", P[i]);
    	fclose(frc);
	}
	frc = NULL;
	*/
	
	for(i = 0; i < ROW; i++)
	{
		if(0 > P[i])
		{
    		temp_sig = 1;
		}
		else
		{
			temp_sig = 0;
		}
		/*
		frc = fopen("Decode2.txt", "a+");
		fprintf(frc, "%d\n", temp_sig);
    		fclose(frc);
		frc = NULL;
		*/
	
    	if(temp_sig != RecvTransSignal[i])
    	{
    		error_count++;
    	}
	}
	error_num = error_num + error_count;
	//printf("Decoded Error: %d\n", error_count);
	//printf("Output the Decoded Results.\n");
	//printf("------------------------------------------------------------------\n");
}

void loadHmatrix()
{
	int i, j;
	FILE *fp;
	int *puncture_place, puncture_num = 0;
	fpos_t pos;
	
	printf("Malloc Memery for H Matrix.\n");
	int row = ROW, col = COL;
	printf("ROW: %d, COL: %d\n", row, col);
	s = (int**)malloc(sizeof(int*) * ROW);
	for (i = 0; i < ROW; i++)
	{
  		s[i] = (int*)malloc(sizeof(int) * COL);
  	}
	RecvSignal = (float*)malloc(sizeof(float) * COL);
	RecvInitSignal = (float*)malloc(sizeof(float) * COL);
	RecvTransSignal = (float*)malloc(sizeof(float) * COL);

	fp = fopen("X.txt", "r");
	for(i = 0; i < COL; i++)
	{
		fscanf(fp, "%f ", &RecvTransSignal[i]);
	}
	for(i = 0; i < COL; i++)
	{
		RecvInitSignal[i] = RecvTransSignal[i];
	}

	fp = fopen("Puncture.txt", "r");
	fseek(fp, 0L, SEEK_END);
	fgetpos(fp, &pos);
	float pos_half = floor((float)(pos/2));
	
	fp = fopen("Puncture.txt", "r");
	puncture_place = (int*)malloc(sizeof(int) * ((int)pos_half));
	for(i = 0; i < pos_half; i++)
	{
		fscanf(fp, "%d ", &puncture_place[i]);
	}
	for(i = 0; i < COL; i++)
	{
		if(puncture_place[puncture_num] == i)
		{
			RecvInitSignal[i] = -1;
			puncture_num++;
		}
	}

	printf("Reading the H Matrix.\n");
	fp = fopen("H.txt", "r");
	for(i = 0; i < ROW; i++)
	{
		for(j = 0; j < COL; j++)
		{
			fscanf(fp, "%d ", &s[i][j]);
		}
	}

	fclose(fp);
	fp = NULL;
	free(puncture_place);
	puncture_place = NULL;
}

void init()
{
	int i, j;

	BER = 0;
	error_num = 0;
	transerror = 0;
	gausscount = 0;

	printf("Malloc the Memory.\n");

	VtoCNum = (int*)malloc(sizeof(int) * COL);
	CtoVNum = (int*)malloc(sizeof(int) * ROW);
	P = (float*)malloc(sizeof(float) * COL);
	for(i = 0; i < COL; i++)
	{
		P[i] = 0;
	}
	for(i = 0; i < COL; i++)
	{
		VtoCNum[i] = 0;
	}
	for(i = 0; i < ROW; i++)
	{
		CtoVNum[i] = 0;
	}

	num_one = 0;
	for(i = 0; i < COL; i++)
	{
		for(j = 0; j < ROW; j++)
		{
			if(1 == s[j][i])num_one++;
		}
	}
	num_one_Cpu = (int*)malloc(sizeof(int));
	*(num_one_Cpu) = num_one;
	
	VtoCIndex= (int*)malloc(sizeof(int) * num_one);
	CtoVIndex= (int*)malloc(sizeof(int) * num_one);
	VtoCSeq = (int*)malloc(sizeof(int) * num_one);
	CtoVSeq = (int*)malloc(sizeof(int) * num_one);

	for(i = 0; i < num_one; i++)
	{
		VtoCIndex[i] = 0;
	}
	for(i = 0; i < num_one; i++)
	{
		CtoVIndex[i] = 0;
	}
	for(i = 0; i < num_one; i++)
	{
		VtoCSeq[i] = 0;
	}
	for(i = 0; i < num_one; i++)
	{
		CtoVSeq[i] = 0;
	}

	VtoCCount = (int*)malloc(sizeof(int) * COL);
	CtoVCount = (int*)malloc(sizeof(int) * ROW);
	for(i = 0; i < COL; i++)
	{
		VtoCCount[i] = 0;
	}
	for(i = 0; i < ROW; i++)
	{
		CtoVCount[i] = 0;
	}

	a1_Cpu = (float*)malloc(sizeof(float));
	a0_Cpu = (float*)malloc(sizeof(float));
	b1_Cpu = (float*)malloc(sizeof(float));
	b0_Cpu = (float*)malloc(sizeof(float));
	a11_Cpu = (float*)malloc(sizeof(float));
	a00_Cpu = (float*)malloc(sizeof(float));

	checkCudaErrors(cudaMalloc((void**)&RecvSigGpu, sizeof(float) * COL));
	checkCudaErrors(cudaMalloc((void**)&RecvProGpu, sizeof(float) * COL));
	checkCudaErrors(cudaMalloc((void**)&VtoCSeqGpu, sizeof(int) * num_one));
	checkCudaErrors(cudaMalloc((void**)&CtoVSeqGpu, sizeof(int) * num_one));
	checkCudaErrors(cudaMalloc((void**)&VtoCInfor, sizeof(float) * num_one));
	checkCudaErrors(cudaMalloc((void**)&CtoVInfor, sizeof(float) * num_one));
	checkCudaErrors(cudaMalloc((void**)&VtoCInitIndexGpu, sizeof(int) * num_one));
	checkCudaErrors(cudaMalloc((void**)&VtoCInitCountGpu, sizeof(int) * COL));
	checkCudaErrors(cudaMalloc((void**)&VtoCNumGpu, sizeof(int) * COL));
	checkCudaErrors(cudaMalloc((void**)&VtoCInitInforGpu, sizeof(float) * num_one));
	checkCudaErrors(cudaMalloc((void**)&VtoCIterInforGpu, sizeof(float) * COL));
	checkCudaErrors(cudaMalloc((void**)&num_one_Gpu, sizeof(int)));
	checkCudaErrors(cudaMalloc((void**)&a1_Gpu, sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&a0_Gpu, sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&b1_Gpu, sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&b0_Gpu, sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&a11_Gpu, sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&a00_Gpu, sizeof(float)));
}

void Exit()
{
	free(RecvSignal);
	RecvSignal = NULL;
	free(RecvInitSignal);
	RecvInitSignal = NULL;
	free(RecvTransSignal);
	RecvTransSignal = NULL;
	
	free(P);
	P = NULL;
	free(VtoCNum);
	VtoCNum = NULL;
	free(CtoVNum);
	CtoVNum = NULL;
	free(VtoCSeq);
	VtoCSeq = NULL;
	free(CtoVSeq);
	CtoVSeq = NULL;
	free(VtoCIndex);
	VtoCIndex = NULL;
	free(CtoVIndex);
	CtoVIndex = NULL;
	free(VtoCCount);
	VtoCCount = NULL;
	free(CtoVCount);
	CtoVCount = NULL;
	free(CtoVTable);
	CtoVTable = NULL;
	free(CtoVTableIndex);
	CtoVTableIndex = NULL;
	free(CtoVTableNum);
	CtoVTableNum = NULL;
	free(VtoCTable);
	VtoCTable = NULL;
	free(VtoCTableIndex);
	VtoCTableIndex = NULL;
	free(VtoCTableNum);
	VtoCTableNum = NULL;
	free(VtoCInitTable);
	VtoCInitTable = NULL;
	free(a1_Cpu);
	a1_Cpu = NULL;
	free(a0_Cpu);
	a0_Cpu = NULL;
	free(b1_Cpu);
	b1_Cpu = NULL;
	free(b0_Cpu);
	b0_Cpu = NULL;
	free(a11_Cpu);
	a11_Cpu = NULL;
	free(a00_Cpu);
	a00_Cpu = NULL;
	
	checkCudaErrors(cudaFree(RecvSigGpu));
	checkCudaErrors(cudaFree(RecvProGpu));
	checkCudaErrors(cudaFree(VtoCInfor));
	checkCudaErrors(cudaFree(CtoVInfor));
	checkCudaErrors(cudaFree(VtoCTableGpu));
	checkCudaErrors(cudaFree(CtoVTableGpu));
	checkCudaErrors(cudaFree(VtoCSeqGpu));
	checkCudaErrors(cudaFree(CtoVSeqGpu));
	checkCudaErrors(cudaFree(VtoCTableIndexGpu));
	checkCudaErrors(cudaFree(CtoVTableIndexGpu));
	checkCudaErrors(cudaFree(VtoCTableNumGpu));
	checkCudaErrors(cudaFree(CtoVTableNumGpu));
	checkCudaErrors(cudaFree(VtoCInitIndexGpu));
	checkCudaErrors(cudaFree(VtoCInitCountGpu));
	checkCudaErrors(cudaFree(VtoCNumGpu));
	checkCudaErrors(cudaFree(VtoCInitInforGpu));
	checkCudaErrors(cudaFree(VtoCIterInforGpu));
	checkCudaErrors(cudaFree(num_one_Gpu));
	checkCudaErrors(cudaFree(VtoCInitTableGpu));
	checkCudaErrors(cudaFree(a1_Gpu));
	checkCudaErrors(cudaFree(a0_Gpu));
	checkCudaErrors(cudaFree(b1_Gpu));
	checkCudaErrors(cudaFree(b0_Gpu));
	checkCudaErrors(cudaFree(a11_Gpu));
	checkCudaErrors(cudaFree(a00_Gpu));
}

int main()
{
	int i, j, k, iter, count, num_point;
	float scale;
	int quanbits;
	float precision;
	int decode_mode;

	//printf("Please Input the Sigma: ");
	//scanf("%f", &SNR);
	printf("Please Input the Start SNR: ");
	scanf("%f", &SNR_start);
	printf("Please Input the Stop SNR: ");
	scanf("%f", &SNR_stop);
	printf("Please Input the SNR Precision: ");
	scanf("%f", &SNR_precision);
	printf("Please Input the number of Frame: ");
	scanf("%d", &count);
	printf("Please Input the number of Iteration: ");
	scanf("%d", &iter);
	printf("Please Input the Scale of LLR: ");
	scanf("%f", &scale);
	printf("Please Input the Quanzation Bits: ");
	scanf("%d", &quanbits);
	printf("Please Input the Decode Mode (0: Standard BP, 1: Min-Sum): ");
	scanf("%d", &decode_mode);
	printf("------------------------------------------------------------------\n");

	float base2 = 2;
	precision = scale / pow(base2, quanbits - 1);
	printf("Precision: %f\n", precision);
	cudaMemcpyToSymbol(precision_Gpu, &precision, sizeof(float));
	cudaMemcpyToSymbol(scale_Gpu, &scale, sizeof(float));
	cudaMemcpyToSymbol(decode_mode_Gpu, &decode_mode, sizeof(int));

	loadHmatrix();
	init();
		
	StaticHMatrix();
	TableInit();
	IterInit();
	CtoVTableBuild();
	VtoCTableBuild();
	VtoCInitTableBuild();

	num_point = (int)((SNR_stop - SNR_start) / SNR_precision) + 1;
	SNR = SNR_start;
	printf("------------------------------------------------------------------\n");

	clock_t start, stop;
	double runtime;
	
	for(k = 0; k < num_point; k++)
	{
		init_genrand(long(time(0)));
		
		printf("SNR: %f\n", SNR);;
		OOK_init(SNR);
		
		printf("InitCodeWords.\n");
		printf("Decoding...");

		start = clock();
		
		for(i = 0; i < count; i++)
		{
			InitCodeWords();
			for(j = 0; j < iter; j++)
			{
				CtoVUpdate();
				VtoCUpdate();
				ProVar();
			}
			ShowLLRResult();
		}
		printf("OK.\n");

		stop = clock();
		runtime = (stop - start) / 1000.0000;

		TER = abs(transerror * 100  * 1e12/ COL / ((double)(count)) / 1e12);
		BER = abs(error_num * 100 * 1e12 / ROW / ((double)(count)) / 1e12);
		printf("BITS_TRANS: %d\n", COL * count);
		printf("ERROR_TRANS: %d\n", transerror);
		printf("ERROR_LEFT: %d\n", error_num);
		printf("TER: %.10lf%%\n", TER);
		printf("BER: %.10lf%%\n", BER);
		printf("Running Time: %fs\n", runtime);
		printf("------------------------------------------------------------------\n");

		FILE *frc;
		frc = fopen("SNR.txt", "a+");
		fprintf(frc, "%f\n", SNR);
	    fclose(frc);
		frc = NULL;
		frc = fopen("TER.txt", "a+");
		fprintf(frc, "%.10lf%%\n", TER);
	    fclose(frc);
		frc = NULL;
		frc = fopen("BER.txt", "a+");
		fprintf(frc, "%.10lf%%\n", BER);
	    fclose(frc);
		frc = NULL;
		
		SNR = SNR + SNR_precision;
		transerror = 0;
		error_num = 0;
	}

	Exit();

	return 0;
}
