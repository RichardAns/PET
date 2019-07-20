// expdist

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

int main(int argc,char *argv[])
{
	int bins = 100;
	int * hist = (int *)calloc(bins,sizeof(int));
	for (int k=0;k<50000;k++){
		double z = (double)rand()/(double)RAND_MAX;
		if(z > 0.0){
			double y =-11.49*log(1.0-z);
			int bin = (int)(2.0*y);
			if(bin>=0 && bin < bins) hist[bin]++;
			else hist[bins-1]++;
		}
		
	}
	for(int k=0;k<bins;k++)printf("%d\n",hist[k]);
	return 0;
}

