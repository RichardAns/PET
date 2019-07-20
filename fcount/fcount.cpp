// fcount
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <vector>

#include "cx_host.h"

template <typename T> size_t count(const char *name,size_t nxy,size_t nz)
{
	std::vector<T> buf(nxy);
	size_t sum = 0;
	FILE *fin = fopen(name,"rb");
	if(!fin) {printf("bad open %s\n",name); return 0;}
	for(int k=0;k<nz;k++){
		size_t count = fread(buf.data(),sizeof(T),nxy,fin);
		if(count != nxy){printf("bad resd %s\n",name); return 0;}
		for(size_t k=0;k<nxy;k++)if(buf[k] != 0) sum++;
	}
	fclose(fin);
	return sum;
}

int main(int argc,char *argv[])
{
	if(argc <2){
		printf("usage: fcount <filename> <nx> <ny> [<nz|1> type|uint ( c s i,u,f l,d )]\n");
		return 0;
	}

	size_t nx = atoi(argv[2]);
	size_t ny = atoi(argv[3]);
	size_t nz = 1; if(argc >4) nz = atoi(argv[4]);
	char  type = 'u';	if(argc>5) type = argv[5][0];

	size_t size = nx*ny;
	size_t sum = 0;
	if(type == 'c')      sum = count<char>(argv[1],size,nz);
	else if(type == 's') sum = count<short>(argv[1],size,nz);
	else if(type == 'i') sum = count<int>(argv[1],size,nz);
	else if(type == 'l') sum = count<long long>(argv[1],size,nz);
	else if(type == 'f') sum = count<float>(argv[1],size,nz);
	else if(type == 'd') sum = count<double>(argv[1],size,nz);
	printf("file %s has %lld non zero values\n",argv[1],sum);

	return 0;
}

