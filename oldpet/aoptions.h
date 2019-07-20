#ifndef AOPTIONS_H_
#define AOPTIONS_H_

// Richard Ansorge, BSS Group Cavendish Laboratory. June 2017
// Copyright 2008-2017 Richard Ansorge.  All rights reserved.

// WE MAKE NO REPRESENTATION ABOUT THE SUITABILITY OF THIS SOURCE 
// CODE FOR ANY PURPOSE. IT IS PROVIDED "AS IS" WITHOUT WARRANTY OF ANY KIND 
//
// Simple class to crack command line parameters.  Keywords are specified locally at point of use. 
// No (possibly complex) initial command line parsing is needed.  Keywords are NOT case sensitive and
//  are forced to lower case internally.  However strings passed by keyword:string are preserved.
//  
//
//  Usage:
//  create instance  of class in main:  AOptions opt(argc,argv,nskip);  
//  then pass opt to all interested functions (prefereably by reference!)
//  the first nskip arguments are ignored by this class (useful for filenames etc)

// examples  use_stuff            turns on flag in code:             if(opt.isset("use_stuff")) dosomething
//           setn:2               get value (int float or string):   if(opt.isset("setn")) n = atoi(opt.get_string("setn"));
//           setpos 1.0 2.0 3.0   several values following keyword:  if(opt.isset("setposn") {k=opt.iset("setposn"); x = opt.fopt(k+1); y=opt.fopt(k+2); z=opt.fopt(k+3);}
//
// NB keywords are searched for using strstr(keyword,argv[.]) using first match. Therefore short keys that are substrings of longer keys MUST appear first.
// One feature is that keywords can include leading - or -- on the command line for people who like that. 
// Single letter keywords are possible must be used with care but are not case sensitive (but you cna change force_lower if you want).  


class AOptions  { 
private:
	int argc;
	char **argv;
	// some global flags
	int is_verbose;   //for debug
	int have_device;  // for cuda - not used
	int istart;
	int force_lower(const char *arg);
	int dual;        // this to permit keyword:val OR keyword val
public:

	// constuctors
	AOptions() {dual = 1; is_verbose = 0; istart = 0; };
	AOptions(int argc_in,const char **argv_in,int start);
	int isset(const char *opt);
	inline void set_device(int dev) {have_device = dev; };
	inline void set_verbose(int verb) {is_verbose = verb; };
	inline void set_dual(int dv) {dual = dv; };
	inline void set_start(int start) {istart = start; };
	inline int device() {return  have_device; };
	inline int verbose() {return  is_verbose; };
	inline int oargc() {return  argc; };
	inline const char *oargv(int n) {return  argv[n]; };
	const char *get_string(const char *opt); 
	int   set_from_opt(const char *opt,int   default_val);
	float set_from_opt(const char *opt,float default_val);
	int iopt(int n);
	float fopt(int n);
	
	//~AOptions() { printf("~Aoptions start/end\n"); };
}; 

AOptions::AOptions(int argc_in,char **argv_in,int start)
{
	argc = argc_in;
	argv = argv_in;
	istart = start;  // dont crack parms before this one (filenames)	
	is_verbose = 0;
	have_device =1;
	dual = 1;
	for (int i=istart; i<argc; i++) force_lower(argv[i]);
}

int AOptions::force_lower(char *opt)
{
	int k=0;
	size_t len = strlen(opt);
	//printf("change %s to ",opt);
	//if (strlen(opt)< 2) return 0;
	while(k < len){
		if(opt[k] == 0 || opt[k] == ':')   break;                        //allow mixed case after :
		if(opt[k] >= 'A' && opt[k] <= 'Z') opt[k] = opt[k] - 'A' + 'a';  // assume ascii
		k++;
	}
	if(is_verbose) printf("opt lower: %s length %d\n",opt,k);
	if(k >256) return 1;
	else return 0;
}

int AOptions::isset(const char *opt)
{
	// returns positve int (true) if found, i+1 may point to an optional parameter
	for (int i=istart; i<argc; i++) if (strstr(argv[i],opt)) {
		if (is_verbose)printf("arg[%d]=%s has match for %s\n",i,argv[i],opt);
		return i;
	}
	return 0;
}

char * AOptions::get_string(const char *head)
{
	//char* none = "n";  // user should test this to set appropriate default.
	int k = isset(head);
	if(k > 0){
		//printf("getopt %s k=%d\n",opt,k);
		char *tail = strstr(argv[k],":");
		if (tail) {
			if (is_verbose)printf("get_string: argv[%d] = %s head match, tail is %s\n",k,head,tail+1);
			return tail+1;
		}
		else return argv[k+1];  // could be an error
	}
	else return NULL;
}

// try to use either optname:value  or optname value
int AOptions::set_from_opt(const char *opt,int default_val)
{
	if (!isset(opt)) return default_val;
	char *tail = get_string(opt);
	if(tail) return atoi(tail);
	else if (dual){
		int k = isset(opt)+1;
		if (k <argc) {
			int i = atoi(argv[k]);
			if (i > -10000000 && i < 10000000) { 
				if (is_verbose)printf("iopt %d val %d\n",k,i);
				return i; 
			}
		}
	}
	return default_val;
}

float AOptions::set_from_opt(const char *opt,float default_val)
{
	if (!isset(opt)) return default_val;
	char *tail = get_string(opt);
	if(tail) return (float)atof(tail);
	else if (dual){
		int k = isset(opt)+1;
		if (k <argc) {
			float t = (float)atof(argv[k]);
			if (t > -1.0e+8 && t < 1.0e+08)  { 
				if (is_verbose)printf("fopt %d val %f\n",k,t);
				return t; 
			}
		}
	}
	return default_val;
}

int AOptions::iopt(int n)   // integer value of parameter n
{
	if (n <argc) {
		int i = atoi(argv[n]);
		if (i > -10000000 && i < 10000000) { 
			if (is_verbose)printf("iopt %d val %d\n",n,i);
			return i; 
		}
	}
	if (is_verbose)printf("iopt using default for n=%d\n",n);
	return -99999;
}

float AOptions::fopt(int n)
{
	if (n <argc) {
		float t = (float)atof(argv[n]);  // float value of parameter n
		if (t > -1.0e+8 && t < 1.0e+08) { 
			if (is_verbose)printf("fopt %d val %f\n",n,t);
			return t; 
		}
	}
	if (is_verbose)printf("fopt using default for n=%d\n",n);
	return -99999.0f;
}
#endif