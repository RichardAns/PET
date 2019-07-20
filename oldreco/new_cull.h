// new_cull taken from culltab

#ifndef COMMAPAD_H_
#define COMMAPAD_H_


int comma_pad(double val, char *cpad)
{
	char* buf = mycalloc<char>(100,"buf");
	sprintf(buf, "%.0f",val);
	int len = (int)strlen(buf);
	int commas = len/3;
	if (commas*3 == len) commas--;
	int tail_out = len+commas;
	int tail_buf = len-1;
	cpad[tail_out--] = 0;
	for (int k = 0; k < len; k++){
		if (k>0 && k%3 == 0) cpad[tail_out--] = ',';
		cpad[tail_out--] = buf[tail_buf--];		
	}
	free(buf);
	return 0;
}

#endif

#ifndef NEWCULL_H_
#define NEWCULL_H_

int new_cull(int xpos,int ypos,int zpos,double cut,double tcut)
{
	if (zpos < 95 || zpos > 96){ printf("zpos %d not supported\n",zpos); return 1; }

	char name[256];
	size_t lors = 0;	
	double *gentab = mycalloc<double>(10, "gentab");
	sprintf(name,"gentrys\\gentrys_%d_%d_%d.raw",xpos,ypos,zpos);
	read_raw<double>(name,gentab,5);


	sprintf(name,"ltab\\sltab_%d_%d_%d.raw",xpos,ypos,zpos);
	Lor *ltab = use_raw_full<Lor>(name,lors,name);  if (!ltab) return 1;
	std::sort<Lor *>(ltab,ltab+lors,lorsort_key);   // resort by key  
	double hitsum = 0.0; // accounting
	double hitkeep = 0.0;
	double hitdrop = 0.0;

	// lookup[i] => k+1  for lor having index=k;  NB lookup is a 3GB array
	uint *lookup = mycalloc<uint>(F120_STride*F120_STride,"lookup"); if (!lookup) return 1;

	quad p;  // proper lor
	quad q;  // flip p
	quad m;  // mirror
	quad w;  // flip m

	
	for (int k = 0; k<lors; k++){
		int index = index_from_key(ltab[k].key);
		if (lookup[index]> 0)printf("Warning key = %d found twice at index %d and %d\n",index_from_key(ltab[k].key),k,lookup[index]-1);  // could comine lors here
		lookup[index] = k+1;
		//lcount[index]++;
	}


	// combine proper and improper master and mirrors
	int cut_kill = 0;
	int mirror_kill = 0;
	int mirror_drop = 0;
	for (int ip=0; ip<F120_STride*F120_STride; ip++){
		int bugs =0;
		int plor = lookup[ip]-1;
		int qlor = 0;
		int mlor = 0;
		int wlor = 0;
		float ivalues = 0.0f;
		if (lookup[ip] > 0){
			//float pval = -1.0f;
			lor_from_index(ip,p);
			//if (p.z1==48 && p.z2 ==48 && ltab[plor].val >2.0f) bugs =1;
			if ( (abs(p.z1 - p.z2) >= F120_TrueNZ) || (abs(p.c1-p.c2) < F120_DCmin) || (abs(p.c1-p.c2) > F120_DCmax) ) {  // drop if delataZ > 47. or delata c not in [72,216]
				lookup[ip] = 0;
				cut_kill++;
				hitdrop += ltab[plor].val;
				continue;
			}
			hitkeep += ltab[plor].val;
			ivalues = 1.0f; // start new master here
			if (bugs) printf("p (%2d %3d)-(%2d %3d) %6.2f",p.z1,p.c1,p.z2,p.c2,ltab[plor].val);

			// combine proper and improper lors in case both present
			flip(p,q);
			int iq = index_from_lor(q);
			if (iq>0 && iq != ip  && lookup[iq] > 0){        // check for q=p, should be impossible since c's must differ
				qlor = lookup[iq]-1;
				ltab[plor].val += ltab[qlor].val;    // add improper value to p, do NOT increment count as part of same lor
				hitkeep += ltab[qlor].val;
				if (bugs) printf(" q (%2d %3d)-(%2d %3d) %6.2f",q.z1,q.c1,q.z2,q.c2,ltab[qlor].val);
				lookup[iq] = 0;                      // drop q
				// ensure p is the proper version - if necessary
				if (p.z1 > q.z1 || (p.z1==q.z1 && p.c1 > q.c1)) ltab[plor].key = ltab[qlor].key;
			}

			// find mirror and remove, but use average value. Special treatment for self mirror 48-48 case for even zbin
			int check = mirror(p,m,zpos);  // check = 1 if 48-48, =2 if out of detector
			if (check == 2) mirror_kill++;
			int im = index_from_lor(m);
			if (check==1) ltab[plor].val *= 0.5f;  //NB added 23/08/17 so can use mirror in subsequent code without special if statements.
			else if (im>0 && !check){
				if (lookup[im] > 0) {
					mlor = lookup[im]-1;
					ltab[plor].val += ltab[mlor].val;
					hitkeep += ltab[mlor].val;
					if (bugs) printf(" m (%2d %3d)-(%2d %3d) %6.2f",m.z1,m.c1,m.z2,m.c2,ltab[mlor].val);
					ivalues = 2.0f;
					lookup[im] = 0;
					mirror_drop++;
				}
			}

			// deal with cases wherre both with proper and improper mirror lors exist (always distict)
			flip(m,w);
			int iw = index_from_lor(w);
			if (iw>0 && !check){
				if (lookup[iw] > 0) {
					wlor = lookup[iw]-1;
					ltab[plor].val += ltab[wlor].val;
					hitkeep += ltab[wlor].val;
					if (bugs) printf(" w (%2d %3d)-(%2d %3d) %6.2f",w.z1,w.c1,w.z2,w.c2,ltab[wlor].val);
					ivalues = 2.0f;
					lookup[iw] = 0;
					mirror_drop++;
				}
			}

			if (ivalues > 0.0f && ltab[plor].val > 0.0f) ltab[plor].val /= ivalues;  // use average of master and mirror
			if (bugs) printf(" pm %6.2f\n",ltab[plor].val);
		}
	}       // end ip loop

	// now use averaged masters
	size_t plors = 0;
	//double lsum = 0.0;
	for (int ip=0; ip<F120_STride*F120_STride; ip++) if (lookup[ip] > 0) plors++;
	if (lors < 10) { printf("%d lors found in new_cull - somethings wrong\n",lors); return 1; }
	
	Lor *pltab = mymalloc<Lor>(plors,"pltab"); if (!pltab) return 1;
	size_t nlors = 0;
	for (int ip=0; ip<F120_STride*F120_STride; ip++) if (lookup[ip] > 0) {
		int plor = lookup[ip]-1;
		pltab[nlors].key = ltab[plor].key;
		pltab[nlors].val = ltab[plor].val;
		nlors++;
	}
	if (nlors != plors) { printf("nlors = %d but plors = %d in new_cull\n",nlors,plors); return 1; }

	std::sort<Lor *>(pltab,pltab+plors,lorsort);   // sort by value	
	double sum = 0.0;
	for (int k=0; k<plors; k++) sum += pltab[k].val;
	cut = sum*(1.0 - tcut);

	// drop smallest values - depends on input sorted by value descending
	double tail = 0.0;
	float cutval = 0.0f;
	size_t kcut = 0;
	for (size_t k = plors-1; k > 0; k--){
		tail += pltab[k].val;
		cutval = pltab[k].val;
		hitkeep -= pltab[k].val;
		hitdrop += pltab[k].val;
		kcut = k;
		if (tail > cut) break;
	}
	while (pltab[kcut].val <= cutval && kcut >0) {
		hitkeep -= pltab[kcut].val;
		hitdrop += pltab[kcut].val;
		kcut--;  //drop more if same value - for consistancy NB possible accounting error introduced here
	}
	plors = kcut;
	hitkeep *= (gentab[3]*1.0e-06);   // restore normalization
	hitdrop *= (gentab[3]*1.0e-06);
	hitsum = hitkeep+hitdrop;
	printf("new_cull %d lors left from %d after culls and value cut at %9.5f cuts %d mirror %d %d\n",plors,lors,cutval,cut_kill,mirror_kill,mirror_drop);
	

	gentab[5] = hitkeep;
	gentab[6] = hitdrop;
	gentab[7] = hitsum;
	gentab[8] = gentab[5] / gentab[2];   // true efficiency
	char g5[100]; comma_pad(hitkeep,g5);
	char g6[100]; comma_pad(hitdrop,g6);
	char g7[100]; comma_pad(hitsum,g7);
	printf("gentab updated keep %s drop %s sum %s eff %.3f %%\n",g5,g6,g7,gentab[8]*100.0);

	sprintf(name,"gentrys\\new_gentrys_%d_%d_%d.raw",xpos,ypos,zpos);
	write_raw<double>(name,gentab,10);

	// finally save culled data
	std::sort<Lor *>(pltab,pltab+plors,lorsort_key);
	sprintf(name,"cull\\cull_%d_%d_%d.raw",xpos,ypos,zpos);
	write_raw(name,pltab,plors);

	free(pltab);
	free(lookup);
	free(ltab);
	return 0;
}

#endif