kernel void force_pairs(global float4* params,
                        global int2* pairs,
                        global float4* spring,                         
                        global float4* pos,
                        global float4* vel,
                        global float4* force,
                        global float4* out) {
  int n = get_global_id(0);
  
  int i = pairs[n].s0;
  int j = pairs[n].s1;
  
  float springConstant = spring[n].s0;
  float damping = spring[n].s1;
  float restLength = spring[n].s2;
  
  float4 a = pos[i];
  float4 b = pos[j];

  float4 va = vel[i];
  float4 vb = vel[j];
  
  float4 a2b = a - b;
  float4 Va2b = va - vb; 
  
  float a2bDistance = distance(a, b);		
  if (0 < a2bDistance) {
    a2b /= a2bDistance;
  }
	
  // spring force is proportional to how much it stretched 		
  float springForce = - (a2bDistance - restLength) * springConstant; 
		
  // want velocity along line b/w a & b, damping force is proportional to this
  float dampingForce = -damping * dot(a2b, Va2b);
		
  float r = springForce + dampingForce;
  float4 f = a2b * r;
  
  force[n] = f;
  out[n] = force[n];
}

kernel void force_total(global int* numPairs,
                        global int* pairsList,
                        global float4* forcePairs,
                        global float4* totalForce,
                        global float4* test,
                        int maxp) {
  int i = get_global_id(0);
  float4 total = (float4)(0, 0, 0, 0);
  
  // Get number of pairs particle i is forming springs with:
  int np = numPairs[i];
  
  // We loop over all the pairs for particle i:  
  for (int n = 0; n < np; n++) {
    // We first get index of pair involving i and the sign 
    // of the interaction (encoded together with the index 
    // to minimize memory access):
    int s = +1;
    int j = pairsList[maxp * i + n];
    
    if (j < 0) {
      s = -1;
      j *= -1;
    }
    j -= 1;
    
    // Getting jth spring force for particle i.
    float4 f = forcePairs[j];
    total += f * (float)(s);
  }
  
  totalForce[i] += total;  
}