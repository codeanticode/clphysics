// Adapted from N-Body Simulation tutorial by Brown Deer Technology:
// http://www.browndeertechnology.com/docs/BDT_OpenCL_Tutorial_NBody-rev3.html
// The Chapter 31 from GPU Gems 3 (Fast N-Body Simulation with CUDA) is also
// useful. Available online:
// http://http.developer.nvidia.com/GPUGems3/gpugems3_ch31.html

kernel void gravity_force(global float4* params,
                          global float4* pos,                          
                          global float4* force,
                          local float4* pcache,
                          float gconst,
                          float inveps) {
  // Getting global and local id's of current work-item:                         
  int i = get_global_id(0);
  int il = get_local_id(0);

  // Global and local sizes:
  int n = get_global_size(0);
  int nl = get_local_size(0);
  
  // Number of work-groups (or thread-blocks):
  int nb = n / nl;
  
  float4 f0 = force[i];  
  float4 fi = (float4)(0, 0, 0, 0);
  float4 p = pos[i]; 
  float m = params[i].y;
   
  // We iterate over all the blocks, in order to calculate the total
  // gravitational force exerted on i by all the other particles:
  for (int jb = 0; jb < nb; jb++) {
  
      // We cache the positions and masses of all the items in the current 
      // block in local memory for fast access during the evaluation loop:
      pcache[il].xyz = pos[jb * nl + il].xyz;
      pcache[il].w = params[jb * nl + il].y;
      // We wait until all the work-items in the current block finish
      // with the last instruction, so that the pcache contains all 
      // the data we need for the evaluation loop below:      
      barrier(CLK_LOCAL_MEM_FENCE); 

      // We evaluate the interaction of i with all the cached positions:
      for (int j = 0; j < nl; j++) {
        // Read a cached particle position:
        float4 p2 = (float4)(pcache[j].xyz, 0);        
        float m2 = pcache[j].w;
        float4 d = p2 - p;
        
        float invr = rsqrt(d.x * d.x + d.y * d.y + d.z * d.z);
        if (invr > inveps) {
          invr = inveps;
        }
        
        float f = gconst * m * m2 * invr * invr * invr;
        fi += f * d;
      }

      // Wait for all the work-items in the current block to finish.
      barrier(CLK_LOCAL_MEM_FENCE);
  }   
   
  force[i] = f0 + fi;
}
