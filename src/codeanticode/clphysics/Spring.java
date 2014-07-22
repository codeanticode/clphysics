/*
  Copyright (c) 2011-14 Andres Colubri

  This library is free software; you can redistribute it and/or
  modify it under the terms of the GNU Lesser General Public
  License as published by the Free Software Foundation; either
  version 2.1 of the License, or (at your option) any later version.

  This library is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
  Lesser General Public License for more details.

  You should have received a copy of the GNU Lesser General
  Public License along with this library; if not, write to the
  Free Software Foundation, Inc., 59 Temple Place, Suite 330,
  Boston, MA  02111-1307  USA
*/

package codeanticode.clphysics;

import static com.jogamp.opencl.CLMemory.Mem.READ_ONLY;
import static com.jogamp.opencl.CLMemory.Mem.READ_WRITE;
import static java.lang.Math.min;

import java.io.IOException;
import java.io.InputStream;
import java.nio.FloatBuffer;
import java.nio.IntBuffer;
import java.util.ArrayList;

import processing.core.PApplet;

import com.jogamp.opencl.CLBuffer;
import com.jogamp.opencl.CLKernel;

/**
 * Implements a spring force with damping term.
 * 
 */
public class Spring extends FewPairsForce {
  CLBuffer<IntBuffer> pairs;
  CLBuffer<FloatBuffer> stringParams;  
  CLBuffer<FloatBuffer> forcePairs; 
  CLBuffer<IntBuffer> pairsList;
  CLBuffer<IntBuffer> numPairsPP;
  CLBuffer<IntBuffer> signPairs;
  CLBuffer<FloatBuffer> out;
  CLKernel accumKernel;
  int maxPairsPP;
  CLBuffer<FloatBuffer> test;
  
  public Spring(ParticleSystem sys, int n) {
    super(sys);

    InputStream stream = this.getClass().getResourceAsStream("spring.cl");    
    try {
      program = sys.context.createProgram(stream).build();
    } catch (IOException e) {
      e.printStackTrace();
    }
    kernel = program.createCLKernel("force_pairs");
    accumKernel = program.createCLKernel("force_total");
    
    numPairs = n;
    localWorkSize = min(sys.device.getMaxWorkGroupSize(), 256); // Local work size dimensions      
    globalWorkSize = sys.roundUp(localWorkSize, numPairs);      // rounded up to the nearest multiple of the localWorkSize     

    pairs = sys.context.createIntBuffer(globalWorkSize * 2, READ_ONLY);
    stringParams = sys.context.createFloatBuffer(globalWorkSize * 4, READ_ONLY);
    forcePairs = sys.context.createFloatBuffer((globalWorkSize + 1) * 4, READ_WRITE);
    
    out = sys.context.createFloatBuffer(globalWorkSize * 4, READ_WRITE);
    
  }
  
  public void setPair(int n, int i, int j, float ks, float d, float r) {
    int pair[] = {i, j};
    float params[] = {ks, d, r, 0};
    
    FloatBuffer fbuf = stringParams.getBuffer();  
    fbuf.position(4 * n);
    fbuf.put(params, 0, 4);    
    fbuf.rewind();
    
    IntBuffer ibuf = pairs.getBuffer();    
    ibuf.position(2 * n);
    ibuf.put(pair, 0, 2); 
    ibuf.rewind();    
  }
  
  public void update() {
    sys.queue.putWriteBuffer(pairs, true)
             .putWriteBuffer(stringParams, true);
        
    // Build list of pairs for each particle.
    maxPairsPP = 0;
    @SuppressWarnings("unchecked")
    ArrayList<Integer>[] pairsPP = new ArrayList[sys.numParticles];
    IntBuffer pbuf = pairs.getBuffer(); 
    for (int p = 0; p < numPairs; p++) {
      int i = pbuf.get();
      int j = pbuf.get();
      
      if (pairsPP[i] == null) {
        pairsPP[i] = new ArrayList<Integer>();
      }
      if (pairsPP[j] == null) {
        pairsPP[j] = new ArrayList<Integer>();
      }
      
      pairsPP[i].add(p + 1);
      pairsPP[j].add(-(p + 1));
      
      maxPairsPP = PApplet.max(maxPairsPP, pairsPP[i].size(), pairsPP[j].size());
    }
    
    test = sys.context.createFloatBuffer(sys.globalWorkSize * 4, READ_WRITE);
    FloatBuffer buf = test.getBuffer();
    for (int i = 0; i < sys.numParticles; i++) {
      buf.put(1); buf.put(2); buf.put(3); buf.put(0);
    }
    buf.rewind();
    sys.queue.putWriteBuffer(test, true);
    
    
    numPairsPP = sys.context.createIntBuffer(sys.globalWorkSize, READ_ONLY);
    pairsList = sys.context.createIntBuffer(sys.globalWorkSize * maxPairsPP, READ_ONLY);    
    
    IntBuffer npBuf = numPairsPP.getBuffer();
    IntBuffer plBuf = pairsList.getBuffer();
    
    PApplet.println("maxPairs: " + maxPairsPP);
    
    for (int i = 0; i < sys.numParticles; i++) {
      int s = pairsPP[i].size();
      
      PApplet.println("Number of pairs for particle " + i + ": " + s);
      npBuf.put(i, s);
            
      Object obj[] = pairsPP[i].toArray();
      int pairs[] = new int[maxPairsPP];
      PApplet.print("   Pairs: ");
      for (int k = 0; k < obj.length; k++) { 
        pairs[k] = (Integer)obj[k];
        int j = pairs[k];
        if (j < 0) j *= -1;
        j -= 1;
        PApplet.print(j + ", ");
      }     
      PApplet.println();
      plBuf.position(i * maxPairsPP);
      plBuf.put(pairs, 0, maxPairsPP);
    }
    npBuf.rewind();
    plBuf.rewind();
    
    sys.queue.putWriteBuffer(numPairsPP, true)
             .putWriteBuffer(pairsList, true);    
  }

  public void apply(CLBuffer<FloatBuffer> pos, 
                    CLBuffer<FloatBuffer> vel,
                    CLBuffer<FloatBuffer> forces) {
    if (!enabled) return;
    
    // Calculating all the i,j contributions.
    kernel.setArg(0, sys.parameters);
    kernel.setArg(1, pairs);
    kernel.setArg(2, stringParams);
    kernel.setArg(3, pos);
    kernel.setArg(4, vel);
    kernel.setArg(5, forcePairs);
    kernel.setArg(6, out);      
    sys.queue.put1DRangeKernel(kernel, 0, globalWorkSize, localWorkSize).finish();

/*
    // Read back forces into host memory for checking purposes:    
    sys.queue.putReadBuffer(out, true);
    FloatBuffer buf = out.getBuffer();
    System.out.println("force from spring calc:");
    for (int i = 0; i < numPairs; i++) {
      System.out.print(buf.get() + ", ");
      System.out.print(buf.get() + ", ");
      System.out.print(buf.get() + ", ");
      System.out.print(buf.get() + " | \n");
    }
    buf.rewind();  
  */
    
    // Accumulation all the pair contributions for each particle.
    accumKernel.setArg(0, numPairsPP);
    accumKernel.setArg(1, pairsList);
    accumKernel.setArg(2, forcePairs);
    accumKernel.setArg(3, forces);
    accumKernel.setArg(4, test);
    accumKernel.setArg(5, maxPairsPP);    
    sys.queue.put1DRangeKernel(accumKernel, 0, sys.globalWorkSize, sys.localWorkSize).finish();
    
/*
    // Read back forces into host memory for checking purposes:    
    sys.queue.putReadBuffer(forces, true);
    buf = forces.getBuffer();
    System.out.println("force from spring accum:");
    for (int i = 0; i < sys.numParticles; i++) {
      System.out.print(buf.get() + ", ");
      System.out.print(buf.get() + ", ");
      System.out.print(buf.get() + ", ");
      System.out.print(buf.get() + " | \n");
    }
    buf.rewind();    
*/
 }
}