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

import java.io.IOException;
import java.io.InputStream;
import java.nio.FloatBuffer;

import com.jogamp.opencl.CLBuffer;

import processing.core.PApplet;

public class Gravity extends AllPairsForce {
  protected float gravConst;
  protected float minPairDist;
  protected float minPairDistInv;
  
  public Gravity(ParticleSystem sys) {
    super(sys); 
    
    enabled = false;
    // CUDA implementation of Barnes Hut approximation:
    // http://www.gpucomputing.net/?q=node/1314
    
    InputStream stream = this.getClass().getResourceAsStream("gravity.cl");    
    try {
      program = sys.context.createProgram(stream).build();
    } catch (IOException e) {
      e.printStackTrace();
    }
    kernel = program.createCLKernel("gravity_force");
    gravConst = 1.0f;
    minPairDist = 1.0f;
    minPairDistInv = 1.0f / minPairDist;
    
    PApplet.println("Loaded gravitational force");
  }
  
  public void setGravitationalConstant(float gc) {
    gravConst = gc;    
  }

  public void setMinPairDistance(float pd) {
    minPairDist = pd;
    minPairDistInv = 1.0f / minPairDist;
  }
  
  public void apply(CLBuffer<FloatBuffer> pos, 
                    CLBuffer<FloatBuffer> vel,
                    CLBuffer<FloatBuffer> forces) {
    if (!enabled) return;
    
    kernel.setArg(0, sys.parameters);
    kernel.setArg(1, pos);
    kernel.setArg(2, forces);
    kernel.setNullArg(3, sys.localWorkSize * 4);    
    kernel.setArg(4, gravConst);
    kernel.setArg(5, minPairDistInv);
    sys.queue.put1DRangeKernel(kernel, 0, sys.globalWorkSize, sys.localWorkSize).finish();
 }  
}
