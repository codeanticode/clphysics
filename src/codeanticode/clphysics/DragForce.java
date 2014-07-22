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

public class DragForce extends SingleForce {
  float dragConstant;
  
  public DragForce(ParticleSystem sys) {
    super(sys); 
    
    enabled = false;
    
    InputStream stream = this.getClass().getResourceAsStream("drag.cl");    
    try {
      program = sys.context.createProgram(stream).build();
    } catch (IOException e) {
      e.printStackTrace();
    }
    kernel = program.createCLKernel("drag_force");
    dragConstant = 0.01f;
  } 
  
  public void setDragConstant(float dc) {
    dragConstant = dc;    
  }
  
  public void apply(CLBuffer<FloatBuffer> pos, 
                    CLBuffer<FloatBuffer> vel,
                    CLBuffer<FloatBuffer> forces) {
    if (!enabled) return;
    
    kernel.setArg(0, sys.parameters);
    kernel.setArg(1, vel);
    kernel.setArg(2, forces);
    kernel.setArg(3, dragConstant);
    sys.queue.put1DRangeKernel(kernel, 0, sys.globalWorkSize, sys.localWorkSize).finish();    
  }  
}
