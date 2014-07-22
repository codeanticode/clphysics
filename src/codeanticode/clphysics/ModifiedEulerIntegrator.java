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

import com.jogamp.opencl.CLKernel;
import com.jogamp.opencl.CLProgram;
import java.io.IOException;
import java.io.InputStream;
import processing.core.PApplet;

public class ModifiedEulerIntegrator implements Integrator {
  ParticleSystem sys;
  CLProgram program;
  CLKernel kernel;
  //CLBuffer<FloatBuffer> out;

  public ModifiedEulerIntegrator(ParticleSystem sys) {
    this.sys = sys;
    
    InputStream stream = this.getClass().getResourceAsStream("modeuler.cl");    
    try {
      program = sys.context.createProgram(stream).build();
    } catch (IOException e) {
      e.printStackTrace();
    }    
    if (sys.clglInterop) {
      kernel = program.createCLKernel("time_step_gl");
    } else {
      kernel = program.createCLKernel("time_step");
    }
    
    //out = sys.context.createFloatBuffer(sys.globalWorkSize * 4, READ_WRITE);    
    PApplet.println("Done initializing integrator");
  }

  public void step(float t) {
    sys.clearForces();
    sys.applyForces();

    if (sys.clglInterop) {
      
    } else {
      kernel.setArg(0, sys.parameters);
      kernel.setArg(1, sys.forces);
      kernel.setArg(2, sys.positions[sys.currRead]);
      kernel.setArg(3, sys.velocities[sys.currRead]);
      kernel.setArg(4, sys.positions[sys.currWrite]);
      kernel.setArg(5, sys.velocities[sys.currWrite]);      
      kernel.setArg(6, t);      
    }
    
    sys.queue.put1DRangeKernel(kernel, 0, sys.globalWorkSize, sys.localWorkSize).finish();
    
    /*
    // Read back forces into host memory for checking purposes:
    sys.queue.putReadBuffer(out, true);
    FloatBuffer buf = out.getBuffer();
    System.out.println("output from modeuler calc:");
    System.out.print(buf.get() + ", ");
    System.out.println(buf.get() + " ");
    buf.rewind();       
        */
  }
}
