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

import static com.jogamp.opencl.CLMemory.Mem.READ_WRITE;

import java.io.IOException;
import java.io.InputStream;
import java.nio.FloatBuffer;
import com.jogamp.opencl.CLBuffer;
import com.jogamp.opencl.CLKernel;
import com.jogamp.opencl.CLProgram;

public class RungeKuttaIntegrator implements Integrator {
  ParticleSystem sys;
  CLProgram program;
  
  CLKernel cpKernel;
  CLKernel upKernel;
  CLKernel tsKernel;

  CLBuffer<FloatBuffer> k1Forces;
  CLBuffer<FloatBuffer> k1Positions;
  CLBuffer<FloatBuffer> k1Velocities;
  CLBuffer<FloatBuffer> k2Forces;
  CLBuffer<FloatBuffer> k2Positions;
  CLBuffer<FloatBuffer> k2Velocities;
  CLBuffer<FloatBuffer> k3Forces;
  CLBuffer<FloatBuffer> k3Positions;
  CLBuffer<FloatBuffer> k3Velocities;
  CLBuffer<FloatBuffer> k4Forces;
  CLBuffer<FloatBuffer> k4Positions;
  CLBuffer<FloatBuffer> k4Velocities;

  public RungeKuttaIntegrator(ParticleSystem sys) {
    this.sys = sys;

    InputStream stream = this.getClass().getResourceAsStream("rungekutta.cl");    
    try {
      program = sys.context.createProgram(stream).build();
    } catch (IOException e) {
      e.printStackTrace();
    }    
    
    cpKernel = program.createCLKernel("copy");
    upKernel = program.createCLKernel("update");
    
    if (sys.clglInterop) {
      tsKernel = program.createCLKernel("time_step_gl");
    } else {
      tsKernel = program.createCLKernel("time_step");
    }    
    
    k1Forces = sys.context.createFloatBuffer(sys.globalWorkSize * 4, READ_WRITE);
    k1Positions = sys.context.createFloatBuffer(sys.globalWorkSize * 4, READ_WRITE);
    k1Velocities = sys.context.createFloatBuffer(sys.globalWorkSize * 4, READ_WRITE);    
    k2Forces = sys.context.createFloatBuffer(sys.globalWorkSize * 4, READ_WRITE);
    k2Positions = sys.context.createFloatBuffer(sys.globalWorkSize * 4, READ_WRITE);
    k2Velocities = sys.context.createFloatBuffer(sys.globalWorkSize * 4, READ_WRITE);
    k3Forces = sys.context.createFloatBuffer(sys.globalWorkSize * 4, READ_WRITE);
    k3Positions = sys.context.createFloatBuffer(sys.globalWorkSize * 4, READ_WRITE);
    k3Velocities = sys.context.createFloatBuffer(sys.globalWorkSize * 4, READ_WRITE);
    k4Forces = sys.context.createFloatBuffer(sys.globalWorkSize * 4, READ_WRITE);
    k4Positions = sys.context.createFloatBuffer(sys.globalWorkSize * 4, READ_WRITE);
    k4Velocities = sys.context.createFloatBuffer(sys.globalWorkSize * 4, READ_WRITE);
  }

  public void step(float deltaT) {
    float t;
    
    // //////////////////////////////////////////////////////
    // get all the k1 values

    cpKernel.setArg(0, sys.positions[sys.currRead]);
    cpKernel.setArg(1, sys.velocities[sys.currRead]);
    cpKernel.setArg(2, k1Positions);
    cpKernel.setArg(3, k1Velocities);
    sys.queue.put1DRangeKernel(cpKernel, 0, sys.globalWorkSize, sys.localWorkSize).finish();

    sys.clearForces(k1Forces);    
    sys.applyForces(k1Positions, k1Velocities, k1Forces);
    
    // //////////////////////////////////////////////////////////////
    // get k2 values

    t = 0.5f * deltaT;
    upKernel.setArg(0, sys.parameters);    
    upKernel.setArg(1, sys.positions[sys.currRead]);
    upKernel.setArg(2, sys.velocities[sys.currRead]);
    upKernel.setArg(3, k1Forces);
    upKernel.setArg(4, k1Velocities);
    upKernel.setArg(5, k2Positions);
    upKernel.setArg(6, k2Velocities);
    upKernel.setArg(7, t);
    sys.queue.put1DRangeKernel(upKernel, 0, sys.globalWorkSize, sys.localWorkSize).finish();    

    sys.clearForces(k2Forces);    
    sys.applyForces(k2Positions, k2Velocities, k2Forces);
      
    // ///////////////////////////////////////////////////
    // get k3 values
    
    t = 0.5f * deltaT;
    upKernel.setArg(0, sys.parameters);    
    upKernel.setArg(1, sys.positions[sys.currRead]);
    upKernel.setArg(2, sys.velocities[sys.currRead]);
    upKernel.setArg(3, k2Forces);
    upKernel.setArg(4, k2Velocities);
    upKernel.setArg(5, k3Positions);
    upKernel.setArg(6, k3Velocities);
    upKernel.setArg(7, t);
    sys.queue.put1DRangeKernel(upKernel, 0, sys.globalWorkSize, sys.localWorkSize).finish();    

    sys.clearForces(k3Forces);    
    sys.applyForces(k3Positions, k3Velocities, k3Forces);

    // ////////////////////////////////////////////////
    // get k4 values

    t = deltaT;
    upKernel.setArg(0, sys.parameters);    
    upKernel.setArg(1, sys.positions[sys.currRead]);
    upKernel.setArg(2, sys.velocities[sys.currRead]);
    upKernel.setArg(3, k3Forces);
    upKernel.setArg(4, k3Velocities);
    upKernel.setArg(5, k4Positions);
    upKernel.setArg(6, k4Velocities);
    upKernel.setArg(7, t);
    sys.queue.put1DRangeKernel(upKernel, 0, sys.globalWorkSize, sys.localWorkSize).finish();    

    sys.clearForces(k4Forces);    
    sys.applyForces(k4Positions, k4Velocities, k4Forces);
    
    // ///////////////////////////////////////////////////////////
    // put them all together and what do you get?

    if (sys.clglInterop) {
      
    } else {
      tsKernel.setArg(0, sys.parameters);      
      tsKernel.setArg(1, sys.positions[sys.currRead]);
      tsKernel.setArg(2, sys.velocities[sys.currRead]);
      tsKernel.setArg(3, sys.positions[sys.currWrite]);
      tsKernel.setArg(4, sys.velocities[sys.currWrite]);    
      tsKernel.setArg(5, k1Velocities);
      tsKernel.setArg(6, k2Velocities);
      tsKernel.setArg(7, k3Velocities);
      tsKernel.setArg(8, k4Velocities);
      tsKernel.setArg(9, k1Forces);
      tsKernel.setArg(10, k2Forces);
      tsKernel.setArg(11, k3Forces);
      tsKernel.setArg(12, k4Forces);
      tsKernel.setArg(13, deltaT);
    }
    
    /*
    // Read back forces into host memory for checking purposes:
    sys.queue.putReadBuffer(sys.positions[sys.currWrite], true);
    System.out.println("Pos:");
    System.out.print(sys.positions[sys.currWrite].getBuffer().get() + ", ");
    System.out.println(sys.positions[sys.currWrite].getBuffer().get() + " ");
    sys.positions[sys.currWrite].getBuffer().rewind();     
    */
    
    sys.queue.put1DRangeKernel(tsKernel, 0, sys.globalWorkSize, sys.localWorkSize).finish();     
  }
}
