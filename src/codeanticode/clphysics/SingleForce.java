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

import java.nio.FloatBuffer;

import com.jogamp.opencl.CLBuffer;
import com.jogamp.opencl.CLKernel;
import com.jogamp.opencl.CLProgram;

public abstract class SingleForce {
  protected ParticleSystem sys;  
  protected CLProgram program;  
  protected CLKernel kernel;
  protected boolean enabled; 

  public SingleForce(ParticleSystem s) {
    sys = s;
  }
  
  public void turnOn() {
    enabled = true;
  }

  public void turnOff() {
    enabled = true;
  }
  
  public abstract void apply(CLBuffer<FloatBuffer> pos, 
                             CLBuffer<FloatBuffer> vel,
                             CLBuffer<FloatBuffer> forces);
}