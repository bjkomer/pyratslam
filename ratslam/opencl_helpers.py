import numpy as np
import pyopencl as cl


def correlate_x( queue ):
  """Computes a correlation in the x direction"""

  textconf = {}
  text = """
  __kernel void fn(
    __global int *len_x,
    __global int *len_y,
    __global int *len_z,
    )
  {
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    const int z = get_global_id(2);

    for ( int idx = 0; idx < filterSize; idx++ ) {
      sum += input[ x + idx ] * filter[ idx ]
    }
    output[ x + y * get_global_size(0) + z * get_global_size(0) * get_global_size(1) ] = sum;
  }
  """

  text = Template( text, output_encoding='ascii').render(**textconf)

  _fn = cl.Program( queue.context, text ).build().fn

def correlate():
  """Computes a wrapping correlation of the posecell network with a filter kernel"""
  correlate_x()
  correlate_y()
  correlate_z()
