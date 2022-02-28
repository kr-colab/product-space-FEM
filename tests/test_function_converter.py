import pytest
import product_fem as pf
import fenics as fx
import numpy as np


class TestFunctionConverters:
    
    def test_1d_converters(self):
        n = 25
        mesh = fx.UnitIntervalMesh(n-1)
        V = fx.FunctionSpace(mesh, 'CG', 1)
        dof_coords = V.tabulate_dof_coordinates().flatten()
        
        # these should all be equivalent
        string = 'exp(-x[0] + sin(x[0]))'
        ndarray = np.array([np.exp(-x + np.sin(x)) for x in dof_coords])
        pyfunc = lambda x: np.exp(-x + np.sin(x))
        
        f_string = pf.string_to_Function(string, V)
        f_ndarray = pf.ndarray_to_Function(ndarray, V)
        f_pyfunc = pf.pyfunc_to_Function(pyfunc, V)
        
        assert fx.assemble((f_string - f_ndarray)**2 * fx.dx) < 1e-10
        assert fx.assemble((f_string - f_pyfunc)**2 * fx.dx) < 1e-10
        assert fx.assemble((f_ndarray - f_pyfunc)**2 * fx.dx) < 1e-10
        
    def test_2d_converters(self):
        nx, ny = 11, 11
        mesh = fx.UnitSquareMesh(nx-1, ny-1)
        V = fx.FunctionSpace(mesh, 'CG', 1)
        dof_coords = V.tabulate_dof_coordinates() # shape (nx*ny, 2)
        
        # these should all be equivalent
        string = 'exp(-x[0]) + exp(-x[1])'
        ndarray = np.array([np.exp(-xy[0]) + np.exp(-xy[1]) for xy in dof_coords])
        pyfunc = lambda x,y: np.exp(-x) + np.exp(-y)
        
        f_string = pf.string_to_Function(string, V)
        f_ndarray = pf.ndarray_to_Function(ndarray, V)
        f_pyfunc = pf.pyfunc_to_Function(pyfunc, V)
        
        assert fx.assemble((f_string - f_ndarray)**2 * fx.dx) < 1e-10
        assert fx.assemble((f_string - f_pyfunc)**2 * fx.dx) < 1e-10
        assert fx.assemble((f_ndarray - f_pyfunc)**2 * fx.dx) < 1e-10
        
        