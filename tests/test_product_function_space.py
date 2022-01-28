import fenics
import product_fem as prod

import pytest

class TestProductFunctionSpace():

    def test_basic_properties(self):
        n = 22
        mesh = fenics.UnitIntervalMesh(n-1)
        V1 = FunctionSpace(mesh, 'CG', 1)
        V = prod.ProductFunctionSpace(V1)

        assert V.marginal_mesh == mesh
        assert V.marginal_function_space == V1


class TestBoundaryConditions():

    def verify_bc(self, bc, V, bc_fn):
        assert bc.product_function_space == V
        assert bc.marginal_function_space == V.marginal_vunction_space

    def test_zero_dirichlet_bc(self):
        n = 22
        mesh = fenics.UnitIntervalMesh(n-1)
        V1 = FunctionSpace(mesh, 'CG', 1)
        V = prod.ProductFunctionSpace(V1)
        bc = prod.ProductDirichletBC(V, 0)

        def bc_fn(x, y):
            return 0.0

        self.verify_bc(bc, V, bc_fn)

    def test_nonzero_dirichlet_bc(self):
        n = 22
        mesh = fenics.UnitIntervalMesh(n-1)
        V1 = FunctionSpace(mesh, 'CG', 1)
        V = prod.ProductFunctionSpace(V1)

        def u_bdry(x, y):
            return 0.5 * (x**2 + x*y + y**2)

        bc = prod.ProductDirichletBC(V, u_bdry)
        self.verify_bc(bc, V, u_bdry)


