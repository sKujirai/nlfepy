import os
import sys
import numpy as np
import logging
from logging import getLogger

# import dmsh
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from nlfepy import Mesh, Material, Variable, Constitutive, Viewer
from nlfepy.interface import PVW
from nlfepy.io import VtuWriter
from nlfepy.util import calc_element_value


def main(mesh_path):

    # Set logger
    logging.basicConfig(level=logging.INFO)
    logger = getLogger("pvw")
    logger.info("Program start...")

    # Read mesh
    logger.info("Setting mesh info...")
    mesh = Mesh()
    mesh.read(mesh_path)

    # Prepare mesh using dmsh
    # geo = dmsh.Rectangle(0., 1., 0., 1.)
    # coords, connectivity = dmsh.generate(geo, 0.1)

    # Mesh class
    # mesh = Mesh()
    # mesh.set_shape(coords=coords.T, connectivity=connectivity)
    # mesh.set_bc(constraint='compression', value=0.001)

    # Set material
    logger.info("Setting material...")
    mater = Material("Al")

    # Physical quantities
    vals = Variable()

    # Set constitutive
    logger.info("Setting constitutive equation...")
    constitutive = Constitutive(mater, nitg=mesh.n_tintgp, val=vals["itg"])

    # Solve the governing equation (Principle of virtual work)
    logger.info("Solving the governing equation...")
    pvw_params = {
        "logging": True,
    }
    pvw = PVW(mesh=mesh, cnst=constitutive, val=vals["point"], params=pvw_params)
    # Solve KU=F
    pvw.solve()
    # Calc. stress
    pvw.calc_stress()

    # Calc. element values
    calc_element_value(mesh=mesh, values=vals)

    # Plot result
    logger.info("Drawing mesh...")
    projection = "3d" if mesh.n_dof == 3 else "2d"
    viewer = Viewer(projection=projection)

    # Check B.C.
    viewer.plot_bc(mesh)
    viewer.save("boundary_conditions.png")
    # viewer.show()

    # Plot result
    viewer.plot(
        mesh=mesh, val=vals["element"]["stress"][:, 1], title="Stress YY",
    )
    viewer.save("result.png")
    # viewer.show()

    # Contour plot
    if mesh.n_dof == 2:
        viewer.contour(
            mesh=mesh, val=vals["point"]["u_disp"][1], title="Displacement Y"
        )
        viewer.save("result_contour.png")
        # viewer.show()

    # Save results
    logger.info("Saving results")
    writer = VtuWriter(mesh=mesh, values=vals)
    writer.write("result.vtu")  # , output_bc=False)

    logger.info("Program end")


if __name__ == "__main__":
    main("tests/data/mesh.vtu")
    # main('tests/data/mesh_mpc.vtu')
    # main('tests/data/mesh_load.vtu')
    # main('tests/data/mesh_3d.vtu')
    # main('tests/data/mesh_3d_load.vtu')
