import os
import sys
import numpy as np
import logging
from logging import getLogger
import shutil
import datetime

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from nlfepy import Mesh, Variable, Constitutive, Viewer
from nlfepy.interface import PVW_UL
from nlfepy.io import VtuWriter


def main(mesh_path):

    # Number of steps
    n_steps = 10

    # Set logger
    logging.basicConfig(level=logging.INFO)
    logger = getLogger("pvw")
    logger.info("Program start...")

    # Read mesh
    logger.info("Setting mesh info...")
    mesh = Mesh()
    mesh.read(mesh_path)

    # Physical quantities
    vals = Variable()

    # Set constitutive
    logger.info("Setting constitutive equation...")
    # cnst_params = {}
    cnst_dict = {
        "Al": ["isotropic"],
        # 'Al': ['j2flow'],  # , cnst_params],
        # 'Al': ['crystal_plasticity'],  # , cnst_params],
    }
    constitutive = Constitutive(cnst_dict, nitg=mesh.n_tintgp, val=vals["itg"])

    # Solve the governing equation (Principle of virtual work)
    logger.info("Solving the governing equation...")
    pvw_params = {
        "logging": False,
    }
    pvw = PVW_UL(mesh=mesh, cnst=constitutive, val=vals["point"], params=pvw_params)

    # Make output dir
    writer = VtuWriter(mesh=mesh, values=vals)
    output_dir = r"out"
    vtu_header = r"result"
    try:
        if os.path.exists(output_dir):
            timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
            shutil.move(output_dir, output_dir + "_" + timestamp)
        os.mkdir(output_dir)
    except PermissionError:
        logger.error("Permission error")
    except Exception:
        logger.error("Failed to create output directory")
        sys.exit(1)

    # Set viewer instance
    projection = "3d" if mesh.n_dof == 3 else "2d"
    viewer = Viewer(projection=projection)

    # Main Loop
    for istep in range(n_steps):
        logger.info("Step No. {} / {}".format(istep + 1, n_steps))
        # Solve global stiffness equation
        pvw.solve()
        # Save result in VTK XML format
        vals["element"]["rand"] = np.random.rand(mesh.n_element, 3)
        writer.write(
            os.path.join(output_dir, vtu_header + "_" + str(istep) + ".vtu"),
            output_bc=False,
        )

    # Plot result
    logger.info("Drawing mesh...")
    projection = "3d" if mesh.n_dof == 3 else "2d"
    viewer = Viewer(projection=projection)
    val = None
    viewer.plot(mesh=mesh, val=val)
    # viewer.save('result.png', transparent=True, dpi=300)
    viewer.show()

    logger.info("Program end")


if __name__ == "__main__":
    # main('tests/data/mesh.vtu')
    main("tests/data/mesh_3d.vtu")
