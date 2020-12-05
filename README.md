# nlfepy
Simple finite element Python library for nonlinear problems

## Installation

```bash
pip install git+https://github.com/sKujirai/nlfepy
```

## Quick start
For the detail procedure, refer to a [sample notebook](./notebook/example_pvw.ipynb)

```python
from nlfepy import Mesh, Material, Variable, Constitutive, Viewer
from nlfepy.interface import PVW
from nlfepy.io import VtuWriter
from nlfepy.util import calc_element_value


# Read mesh
mesh = Mesh()
mesh.read(mesh_path)

# or generate mesh using dmsh
# import dmsh
# geo = dmsh.Rectangle(0., 1., 0., 1.)
# coords, connectivity = dmsh.generate(geo, 0.1)
# mesh = Mesh()
# mesh.set_shape(coords=coords.T, connectivity=connectivity)
# mesh.set_bc(constraint='compression', value=0.001)

# Set material
mater = Material('Al')

# Physical quantities
vals = Variable()

# Set constitutive equation class
constitutive = Constitutive(
    mater,
    nitg=mesh.n_tintgp,
    val=vals['itg']
)

# Solve a governing equation (Principle of virtual work)
pvw = PVW(
    mesh=mesh,
    cnst=constitutive,
    val=vals['point'],
)

# Solve KU=F
pvw.solve()
# Calc. stress (optional)
pvw.calc_stress()

# Calc. element values (for output)
calc_element_value(mesh=mesh, values=vals)

# Plot results
viewer = Viewer()
viewer.plot(
    mesh=mesh,
    val=vals['element']['stress'][:, 1],
    title='Stress YY',
)
viewer.save('result.png', transparent=True, dpi=300)

# Contour plot
viewer.contour(
    mesh=mesh,
    val=vals['point']['u_disp'][1],
    title='Displacement Y'
)

# Save results in VTK XML format
writer = VtuWriter(mesh=mesh, values=vals)
writer.write('result.vtu')
```
