# PyTorch MD utils

A collection of Torchscript compatible PyTorch functions useful for Molecular Simulation.
You can pip install this repo, or copy and paste the python code. 

## Install
```bash
git clone <repo_url>
cd <repo_name>
pip install .
```

## Usage


### Neighborlist


It outputs neighbors and shifts in the same format as ASE 
https://wiki.fysik.dtu.dk/ase/ase/neighborlist.html#ase.neighborlist.primitive_neighbor_list

neighbors, distances, shifts = simple_nl(..)
is equivalent to
    
[i, j], d, S = primitive_neighbor_list( quantities="ijdS", ...)

```python
from pytorch_md_utils import simple_nl

nl, distances, unit_shifts = simple_nl(positions, cell, True, cutoff)

```

### Ewald sum

Compute the Coulomb energy for particles in a periodic cell. This function is torchscipt compatible
The forces due to this energy can calculated with Autograd.

```python
from pytorch_md_utils import ewald


energy = ewald(positions, charges, cell)

energy.backward()
forces = -positions.grad
```