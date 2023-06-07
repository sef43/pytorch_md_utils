import torch
from typing import Tuple


@torch.jit.script
def simple_nl(positions: torch.Tensor, cell: torch.Tensor, pbc: bool, cutoff: float, sorti: bool=False, halfList: bool=False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """simple torchscriptable neighborlist. 
    
    It aims are to be correct, clear, and torchscript compatible.
    It is O(n^2) but with pytorch vectorisation the prefactor is small.
    It outputs neighbors and shifts in the same format as ASE 
    https://wiki.fysik.dtu.dk/ase/ase/neighborlist.html#ase.neighborlist.primitive_neighbor_list

    neighbors, distances, shifts = simple_nl(..)
    is equivalent to
    
    [i, j], d, S = primitive_neighbor_list( quantities="ijdS", ...)

    Limitations:
        - either no PBCs or PBCs in all x,y,z
        - cutoff must be less than half the smallest box length
        - cell must be rectangular or triclinic in OpenMM format:
        http://docs.openmm.org/development/userguide/theory/05_other_features.html#periodic-boundary-conditions

    Parameters
    ----------
    positions: torch.Tensor
        Coordinates, shape [N,3]
    cell: torch.Tensor
        Triclinic unit cell, shape [3,3], must be in OpenMM format: http://docs.openmm.org/development/userguide/theory/05_other_features.html#periodic-boundary-conditions 
    pbc: bool
        should PBCs be applied
    cutoff: float
        Distances beyond cutoff are not included in the neighborlist
    soti: bool=False
        if True the returned neighborlist will be sorted in the i index. The default is False (no sorting).
    halfList: bool=False
        if True only half of the list where j>i will be returned. The default is False (full i,j list returned). 
        
    Returns
    -------
    neighbors: torch.Tensor
        neighbor list, shape [2, number of neighbors]
    distances: torch.Tensor
        pair distance vector, shape [number of neighbors]
    shifts: torch.Tensor
        shift vector, shape [number of neighbors, 3], From ASE docs: 
        shift vector (number of cell boundaries crossed by the bond between atom i and j). 
        With the shift vector S, the distances D between atoms can be computed from:
        D = positions[j]-positions[i]+S.dot(cell)
    """

    num_atoms = positions.shape[0]
    device=positions.device

    # get i,j indices where j>i
    uij = torch.triu_indices(num_atoms, num_atoms, 1, device=device)
    triu_deltas = positions[uij[0]] - positions[uij[1]]

    wrapped_triu_deltas=triu_deltas.clone()

    if pbc:
        # using method from: https://github.com/openmm/NNPOps/blob/master/src/pytorch/neighbors/getNeighborPairsCPU.cpp
        wrapped_triu_deltas -= torch.outer(torch.round(wrapped_triu_deltas[:,2]/cell[2,2]), cell[2])
        wrapped_triu_deltas -= torch.outer(torch.round(wrapped_triu_deltas[:,1]/cell[1,1]), cell[1])
        wrapped_triu_deltas -= torch.outer(torch.round(wrapped_triu_deltas[:,0]/cell[0,0]), cell[0])

        # From ASE docs:
        # wrapped_delta = pos[i] - pos[j] - shift.cell
        # => shift = ((pos[i]-pos[j]) - wrapped_delta).cell^-1
        shifts = torch.mm(triu_deltas - wrapped_triu_deltas, torch.linalg.inv(cell))

    else:
        shifts = torch.zeros(triu_deltas.shape, device=device)
    
    triu_distances = torch.linalg.norm(wrapped_triu_deltas, dim=1)

    # filter
    mask = triu_distances > cutoff
    uij = uij[:,~mask]    
    shifts = shifts[~mask, :]
    triu_distances = triu_distances[~mask]

    if halfList:
        return uij, triu_distances, shifts


    # get the ij pairs where j<i
    lij = torch.stack((uij[1], uij[0]))
    neighbors = torch.hstack((uij, lij))
    shifts = torch.vstack((shifts, -shifts))
    distances = torch.hstack((triu_distances, triu_distances))
    
    if sorti:
        idx = torch.argsort(neighbors[0])
        neighbors = neighbors[:,idx]
        shifts = shifts[idx,:]
        distances = distances[idx]

    return neighbors, distances, shifts
