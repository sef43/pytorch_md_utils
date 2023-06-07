import pytest
import openmm as mm
import openmm.app as app
import numpy as np
from openmm.unit import *
from sys import stdout
import torch
from pytorch_md_utils import ewald

@pytest.mark.parametrize("N", [10, 100, 1000])
@pytest.mark.parametrize("testGrad", [False, True])
def testEwald(N, testGrad):

    # make a box of N charged particles in a box of shape 2,2,3

    topology=app.Topology()
    chain=topology.addChain()
    for i in range(N):
        residue=topology.addResidue('Ar', chain)
        topology.addAtom('Ar', app.element.get_by_symbol('Ar'), residue)

    cell = np.array([[2,0,0],
                     [0,2.5,0],
                     [0,0,3]])
    topology.setPeriodicBoxVectors(cell)

    # make sure positions are also outside of box to test PBC wrapping
    positions = (np.random.random(size=(N,3)))*10.0
    charges = [1 for q in range(N//2)] + [-1 for q in range(N//2)]

    # compute energy using OpenMM
    system = mm.System()
    for atom in topology.atoms():
        system.addParticle(18.0)
    system.setDefaultPeriodicBoxVectors(*cell)

    force = mm.NonbondedForce()
    for i in range(N):
        force.addParticle(charges[i], 1.0, 0.0)

    force.setNonbondedMethod(mm.NonbondedForce.Ewald)
    system.addForce(force)
    integrator = mm.LangevinMiddleIntegrator(300*kelvin, 1/picosecond, 0.001*picoseconds)
    simulation = app.Simulation(topology, system, integrator, platform=mm.Platform.getPlatformByName('Reference'))
    simulation.context.setPositions(positions)
    state = simulation.context.getState(getEnergy=True, getForces=True)
    
    validEnergy=state.getPotentialEnergy()._value
    validForces=state.getForces()._value


    tensorPos = torch.tensor(positions, requires_grad=testGrad)
    tensorCell = torch.tensor(cell)
    tensorCharges = torch.tensor(charges)
    testEnergy = ewald(tensorPos, tensorCharges, tensorCell)
    
    
    assert(np.isclose(validEnergy, testEnergy.detach().numpy(), rtol=1e-4))

    if testGrad:
        testEnergy.backward()
        testForces = -tensorPos.grad
        assert(np.allclose(validForces, testForces, rtol=1e-3))


