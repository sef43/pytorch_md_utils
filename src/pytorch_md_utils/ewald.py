import torch
from typing import Tuple
from .neighborlist import simple_nl



@torch.jit.script
def _EwaldErrorFunction(width: float, alpha: float, target: float, arg: int) -> float:
    """
    from OpenMM NonbondedForceImpl.cpp

    Ewald error function used to set kmax based on chosen tolerance
    see: http://docs.openmm.org/latest/userguide/theory/02_standard_forces.html#coulomb-interaction-with-ewald-summation
    """
    temp = arg*torch.pi/(width*alpha);
    return target-0.05*torch.sqrt(width*alpha)*arg*torch.exp(-temp*temp);

@torch.jit.script
def _findZero(width: float, alpha: float, target: float, initialGuess: int) -> int:
    """
    from OpenMM NonbondedForceImpl.cpp

    finds the smallest value of kmax that minimized the ewald error function
    see: http://docs.openmm.org/latest/userguide/theory/02_standard_forces.html#coulomb-interaction-with-ewald-summation
    """
    arg = initialGuess;
    value = _EwaldErrorFunction(width, alpha, target, arg)
    if (value > 0.0):
        while (value > 0.0 and arg > 0):
            arg=arg-1
            value = _EwaldErrorFunction(width, alpha, target, arg)
        return arg+1
    while (value < 0.0):
        arg=arg+1
        value = _EwaldErrorFunction(width, alpha, target, arg)
    return arg

@torch.jit.script
def _getParams(cell: torch.Tensor, tol: float, cutoff: float) -> Tuple[float, int, int, int]:
    """ get optimal Ewald parameters based on given tolerance
    
    """

    # Set alpha and max number of k  based on tolerance
    # Copied from OpenMM
    alphaEwald = torch.sqrt(-torch.log(torch.tensor(2*tol)))/cutoff
    kmaxx = _findZero(cell[0,0], alphaEwald, tol, 10)
    kmaxy = _findZero(cell[1,1], alphaEwald, tol, 10)
    kmaxz = _findZero(cell[2,2], alphaEwald, tol, 10)
    if (kmaxx%2 == 0):
        kmaxx+=1
    if (kmaxy%2 == 0):
        kmaxy+=1
    if (kmaxz%2 == 0):
        kmaxz+=1

    return alphaEwald, kmaxx, kmaxy, kmaxz



@torch.jit.script
def ewald(r: torch.Tensor, q: torch.Tensor, cell: torch.Tensor,  tol: float=0.0005, cutoff: float=1.0, coulombConst: float=138.935457) -> torch.Tensor:
    """ Torchscript compatible Ewald summation 

    Compute the Coulomb energy for particles in a periodic cell. This function is torchscipt compatible
    The forces due to this energy can calculated with Autograd.

    Limitations:
        - must have PBCs in all x,y,z
        - cutoff must be less than half the smallest box length
        - cell must be rectangular
        - currently exclusions are not implemented

    Parameters
    ----------
    r: torch.Tensor
        Coordinates, shape [N,3]
    q: torch.Tensor
        Charges, shape [N]
    cell: torch.Tensor
        Rectangular unit cell, shape [3,3]
    tol: float=0.0005
        tolerance used to control accuracy and number of K space vectors used.
        see: http://docs.openmm.org/latest/userguide/theory/02_standard_forces.html#coulomb-interaction-with-ewald-summation
    cutoff: float=1.0 nm
        Cutoff distance for real space calculation. 
    coulombConst: float=138.935457 (default units of kj/mol nm = OpenMM default units)
        Value of Coulombs constant (1/(4pi epsilon_0)) to use.
        

    Returns
    -------
    energy: torch.Tensor
        The total Coulomb energy, shape [1]
    """

    device = r.device
    dtype = r.dtype

    alphaEwald, kmaxx, kmaxy, kmaxz = _getParams(cell, tol, cutoff)

    # define constants
    # Copied from OpenMM
    PI = torch.pi
    SQRT_PI = torch.sqrt(torch.tensor(PI, device=device, dtype=dtype))
    TWO_PI = 2*PI
    ONE_4PI_EPS0 = coulombConst
    epsilon     =  1.0
    factorEwald = -1.0 / (4.0*alphaEwald*alphaEwald)
    recipCoeff  = ONE_4PI_EPS0*2*PI/(cell[0, 0] * cell[1,1] * cell[2,2]) /epsilon

    # compute recip space energy method modified from OpenMM reference platform.

    # setup k vectors
    recipBoxSize = TWO_PI/torch.diag(cell)

    rx = torch.arange(-kmaxx+1, kmaxx, device=device, dtype=torch.int32)
    ry = torch.arange(-kmaxy+1, kmaxy, device=device, dtype=torch.int32)
    rz = torch.arange(-kmaxz+1, kmaxz, device=device, dtype=torch.int32)

    kvecs = torch.cartesian_prod(rx, ry, rz)

    temp = torch.sum(kvecs*kvecs, dim=-1)

    # remove (0,0,0)
    mask = temp ==0
    kvecs = kvecs[~mask]
    kvecs=kvecs.to(r.dtype)
    kvecs = kvecs*recipBoxSize[None,:]

    k2 = torch.sum(kvecs**2, dim=-1)
    
    ak = torch.exp(k2*factorEwald) / k2

    kr = torch.mm(kvecs, r.T)

    real_eikr = torch.sum(q*torch.cos(kr), dim=-1)
    imag_eikr = torch.sum(q*torch.sin(kr), dim=-1)

    S2 = real_eikr**2+imag_eikr**2

    recipEnergy = recipCoeff*torch.sum( ak*S2)

    # direct space
    nl, distances, _ = simple_nl(r, cell, True, cutoff, halfList=True)
    qq = q[nl[0]]*q[nl[1]]
    directEnergy = ONE_4PI_EPS0*torch.sum(qq*torch.erfc(alphaEwald*distances)/distances)

    # self 
    selfEnergy = -ONE_4PI_EPS0*alphaEwald*torch.sum(q**2)/SQRT_PI

    return recipEnergy + directEnergy + selfEnergy

    