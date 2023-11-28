import torch
from torch.autograd import grad
#import numpy as np

from pykeops.torch import Vi, Vj


def GaussKernel(sigma):
    x, y, b = Vi(0, 3), Vj(1, 3), Vj(2, 3)
    gamma = 1 / (sigma * sigma)
    D2 = x.sqdist(y)
    K = (-D2 * gamma).exp()
    return (K * b).sum_reduction(axis=1)


###################################################################
# Define "Gaussian-CauchyBinet" kernel :math:`(K(x,y,u,v)b)_i = \sum_j \exp(-\gamma\|x_i-y_j\|^2) \langle u_i,v_j\rangle^2 b_j`

def GaussLinKernel_current(sigma):
    x, y, u, v, b = Vi(0, 3), Vj(1, 3), Vi(2, 3), Vj(3, 3), Vj(4, 1)
    gamma = 1 / (sigma * sigma)
    D2 = y.sqdist(x)
    # K = (u * v).sum()
    K = (-D2 * gamma).exp() * (u * v).sum()
    return (K * b).sum_reduction(axis=1)


def ConstantLinKernel_current():
    x, y, u, v, b = Vi(0, 3), Vj(1, 3), Vi(2, 3), Vj(3, 3), Vj(4, 1)
    D2 = (y - x) ** 4
    K = (-D2).exp() * (u * v).sum()
    return (K * b).sum_reduction(axis=1)


def LaplaceLinKernel_current(sigma):
    x, y, u, v, b = Vi(0, 3), Vj(1, 3), Vi(2, 3), Vj(3, 3), Vj(4, 1)
    gamma = 1 / sigma
    D2 = (y - x).abs()
    K = (-D2 * gamma).exp() * (u * v).sum()
    return (K * b).sum_reduction(axis=1)


def CauchyLinKernel_current(sigma):
    x, y, u, v, b = Vi(0, 3), Vj(1, 3), Vi(2, 3), Vj(3, 3), Vj(4, 1)
    gamma = 1 / (sigma * sigma)
    D2 = (y - x) ** 2
    K = 1 / (1 + D2 * gamma) * (u * v).sum()
    return (K * b).sum_reduction(axis=1)


def QuadLinKernel_current(sigma):
    x, y, u, v, b = Vi(0, 3), Vj(1, 3), Vi(2, 3), Vj(3, 3), Vj(4, 1)
    gamma = 1 / (2 * sigma)
    D2 = y.sqdist(x)
    K = D2 * (u * v).sum()
    return (K * b).sum_reduction(axis=1)


def LinLinKernel_current(sigma):
    x, y, u, v, b = Vi(0, 3), Vj(1, 3), Vi(2, 3), Vj(3, 3), Vj(4, 1)
    gamma = 1 / (sigma)
    D = (y - x).abs()
    K = D * gamma * (u * v).sum()
    return (K * b).sum_reduction(axis=1)


def GaussSquaredKernel_varifold_unoriented(sigma):
    x, y, u, v, b = Vi(0, 3), Vj(1, 3), Vi(2, 3), Vj(3, 3), Vj(4, 1)
    gamma = 1 / (sigma * sigma)
    D2 = y.sqdist(x)
    K = (-D2 * gamma).exp() * ((u * v) ** 2).sum()
    return (K * b).sum_reduction(axis=1)


def GibbsKernel_varifold_oriented(sigma, sigma_n):
    x, y, u, v, b = Vi(0, 3), Vj(1, 3), Vi(2, 3), Vj(3, 3), Vj(4, 1)
    gamma = 1 / (sigma * sigma)
    gamma2 = 1 / (sigma_n * sigma_n)
    D2 = (y - x) ** 2
    n = (u * v)
    K = (-D2 * gamma).exp() * ((n - 1) * gamma2).exp().sum()
    return (K * b).sum_reduction(axis=1)


def GaussLinKernel_varifold_unoriented(sigma):
    x, y, u, v, b = Vi(0, 3), Vj(1, 3), Vi(2, 3), Vj(3, 3), Vj(4, 1)
    gamma = 1 / (sigma * sigma)
    D2 = (y - x) ** 2
    K = (-D2 * gamma).exp() * ((u * v).abs()).sum()
    return (K * b).sum_reduction(axis=1)


def LaplaceSquaredKernel_varifold_unoriented(sigma):
    x, y, u, v, b = Vi(0, 3), Vj(1, 3), Vi(2, 3), Vj(3, 3), Vj(4, 1)
    gamma = 1 / sigma
    D2 = (y - x).abs()
    K = (-D2 * gamma).exp() * (u * v).sum() ** 2
    return (K * b).sum_reduction(axis=1)


def LaplaceLinKernel_varifold_unoriented(sigma):
    x, y, u, v, b = Vi(0, 3), Vj(1, 3), Vi(2, 3), Vj(3, 3), Vj(4, 1)
    gamma = 1 / sigma
    D2 = (y - x).abs()
    K = (-D2 * gamma).exp() * (u * v).sum().abs()
    return (K * b).sum_reduction(axis=1)


##################################################################
# Custom ODE solver, for ODE systems which are defined on tuples
def RalstonIntegrator():
    def f(ODESystem, x0, nt, deltat=1.0):
        x = tuple(map(lambda x: x.clone(), x0))
        dt = deltat / nt
        l = [x]
        for i in range(nt):
            xdot = ODESystem(*x)
            xi = tuple(map(lambda x, xdot: x + (2 * dt / 3) * xdot, x, xdot))
            xdoti = ODESystem(*xi)
            x = tuple(
                map(
                    lambda x, xdot, xdoti: x + (0.25 * dt) * (xdot + 3 * xdoti),
                    x,
                    xdot,
                    xdoti,
                )
            )
            l.append(x)
        return l

    return f


def Hamiltonian(K):
    def H(p, q):
        # print(K(q, q, p))
        return 0.5 * (p * K(q, q, p)).sum()

    return H


def HamiltonianSystem(K):
    H = Hamiltonian(K)

    def HS(p, q):
        Gp, Gq = grad(H(p, q), (p, q), create_graph=True)
        return -Gq, Gp

    return HS


#####################################################################
# Shooting approach


def Shooting(p0, q0, K, nt=10, Integrator=RalstonIntegrator()):
    return Integrator(HamiltonianSystem(K), (p0, q0), nt)


def Flow(x0, p0, q0, K, deltat=1.0, Integrator=RalstonIntegrator()):
    HS = HamiltonianSystem(K)

    def FlowEq(x, p, q):
        return (K(x, q, p),) + HS(p, q)

    return Integrator(FlowEq, (x0, p0, q0), deltat)[0]


def LDDMMloss(K, dataloss, gamma=1):
    def loss(p0, q0):
        p, q = Shooting(p0, q0, K)[-1]
        # print(Hamiltonian(K)(p0, q0))
        return gamma * Hamiltonian(K)(p0, q0) + dataloss(q)

    return loss


#####################################################################
# Varifold data attachment loss for surfaces

# VT: vertices coordinates of target surface,
# FS,FT : Face connectivity of source and target surfaces
# K kernel
def get_center_length_normal(F, V):
    V0, V1, V2 = (
        V.index_select(0, F[:, 0]),
        V.index_select(0, F[:, 1]),
        V.index_select(0, F[:, 2]),
    )
    centers, normals = (V0 + V1 + V2) / 3, 0.5 * torch.cross(V1 - V0, V2 - V0)
    length = (normals ** 2).sum(dim=1)[:, None].clamp_(min=1e-10).sqrt()
    return centers, length, normals / (length)
def lossVarifoldSurf(FS, VT, FT, K):
    """Compute varifold distance between two meshes
    Input:
        - FS: face connectivity of source mesh
        - VT: vertices of target mesh [nVx3 torch tensor]
        - FT: face connectivity of target mesh [nFx3 torch tensor]
        - K: kernel
    Output:
        - loss: function taking VS (vertices coordinates of source mesh)
    """

    CT, LT, NTn = get_center_length_normal(FT, VT)

    cst = (LT * K(CT, CT, NTn, NTn, LT)).sum()

    def loss(VS):
        CS, LS, NSn = get_center_length_normal(FS, VS)
        return (
                cst
                + (LS * K(CS, CS, NSn, NSn, LS)).sum()
                - 2 * (LS * K(CS, CT, NSn, NTn, LT)).sum()
        )

    return loss


def VarifoldSurfPS(FS, VT, FT, K):
    """Compute varifold distance between two meshes
    Input:
        - FS: face connectivity of source mesh
        - VT: vertices of target mesh [nVx3 torch tensor]
        - FT: face connectivity of target mesh [nFx3 torch tensor]
        - K: kernel
    Output:
        - loss: function taking VS (vertices coordinates of source mesh)
    """

    def get_center_length_normal(F, V):
        V0, V1, V2 = (
            V.index_select(0, F[:, 0]),
            V.index_select(0, F[:, 1]),
            V.index_select(0, F[:, 2]),
        )
        centers, normals = (V0 + V1 + V2) / 3, 0.5 * torch.cross(V1 - V0, V2 - V0)
        length = (normals ** 2).sum(dim=1)[:, None].sqrt()
        return centers, length, normals / (length + 1e-20)

    CT, LT, NTn = get_center_length_normal(FT, VT)

    cst = (LT * K(CT, CT, NTn, NTn, LT)).sum()

    def PS(VS):
        CS, LS, NSn = get_center_length_normal(FS, VS)
        return ((LS * K(CS, CT, NSn, NTn, LT)).sum())

    return PS


def lossVarifoldSurf_cst(FS, VT, FT, K):
    """Compute varifold distance between two meshes
    Input:
        - FS: face connectivity of source mesh
        - VT: vertices of target mesh [nVx3 torch tensor]
        - FT: face connectivity of target mesh [nFx3 torch tensor]
        - K: kernel
    Output:
        - loss: function taking VS (vertices coordinates of source mesh)
    """

    def get_center_length_normal(F, V):
        V0, V1, V2 = (
            V.index_select(0, F[:, 0]),
            V.index_select(0, F[:, 1]),
            V.index_select(0, F[:, 2]),
        )
        normals = 0.5 * torch.cross(V1 - V0, V2 - V0)
        length = (normals ** 2).sum(dim=1)[:, None].sqrt()
        return length, normals / (length + 1e-21)

    LT, NTn = get_center_length_normal(FT, VT)
    cst = (LT * K(NTn, NTn, LT)).sum()

    def loss(VS):
        LS, NSn = get_center_length_normal(FS, VS)
        return (
                cst
                + (LS * K(NSn, NSn, LS)).sum()
                - 2 * (LS * K(NSn, NTn, LT)).sum()
        )

    return loss
