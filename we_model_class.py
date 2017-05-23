# Reference: "How Model Complexity Influences Sea Ice Stability",
# T.J.W. Wagner & I. Eisenman, J Clim 28,10 (2015)
# The addition of a deep ocean coupling is done by Erik B. Myklebust (2017),
# see appendix A in thesis for details.

import numpy as np
import scipy as sp

from tqdm import *  # pip install tqdm

from scipy.sparse.linalg import spsolve
from scipy.sparse import csc_matrix
from scipy.linalg import solve

class model:
    def __init__(self, grid_size=400, forcing=np.zeros(100), load_from_init_file=True):
        self.load_from_file = load_from_init_file

        self.ma_length = ma_length

        self.grid_size = grid_size
        self.forcing = forcing
        self.tmax = len(forcing)

        self.D = 0.66  # diffusivity for heat transport (W m^-2 K^-1)
        self.S1 = 338  # insolation seasonal dependence (W m^-2)

        self.A = 191.5  ## OLR when temp = 0 (W m^-2)

        self.B = 2.5   # OLR temperature dependence (W m^-2 K^-1)
        self.cw = 7.3  # ocean mixed layer heat capacity (W yr m^-2 K^-1)
        self.S0 = 420  # insolation at equator (W m^-2)
        self.S2 = 240  # insolation spatial dependence (W m^-2)
        self.a0 = 0.7  # ice-free co-albedo at equator
        self.a2 = 0.1  # ice=free co-albedo spatial dependence
        self.ai = 0.4  # co-albedo where there is sea ice
        self.k = 2  # sea ice thermal conductivity (W m^-2 K^-1)
        self.Lf = 9.5  # sea ice latent heat of fusion (W yr m^-3)
        self.cg = 0.01 * self.cw  # ghost layer heat capacity(W yr m^-2 K^-1)
        self.tau = 1e-5  # ghost layer coupling timescale (yr)
        self.Fb = 4.0

        self.time_steps = 1000
        self.dt = 1.0 / self.time_steps

        self.dx = 1.0 / self.grid_size
        self.xgrid = np.arange(self.dx / 2, 1 + self.dx / 2, self.dx)

        # Deep ocean
        self.cd = 106
        self.coupling = 5 * 0.73 * (-np.tanh(10 * (self.xgrid - 0.2)) + 1) / 2


    def create_matrix(self):
        xb = np.arange(0, 1 - self.dx, self.dx)

        lam = self.D * (1 - xb**2) / self.dx**2
        L1 = np.append(0, -lam)
        L2 = np.append(-lam, 0)
        L3 = -L1 - L2
        diffop = - \
            np.diag(L3) - np.diag(L2[:self.grid_size - 1],
                                  1) - np.diag(L1[1:self.grid_size], -1)
        return diffop

    def create_seasonal(self):
        ty = np.arange(self.dt / 2, 1 + self.dt / 2, self.dt)
        S = (np.tile(self.S0 - self.S2 * self.xgrid**2, [self.time_steps, 1]) - np.tile(self.S1 * np.cos(
            2 * np.pi * ty), [self.grid_size, 1]).T * np.tile(self.xgrid, [self.time_steps, 1]))
        return S

    def run(self):
        diffop = self.create_matrix()

        cg_tau = self.cg / self.tau
        dt_tau = self.dt / self.tau
        dc = dt_tau * cg_tau
        kappa = (1 + dt_tau) * np.identity(self.grid_size) - \
            self.dt * diffop / self.cg

        S = self.create_seasonal()

        M = self.B + cg_tau

        def ice_volume(mat):
            ice_thickness = (-mat / self.Lf) * (mat < 0)
            vol = []
            for i in range(0, len(ice_thickness[:, 0])):
                tmp = self.xgrid * ice_thickness[i, :]
                vol.append(2 * np.pi * self.dx * np.sum(tmp))
            return vol

        def ocean_heat(deep_temp):
            qw = np.mean(self.coupling * deep_temp)
            return qw

        def albedo(x, xe):
            gamma = 150
            out = self.ai + (self.a0 - self.a2 * x**2 -
                             self.ai) / (np.exp(gamma * (x - xe)) + 1)
            return out

        self.alb = list(map(lambda xe: albedo(self.xgrid, xe), self.xgrid))

        def find_nearest(array, value):
            idx = (np.abs(array - value)).argmin()
            return idx

        def ice_edge(ent):
            idx = find_nearest(ent, 0)
            return idx

        def step_temp_zero(ent, f):
            g = -self.k * self.Lf / ent
            tmp = f / (self.B + cg_tau + g)
            return tmp

        def step_ent(ent, incomming, outgoing):
            tmp = ent + self.dt * (incomming - outgoing)
            return tmp

        def step_temp(ent, temp_zero):
            tmp = ent / self.cw * (ent >= 0) + temp_zero * \
                (temp_zero < 0) * (ent < 0)
            return tmp

        def step_deep_temp(temp, deep_temp):
            tmp = deep_temp + self.dt / self.cd * \
                (self.coupling * (temp - deep_temp))
            return tmp

        def step_ghost_temp(ent, temp_zero, ghost_temp, C):
            g = -self.k * self.Lf / ent
            diag = 1 / (self.B + cg_tau + g) * (temp_zero < 0) * (ent < 0)
            mat = -dc * np.diag(diag)
            matrix = (kappa + mat)
            e = ent / self.cw * (ent >= 0)
            v = C / (self.B + cg_tau + g) * (temp_zero < 0) * (ent < 0)
            vector = ghost_temp + dt_tau * (e + v)
            sol = spsolve(csc_matrix(matrix),vector)
            return sol

        def next_step(i, forc, prev_ent, prev_temp, prev_deep_temp, prev_ghost_temp):
            qw = self.Fb

            alpha = self.alb[int(ice_edge(prev_ent))]
            C = alpha * S[i, :] - self.A + cg_tau * prev_ghost_temp + forc
            temp_zero = step_temp_zero(prev_ent, C)

            temp = step_temp(prev_ent, temp_zero)

            deep_temp = step_deep_temp(temp, prev_deep_temp)

            incomming = alpha * \
                S[i, :] + cg_tau * prev_ghost_temp + \
                self.coupling * deep_temp + forc + qw
            outgoing = self.A + (self.B + cg_tau + self.coupling) * temp
            ent = step_ent(prev_ent, incomming, outgoing)

            C = alpha * S[i, :] - self.A + forc
            ghost_temp = step_ghost_temp(ent, temp_zero, prev_ghost_temp, C)

            return ent, temp, deep_temp, ghost_temp

        def initial(xgrid):
            return 7.5 + 20 * (1 - 2 * xgrid**2)

        outTemp = []
        outEnt = []
        outDeepTemp = []
        iceOut = []

        temp = initial(self.xgrid)
        ghost_temp = temp
        deep_temp = temp
        ent = self.cw * temp

        import os.path
        if self.load_from_file == True and os.path.isfile("initial.csv") == True:
            print("Loading initial from file")
            initial_condition = np.loadtxt("initial.csv", delimiter=',')
            temp = initial_condition[1, :]
            ghost_temp = temp
            deep_temp = initial_condition[2, :]
            ent = initial_condition[0, :]

        else:
            temp = initial(self.xgrid)
            ghost_temp = temp
            deep_temp = temp
            ent = self.cw * temp
            # spin up
            spin_up = 2000
            print("Spin up with initial forcing")
            for j in tqdm(range(0, spin_up)):
                forc = self.forcing[0]
                for i in range(0, self.time_steps):
                    ent, temp, deep_temp, ghost_temp = next_step(
                        i, forc, ent, temp, deep_temp, ghost_temp)
            outInit = [ent, temp, deep_temp]
            np.savetxt("initial.csv", np.asarray(outInit), delimiter=',')

        append_Ent = outEnt.append
        append_Temp = outTemp.append
        append_DeepTemp = outDeepTemp.append
        append_ice = iceOut.append

        print("Running")
        for j in tqdm(range(0, self.tmax)):
            forc = self.forcing[j]
            append_Ent(ent)
            for i in range(0, self.time_steps):
                if i % 10 == 0:
                    append_Temp(temp)
                    append_DeepTemp(deep_temp)
                    append_ice(self.xgrid[ice_edge(ent)])
                ent, temp, deep_temp, ghost_temp = next_step(
                    i, forc, ent, temp, deep_temp, ghost_temp)

        self.ice_edge = np.asarray(iceOut)
        self.temp = np.asarray(outTemp)
        self.ent = np.asarray(outEnt)
        self.deep_temp = np.asarray(outDeepTemp)
        self.volume = np.asarray(ice_volume(self.ent))
