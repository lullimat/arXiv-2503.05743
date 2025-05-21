import numpy as np
from functools import reduce
import sympy as sp

from idpy.IdpyCode import CUDA_T, OCL_T, IDPY_T
from idpy.IdpyCode import GetTenet, GetParamsClean, CheckOCLFP
from idpy.IdpyCode import IdpyMemory

from idpy.IdpyCode.IdpyCode import IdpyKernel, IdpyFunction, IdpyLoop
from idpy.IdpyCode.IdpySims import IdpySims

from idpy.Utils.NpTypes import NpTypes

from idpy.LBM.LBM import RootLB, ShanChenMultiPhase
from idpy.LBM.LBM import InitFStencilWeights, InitDimSizesStridesVolume, InitStencilWeights
from idpy.LBM.LBM import F_IndexFromPos, F_PosFromIndex, F_PointDistanceCenterFirst
from idpy.LBM.LBM import F_PosFromIndexDIM, F_NFlatProfile
from idpy.LBM.LBM import IndexFromPos, PosFromIndex
from idpy.LBM.LBM import K_InitFlatInterface, K_InitRadialInterface
from idpy.LBM.LBM import K_InitPopulations, F_NFlatProfilePeriodicR, F_NFlatProfilePeriodic
from idpy.LBM.LBM import LBMTypes, AllTrue, OneTrue
from idpy.LBM.LBM import K_StreamPeriodic, M_SwapPop, K_ComputeMoments, K_ComputePsi
from idpy.LBM.LBM import ComputeCenterOfMass, K_Collision_ShanChenGuoMultiPhase
import math

import matplotlib.pyplot as plt

NPT = NpTypes()

def CheckCenterOfMassDeltaPConvergence(lbm):
    _first_flag = False
    if 'cm_conv' not in lbm.aux_vars:
        lbm.sims_vars['cm_conv'] = []
        lbm.aux_vars.append('cm_conv')

        lbm.sims_vars['cm_coords'] = []
        lbm.aux_vars.append('cm_coords')        

        lbm.sims_vars['delta_p'] = []
        lbm.aux_vars.append('delta_p')

        lbm.sims_vars['p_in'], lbm.sims_vars['p_out'] = \
                        [], []
        lbm.aux_vars.append('p_out')
        lbm.aux_vars.append('p_in')

        lbm.sims_vars['is_centered_seq'] = []
        lbm.aux_vars.append('is_centered_seq')
        
        _first_flag = True
        
    _p_in, _p_out, _delta_p = lbm.DeltaPLaplace()
    print("p_in: ", _p_in, "p_out: ", _p_out, "delta_p: ", _delta_p)
    print()

    _chk, _break_f = [], False
    if not _first_flag:
        _delta_delta_p = _delta_p - lbm.sims_vars['delta_p'][-1]
        _delta_p_in = _p_in - lbm.sims_vars['p_in'][-1]
        _delta_p_out = _p_out - lbm.sims_vars['p_out'][-1]
        
        ##_chk += [not lbm.sims_vars['is_centered']]
        _chk += [abs(_delta_p) < 1e-9]
        _chk += [abs(_delta_delta_p / _delta_p) < 1e-5]
        _chk += [math.isnan(_delta_p)]

        _break_f = OneTrue(_chk)        

        print("Center of mass: ", lbm.sims_vars['cm_coords'])
        print("delta delta_p: ", _delta_delta_p,
              "delta p_in: ", _delta_p_in,
              "delta p_out: ", _delta_p_out)
        
        print(_chk)
        print()

    lbm.sims_vars['cm_conv'].append(np.copy(lbm.sims_vars['cm_coords']))
    lbm.sims_vars['delta_p'].append(float(_delta_p))
    lbm.sims_vars['p_in'].append(float(_p_in))
    lbm.sims_vars['p_out'].append(float(_p_out))
    lbm.sims_vars['is_centered_seq'].append(lbm.sims_vars['is_centered'])
    
    return _break_f


def MPFlatPT_E4(lbm):
    if 'psi_f' not in lbm.sims_vars:
        lbm.sims_vars['psi_f'] = \
            sp.lambdify(lbm.sims_vars['n_sym'], lbm.sims_vars['psi_sym'])

    '''
    need to write it with kernels?
    After all it only applies to flat interfaces
    that are usually simulated with small systems
    '''
    _p_xx, _p_yy = 0, 0
    _direction = lbm.sims_vars['direction']
    _dim_sizes = lbm.sims_vars['dim_sizes']
    _dim_strides = lbm.sims_vars['dim_strides']
    _G = lbm.sims_vars['SC_G']
    _c2 = lbm.sims_vars['c2']
    _dim = lbm.sims_vars['DIM']
    
    _pos = np.copy(lbm.sims_vars['dim_center'])
    _pos[_direction] = 0
    
    _n_swap = lbm.sims_idpy_memory['n'].D2H()
    
    _n = np.zeros(_dim_sizes[_direction])
    for i in range(_dim_sizes[_direction]):
        _index = IndexFromPos(_pos, _dim_strides)
        _n[i] = _n_swap[_index]
        _pos[_direction] += 1
    
    _psi = lbm.sims_vars['psi_f'](_n)
    _psi_m1 = lbm.sims_vars['psi_f'](np.append(_n[1:], _n[0]))
    _psi_p1 = lbm.sims_vars['psi_f'](np.append(_n[-1], _n[:-1]))
        
    _p_xx += _c2 * _n
    _p_xx += _G * _psi * (_psi_p1 + _psi_m1) / 4

    _p_yy += _c2 * _n + _G * (_psi ** 2) / 3
    _p_yy += _G * _psi * (_psi_p1 + _psi_m1) / 12

    del _n

    return _p_xx, _p_yy

MultiPhaseFlatPT = {'D2E4': MPFlatPT_E4}

def MPFlatPTMax(lbm, max_l2 = 8):
    '''
    This implementation only works for weights
    that are correctly indexed by the squared index
    Need to compute the weights for three dimensions
    starting from two
    '''
    if 'psi_f' not in lbm.sims_vars:
        lbm.sims_vars['psi_f'] = \
            sp.lambdify(lbm.sims_vars['n_sym'], lbm.sims_vars['psi_sym'])

    '''
    getting weights as a function of the squared length
    '''
    _le = len(lbm.sims_vars['E_list'])
    _dim = lbm.sims_vars['DIM']
    _l2_list = np.zeros(_le//_dim, dtype = np.int32)
    _w_l2_list = np.zeros(max_l2 + 1)

    _l2_swap = 0
    for c_i in range(_le):
        _l2_swap += int(lbm.sims_vars['E_list'][c_i] ** 2)

        if (c_i + 1) % _dim == 0:
            if (c_i + 1) == _dim or _l2_list[(c_i - _dim) // _dim] != _l2_swap:
                print(_l2_swap, lbm.sims_vars['EW_list'][c_i // _dim])

                _w_l2_list[_l2_swap] = \
                    lbm.sims_vars['EW_list'][c_i // _dim]

            _l2_list[c_i // _dim] = _l2_swap
            _l2_swap = 0
            
    _l2_list = np.unique(_l2_list)
    
    if _l2_list[-1] > max_l2:
        print("max_l2: ", max_l2)
        raise Exception("Number of weights beyond maximum")

    '''
    defining the constants
    '''
    _c_xx_plus1 = _w_l2_list[1]/2 + _w_l2_list[2] + _w_l2_list[5]
    _c_xx_plus2 = _w_l2_list[4] + 2 * _w_l2_list[5] + 2 * _w_l2_list[8]
    _c_xx_cross1 = 2 * _w_l2_list[4] + 4 * _w_l2_list[5] + 4 * _w_l2_list[8]
    print("_c_xx_plus1: ", _c_xx_plus1, "_c_xx_plus2: ", _c_xx_plus2, "_c_xx_cross1: ", _c_xx_cross1)
    print()
    _c_yy_local = _w_l2_list[1] + 4 * _w_l2_list[4]
    _c_yy_plus1 = _w_l2_list[2] + 4 * _w_l2_list[5]
    _c_yy_plus2 = _w_l2_list[5] / 2 + 2 * _w_l2_list[8]
    _c_yy_cross1 = _w_l2_list[5] + 4 * _w_l2_list[8]
    print("_c_yy_local: ", _c_yy_local)
    print("_c_yy_plus1: ", _c_yy_plus1, "_c_yy_plus2: ", _c_yy_plus2, "_c_yy_cross1: ", _c_xx_cross1)
    print()
    
    '''
    computing the components
    '''
    
    _p_xx, _p_yy = 0, 0
    _direction = lbm.sims_vars['direction']
    _dim_sizes = lbm.sims_vars['dim_sizes']
    _dim_strides = lbm.sims_vars['dim_strides']
    _G = lbm.sims_vars['SC_G']
    _c2 = lbm.sims_vars['c2']
    _dim = lbm.sims_vars['DIM']

    _pos = np.copy(lbm.sims_vars['dim_center'])
    _pos[_direction] = 0

    _n_swap = lbm.sims_idpy_memory['n'].D2H()

    _n = np.zeros(_dim_sizes[_direction])
    for i in range(_dim_sizes[_direction]):
        _index = IndexFromPos(_pos, _dim_strides)
        _n[i] = _n_swap[_index]
        _pos[_direction] += 1

    _psi = lbm.sims_vars['psi_f'](_n)
    _psi_m1 = np.append(_psi[1:], _psi[0])
    _psi_p1 = np.append(_psi[-1], _psi[:-1])
    _psi_m2 = np.append(_psi_m1[1:], _psi_m1[0])
    _psi_p2 = np.append(_psi_p1[-1], _psi_p1[:-1])

    _p_xx += _c2 * _n + _G * _c_xx_plus1 * _psi * (_psi_p1 + _psi_m1)
    _p_xx += _G * _c_xx_plus2 * _psi * (_psi_p2 + _psi_m2)
    _p_xx += _G * _c_xx_cross1 * _psi_p1 * _psi_m1

    _p_yy += _c2 * _n + _G * _c_yy_local * (_psi ** 2)
    _p_yy += _G * _c_yy_plus1 * _psi * (_psi_p1 + _psi_m1)
    _p_yy += _G * _c_yy_plus2 * _psi * (_psi_p2 + _psi_m2)
    _p_yy += _G * _c_yy_cross1 * _psi_p1 * _psi_m1

    del _n

    return _p_xx, _p_yy

def GetSnapshotN(lbm):
    first_flag = False

    if 'snapshots_n_k' not in lbm.sims_idpy_memory:
        lbm.sims_idpy_memory['snapshots_n_k'] = 0
        first_flag = True

    _n_swap = np.copy(lbm.sims_idpy_memory['n'].D2H())
    _n_swap = _n_swap.reshape(np.flip(lbm.sims_vars['dim_sizes']))
    _dim = len(lbm.sims_vars['dim_sizes'])

    if _dim == 3:
        _n_swap = _n_swap[lbm.sims_vars['dim_sizes'][2]//2,:,:] 

    _k_fig = lbm.sims_idpy_memory['snapshots_n_k']
    _fig = plt.figure()
    plt.imshow(_n_swap, origin = 'lower')
    plt.savefig('./snapshot_' + ('%010d' % (_k_fig)) + '.png', dpi = 150)
    plt.close()

    lbm.sims_idpy_memory['snapshots_n_k'] += 1
    
    return False

class PShanChenMultiPhase(ShanChenMultiPhase):
    def __init__(self, *args, **kwargs):
        if not hasattr(self, 'params_dict'):
            self.params_dict = {}

        self.kwargs = GetParamsClean(kwargs, [self.params_dict],
                                     needed_params = ['psi_sym'])
        ShanChenMultiPhase.__init__(self, *args, **kwargs)

        if 'psi_sym' not in self.params_dict:
            raise Exception("Missing sympy expression for the pseudo-potential, parameter 'psi_sym'")
        else:
            self.sims_vars['psi_sym'] = self.params_dict['psi_sym']
            self.sims_vars['n_sym'] = sp.symbols('n')

    def CapillaryWaveFileName(self):
        return 'lbm_mp_' + str(self.sims_vars['dim_sizes']) + '_G_' + str(self.sims_vars['SC_G'])
            
    def PopulationsDump(self):
        _file_name = self.CapillaryWaveFileName()
        self.sims_dump_idpy_memory += ['n', 'u']
        self.DumpPopSnapshot(file_name = _file_name + '.hdf5')
            
    def MainLoopSCF(self, time_steps, convergence_functions = []):
        _all_init = []
        for key in self.init_status:
            _all_init.append(self.init_status[key])

        if not AllTrue(_all_init):
            print(self.init_status)
            raise Exception("Hydrodynamic variables/populations not initialized")
        
        _K_ComputeMoments = K_ComputeMoments(custom_types = self.custom_types.Push(),
                                             constants = self.constants,
                                             optimizer_flag = self.optimizer_flag)

        _K_ComputePsi = K_ComputePsi(custom_types = self.custom_types.Push(),
                                     constants = self.constants,
                                     psi_code = self.params_dict['psi_code'],
                                     optimizer_flag = self.optimizer_flag)

        _K_Collision_ShanChenMultiPhase = \
            K_Collision_ShanChenMultiPhase(custom_types = self.custom_types.Push(),
                                           constants = self.constants,
                                           f_classes = [F_PosFromIndex,
                                                        F_IndexFromPos],
                                           optimizer_flag = self.optimizer_flag)
        
        _K_StreamPeriodic = K_StreamPeriodic(custom_types = self.custom_types.Push(),
                                             constants = self.constants,
                                             f_classes = [F_PosFromIndex,
                                                          F_IndexFromPos],
                                             optimizer_flag = self.optimizer_flag)
        
        self._MainLoop = \
            IdpyLoop(
                [self.sims_idpy_memory],
                [
                    [
                        (_K_ComputeMoments(tenet = self.tenet,
                                           grid = self.sims_vars['grid'],
                                           block = self.sims_vars['block']), ['n', 'u', 'pop',
                                                                              'XI_list', 'W_list']),
                        (_K_ComputePsi(tenet = self.tenet,
                                       grid = self.sims_vars['grid'],
                                       block = self.sims_vars['block']), ['psi', 'n']),

                        (_K_Collision_ShanChenMultiPhase(tenet = self.tenet,
                                                         grid = self.sims_vars['grid'],
                                                         block = self.sims_vars['block']),
                         ['pop', 'u', 'n', 'psi',
                          'XI_list', 'W_list',
                          'E_list', 'EW_list',
                          'dim_sizes', 'dim_strides']),
                        
                        (_K_StreamPeriodic(tenet = self.tenet,
                                           grid = self.sims_vars['grid'],
                                           block = self.sims_vars['block']), ['pop_swap', 'pop',
                                                                              'XI_list', 'dim_sizes',
                                                                              'dim_strides']),
                        (M_SwapPop(tenet = self.tenet), ['pop_swap', 'pop'])
                    ]
                ]
            )

        '''
        now the loop: need to implement the exit condition
        '''
        old_step = 0
        for step in time_steps[1:]:
            print(step, step - old_step)
            '''
            Very simple timing, reasonable for long executions
            '''
            self._MainLoop.Run(range(step - old_step))
            
            old_step = step
            if len(convergence_functions):
                checks = []
                for c_f in convergence_functions:
                    checks.append(c_f(self))

                if OneTrue(checks):
                    break
              
    def MainLoopKSF(self, time_steps, convergence_functions = []):
        _all_init = []
        for key in self.init_status:
            _all_init.append(self.init_status[key])
            
        if not AllTrue(_all_init):
            print(self.init_status)
            raise Exception("Hydrodynamic variables/populations not initialized")
        
        _K_ComputeMoments = K_ComputeMoments(custom_types = self.custom_types.Push(),
                                             constants = self.constants,
                                             optimizer_flag = self.optimizer_flag)

        _K_ComputePsi = K_ComputePsi(custom_types = self.custom_types.Push(),
                                     constants = self.constants,
                                     psi_code = self.params_dict['psi_code'],
                                     optimizer_flag = self.optimizer_flag)

        _K_Collision_Kupershtokh = \
            K_Collision_Kupershtokh(custom_types = self.custom_types.Push(),
                                    constants = self.constants,
                                    f_classes = [F_PosFromIndex,
                                                 F_IndexFromPos],
                                    optimizer_flag = self.optimizer_flag)
        
        _K_StreamPeriodic = K_StreamPeriodic(custom_types = self.custom_types.Push(),
                                             constants = self.constants,
                                             f_classes = [F_PosFromIndex,
                                                          F_IndexFromPos],
                                             optimizer_flag = self.optimizer_flag)
        
        self._MainLoop = \
            IdpyLoop(
                [self.sims_idpy_memory],
                [
                    [
                        (_K_ComputeMoments(tenet = self.tenet,
                                           grid = self.sims_vars['grid'],
                                           block = self.sims_vars['block']), ['n', 'u', 'pop',
                                                                              'XI_list', 'W_list']),
                        (_K_ComputePsi(tenet = self.tenet,
                                       grid = self.sims_vars['grid'],
                                       block = self.sims_vars['block']), ['psi', 'n']),

                        (_K_Collision_Kupershtokh(tenet = self.tenet,
                                                  grid = self.sims_vars['grid'],
                                                  block = self.sims_vars['block']),
                         ['pop', 'u', 'n', 'psi',
                          'XI_list', 'W_list',
                          'E_list', 'EW_list',
                          'dim_sizes', 'dim_strides']),
                        
                        (_K_StreamPeriodic(tenet = self.tenet,
                                           grid = self.sims_vars['grid'],
                                           block = self.sims_vars['block']), ['pop_swap', 'pop',
                                                                              'XI_list', 'dim_sizes',
                                                                              'dim_strides']),
                        (M_SwapPop(tenet = self.tenet), ['pop_swap', 'pop'])
                    ]
                ]
            )

        '''
        now the loop: need to implement the exit condition
        '''
        old_step = 0
        for step in time_steps[1:]:
            print(step, step - old_step)
            '''
            Very simple timing, reasonable for long executions
            '''
            self._MainLoop.Run(range(step - old_step))
            
            old_step = step
            if len(convergence_functions):
                checks = []
                for c_f in convergence_functions:
                    checks.append(c_f(self))

                if OneTrue(checks):
                    break

    def MainLoopDebug(self, time_steps, convergence_functions = []):
        _all_init = []
        for key in self.init_status:
            _all_init.append(self.init_status[key])

        if not AllTrue(_all_init):
            print(self.init_status)
            raise Exception("Hydrodynamic variables/populations not initialized")
        
        _K_ComputeMoments = K_ComputeMoments(custom_types = self.custom_types.Push(),
                                             constants = self.constants,
                                             optimizer_flag = self.optimizer_flag)

        _K_ComputePsi = K_ComputePsi(custom_types = self.custom_types.Push(),
                                     constants = self.constants,
                                     psi_code = self.params_dict['psi_code'],
                                     optimizer_flag = self.optimizer_flag)

        _K_Collision_ShanChenGuoMultiPhase = \
            K_Collision_ShanChenGuoMultiPhase(custom_types = self.custom_types.Push(),
                                              constants = self.constants,
                                              f_classes = [F_PosFromIndex,
                                                           F_IndexFromPos],
                                              optimizer_flag = self.optimizer_flag)
        
        _K_StreamPeriodic = K_StreamPeriodic(custom_types = self.custom_types.Push(),
                                             constants = self.constants,
                                             f_classes = [F_PosFromIndex,
                                                          F_IndexFromPos],
                                             optimizer_flag = self.optimizer_flag)
        
        self._MainLoop = \
            IdpyLoop(
                [self.sims_idpy_memory],
                [
                    [
                        (_K_ComputeMoments(tenet = self.tenet,
                                           grid = self.sims_vars['grid'],
                                           block = self.sims_vars['block']), ['n', 'u', 'pop',
                                                                              'XI_list', 'W_list']),
                        (_K_ComputePsi(tenet = self.tenet,
                                       grid = self.sims_vars['grid'],
                                       block = self.sims_vars['block']), ['psi', 'n']),
                        
                        (_K_StreamPeriodic(tenet = self.tenet,
                                           grid = self.sims_vars['grid'],
                                           block = self.sims_vars['block']), ['pop_swap', 'pop',
                                                                              'XI_list', 'dim_sizes',
                                                                              'dim_strides']),
                        (M_SwapPop(tenet = self.tenet), ['pop_swap', 'pop'])
                    ]
                ]
            )
        '''
        now the loop: need to implement the exit condition
        '''
        old_step = time_steps[0]
        for step in time_steps[1:]:
            print(step, step - old_step)
            '''
            Very simple timing, reasonable for long executions
            '''
            self._MainLoop.Run(range(step - old_step))
            
            old_step = step
            if len(convergence_functions):
                checks = []
                for c_f in convergence_functions:
                    checks.append(c_f(self))

                if OneTrue(checks):
                    break
    
    

'''
Device Functions
'''
class F_NFlatProfileR(IdpyFunction):
    def __init__(self, custom_types = None, f_type = 'NType'):
        IdpyFunction.__init__(self, custom_types = custom_types, f_type = f_type)
        self.params = {'SType x': ['const'],
                       'LengthType x0': ['const'],
                       'LengthType w0': ['const'],
                       'LengthType w1': ['const']}

        self.functions[IDPY_T] = """
        return tanh((LengthType)(x - (x0 - 0.5 * w0))) - tanh((LengthType)(x - (x0 + 0.5 * w1)));
        """


'''
Kernels
'''

class K_Collision_ShanChenMultiPhase(IdpyKernel):
    def __init__(self, custom_types = {}, constants = {}, f_classes = [],
                 optimizer_flag = None):
        IdpyKernel.__init__(self, custom_types = custom_types,
                            constants = constants, f_classes = f_classes,
                            optimizer_flag = optimizer_flag)
        self.SetCodeFlags('g_tid')
        self.params = {'PopType * pop': ['global', 'restrict'],
                       'UType * u': ['global', 'restrict'],
                       'NType * n': ['global', 'restrict', 'const'],
                       'PsiType * psi': ['global', 'restrict', 'const'],
                       'SType * XI_list': ['global', 'restrict', 'const'],
                       'WType * W_list': ['global', 'restrict', 'const'],
                       'SType * E_list': ['global', 'restrict', 'const'],
                       'WType * EW_list': ['global', 'restrict', 'const'],
                       'SType * dim_sizes': ['global', 'restrict', 'const'],
                       'SType * dim_strides': ['global', 'restrict', 'const']}
        
        self.kernels[IDPY_T] = """
        if(g_tid < V){
            // Getting thread position
            SType g_tid_pos[DIM];
            F_PosFromIndex(g_tid_pos, dim_sizes, dim_strides, g_tid);

            // Computing Shan-Chen Force
            SCFType F[DIM]; SType neigh_pos[DIM]; UType lu_post[DIM];
            for(int d=0; d<DIM; d++){F[d] = lu_post[d] = 0.;}

            PsiType lpsi = psi[g_tid];

            for(int qe=0; qe<QE; qe++){
                // Compute neighbor position
                for(int d=0; d<DIM; d++){
                    neigh_pos[d] = ((g_tid_pos[d] + E_list[d + qe*DIM] + dim_sizes[d]) % dim_sizes[d]);
                }
                // Compute neighbor index
                SType neigh_index = F_IndexFromPos(neigh_pos, dim_strides);
                // Get the pseudopotential value
                PsiType npsi = psi[neigh_index];
                // Add partial contribution
                for(int d=0; d<DIM; d++){F[d] += E_list[d + qe*DIM] * EW_list[qe] * npsi;}
            }
            for(int d=0; d<DIM; d++){F[d] *= -SC_G * lpsi;}

            // Local density and velocity for shift and equilibrium
            NType ln = n[g_tid]; UType lu[DIM];

            // Shan-Chen velocity shift & Copy to global memory
            for(int d=0; d<DIM; d++){ 
                lu[d] = u[g_tid + V*d] + F[d] / ln / OMEGA;
            }

            // Compute square norm of Guo shifted velocity
            UType u_dot_u = 0.;
            for(int d=0; d<DIM; d++){u_dot_u += lu[d]*lu[d];}

            // Cycle over the populations: equilibrium + Shan Chen
            for(int q=0; q<Q; q++){
                UType u_dot_xi = 0.; 
                for(int d=0; d<DIM; d++){
                    u_dot_xi += lu[d] * XI_list[d + q*DIM];
                }

                PopType leq_pop = 1.;

                // Equilibrium population
                leq_pop += + u_dot_xi*CM2 + 0.5*u_dot_xi*u_dot_xi*CM4;
                leq_pop += - 0.5*u_dot_u*CM2;
                leq_pop = leq_pop * ln * W_list[q];

                pop[g_tid + q*V] = \
                    pop[g_tid + q*V]*(1. - OMEGA) + leq_pop*OMEGA;

                for(int d=0; d<DIM; d++){
                   lu_post[d] += pop[g_tid + q*V] * XI_list[d + q*DIM];
                }

             }

            for(int d=0; d<DIM; d++){ 
                u[g_tid + V*d] = 0.5 * (u[g_tid + V*d] + lu_post[d] / ln);
            }

        }
        """

class K_Collision_Kupershtokh(IdpyKernel):
    def __init__(self, custom_types = {}, constants = {}, f_classes = [],
                 optimizer_flag = None):
        IdpyKernel.__init__(self, custom_types = custom_types,
                            constants = constants, f_classes = f_classes,
                            optimizer_flag = optimizer_flag)
        self.SetCodeFlags('g_tid')
        self.params = {'PopType * pop': ['global', 'restrict'],
                       'UType * u': ['global', 'restrict'],
                       'NType * n': ['global', 'restrict', 'const'],
                       'PsiType * psi': ['global', 'restrict', 'const'],
                       'SType * XI_list': ['global', 'restrict', 'const'],
                       'WType * W_list': ['global', 'restrict', 'const'],
                       'SType * E_list': ['global', 'restrict', 'const'],
                       'WType * EW_list': ['global', 'restrict', 'const'],
                       'SType * dim_sizes': ['global', 'restrict', 'const'],
                       'SType * dim_strides': ['global', 'restrict', 'const']}
        
        self.kernels[IDPY_T] = """
        if(g_tid < V){
            // Getting thread position
            SType g_tid_pos[DIM];
            F_PosFromIndex(g_tid_pos, dim_sizes, dim_strides, g_tid);

            // Computing Kupershtokh Force
            SCFType F[DIM]; SType neigh_pos[DIM]; UType lu_post[DIM];
            for(int d=0; d<DIM; d++){F[d] = lu_post[d] = 0.;}

            PsiType lpsi = psi[g_tid];

            for(int qe=0; qe<QE; qe++){
                // Compute neighbor position
                for(int d=0; d<DIM; d++){
                    neigh_pos[d] = ((g_tid_pos[d] + E_list[d + qe*DIM] + dim_sizes[d]) % dim_sizes[d]);
                }
                // Compute neighbor index
                SType neigh_index = F_IndexFromPos(neigh_pos, dim_strides);
                // Get the pseudopotential value
                PsiType npsi = psi[neigh_index];
                // Add partial contribution
                for(int d=0; d<DIM; d++){F[d] += E_list[d + qe*DIM] * EW_list[qe] * npsi;}
            }
            for(int d=0; d<DIM; d++){F[d] *= -SC_G * lpsi;}

            // Local density and velocity for shift and equilibrium
            NType ln = n[g_tid]; UType lu[DIM]; UType lu_bare[DIM];

            // Kupershtokh shifted and non-shifted velocities & Copy to global memory
            for(int d=0; d<DIM; d++){ 
                lu[d] = u[g_tid + V*d] + F[d] / ln;
                lu_bare[d] = u[g_tid + V*d];
            }

            // Compute square norm of shifted and unshifted velocities
            UType u_dot_u = 0.; UType u_dot_u_bare = 0.;
            for(int d=0; d<DIM; d++){
                u_dot_u += lu[d]*lu[d];
                u_dot_u_bare += lu_bare[d]*lu_bare[d];
            }

            // Cycle over the populations: equilibrium + Kupershtokh
            for(int q=0; q<Q; q++){
                UType u_dot_xi = 0.; UType u_dot_xi_bare = 0.; 
                for(int d=0; d<DIM; d++){
                    u_dot_xi += lu[d] * XI_list[d + q*DIM];
                    u_dot_xi_bare += lu_bare[d] * XI_list[d + q*DIM];
                }

                PopType leq_pop = 1.; PopType leq_pop_bare = 1.;

                // Equilibrium population with shifted velocity
                leq_pop += + u_dot_xi*CM2 + 0.5*u_dot_xi*u_dot_xi*CM4;
                leq_pop += - 0.5*u_dot_u*CM2;
                leq_pop = leq_pop * ln * W_list[q];

                // Equilibrium population with no-shifted velocity
                leq_pop_bare += + u_dot_xi_bare*CM2 + 0.5*u_dot_xi_bare*u_dot_xi_bare*CM4;
                leq_pop_bare += - 0.5*u_dot_u_bare*CM2;
                leq_pop_bare = leq_pop_bare * ln * W_list[q];

                pop[g_tid + q*V] = \
                    pop[g_tid + q*V]*(1. - OMEGA) - leq_pop_bare*(1. - OMEGA) + leq_pop;

                for(int d=0; d<DIM; d++){
                   lu_post[d] += pop[g_tid + q*V] * XI_list[d + q*DIM];
                }

             }

            for(int d=0; d<DIM; d++){ 
                u[g_tid + V*d] = 0.5 * (u[g_tid + V*d] + lu_post[d] / ln);
            }

        }
        """