# -*- coding: utf-8 -*-
"""
Created on Fri Jun 06 15:18:41 2014

@author: hdemarch
"""

#import os
from importlib import reload

#os.chdir('H:\\dev_python\\hdm\\optimaltrading')
#os.chdir('/home/hdemarch/dev_python/hdm/optimaltrading')
import optimizers_Long_Signal as oopt

#import storage as sto

reload(oopt)
#reload(sto)

print(oopt.NonParametric.needed_conf)


import numpy as np
#DiffPerf = np.zeros((5,5,4))

#for i in range(5): bnv     
#    for j in range(5):

#Parameters
T, Qf, P0, sigma = 100, 10, 100. , 0.05
Gain, RiskAvers = 0.32 , 0.0
Limpact, Lmax = 0.1 , 40
Lsup = 12#int(Lmax*Qf*2./T)
#End of parameters
        
f_final_cost = lambda Q: (1.)*np.sqrt(np.abs(Q))
conf  = {'gain_function_type':'quad',
            'f_Gain': lambda q: 0, 'f_dGain': lambda q: 0,
              'f_gain': lambda q: 0,'f_dgain': lambda q: 0,
                'f_PnL': lambda P,Q,M,Qf,M0 : M+(Q-Qf)*P*max(0,1-np.sign(Q-Qf)*f_final_cost((Q-Qf))),
                 'f_utility': lambda PnL,M0 : 0,'f_final_Impact' : f_final_cost,
                      'f_gain_Impact' : lambda Q: min(1.,(0.0)*np.sqrt(Q)),'f_market_Impact' : lambda Q : (0.0)*1./np.sqrt(np.abs(Q)+1.)}
        
npopt = oopt.NonParametric()
                           
npopt.set_conf( conf)
npopt.set_gain_function(sigma,Gain)
npopt.set_utility_function(RiskAvers)
print(npopt.check_conf())
        
#npopt.optim('explicit_quad' ,  sigma=sigma, T=T, P0= P0, M0=Qf*P0, Qf=Qf, Q0=0, Gain = Gain , RiskAvers = RiskAvers)
#
#
#        
#npopt.phases_filling()
#
#npopt.evolution_Q()
#        
#npopt.plot_evolQ(fignum=1)
#        
#sims = npopt.controlled_trajectories( 4, fignum=2)
#
#npopt.E_Alterna_quantile_t_Q()
#
#npopt.Perf_StandardDev_t_Q()
#
#npopt.Alterna_quantile_t_Q_phase_filling()
#
#npopt.plot_arrow(fignum=5)
#        
#npopt.evolution_EU(500)
#        
#npopt.plot_evolEU(fignum=3)

        
#npopt.optim_Alterna_quantile_t_Q_L(mode_g = 'explicit' ,  sigma=sigma, T=T, P0= P0, M0=Qf*P0, Qf=Qf, Q0=0, Gain = Gain , RiskAvers = RiskAvers,
#                                 Limpact = Limpact, Lmax = Lmax, Lsup = Lsup)
               

#sims = npopt.Alterna_quantile_t_Q_L_controlled_trajectories( 1, fignum=1)
#
#
#npopt.Perf_StandardDev_t_Q_L()
#
#npopt.evolutionQ_t_Q_L()
#npopt.plot_evolQ_t_Q_L(fignum=5,fz = 20)






#npopt.Alterna_quantile_t_Q_phase_filling()
#npopt.plot_arrow(fignum=2)
#npopt.plot_arrow(fignum=6,Color = 'g')
#plot(npopt.quant_t_Q[:,npopt.Qf-2])
#A = npopt.quant_t_Q[:,npopt.Qf-2]
#npopt.evolution_EPerf_t_Q(8)

#npopt.plot_evolEPerf_t_Q(fignum=7)

        
#npopt.optimize_r(mode_r = 'extract',mode_g = 'explicit' ,  sigma=sigma, T=T, P0= P0, M0=Qf*P0, Qf=Qf, Q0=0, Gain = Gain , RiskAvers = RiskAvers)
#npopt.Perf_StandardDev_t_Q()
#
#npopt.Alterna_quantile_t_Q_phase_filling()
#sims = npopt.Alterna_quantile_t_Q_controlled_trajectories( 4, fignum=2)
#npopt.plot_arrow(fignum=3)        
        
#npopt.optimize_r(mode_r = 'basic',mode_g = 'explicit' ,  sigma=sigma, T=T, P0= P0, M0=Qf*P0, Qf=Qf, Q0=0, Gain = Gain , RiskAvers = RiskAvers)
#npopt.Perf_StandardDev_t_Q()
#npopt.Alterna_quantile_t_Q_phase_filling()

#sims = npopt.Alterna_quantile_t_Q_controlled_trajectories( 4, fignum=3)

#npopt.plot_arrow(fignum=2)


#npopt.param_lin_t_Q(2.,mode_g = 'explicit' ,  sigma=sigma, T=T, P0= P0, M0=Qf*P0, Qf=Qf, Q0=0, Gain = Gain , RiskAvers = RiskAvers)
#
#npopt.Perf_StandardDev_t_Q()
#npopt.Alterna_quantile_t_Q_phase_filling()
#sims = npopt.Alterna_quantile_t_Q_controlled_trajectories( 4, fignum=4)
#npopt.plot_arrow(fignum=5)



#npopt.optimize_r(mode_r = 'choice', mode_g = 'explicit' ,  sigma=sigma, T=T, P0= P0, M0=Qf*P0, Qf=Qf, Q0=0,Gain = Gain, RiskAvers = RiskAvers ,
#                                 Limpact = Limpact, Lmax = Lmax ,r_choice = 1.)
#plot(npopt.quant_t_Q[:,npopt.Qf-2])
#plot(npopt.quant_t_Q[:,npopt.Qf-2]/A)
#npopt.Perf_StandardDev_t_Q()
#npopt.Alterna_quantile_t_Q_phase_filling()
#sims = npopt.Alterna_quantile_t_Q_controlled_trajectories( 4, fignum=5)
#npopt.plot_arrow(fignum=4)
#npopt.plot_arrow(fignum=6,Color = 'r')
#
#npopt.affiche()

#print('comp perf', (npopt.dic_output_params['Perf          ']-npopt.dic_output_params['Perf_Optim_old'])/npopt.dic_output_params['Perf          '])
#print('comp Ecartype', (npopt.dic_output_params['EcarType          ']-npopt.dic_output_params['EcarType_Optim_old'])/npopt.dic_output_params['EcarType          '])
#print('comp perf Estim', (npopt.dic_output_params['Perf_Optim_old']-npopt.dic_output_params['Perf_Optim_t_Q'])/npopt.dic_output_params['Perf_Optim_old'])
#print('comp Ecartype Estim', (npopt.dic_output_params['EcarType_Optim_old']-npopt.dic_output_params['EcarType_Optim_t_Q'])/npopt.dic_output_params['EcarType_Optim_old'])


#npopt.param_Merdique(mode_g = 'explicit' ,  sigma=sigma, T=T, P0= P0, M0=Qf*P0, Qf=Qf, Q0=0,Gain = Gain , RiskAvers = RiskAvers,
#                                 Limpact = Limpact, Lmax = Lmax)
#npopt.Perf_StandardDev_t_Q()


#print("\n", npopt.list_storage( experiment_name = '2014_06_13_12_28_05'))


#npopt2 = oopt.NonParametric(filename = 'H:\dev_python\projects\HJB Market orders\data\optimizers_nonparametric_2014_06_13_12_28_05.h5')