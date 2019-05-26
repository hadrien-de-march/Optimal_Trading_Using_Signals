# -*- coding: utf-8 -*-
"""
Created on Fri Jun 06 15:18:41 2014

@author: hdemarch
"""

#import os

#os.chdir('H:\\dev_python\\hdm\\optimaltrading')
#os.chdir('/home/hdemarch/dev_python/hdm/optimaltrading')
import optimizers as oopt

import matplotlib.pyplot as plt

#import storage as sto

#reload(oopt)
#reload(sto)

print(oopt.NonParametric.needed_conf)


import numpy as np
#DiffPerf = np.zeros((5,5,4))

#for i in range(5):
#    for j in range(5):

#Parameters
T, Qf, P0, sigma = 20, 7, 100. , 0.25
Gain, RiskAvers = 1. , 0.0
#End of parameters

one_exp = 1
printing = 1


if one_exp:
    Nb_exp = 1
    exp_init = Qf
    exp_fin = Qf
else:
    Nb_exp = T-1
    exp_init = 1
    exp_fin = T-1
    Qfs = np.zeros(Nb_exp)
    eff_ratio_bor_bets = np.zeros(Nb_exp)
    eff_ratio_bors = np.zeros(Nb_exp)
    eff_ratio_bor_apps = np.zeros(Nb_exp)
    eff_ratio_frees = np.zeros(Nb_exp)
    eff_ratio_free_apps = np.zeros(Nb_exp)
    eff_ratio_newtests = np.zeros(Nb_exp)
    eff_ratio_basics = np.zeros(Nb_exp)
 
for n in range(Nb_exp):
    Qf = int(exp_init+n*(exp_fin-exp_init)*1./(Nb_exp-1+1e-10)+1e-10)
       
    f_final_cost = lambda Q: (1000000.)*np.sqrt(np.abs(Q))
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
        
    npopt.optim_Alterna_quantile_t_Q(mode_g = 'explicit' ,  sigma=sigma, T=T, P0= P0, M0=Qf*P0, Qf=Qf, Q0=0, Gain = Gain , RiskAvers = RiskAvers)
               
#npopt.Alterna_quantile_t_Q_phase_filling()
##sims = npopt.Alterna_quantile_t_Q_controlled_trajectories( 5, fignum=1)
    if one_exp and printing:
        npopt.plot_arrow(fignum=2)
        npopt.plot_arrow(fignum=4,Color = 'g')#constrained
        npopt.plot_arrow(fignum=5,Color = 'g')#deteministic
        npopt.plot_arrow(fignum=6,Color = 'g')#con/uncon calibrated mixed effect
        npopt.plot_arrow(fignum=7,Color = 'g')#unconstrained calibrated
        npopt.plot_arrow(fignum=8,Color = 'g')#unconstrained
        npopt.plot_arrow(fignum=9,Color = 'g')#new test
        npopt.plot_arrow(fignum=10,Color = 'g')#basic callib
        
#
    npopt.Perf_StandardDev_t_Q()
#plot(npopt.quant_t_Q[:,npopt.Qf-2])
#A = npopt.quant_t_Q[:,npopt.Qf-2]

##npopt.evolutionQ_t_Q()
##file_name = "C:\Users\User\Dropbox\These\Paper Lehalle\Python\Dataframes\Evol.csv"
##npopt.df_evolQ_t_QDump.to_csv(file_name, sep='\t')
##npopt.plot_evolQ_t_Q(fignum=5,fz = 20)

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



    npopt.optimize_r(mode_r = 'choice', mode_g = 'explicit' ,  sigma=sigma, T=T, P0= P0, M0=Qf*P0, Qf=Qf, Q0=0,Gain = Gain, RiskAvers = RiskAvers ,r_choice = 1.)
##plot(npopt.quant_t_Q[:,npopt.Qf-2])
##plot(npopt.quant_t_Q[:,npopt.Qf-2]/A)
    if one_exp and printing:
        npopt.plot_arrow(fignum=3)
        npopt.plot_arrow(fignum=5,Color = 'r')
    npopt.Perf_StandardDev_t_Q()
##npopt.Alterna_quantile_t_Q_phase_filling()
##sims = npopt.Alterna_quantile_t_Q_controlled_trajectories( 4, fignum=5)
#npopt.plot_arrow(fignum=4)
#npopt.plot_arrow(fignum=6,Color = 'r')
##
##npopt.affiche()
#
##print 'comp perf', (npopt.dic_output_params['Perf          ']-npopt.dic_output_params['Perf_Optim_old'])/npopt.dic_output_params['Perf          ']
##print 'comp Ecartype', (npopt.dic_output_params['EcarType          ']-npopt.dic_output_params['EcarType_Optim_old'])/npopt.dic_output_params['EcarType          ']
##print 'comp perf Estim', (npopt.dic_output_params['Perf_Optim_old']-npopt.dic_output_params['Perf_Optim_t_Q'])/npopt.dic_output_params['Perf_Optim_old']
##print 'comp Ecartype Estim', (npopt.dic_output_params['EcarType_Optim_old']-npopt.dic_output_params['EcarType_Optim_t_Q'])/npopt.dic_output_params['EcarType_Optim_old']


    npopt.optimize_theory(mode_g = 'explicit' ,  sigma=sigma, T=T, P0= P0,
                      M0=Qf*P0, Qf=Qf, Q0=0,Gain = Gain, RiskAvers = RiskAvers,
                      formula = "border_better")
    if one_exp and printing:
        npopt.plot_arrow(fignum=3)
        npopt.plot_arrow(fignum=6,Color = 'r')
    npopt.Perf_StandardDev_t_Q()

    
    npopt.optimize_theory(mode_g = 'explicit' ,  sigma=sigma, T=T, P0= P0,
                      M0=Qf*P0, Qf=Qf, Q0=0,Gain = Gain, RiskAvers = RiskAvers,
                      formula = "new_test")
    npopt.Perf_StandardDev_t_Q()
    if one_exp and printing:
        npopt.plot_arrow(fignum=3)
        npopt.plot_arrow(fignum=9,Color = 'r')
    npopt.Perf_StandardDev_t_Q()
    
    npopt.optimize_theory(mode_g = 'explicit' ,  sigma=sigma, T=T, P0= P0,
                      M0=Qf*P0, Qf=Qf, Q0=0,Gain = Gain, RiskAvers = RiskAvers,
                      formula = "border")
    npopt.Perf_StandardDev_t_Q()
    if one_exp and printing:
        npopt.plot_arrow(fignum=3)
        npopt.plot_arrow(fignum=4,Color = 'r')
    npopt.Perf_StandardDev_t_Q()

    npopt.optimize_theory(mode_g = 'explicit' ,  sigma=sigma, T=T, P0= P0,
                      M0=Qf*P0, Qf=Qf, Q0=0,Gain = Gain, RiskAvers = RiskAvers,
                      formula = "border_approx")
    npopt.Perf_StandardDev_t_Q()

    npopt.optimize_theory(mode_g = 'explicit' ,  sigma=sigma, T=T, P0= P0,
                      M0=Qf*P0, Qf=Qf, Q0=0,Gain = Gain, RiskAvers = RiskAvers,
                      formula = "free")
    npopt.Perf_StandardDev_t_Q()
    if one_exp and printing:
        npopt.plot_arrow(fignum=3)
        npopt.plot_arrow(fignum=8,Color = 'r')
    npopt.Perf_StandardDev_t_Q()

    npopt.optimize_theory(mode_g = 'explicit' ,  sigma=sigma, T=T, P0= P0,
                      M0=Qf*P0, Qf=Qf, Q0=0,Gain = Gain, RiskAvers = RiskAvers,
                      formula = "free_approx")
    if one_exp and printing:
        npopt.plot_arrow(fignum=3)
        npopt.plot_arrow(fignum=7,Color = 'r')
    npopt.Perf_StandardDev_t_Q()
        
    npopt.optimize_theory(mode_g = 'explicit' ,  sigma=sigma, T=T, P0= P0,
                      M0=Qf*P0, Qf=Qf, Q0=0,Gain = Gain, RiskAvers = RiskAvers,
                      formula = "basic_shift")
    if one_exp and printing:
        npopt.plot_arrow(fignum=3)
        npopt.plot_arrow(fignum=10,Color = 'r')
    npopt.Perf_StandardDev_t_Q()

    perf_opt = npopt.dic_output_params['Perf_Optim_t_Q']
    perf_det = npopt.dic_output_params['Perf_r_cho_t_Q']
    perf_th_bor_bet =  npopt.dic_output_params['Perf_Theory_border_better']
    perf_th_bor =  npopt.dic_output_params['Perf_Theory_border']
    perf_th_newtest =  npopt.dic_output_params['Perf_Theory_new_test']
    perf_th_bor_app =  npopt.dic_output_params['Perf_Theory_border_approx']
    perf_th_free = npopt.dic_output_params['Perf_Theory_free']
    perf_th_free_app = npopt.dic_output_params['Perf_Theory_free_approx']
    perf_th_basic = npopt.dic_output_params['Perf_Theory_basic_shift']
    eff_ratio_bor = 1.-(perf_opt-perf_th_bor)/(perf_opt-perf_det)
    eff_ratio_newtest = 1.-(perf_opt-perf_th_newtest)/(perf_opt-perf_det)
    eff_ratio_bor_bet = 1.-(perf_opt-perf_th_bor_bet)/(perf_opt-perf_det)
    eff_ratio_bor_app = 1.-(perf_opt-perf_th_bor_app)/(perf_opt-perf_det)
    eff_ratio_free = 1.-(perf_opt-perf_th_free)/(perf_opt-perf_det)
    eff_ratio_free_app = 1.-(perf_opt-perf_th_free_app)/(perf_opt-perf_det)
    eff_ratio_basic = 1.-(perf_opt-perf_th_basic)/(perf_opt-perf_det)
    print("efficiency ratio border effect", eff_ratio_bor)
    print("efficiency ratio mixed effect", eff_ratio_bor_bet)
    print("efficiency ratio border effect approximate", eff_ratio_bor_app)
    print("efficiency ratio free", eff_ratio_free)
    print("efficiency ratio free approximate", eff_ratio_free_app)
    print("efficiency ratio new test", eff_ratio_newtest)
    print("efficiency ratio basic shift", eff_ratio_basic)
    if not one_exp:
        Qfs[n] = Qf
        eff_ratio_bors[n] = eff_ratio_bor
        eff_ratio_bor_bets[n] = eff_ratio_bor_bet
        eff_ratio_bor_apps[n] = eff_ratio_bor_app
        eff_ratio_frees[n] = eff_ratio_free
        eff_ratio_free_apps[n] = eff_ratio_free_app
        eff_ratio_newtests[n] = eff_ratio_newtest
        eff_ratio_basics[n] = eff_ratio_basic


if not one_exp:
    #plt.plot(Qfs, eff_ratio_bors, label = "constrained")
    #plt.plot(Qfs, eff_ratio_bor_bets, label = "con/uncon calibrated mixed effect")
    plt.plot(Qfs, eff_ratio_bor_apps, label = "constrained calibrated")
    plt.plot(Qfs, eff_ratio_frees, label = "unconstrained")
    plt.plot(Qfs, eff_ratio_free_apps, label = "unconstrained calibrated")
    #plt.plot(Qfs, eff_ratio_newtests, label = "new test")
    #plt.plot(Qfs, 0*Qfs+1, label = "1")
    plt.plot(Qfs, eff_ratio_basics, label = "basic calibrated")
    plt.legend()
    plt.show()
    print("efficiency ratio border effect", eff_ratio_bors)
    print("efficiency ratio mixed effect", eff_ratio_bor_bets)
    print("efficiency ratio border effect approximate", eff_ratio_bor_apps)
    print("efficiency ratio free", eff_ratio_frees)
    print("efficiency ratio free approximate", eff_ratio_free_apps)
    print("efficiency ratio new test", eff_ratio_newtests)
    print("efficiency ratio basic", eff_ratio_basics)


#npopt.param_Merdique(mode_g = 'explicit' ,  sigma=sigma, T=T, P0= P0, M0=Qf*P0, Qf=Qf, Q0=0,Gain = Gain , RiskAvers = RiskAvers)
#npopt.Perf_StandardDev_t_Q()
#
#TableComp = np.zeros((3,5))
#
#for i in range(3):
#    for j in range(5):
#
#        npopt.param_App_Disc(mode_g = 'explicit' ,  sigma=sigma, T=T, P0= P0, M0=Qf*P0, Qf=Qf, Q0=0,Gain = Gain , RiskAvers = RiskAvers,
#                                 prec = 1+i, line = 2+2*j)
#        npopt.Perf_StandardDev_t_Q()
#        TableComp[i,j] = npopt.dic_output_params['Perf_Discrete ']
#
#print "Perf Opt" , npopt.dic_output_params['Perf_Optim_t_Q']
#print "Perf r=1" , npopt.dic_output_params['Perf_r_cho_t_Q']
#print(TableComp)

        


#npopt.store()

#print "\n", npopt.list_storage( experiment_name = '2014_06_13_12_28_05')


#npopt2 = oopt.NonParametric(filename = 'H:\dev_python\projects\HJB Market orders\data\optimizers_nonparametric_2014_06_13_12_28_05.h5')
