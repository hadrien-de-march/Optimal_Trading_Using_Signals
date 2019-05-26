"""
Optimizers
============
A library of optimizers for optimal trading with signals

typical import

>>> import hdm.optimaltrading.optimizers as oopt

"""

# helpers

    

# the main stochastic control optimizer
class NonParametric(object):
    """
    The main class to implement backward stochastic optimization

    At creation it just needs parameters linked to the optimizer itself: Nmax, Nmax3 and epsilon.
    Then you need to set its configuration. Needed parameters are available in the needed_conf variable.

    .. code-block:: python
    
       reload(oopt)
       print oopt.NonParametric.needed_conf
       npopt = oopt.NonParametric()
       sigma = 2.
       import numpy as np
       f_final_cost = lambda a: 0.31*np.sqrt(np.abs(a))
       conf  = {'f_gain': lambda q: 0.46*sigma*(1.-q), 'f_dgain': lambda q: -0.46*sigma,
                    'f_utility': lambda P,Q,M,Qf,M0 : M+(Q-Qf)*P*max(0,(1-np.sign(Q-Qf)*f_final_cost(Q-Qf))) }
       npopt.set_conf( conf)
       print npopt.check_conf()
       Qf, P0 = 6, 100.
       npopt.optim('explicit', sigma=sigma, T=10, P0= P0, M0=Qf*P0, Qf=Qf, Q0=0 )

       sims = npopt.controlled_trajectories( 10)
    
    """
    needed_conf = ['gain_function_type','f_Gain', 'f_utility', 'f_dGain','f_gain','f_dgain','f_PnL' ,'f_final_Impact','f_gain_Impact','f_market_Impact']

    def __init__(self, Nmax=100,epsilon=0.00001,Nmax3 = 100000, filename=None):
        self.Nmax=Nmax
        self.delta = 1./(Nmax)
        self.Nmax3 = Nmax3
        self.epsilon = epsilon
        self.conf = {}
        self.last_sim = None
        self.EUgrid   = None
        self.virgin = True

        if filename is not None:      
            print("recovering internal state from file <%s>..." % filename)
            import pandas as pd
            storage = pd.HDFStore( filename) 
            df_output_params = storage.get( 'dic_output_params')
            self.dic_output_params = self.df2dic( df_output_params)
            df_input_params = storage.get( 'dic_input_params')
            self.dic_input_params  = self.df2dic( df_input_params)
            import pdb
            pdb.set_trace()
            p4_qg = storage.get('quantgrid')
            
            self.quantgrid = p4_qg.values
            self.EUgrid    = storage.get('EUgrid').values
            storage.close()
            

    def set_conf(self, k, v=None):
        """
        set a configuration parameter (or several if a dictionary is given)
        """
        if isinstance(k, dict):
            for c,v in k.items():
                self.set_conf(c,v)
            return
        if k in self.needed_conf:
            self.conf[k] = v
        else:
            print("unknown configuration parameter <%s>..." % k)

    def get_conf(self, k, v_def=None):
        """
        get the value of a configuration parameter (or several if a list is given); default value is None
        """
        if isinstance( k, list):
            return [self.get_conf(c,v_def) for c in k]
        
        return self.conf.get(k, v_def)

        
    def set_gain_function(self,sigma,Gain):
        """
        Give gain function types
        3 defined : the quadratic, the square root and the logarithm
        """
        self.gain_type = self.conf['gain_function_type']
        import numpy as np
        G = sigma*Gain
        if self.gain_type=='quad':
            self.conf['f_Gain'] = lambda q: G*(1.-q)
            self.conf['f_dGain'] = lambda q: -G
            self.conf['f_gain'] = lambda q: G*(1.-q)*q
            self.conf['f_dgain'] = lambda q: G*(1.-2.*q)

        elif self.gain_type=='sqrt':
            self.conf['f_Gain'] = lambda q: G*(1.-np.sqrt(q))/np.sqrt(q)
            self.conf['f_dGain'] = lambda q: -G/(2.*(q)*np.sqrt(q))
            self.conf['f_gain'] = lambda q: G*(1.-np.sqrt(q))*np.sqrt(q)
            self.conf['f_dgain'] = lambda q: G*(1./(2*np.sqrt(q))-1.)
            
        if self.gain_type=='log':
            self.conf['f_Gain'] = lambda q: -G*np.e/4.*np.log(q)
            self.conf['f_dGain'] = lambda q: -G*np.e/(q*4)
            self.conf['f_gain'] = lambda q: -G*np.e/4.*np.log(q**q)
            self.conf['f_dgain'] = lambda q: -G*np.e/4.*(np.log(q)+1.)

            
            
    def set_utility_function(self,RiskAvers):
        """
        Sets utility function
        2 types : zero risk aversion or exponential risk aversion
        """
        import numpy as np
        if RiskAvers == 0.:
            self.conf['f_utility'] = lambda PnL,M0 : PnL/M0
        elif RiskAvers > 0. :
            self.conf['f_utility'] = lambda PnL,M0 : (1.-np.exp(-RiskAvers*PnL/M0))/RiskAvers
        else:
            print('Risk Aversion', RiskAvers)
            raise RuntimeError("negative risk aversion !")

            


    def check_conf(self):
        """
        check if configuration parameters are missing and returns their list
        """
        missing = []
        for k in self.needed_conf:
            if k not in self.conf.keys():
                missing.append( k)

        if len( missing)==0:
            if self.dg(3./4)>=self.dg(1./4):
                raise RuntimeError("The gain function is not concave")

        return missing
        
            
    def G(self, q):
        """
        helper to return the gain
        """
        return self.conf['f_Gain'](q)
        
    def dG(self, q):
        """
        helper to return the derivative of the gain
        """
        return self.conf['f_dGain'](q)
        
    def g(self, q):
        """
        helper to return the gain * q
        """
        return self.conf['f_gain'](q)
        
    def dg(self, q):
        """
        helper to return the derivative of the gain * q
        """
        return self.conf['f_dgain'](q)
        
    def gImpact(self,Q):
        return self.conf['f_gain_Impact'](Q)
        
    def mImpact(self,Q):
        return self.conf['f_market_Impact'](Q)
        
        
    def PnL(self,Qf,M0):
        return lambda P,Q,M: self.conf['f_PnL'](P,Q,M,Qf,M0)
        
    def Utility(self , M0):
        return lambda PnL : self.conf['f_utility'](PnL,M0)


    def U(self, Qf, M0):
        """
        helper for the utility function
        """
        return lambda P,Q,M: self.conf['f_utility'](self.conf['f_PnL'](P,Q,M,Qf,M0),M0)
        
    
    
    def f_dgInv(self, mode):
       """
       a metha function with different gain inverting methods
       """
       import numpy as np
       G = self.Gain*self.sigma      
       
       # explicit mode (2)
       if mode=='explicit':
           # explicit quadratic mode (2.0)
           if self.gain_type=='quad':
              def dgInv(x):
                 if x<=-G:
                     return 1.
                 elif x>=G:
                     return 0.
                 else:
                     return 0.5-x/(2.*G)
              return dgInv
            
         # explicit root mode (2bis)
           if self.gain_type=='sqrt':
              def dgInv(x):
                 if x<=-G:
                     return 1.
                 else:
                     return 1./(4.*(x/G+1.)**2)
              return dgInv    


         # explicit exp mode (2tierce)
           if self.gain_type=='log':
              def dgInv(x):
                 if x<=-G*np.e/4.:
                     return 1.
                 else:
                     return np.exp(-(x*4./(G*np.e)+1))
              return dgInv
        
           else:
               print('gain_type',self.gain_type)
               raise RuntimeError("This type of gain does not have an explicit resolution !!")
        
        
       # tabulated mode (3)
       if mode=='tabulated':
            
            print("Pre-processing for optimization running...")
            delta3 = (1.-2*self.epsilon)*1./self.Nmax3

            dgmin = self.dg(1./2-(self.epsilon-1./2))
            print("dgmin=", dgmin)

            dgmax = self.dg(1./2+(self.epsilon-1./2))
            print("dgmax=", dgmax)

            Delta = (dgmax-dgmin)/self.Nmax3
            print("Delta=",Delta)
            self.dgInvData = np.zeros(self.Nmax3+1)
            level = self.epsilon

            for k in range(self.Nmax3+1):
               while (self.dg(level)>-dgmin+(self.Nmax3-k)*Delta):
                   level = level + delta3
               self.dgInvData[self.Nmax3-k]=np.min(level,1-self.epsilon)

            def dgInv(x):
               if x<=dgmin:
                   return 1.
               elif x>=dgmax:
                   return 0.
               else:
                   return self.dgInvData[np.floor((x-dgmin)/Delta)]
            return dgInv

       # HADRIEN: 
       print("no specific pre processing for dgInv?!?")
       return None
        
    def quant(self, t,P,Q,M):
        """
        Returns the optimal quantile
        """
        import numpy as np
        return self.quantgrid[t,np.around((P-self.P0)/self.sigma+t),Q,np.around((self.M0-M)/self.sigma)]
    
    def EU(self, t,P,Q,M):
        """
        Returns the optimal quantile
        """
        import numpy as np
        return self.EUgrid[t,np.around((P-self.P0)/self.sigma+t),Q,np.around((self.M0-M)/self.sigma)]
        
    def access(self, t,P,Q,M):
        """
        Returns an intuitive phase
        """
        import numpy as np
        return self.accessgrid[t,np.around((P-self.P0)/self.sigma+t),Q,np.around((self.M0-M)/self.sigma)]
        
    def index_u(self,t,P):
        import numpy as np
        return np.around((P-self.P0)/self.sigma+t)
            
    def index_v(self,t,M):
        import numpy as np
        return np.around((self.M0-M)/self.sigma)
        
        
    def pPhase(self,t,P,Q,M):
        import numpy as np
        return self.pPhasegrid[t,np.around((P-self.P0)/self.sigma+t),Q,np.around((self.M0-M)/self.sigma)]


    def pPhase_Alterna_q_t_Q(self,t,P,Q,M):
        import numpy as np
        return self.pPhasegrid_Alterna_q_t_Q[t,np.around((P-self.P0)/self.sigma+t),Q,np.around((self.M0-M)/self.sigma)]
        
        
        
        
    def explicit_horror(self,K):
        """
        Horror to optimize r
        """
        return 0.5*(1.0*K*(0.0833333333333333*K**2*(-0.125*K**3 + 5.0625*K**2 + 5.90625*K + 0.5*(-6.75*K**5 + 81.0*K**4 + 185.625*K**3 + 121.5*K**2 + 11.390625*K)**0.5 + 0.421875)**(-0.333333333333333) + 0.75*K*(-0.125*K**3 + 5.0625*K**2 + 5.90625*K + 0.5*(-6.75*K**5 + 81.0*K**4 + 185.625*K**3 + 121.5*K**2 + 11.390625*K)**0.5 + 0.421875)**(-0.333333333333333) + 0.333333333333333*K + 0.1875*(-0.125*K**3 + 5.0625*K**2 + 5.90625*K + 0.5*(-6.75*K**5 + 81.0*K**4 + 185.625*K**3 + 121.5*K**2 + 11.390625*K)**0.5 + 0.421875)**(-0.333333333333333) + 0.333333333333333*(-0.125*K**3 + 5.0625*K**2 + 5.90625*K + 0.5*(-6.75*K**5 + 81.0*K**4 + 185.625*K**3 + 121.5*K**2 + 11.390625*K)**0.5 + 0.421875)**0.333333333333333 + 0.5)**(-0.5) + 1.0*K + 0.5*(0.0833333333333333*K**2*(-0.125*K**3 + 5.0625*K**2 + 5.90625*K + 0.5*(-6.75*K**5 + 81.0*K**4 + 185.625*K**3 + 121.5*K**2 + 11.390625*K)**0.5 + 0.421875)**(-0.333333333333333) + 0.75*K*(-0.125*K**3 + 5.0625*K**2 + 5.90625*K + 0.5*(-6.75*K**5 + 81.0*K**4 + 185.625*K**3 + 121.5*K**2 + 11.390625*K)**0.5 + 0.421875)**(-0.333333333333333) + 0.333333333333333*K + 0.1875*(-0.125*K**3 + 5.0625*K**2 + 5.90625*K + 0.5*(-6.75*K**5 + 81.0*K**4 + 185.625*K**3 + 121.5*K**2 + 11.390625*K)**0.5 + 0.421875)**(-0.333333333333333) + 0.333333333333333*(-0.125*K**3 + 5.0625*K**2 + 5.90625*K + 0.5*(-6.75*K**5 + 81.0*K**4 + 185.625*K**3 + 121.5*K**2 + 11.390625*K)**0.5 + 0.421875)**0.333333333333333 + 0.5)**(-0.5) - 1.0*(0.0833333333333333*K**2*(-0.125*K**3 + 5.0625*K**2 + 5.90625*K + 0.5*(-6.75*K**5 + 81.0*K**4 + 185.625*K**3 + 121.5*K**2 + 11.390625*K)**0.5 + 0.421875)**(-0.333333333333333) + 0.75*K*(-0.125*K**3 + 5.0625*K**2 + 5.90625*K + 0.5*(-6.75*K**5 + 81.0*K**4 + 185.625*K**3 + 121.5*K**2 + 11.390625*K)**0.5 + 0.421875)**(-0.333333333333333) + 0.333333333333333*K + 0.1875*(-0.125*K**3 + 5.0625*K**2 + 5.90625*K + 0.5*(-6.75*K**5 + 81.0*K**4 + 185.625*K**3 + 121.5*K**2 + 11.390625*K)**0.5 + 0.421875)**(-0.333333333333333) + 0.333333333333333*(-0.125*K**3 + 5.0625*K**2 + 5.90625*K + 0.5*(-6.75*K**5 + 81.0*K**4 + 185.625*K**3 + 121.5*K**2 + 11.390625*K)**0.5 + 0.421875)**0.333333333333333 + 0.5)**1.0 + 1.5)**0.5 + 0.5*(0.0833333333333333*K**2*(-0.125*K**3 + 5.0625*K**2 + 5.90625*K + 0.5*(-6.75*K**5 + 81.0*K**4 + 185.625*K**3 + 121.5*K**2 + 11.390625*K)**0.5 + 0.421875)**(-0.333333333333333) + 0.75*K*(-0.125*K**3 + 5.0625*K**2 + 5.90625*K + 0.5*(-6.75*K**5 + 81.0*K**4 + 185.625*K**3 + 121.5*K**2 + 11.390625*K)**0.5 + 0.421875)**(-0.333333333333333) + 0.333333333333333*K + 0.1875*(-0.125*K**3 + 5.0625*K**2 + 5.90625*K + 0.5*(-6.75*K**5 + 81.0*K**4 + 185.625*K**3 + 121.5*K**2 + 11.390625*K)**0.5 + 0.421875)**(-0.333333333333333) + 0.333333333333333*(-0.125*K**3 + 5.0625*K**2 + 5.90625*K + 0.5*(-6.75*K**5 + 81.0*K**4 + 185.625*K**3 + 121.5*K**2 + 11.390625*K)**0.5 + 0.421875)**0.333333333333333 + 0.5)**0.5










    def Init_dics(self , mode, sigma , T , P0, M0 , Q0 , Qf , Gain , RiskAvers , experiment_name=None):
        """
        To initiate the informations of the non_parametric
        Be careful : Problem with the Gain definition
        """
        import numpy as np
        self.T = T
        self.P0 = P0
        self.M0 = M0
        self.Qf = Qf
        self.Q0 = Q0
        self.sigma = sigma
        self.Gain = Gain
        self.RiskAvers = RiskAvers
        
        if experiment_name is None:
            import datetime
            experiment_name = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
        
    
        self.ExperimentNumber = experiment_name
    
        if P0*Qf != M0:
            print("Warning : Initial money is different from naive price")
    
        if P0-T*sigma<0:
#           raise RuntimeError("Potentially negative price")
            print("Warning : Potentially negative price")
    
        self.dgInv = self.f_dgInv( mode)
        self.Pbar = int(np.around(self.P0/self.sigma))
        
        # initializations
        self.Impactfinal = self.conf['f_final_Impact'](1.)
    
        self.Impactmarket = max(self.mImpact(0.),self.mImpact(1.))
    
        self.Impactgain = self.gImpact(1.)
    
        self.dic_input_params = {'T': T, 'P0': P0, 'Qf': Qf, 'M0' : M0, 'sigma': sigma, 'GainCoeff': Gain,
                                 'ImpactCoeff Final': self.Impactfinal,
                                 'ImpactCoeff Market':self.Impactmarket,'ImpactCoeff Gain':self.Impactgain, 
                                 'ExperimentNumber': experiment_name,'Aversion au risque' : self.RiskAvers}
                  
        self.dic_output_params =  {'Prix Naif' : M0}
        
        self.massData = (T+1)*(T+1)*(2*T+1)*(self.Pbar*T+((T-1)*T)/2+1)
        
        self.virgin = False
        
        return
        











        
        
    def access_filling(self):
        """
        fills the access matrix
        """
        import numpy as np
        T=self.T
        print("filling the access")
        #Stockage des etats accessibles
        self.accessgrid = np.zeros((T+1,2*T+1,T+1,self.Pbar*T+((T-1)*T)/2+1))
        self.accessgrid[0,0,0,0] = 1
        
        #Remplissage de la accessgrid
        for t in range(T):
            print("time is going up")
            print(t)
            print("u now")
            for u in range(2*t+1):
                print(u)
                P=self.P0+(u-t)*self.sigma
                for Q in range(t+1):
                  for v in range(min(self.Pbar*t+((t-1)*t)/2+1,Q*(int(2*np.ceil((self.P0+P)/self.sigma))+2*t+Q)/4+1)):
                   M=self.M0-v*self.sigma
        
                   if self.access(t,P,Q,M)!=0:
                       
                      CurrentProb = self.access(t,P,Q,M)
                      
              
                      self.accessgrid[t+1,self.index_u(t+1,P+self.sigma),Q+1,self.index_v(t+1,M-P)] += CurrentProb
                      self.accessgrid[t+1,self.index_u(t+1,P-self.sigma),Q+1,self.index_v(t+1,M-P)] += CurrentProb  
                      self.accessgrid[t+1,self.index_u(t+1,P+self.sigma),Q,self.index_v(t+1,M)] += CurrentProb
                      self.accessgrid[t+1,self.index_u(t+1,P-self.sigma),Q,self.index_v(t+1,M)] += CurrentProb
                      
        return










    
   
    def optim(self, mode_g, sigma, T, P0, M0, Q0, Qf, Gain , RiskAvers , experiment_name=None ):
        """
        optimization

        :param mode: the optimization mode in 'basic', 'explicit', 'tabulated'
        """
        import numpy as np
        import pprint
        
        if self.virgin:
            self.Init_dics(mode_g , sigma , T , P0, M0 , Q0 , Qf , Gain , RiskAvers , experiment_name=None)
        
        # initializations
        U = self.U(self.Qf,self.M0)
        
        def nonZero(x):
            return (x!=0)*1
        
        #La matrice des esperances de U avec les termes a partir de t
        self.EUgrid    = np.zeros((T+1,2*T+1,T+1,self.Pbar*T+((T-1)*T)/2+1))

        #La matrice des Quantiles optimaux
        self.quantgrid = np.zeros((T+1,2*T+1,T+1,self.Pbar*T+((T-1)*T)/2+1))
        self.quantgrid = self.quantgrid-1
        
        #Remplissage de la matrice d'accessibilite
        
        self.access_filling()
        
        # initialisation de l'algo backwards
        print("Initial Utility computing")
        for u in range(2*T+1):
            print(u)
            P = self.P0+(u-T)*self.sigma
            for Q in range(T+1):
                for v in range(min(self.Pbar*T+((T-1)*T)/2+1,Q*(int(2*np.ceil((self.P0+P)/self.sigma))+2*T+Q)/4+1)):
                       M = self.M0 - v*self.sigma
                       if self.access(T,P,Q,M)!=0:
                           self.EUgrid[T,u,Q,v]=U(P,Q,M)
                           self.quantgrid[T,u,Q,v] = 0

        #Propagation de l'algo backwards. pour ne pas se perdre, les t,q et x correspondants sont rappeles
        for j in range(T):
            t=T-1-j
            print("time is going back")
            print(t)
            print("u now")
            for u in range(2*t+1):
                print(u)
                P=self.P0+(u-t)*self.sigma
                for Q in range(t+1):
                  for v in range(min(self.Pbar*t+((t-1)*t)/2+1,Q*(int(2*np.ceil((self.P0+P)/self.sigma))+2*t+Q)/4+1)):
                   M=self.M0-v*self.sigma
                   
                   if self.access(t,P,Q,M)!=0:

                       lambdaSupPlus  = self.EU(t+1,P+self.sigma,Q+1,M-P)
                       lambdaSupMoins = self.EU(t+1,P-self.sigma,Q+1,M-P)
                       lambdaInfPlus  = self.EU(t+1,P+self.sigma,Q,M)
                       lambdaInfMoins = self.EU(t+1,P-self.sigma,Q,M)
                       

                       if mode_g == 'basic': 
                            q = 0.
                            qOpt = 1.
                            EvolOpt = (lambdaSupPlus+lambdaSupMoins)/2.
            #L'algorithme "naif" d'optimisation
                            for i in range(self.Nmax):
                               PSupPlus = (q+(self.mImpact(Q)*q+(1.-self.gImpact(Q))*self.g(q)/self.sigma))/2.
                               PSupMoins = (q-(self.mImpact(Q)*q+(1.-self.gImpact(Q))*self.g(q)/self.sigma))/2.
                               PInfPlus = (1.-q+(-(1.-self.gImpact(Q))*self.g(q)/self.sigma))/2.
                               PInfMoins = (1.-q-(-(1.-self.gImpact(Q))*self.g(q)/self.sigma))/2.

            #On ajoute les morceaux du candidat au fur et a mesure
                               Candidat = PSupPlus*lambdaSupPlus
                               Candidat += PSupMoins*lambdaSupMoins
                               Candidat += PInfPlus*lambdaInfPlus
                               Candidat += PInfMoins*lambdaInfMoins

#                           if PSupPlus<0:
#                               raise RuntimeError("Probleme sur les probabilites 1")
#                           if PSupMoins<0:
#                               raise RuntimeError("Probleme sur les probabilites 2")
#                           if PInfPlus<0:
#                               raise RuntimeError("Probleme sur les probabilites 3")
#                           if PInfPlus<0:
#                               raise RuntimeError("Probleme sur les probabilites 4")
#                           if np.around(PSupPlus+PSupMoins,6) != np.around(q,6):
#                               raise RuntimeError("Probleme sur les probabilites 5")
#                           if np.around(PInfPlus+PInfMoins,6) != np.around(1-q,6):
#                               raise RuntimeError("Probleme sur les probabilites 6") 
#                           if np.around(PSupPlus+PInfPlus,6) != 1./2:
#                               raise RuntimeError("Probleme sur les probabilites 7")
#                           if np.around(PSupMoins+PInfMoins,6) != 1./2:
#                               raise RuntimeError("Probleme sur les probabilites 8")                   
                               if Candidat>EvolOpt:
                                   qOpt = q
                                   EvolOpt = Candidat
                               q=q+self.delta


                       else:

                         a=lambdaSupPlus*(1.+self.mImpact(Q))+lambdaSupMoins*(1.-self.mImpact(Q))-lambdaInfPlus-lambdaInfMoins
                         b=(1.-self.gImpact(Q))*(lambdaSupPlus-lambdaSupMoins-lambdaInfPlus+lambdaInfMoins)/self.sigma
                         if b==0:
                             if a>0:
                                 qOpt = 1.
                             else:
                                 qOpt = 0.
#                             print("b egal a zerooooooooooooooooo")
                         elif (a<=0)&(b<0):             
                             qOpt = 0.
#                             print("a<=0")
                         elif (a>0)&(b<0) :
                             qOpt = 1.
#                         print("a>0")
                         else:
                             qOpt = self.dgInv(-a/b)

                         PSupPlus = (qOpt+(self.mImpact(Q)*qOpt+(1.-self.gImpact(Q))*self.g(qOpt)/self.sigma))/2.
                         PSupMoins = (qOpt-(self.mImpact(Q)*qOpt+(1.-self.gImpact(Q))*self.g(qOpt)/self.sigma))/2.
                         PInfPlus = (1.-qOpt+(-(1.-self.gImpact(Q))*self.g(qOpt)/self.sigma))/2.
                         PInfMoins = (1.-qOpt-(-(1.-self.gImpact(Q))*self.g(qOpt)/self.sigma))/2.

#                     if PSupPlus<0:
#                         raise RuntimeError("Probleme sur les probabilites 1")
#                     if PSupMoins<0:
#                         raise RuntimeError("Probleme sur les probabilites 2")
#                     if PInfPlus<0:
#                         raise RuntimeError("Probleme sur les probabilites 3")
#                     if PInfPlus<0:
#                         raise RuntimeError("Probleme sur les probabilites 4")
#                     if np.around(PSupPlus+PSupMoins,6) != np.around(qOpt,6):
#                         raise RuntimeError("Probleme sur les probabilites 5")
#                     if np.around(PInfPlus+PInfMoins,6) != np.around(1-qOpt,6):
#                         raise RuntimeError("Probleme sur les probabilites 6") 
#                     if np.around(PSupPlus+PInfPlus,6) != 1./2:
#                         raise RuntimeError("Probleme sur les probabilites 7")
#                     if np.around(PSupMoins+PInfMoins,6) != 1./2:
#                         raise RuntimeError("Probleme sur les probabilites 8")   

                         EvolOpt = PSupPlus*lambdaSupPlus
                         EvolOpt += PSupMoins*lambdaSupMoins
                         EvolOpt += PInfPlus*lambdaInfPlus
                         EvolOpt += PInfMoins*lambdaInfMoins

                       self.EUgrid[t,u,Q,v]=EvolOpt
                       self.quantgrid[t,u,Q,v]=qOpt


        self.Opti = self.EU(0,P0,Q0,M0)

#        qnonext=nonZero((self.quantgrid+1)*(1-self.quantgrid)*self.quantgrid)


        self.qfill = nonZero(self.quantgrid+1)

        Ratiofill = np.sum(self.qfill)/self.massData
        print(Ratiofill)


        self.dic_output_params['EU'] = self.Opti
#        self.dic_output_params['directionnalite'] =  np.sum(qnonext)          
#        self.dic_output_params['Remplissage Phases'] =  Ratiofill          
#        self.dic_output_params['inutilite'] =1/Ratiofill

            
        pprint.pprint( self.dic_input_params)

        pprint.pprint( self.dic_output_params)
        
        self.T  = T
       
        return





    def controlled_trajectories(self, NbSimul, plot=True, fignum=1):
        """
        Generate random trajectories and controls them
        """
        import numpy as np
        Simul = np.zeros((NbSimul,self.T+1,5))

        for i in range(NbSimul):

            Simul[i,0,1]=self.P0
            Simul[i,0,2]=self.Q0
            Simul[i,0,3]=self.M0

            for j in range(self.T):
                t=j+1
                P=Simul[i,t-1,1]
                Q=Simul[i,t-1,2]
                M=Simul[i,t-1,3]
                Simul[i,t-1,4] = self.EU(t-1,P,Q,M)
                Simul[i,t-1,0]=self.quant(t-1,P,Q,M)
                q=Simul[i,t-1,0]

                if q==-1:
                 raise RuntimeError("Exit of the known domain...")

                PSupPlus = (1.+self.mImpact(Q)+(1.-self.gImpact(Q))*self.G(q)/self.sigma)/2
                PInfPlus = (1.-(1.-self.gImpact(Q))*self.G(q)*q/self.sigma/(1-q))/2

                buy = np.random.binomial(1,q)

                if buy == 1:
                    d=np.random.binomial(1,PSupPlus)
                else:
                    d=np.random.binomial(1,PInfPlus)

                Simul[i,t,1]=P+2*(d-1./2)*self.sigma
                Simul[i,t,2]=Q+buy
                Simul[i,t,3]=M-buy*P



            P=Simul[i,self.T,1]
            Q=Simul[i,self.T,2]
            M=Simul[i,self.T,3]        
            Simul[i,self.T,4] = self.EU(self.T,P,Q,M)

        if plot:
            self.plot_sim( Simul, Qf=self.Qf, T=self.T, fignum=fignum)

        return Simul





    @staticmethod
    def plot_sim(sim, k=None, Qf=None, T=None, fignum=1):
        """

        """
        import matplotlib.pyplot as plt
        import numpy as np
        fig = plt.figure(fignum)
        plt.clf()
        ax1 = plt.subplot(3,1,1)
        if k is None:
            for i in range(sim.shape[0]):
                plt.plot(sim[i,:,1], lw=3, alpha=.3)
        else:
            plt.plot(sim[k,:,1])
            
        ax2 = plt.subplot(3,1,2, sharex=ax1)
        if k is None:
            for i in range(sim.shape[0]):
                plt.plot(sim[i,:,2], lw=3, alpha=.3)
        else:
            plt.plot(sim[k,:,2])
        plt.plot((Qf)+np.zeros(T+1))
        
        ax3 = plt.subplot(3,1,3, sharex=ax1)
        if k is None:
            for i in range(sim.shape[0]):
                plt.plot(sim[i,:,4], lw=3, alpha=.3)
        else:
            plt.plot(sim[k,:,4])
        plt.plot(np.zeros(T+1))


        return fig, ax1, ax2, ax3






   
    def phases_filling(self):
        
        import numpy as np        
        import pprint
        
        PnL=self.PnL(self.Qf,self.M0)
            
            
        T  = self.T
        def nonZero(x):
            return (x!=0)*1

                  
        self.pPhasegrid=np.zeros((T+1,2*T+1,T+1,self.Pbar*T+((T-1)*T)/2+1))

        
        #initialisation
        
        self.pPhasegrid[0,0,0,0]=1
        
        #Remplissage forward de la matrice
        
        
        for t in range(T):
            print("time is going up")
            print(t)
            print("u now")
            for u in range(2*t+1):
                print(u)
                P=self.P0+(u-t)*self.sigma
                for Q in range(t+1):
                  for v in range(min(self.Pbar*t+((t-1)*t)/2+1,Q*(int(2*np.ceil((self.P0+P)/self.sigma))+2*t+Q)/4+1)):
                   M=self.M0-v*self.sigma
        
                   if self.pPhase(t,P,Q,M)!=0:
                       
                      CurrentProb = self.pPhase(t,P,Q,M)
                      
                      if (CurrentProb>1) or (CurrentProb<0):
                           print(CurrentProb)
                           raise RuntimeError("echec de probabilite")
                      
                         
                      q=self.quant(t,P,Q,M)
                      
                      if q ==-1:
                          raise RuntimeError("etat inaccessible")
              
                      PSupPlus = (q+(self.mImpact(Q)*q+(1.-self.gImpact(Q))*self.g(q)/self.sigma))/2
                      PSupMoins = (q-(self.mImpact(Q)*q+(1.-self.gImpact(Q))*self.g(q)/self.sigma))/2
                      PInfPlus = (1-q+(-(1.-self.gImpact(Q))*self.g(q)/self.sigma))/2
                      PInfMoins = (1-q-(-(1.-self.gImpact(Q))*self.g(q)/self.sigma))/2
              
                      if PSupPlus>0:
                         self.pPhasegrid[t+1,self.index_u(t+1,P+self.sigma),Q+1,self.index_v(t+1,M-P)] += PSupPlus*CurrentProb
                      if PSupMoins>0:             
                         self.pPhasegrid[t+1,self.index_u(t+1,P-self.sigma),Q+1,self.index_v(t+1,M-P)] += PSupMoins*CurrentProb  
                      if PInfPlus>0:
                         self.pPhasegrid[t+1,self.index_u(t+1,P+self.sigma),Q,self.index_v(t+1,M)] += PInfPlus*CurrentProb
                      if PInfMoins>0:             
                         self.pPhasegrid[t+1,self.index_u(t+1,P-self.sigma),Q,self.index_v(t+1,M)] += PInfMoins*CurrentProb           



#        self.PhasefillOpt = nonZero(self.pPhasegrid)

        self.FinalPnL = np.zeros((2*T+1,T+1,self.Pbar*T+((T-1)*T)/2+1))

        print("Final PnL computation")

        for u in range(2*T+1):
                print(u)
                P=self.P0+(u-T)*self.sigma
                for Q in range(T+1):
                  for v in range(min(self.Pbar*T+((T-1)*T)/2+1,Q*(int(2*np.ceil((self.P0+P)/self.sigma))+2*T+Q)/4+1)):
                   M=self.M0-v*self.sigma

                   if self.pPhase(T,P,Q,M)!=0:
                       self.FinalPnL[u,Q,v] = PnL(P,Q,M)

        MoyCarr = np.sum(self.pPhasegrid[T,:,:,:]*self.FinalPnL**2)

        Moy = np.sum(self.pPhasegrid[T,:,:,:]*self.FinalPnL)

        EcarType = np.sqrt(MoyCarr - Moy**2)

#        massDataOpt = np.sum(self.PhasefillOpt)
#
#        RatioPhasefillOpt = massDataOpt/self.massData
#
#
#
#        self.dic_output_params['RemplissagePostOpt']=RatioPhasefillOpt
#        self.dic_output_params['inutilitePostOpt']=1/RatioPhasefillOpt
        self.dic_output_params['EcarType          ']=EcarType/self.M0
        self.dic_output_params['Perf          ']=Moy/self.M0
        print("done")


        pprint.pprint(self.dic_input_params)
        print(" ")
        pprint.pprint(self.dic_output_params)
        print(" ")        
        
    




    
        
        
    def evolution_Q(self):  
        
        print("evolQ is computing")    

        import numpy as np
        import pandas as pd
        T = self.T        
        
        self.evolQ = np.zeros((T+1,T+1))

        for t in range(T+1):
#            print("time is going up")
#            print(t)
            for Q in range(t+1):
                self.evolQ[t,Q]=np.sum(self.pPhasegrid[t,:,Q,:])
        
        
        self.df_evolQDump = pd.DataFrame(self.evolQ,index=range(T+1),columns=range(T+1) ) 


    def plot_evolQ(self, k=0, Qf=None, T=None, fignum=1, ylim=None, fz=20):
        """

        """
        
        # import cal.plot as cplot
        import numpy as np
        import matplotlib.pyplot as plt
        from matplotlib import pylab
        
        import matplotlib
        matplotlib.rc('xtick', labelsize=fz) 
        matplotlib.rc('ytick', labelsize=fz)

        fig = plt.figure( fignum)
        plt.figure(1)
        plt.clf()
        plt.contourf(self.df_evolQDump.columns,self.df_evolQDump.index,self.df_evolQDump.values.T,levels = np.linspace(0,1,15),
                     cmap=pylab.get_cmap('YlOrRd')) # 
        plt.xlabel('Time',fontsize=fz)
        plt.ylabel('Qt',fontsize=fz)
        plt.title('Density of Bought Quantity (Qt)',fontsize=1.5*fz)

        plt.tight_layout()
        if ylim is not None:
            plt.ylim(ylim)

        return fig
      
        
    def evolution_M(self):  

        import numpy as np
        import pandas as pd
        T = self.T
        
        

        self.evolM = np.zeros((T+1,self.Pbar*T+((T-1)*T)/2+1))
        
        for t in range(T+1):
            print("time is going up")
            print(t)
            for v in range(self.Pbar*t+((t-1)*t)/2+1):
               M=self.M0-v*self.sigma
               self.evolM[t,v]=np.sum(self.pPhasegrid[t,:,:,self.index_v(t,M)])
        

        self.df_evolMDump = pd.DataFrame(self.evolM,index=range(T+1),columns=range(self.Pbar*T+((T-1)*T)/2+1) ) 







        
        
    def evolution_EU(self, NpasEU):
        
        print("evolEU is computing")
        
        import numpy as np
        import pandas as pd
        T = self.T


        FindMax = self.PhasefillOpt*self.EUgrid

        self.EUmax = self.Opti

        self.EUmin = self.Opti

#        print("u now")
        for u in range(2*T+1):
#            print(u)
            P=self.P0+(u-T)*self.sigma
            for Q in range(T+1):
              for v in range(min(self.Pbar*T+((T-1)*T)/2+1,Q*(int(2*np.ceil((self.P0+P)/self.sigma))+2*T+Q)/4+1)):
                M=self.M0-v*self.sigma
                
                Comp = FindMax[T,u,Q,v]
                if Comp > self.EUmax:
                    self.EUmax = Comp
                elif Comp < self.EUmin:
                    self.EUmin = Comp
            
        deltaEU = (self.EUmax-self.EUmin)/NpasEU


        self.evolEU = np.zeros((T+1,NpasEU+1))

        def index_w(EspU):
            return np.around((EspU-self.EUmin)/deltaEU)
    
        for t in range(T+1):
#            print("time is going up")
#            print(t)
#            print("u now")
            for u in range(2*t+1):
#                print(u)
                P=self.P0+(u-t)*self.sigma
                for Q in range(t+1):
                  for v in range(min(self.Pbar*t+((t-1)*t)/2+1,Q*(int(2*np.ceil((self.P0+P)/self.sigma))+2*t+Q)/4+1)):
                   M=self.M0-v*self.sigma
                   
                   if self.pPhasegrid[t,self.index_u(t,P),Q,self.index_v(t,M)]>0:
                       self.evolEU[t,index_w(self.EU(t,P,Q,M))] += self.pPhasegrid[t,self.index_u(t,P),Q,self.index_v(t,M)]


        self.df_evolEUDump = pd.DataFrame(self.evolEU,index=range(T+1),columns=range(NpasEU+1) )


    def plot_evolEU(self, k=0, Qf=None, T=None, fignum=1, fz=20):
        """

        """
        
        import numpy as np
        import matplotlib.pyplot as plt
        from matplotlib import pylab
        
        import matplotlib
        matplotlib.rc('xtick', labelsize=fz) 
        matplotlib.rc('ytick', labelsize=fz)

        #fig = cplot.Figure( key_name)
        fig = plt.figure(fignum)
        plt.clf()
        plt.contourf(self.df_evolEUDump.columns,self.df_evolEUDump.index,self.df_evolEUDump.values,levels = np.linspace(0,.2,50),cmap=pylab.get_cmap('YlOrRd')) # levels = np.linspace(0,1,50),
        plt.xlabel('EU',fontsize=fz)
        plt.ylabel('Time',fontsize=fz)
        # plt.ylim((0,5))
        plt.title('Density of the value function EU (capped at 0.2)',fontsize=1.5*fz)
        
        plt.tight_layout()
            
        return fig


#        # import cal.plot as cplot
#        import numpy as np
#        import matplotlib.pyplot as plt
#        from matplotlib import pylab
#        
#        import matplotlib
#        matplotlib.rc('xtick', labelsize=fz) 
#        matplotlib.rc('ytick', labelsize=fz)
#
#        fig = plt.figure( fignum)
#        plt.figure(1)
#        plt.clf()
#        plt.contourf(self.df_evolQDump.columns,self.df_evolQDump.index,self.df_evolQDump.values.T,levels = np.linspace(0,1,15),
#                     cmap=pylab.get_cmap('YlOrRd')) # 
#        plt.xlabel('Time',fontsize=fz)
#        plt.ylabel('Qt',fontsize=fz)
#        plt.title('Density of Bought Quantity (Qt)',fontsize=1.5*fz)
#
#        plt.tight_layout()
#        if ylim is not None:
#            plt.ylim(ylim)
#
#        return fig



















      
    def deltaEU_VS_q(self):
        
        import numpy as np
        import pandas as pd
        T = self.T

        PrintArraydtEUlong = np.zeros(self.massDataOpt)

        PrintArrayqlong = np.zeros(self.massDataOpt)

        i=0

        for t in range(T):
            print("time is going up")
            print(t)
            print("u now")
            for u in range(2*t+1):
                print(u)
                P=self.P0+(u-t)*self.sigma
                for Q in range(t+1):
                  for v in range(min(self.Pbar*t+((t-1)*t)/2+1,Q*(int(2*np.ceil((self.P0+P)/self.sigma))+2*t+Q)/4+1)):
                   M=self.M0-v*self.sigma
                   
                   if self.pPhasegrid[t,self.index_u(t,P),Q,self.index_v(t,M)]>0:
                       PrintArraydtEUlong[i] = self.EU(t+1,P,Q,M)-self.EU(t,P,Q,M)
                       PrintArrayqlong[i] = self.quant(t,P,Q,M)
                       i += 1


               
        self.PrintArraydtEU = np.zeros(i)

        self.PrintArrayq = np.zeros(i)

        for j in range(i):
           self.PrintArraydtEU[j]= PrintArraydtEUlong[j]
           self.PrintArrayq[j]= PrintArrayqlong[j]



        self.df_DeltaEUVSquantDump = pd.DataFrame({'dEU_dt': self.PrintArraydtEU,'q': self.PrintArrayq}) 
 




       
        
    def dEU_dQ_VS_q(self):
        

        import numpy as np
        import pandas as pd
        T = self.T

        
        PrintArraydQEUlong = np.zeros(self.massDataOpt)

        PrintArrayqlong = np.zeros(self.massDataOpt)

        i=0

        for t in range(T):
            print("time is going up")
            print(t)
            print("u now")
            for u in range(2*t+1):
                print(u)
                P=self.P0+(u-t)*self.sigma
                for Q in range(t+1):
                  for v in range(min(self.Pbar*t+((t-1)*t)/2+1,Q*(int(2*np.ceil((self.P0+P)/self.sigma))+2*t+Q)/4+1)):
                   M=self.M0-v*self.sigma
                   
                   if self.pPhasegrid[t,self.index_u(t,P),Q,self.index_v(t,M)]>0:
                       PrintArraydQEUlong[i] = self.EU(t,P,Q+1,M)-self.EU(t,P,Q,M)
                       PrintArrayqlong[i] = self.quant(t,P,Q,M)
                       i += 1
               
               


        self.PrintArraydQEU = np.zeros(i)

        self.PrintArrayq = np.zeros(i)

        for j in range(i):
           self.PrintArraydQEU[j]= PrintArraydQEUlong[j]
           self.PrintArrayq[j]= PrintArrayqlong[j]



        self.df_dEU_dQVSquantDump = pd.DataFrame({'dEU_dQ':self.PrintArraydQEU,'q': self.PrintArrayq}) 
        
    
    
    def store(self, filename=None, root='H:\dev_python\projects\HJB Market orders\data\ '.strip(), prefix='optimizers_nonparametric'):
        """
        linux = {'root': '/mnt/research-safe/fellows/hdemarch/dev_python/projects/HJB Market orders/data/'}
        """
        import pandas as pd
        import numpy  as np
        dic_to_save = {'dic_input_params' : ('df', [self.dic_input_params]), 
                            'dic_output_params': ('df',[self.dic_output_params]), 
                            'quantgrid'        : ('p4', self.quantgrid), 
                            'EUgrid'           : ('p4', self.EUgrid) }
        
        if filename is None:
            filename = '%s.h5' % self.ExperimentNumber
        full_file = '%(root)s%(prefix)s_%(filename)s' % {'root': root, 'prefix': prefix, 'filename': filename}
        storage = pd.HDFStore( full_file) 
        for k, kv in dic_to_save.items():
            print("storing <%s> in <%s>..." % ( k, full_file ) )
            if kv[0]=='df':
                val = pd.DataFrame( kv[1])
            if kv[0]=='p4':
                shp = kv[1].shape
                print("storing shape")
                storage.append("%s_shape" % k, pd.DataFrame(np.array(shp)))
                print("reshaping data")
                val = pd.DataFrame( np.reshape(kv[1], (np.prod( shp),1)))
                print("reshape done")
            storage.append(k, val)
        storage.close()
        
        return full_file
        

    @staticmethod
    def list_storage(experiment_name=None, filename=None, root='H:\dev_python\projects\HJB Market orders\data\ '.strip(), prefix='optimizers_nonparametric'):
        import pandas as pd
        if filename is None:
            filename = '%s.h5' % experiment_name
        full_file = '%(root)s%(prefix)s_%(filename)s' % {'root': root, 'prefix': prefix, 'filename': filename}
        storage = pd.HDFStore( full_file) 
        print(storage)
        keys = storage.keys()
        storage.close()
        return [k[1:] for k in keys]
        
    @staticmethod
    def df2dic( df):
        return {c: df[c].values[0] for c in df.columns}
        
    
    
    
    
    
    
    
    
    
    def E_Alterna_quantile_t_Q(self):
        import numpy as np
        T=self.T
        self.quant_t_Q = np.zeros((T+1,T+1))
        self.quant_t_Q += -1
        for t in range(T):
            for Q in range(t+1):
                TotProba = np.sum(self.pPhasegrid[t,:,Q,:])
                if TotProba != 0:
                    self.quant_t_Q[t,Q] = np.sum(self.quantgrid[t,:,Q,:]*self.pPhasegrid[t,:,Q,:])/TotProba
        
        self.quant_t_Q[T,:]=0
        self.LatestOpt = 'E_Alterna_quantile_t_Q'
        return
                    
                    
                    
                    
    def optim_Alterna_quantile_t_Q(self, mode_g, sigma, T, P0, M0, Q0, Qf, Gain , RiskAvers , experiment_name=None ):
        """
        optimization

        :param mode: the optimization mode_g in 'basic', 'explicit', 'tabulated'
        """
        
        import numpy as np
        import pprint
        
        if self.virgin:
            self.Init_dics(mode_g , sigma , T , P0, M0 , Q0 , Qf , Gain , RiskAvers , experiment_name=None)
            
        # initializations
            
        PnL=self.PnL(self.Qf,self.M0)
        Utility = self.Utility(self.M0)
        
        def Penalization(P,Q):
            return (PnL(P,Q,0)-P*(Q-self.Qf))
        
        def nonZero(x):
            return (x!=0)*1
        
        #La matrice des esperances de U avec les termes a partir de t
        self.EU_t_Q    = np.zeros((T+1,T+1))
        
        #La matrice des esperances de Perf. Elle sera utile
        
        self.bkwEPerf_t_Q = np.zeros((T+1,T+1))

        #La matrice des Quantiles optimaux
        self.quant_t_Q = np.zeros((T+1,T+1))
        self.quant_t_Q = self.quant_t_Q-1

        # initialisation de l'algo backwards
        print("Initial Utility computing")
        for Q in range(T+1):
                   self.EU_t_Q[T,Q]=Utility(Penalization(self.P0,Q))
                   self.quant_t_Q[T,Q] = 0
                   self.bkwEPerf_t_Q[T,Q] = Penalization(self.P0,Q)/self.M0

        #Propagation de l'algo backwards. pour ne pas se perdre, les t,q et x correspondants sont rappeles
        for j in range(T):
            t=T-1-j
            print("time is going back")
            print(t)
            for Q in range(t+1):
                   lambdaSup  = self.EU_t_Q[t+1,Q+1]
                   lambdaInf = self.EU_t_Q[t+1,Q]

                   if mode_g == 'basic': 
                        q = 0.
                        qOpt = 1.
                        EvolOpt = lambdaSup
            #L'algorithme "naif" d'optimisation
                        for i in range(self.Nmax):
                           PSup = q
                           PInf = 1-q

            #On ajoute les morceaux du candidat au fur et a mesure
                           Candidat = PSup*lambdaSup+(1.-self.gImpact(Q))*self.g(q)/self.M0
                           Candidat += -PSup*self.RiskAvers/2.*self.sigma**2*(2.*(Q-self.Qf)+1.)/M0**2
                           Candidat += -q*(self.Qf-Q-1.)*self.sigma*self.mImpact(Q)/self.M0
                           Candidat += PInf*lambdaInf

                 
                           if Candidat>EvolOpt:
                               qOpt = q
                               EvolOpt = Candidat
                           q=q+self.delta


                   else:

                     a=lambdaSup   -(self.Qf-Q-1.)*self.sigma*self.mImpact(Q)/self.M0 
                     a+= -self.RiskAvers/2.*self.sigma**2*(2.*(Q-self.Qf)+1.)/self.M0**2    -lambdaInf
                     b=(1.-self.gImpact(Q))/self.M0
                     
#                     print 't',t,'Q',Q,'lambdaInf',lambdaInf,'lambdaSup',lambdaSup,'a',a,'b',b
                     qOpt = self.dgInv(-a/b)
#                     print 'qOpt', qOpt

                     PSup = qOpt
                     PInf = 1-qOpt

                     EvolOpt = PSup*lambdaSup
                     EvolOpt += PSup*(-(self.Qf-Q-1.)*self.sigma*self.mImpact(Q)/self.M0)
                     EvolOpt += PSup*(-self.RiskAvers/2*self.sigma**2*(2*(Q-self.Qf)+1)/self.M0**2)
                     EvolOpt += PInf*lambdaInf
                     EvolOpt += (1.-self.gImpact(Q))*self.g(qOpt)/self.M0
                     
                   
                   dPerf = qOpt*(-(self.Qf-Q-1.)*self.sigma*self.mImpact(Q)/self.M0)
                   dPerf += (1.-self.gImpact(Q))*self.g(qOpt)/self.M0


                   EvolOpt+=-self.RiskAvers/2*self.sigma**2*(Q-self.Qf)**2/self.M0**2
                   self.EU_t_Q[t,Q]=EvolOpt
                   self.quant_t_Q[t,Q]=qOpt
                   self.bkwEPerf_t_Q[t,Q]=qOpt*self.bkwEPerf_t_Q[t+1,Q+1] + (1.-qOpt)*self.bkwEPerf_t_Q[t+1,Q] + dPerf


        self.Opti_t_Q = self.EU_t_Q[0,self.Q0]


        self.dic_output_params['EU_t_Q']=self.Opti_t_Q
                    

        pprint.pprint( self.dic_input_params)

        pprint.pprint( self.dic_output_params)
        
        self.T  = T
        
        self.LatestOpt = 'optim_Alterna_quantile_t_Q'
       
        return
        
        


    def Alterna_quantile_t_Q_controlled_trajectories(self, NbSimul, plot=True, fignum=1):
        """
        Generate random trajectories and controls them for the alternative quantile
        """
        import numpy as np
        Simul = np.zeros((NbSimul,self.T+1,5))

        for i in range(NbSimul):

            Simul[i,0,1]=self.P0
            Simul[i,0,2]=self.Q0
            Simul[i,0,3]=self.M0
            CurrentPerf = 0

            for j in range(self.T):
                t=j+1
                P=Simul[i,t-1,1]
                Q=int(Simul[i,t-1,2])
                M=Simul[i,t-1,3]
                a = np.around((P-self.P0)/self.sigma+t)
                if a<0:
                   print(a)
                   print(self.EU_t_Q[self.T,])
                   raise RuntimeError("The final penalization is too much")
#                Simul[i,t-1,4] = self.EU_t_Q[t-1,Q]+M-(self.Qf-Q)*P
                Simul[i,t-1,4] = CurrentPerf + self.bkwEPerf_t_Q[t-1,Q]
                Simul[i,t-1,0]= self.quant_t_Q[t-1,Q]
                q=Simul[i,t-1,0]

                if q==-1:
                 raise RuntimeError("Exit of the known domain...")

                PSupPlus = (1+self.mImpact(Q)+(1.-self.gImpact(Q))*self.G(q)/self.sigma)/2.
                
                if q==1.:
                    PInfPlus = 0.
                else:
                    PInfPlus = (1.-(1.-self.gImpact(Q))*self.G(q)*q/self.sigma/(1.-q))/2.
#                print "q",q
#                print "PInfPlus",PInfPlus
                
#                print "PInfPlus",PInfPlus,"i",i,"q",q,"t",t,"P",P,"Q",Q,"M",M
                buy = np.random.binomial(1,q)

                if buy == 1:
                    d=np.random.binomial(1,PSupPlus)
                else:
                    d=np.random.binomial(1,PInfPlus)

                Simul[i,t,1]=P+2*(d-1./2)*self.sigma
                Simul[i,t,2]=Q+buy
                Simul[i,t,3]=M-buy*P
                CurrentPerf += buy*2*(d-1./2)*self.sigma/self.M0
                CurrentPerf += (Q-self.Qf)*2*(d-1./2)*self.sigma/self.M0



            P=Simul[i,self.T,1]
            Q=Simul[i,self.T,2]
            M=Simul[i,self.T,3]        
            Simul[i,self.T,4] = self.PnL(self.Qf,self.M0)(P,Q,M)/self.M0

        if plot:
            self.plot_sim( Simul, Qf=self.Qf, T=self.T, fignum=fignum)

        return Simul
        

    def Alterna_quantile_t_Q_phase_filling(self):
        
        
        print("Phase alternative")
        import numpy as np        
        import pprint
            
        T  = self.T
        def nonZero(x):
            return (x!=0)*1
            
        PnL=self.PnL(self.Qf,self.M0)

                  
        self.pPhasegrid_Alterna_q_t_Q=np.zeros((T+1,2*T+1,T+1,self.Pbar*T+((T-1)*T)/2+1))
        
        #initialisation
        
        self.pPhasegrid_Alterna_q_t_Q[0,0,0,0]=1
        
        #Remplissage forward de la matrice
        
        
        for t in range(T):
            print("Alterna time is going up")
            print(t)
            print("u now")
            for u in range(2*t+1):
                print(u)
                P=self.P0+(u-t)*self.sigma
                for Q in range(t+1):
                  for v in range(min(self.Pbar*t+((t-1)*t)/2+1,Q*(int(2*np.ceil((self.P0+P)/self.sigma))+2*t+Q)/4+1)):
                   M=self.M0-v*self.sigma
        
                   if self.pPhase_Alterna_q_t_Q(t,P,Q,M)!=0:
                       
                      CurrentProb = self.pPhase_Alterna_q_t_Q(t,P,Q,M)
                      
                      if (CurrentProb>1) or (CurrentProb<0):
                           print(CurrentProb)
                           raise RuntimeError("echec de probabilite")
                      
                         
                      q=self.quant_t_Q[t,Q]
                      
                      if q ==-1:
                          raise RuntimeError("etat inaccessible")
              
                      PSupPlus = (q+(self.mImpact(Q)*q+(1.-self.gImpact(Q))*self.g(q)/self.sigma))/2.
                      PSupMoins = (q-(self.mImpact(Q)*q+(1.-self.gImpact(Q))*self.g(q)/self.sigma))/2.
                      PInfPlus = min(1.,(1-q+(-(1.-self.gImpact(Q))*self.g(q)/self.sigma))/2.)
                      PInfMoins = max(0.,(1-q-(-(1.-self.gImpact(Q))*self.g(q)/self.sigma))/2.)
              
                      if PSupPlus>0:
                         self.pPhasegrid_Alterna_q_t_Q[t+1,self.index_u(t+1,P+self.sigma),Q+1,self.index_v(t+1,M-P)] += PSupPlus*CurrentProb
                      if PSupMoins>0:             
                         self.pPhasegrid_Alterna_q_t_Q[t+1,self.index_u(t+1,P-self.sigma),Q+1,self.index_v(t+1,M-P)] += PSupMoins*CurrentProb  
                      if PInfPlus>0:
                         self.pPhasegrid_Alterna_q_t_Q[t+1,self.index_u(t+1,P+self.sigma),Q,self.index_v(t+1,M)] += PInfPlus*CurrentProb
                      if PInfMoins>0:             
                         self.pPhasegrid_Alterna_q_t_Q[t+1,self.index_u(t+1,P-self.sigma),Q,self.index_v(t+1,M)] += PInfMoins*CurrentProb           



        self.FinalPnL_Alterna_q_t_Q = np.zeros((2*T+1,T+1,self.Pbar*T+((T-1)*T)/2+1))

        print("Final PnL computation Alterna t Q")

        for u in range(2*T+1):
                print(u)
                P=self.P0+(u-T)*self.sigma
                for Q in range(T+1):
                  for v in range(min(self.Pbar*T+((T-1)*T)/2+1,Q*(int(2*np.ceil((self.P0+P)/self.sigma))+2*T+Q)/4+1)):
                   M=self.M0-v*self.sigma

                   if self.pPhase_Alterna_q_t_Q(T,P,Q,M)!=0:
                       self.FinalPnL_Alterna_q_t_Q[u,Q,v] = PnL(P,Q,M)

#        self.PhasefillOpt_Alterna_q_t_Q = nonZero(self.pPhasegrid_Alterna_q_t_Q)

        MoyCarr = np.sum(self.pPhasegrid_Alterna_q_t_Q[T,:,:,:]*self.FinalPnL_Alterna_q_t_Q**2)

        Moy = np.sum(self.pPhasegrid_Alterna_q_t_Q[T,:,:,:]*self.FinalPnL_Alterna_q_t_Q)

        EcarType = np.sqrt(MoyCarr - Moy**2)

#        massDataOpt = np.sum(self.PhasefillOpt_Alterna_q_t_Q)
#
#        RatioPhasefillOpt = massDataOpt/self.massData
        
        if self.LatestOpt == 'optim_Alterna_quantile_t_Q':
            
            PerfWord = 'Perf_Optim_old'
            EcarTypeWord = 'EcarType_Optim_old'
            
        if self.LatestOpt == 'E_Alterna_quantile_t_Q':
            
            PerfWord = 'Perf_Eq_old   '
            EcarTypeWord = 'EcarType_Eq_old   '           
            
        if self.LatestOpt == 'optimize_r_basic':

            PerfWord = 'Perf_r_bas_old'
            EcarTypeWord = 'EcarType_r_bas_old'

            
        if self.LatestOpt == 'optimize_r_approx':

            PerfWord = 'Perf_r_prx_old'
            EcarTypeWord = 'EcarType_r_prx_old'

            
        if self.LatestOpt == 'optimize_r_explicit':

            PerfWord = 'Perf_r_exp_old'
            EcarTypeWord = 'EcarType_r_exp_old'

            
        if self.LatestOpt == 'optimize_r_extract':

            PerfWord = 'Perf_r_ext_old'
            EcarTypeWord = 'EcarType_r_ext_old'
            
            
        if self.LatestOpt == 'optimize_linear':

            PerfWord = 'Perf_lin_old  '
            EcarTypeWord = 'EcarType_lin_old  '
            
        if self.LatestOpt == 'optimize_r_choice':

            PerfWord = 'Perf_r_cho_old'
            EcarTypeWord = 'EcarType_r_cho_old'

            
#        self.dic_output_params[RatioPhasefillOptWord]=RatioPhasefillOpt
#        self.dic_output_params[InutiliteWord]=1/RatioPhasefillOpt
        self.dic_output_params[EcarTypeWord]=EcarType/self.M0
        self.dic_output_params[PerfWord]=Moy/self.M0
        
        
#        self.dic_output_params['Comp_Perf_VS_Alterna_q_t_Q']= (self.dic_output_params['Perf_Alterna_q_t_Q']-self.dic_output_params['Perf'])/ self.dic_output_params['Perf'] 
#        self.dic_output_params['Comp_EcarType_VS_Alterna_q_t_Q']= (self.dic_output_params['EcarType_Alterna_q_t_Q']-self.dic_output_params['EcarType'])/ self.dic_output_params['EcarType']
        


        pprint.pprint(self.dic_input_params)
        print(" ")
        pprint.pprint(self.dic_output_params)
        print(" ")
    




    def plot_arrow(self, fignum=1 , Color = 'k'):
        """

        """
        import matplotlib.pyplot as plt
#        from scipy import *
#        from scipy import integrate
#        from scipy.integrate import ode
        import numpy as np
        
        fig = plt.figure(fignum)
        ax=fig.add_subplot(111)

        ## Vector field function
        def vf(t,Q):
          dx=np.zeros(2)
          dx[0]=1
          dx[1]=self.quant_t_Q[t,Q]
          return dx
        def V_use(t,Q):
            return self.quant_t_Q[t,Q]

        #Vector field
        X,Y = np.meshgrid( np.linspace(0,self.T,self.T+1),np.linspace(0,self.Qf,self.Qf+1) )
        U = 1.
        V=np.zeros((self.Qf+1,self.T+1))
        for i in range (self.Qf+1):
            for j in range (self.T+1):
                t = int(X[i,j])
                Q = int(Y[i,j])
                V[i,j] = V_use(t,Q)
                if self.T-t<self.Qf-Q:
                    V[i,j] = -1.
        #Normalize arrows
        N = np.sqrt(U**2+V**2)
        U2, V2 = U/N*(1.-1.*(V==-1.)), V/N*(1.-1.*(V==-1.))
        ax.quiver( X,Y,U2,V2,color = Color,alpha=.7)


        plt.xlim([0,self.T])
        plt.ylim([0,self.Qf])
        plt.xlabel(r"$t$")
        plt.ylabel(r"$Q$")
        plt.show()




    def phase_filling_t_Q(self):
        import numpy as np
        T=self.T
        print("Fast phase filling")
        self.pPhase_t_Q = np.zeros((T+1,T+1))
        self.pPhase_t_Q[0,0] = 1.
        for t in range(T):
            for Q in range(t+1):
                if self.pPhase_t_Q[t,Q]!=0:
                    CurrentProb = self.pPhase_t_Q[t,Q]
                    q=self.quant_t_Q[t,Q]
                    if q==-1:
                        raise RuntimeError("Inaccessible state")
                    self.pPhase_t_Q[t+1,Q] += CurrentProb*(1.-q)
                    self.pPhase_t_Q[t+1,Q+1] += CurrentProb*q
        
        return
        
        
    def Perf_StandardDev_t_Q(self):
        """
        Compute the Performance and the Standard deviation, only based on a t and Q state space
        Careful : for not well bounded gain function the Variance is wrong
        """
        import pprint
        import numpy as np
        T = self.T
        
        PnL=self.PnL(self.Qf,self.M0)
        
        def Penalization(P,Q):
            return (PnL(P,Q,0)-P*(Q-self.Qf))
        
        self.phase_filling_t_Q()
        Perf_t = 0
        Var_t = 0
        for t in range(T):
            dPerf_t = 0
            dCarr_t = 0
            for Q in range(t+1):
                Prob = self.pPhase_t_Q[t,Q]
                q = self.quant_t_Q[t,Q]
                dPerf_t += Prob*(1.-self.gImpact(Q))*self.g(q)
                dPerf_t += -Prob*(self.Qf-Q-1.)*q*self.sigma*self.mImpact(Q)
                dCarr_t += Prob*self.sigma**2*(Q-self.Qf)**2
                dCarr_t += Prob*self.sigma**2*(2*(Q-self.Qf)+1)*q
            Perf_t+=dPerf_t
            Var_t += dCarr_t-dPerf_t**2
                
                
#        if True:
#            Perf_t = np.zeros((T+1,T+1))
#            Carr_t = np.zeros((T+1,T+1))
#            for t in range(T):
#                for Q in range(t+1):
#                    Prob = self.pPhase_t_Q[t,Q]
#                    q = self.quant_t_Q[t,Q]
#                    Perf_t[t,Q] += Prob*(1.-self.gImpact(Q))*self.g(q)
#                    Perf_t[t,Q] += -Prob*(self.Qf-Q-1.)*q*self.sigma*self.mImpact(Q)
#                    Perf_t[t+1,Q+1] += q*Perf_t[t,Q]
#                    Perf_t[t+1,Q] += (1-q)*Perf_t[t,Q]
#                    
#                    Carr_t[t,Q] += Prob*self.sigma**2*(Q-self.Qf)**2
#                    Carr_t[t,Q] += Prob*self.sigma**2*(2*(Q-self.Qf)+1)*q
#                    Carr_t[t+1,Q+1] += q*Carr_t[t,Q]
#                    Carr_t[t+1,Q] += (1-q)*Carr_t[t,Q]
#                    
#                    Carr_t[t+1,Q+1] += 2*Perf_t[t,Q]*(Q+1.-self.Qf)*((1.-self.gImpact(Q))*self.g(q)+q*self.sigma*self.mImpact(Q))
#                    Carr_t[t+1,Q] += 2*Perf_t[t,Q]*(Q-self.Qf)*(-(1.-self.gImpact(Q))*self.g(q))
#
#                
#            Perf = np.sum(Perf_t[T-1,:])
#            EcarType = np.sqrt(np.sum(Carr_t[T-1,:])-Perf**2)
        
        # Mouvements dus a la penalisation finale
        self.Penal = np.zeros(T+1)
        print("Fast final penalization")
        for Q in range(T+1):
                   self.Penal[Q]=Penalization(self.P0,Q)
                   
        Moy = self.Penal*self.pPhase_t_Q[T,:]
        Carr = (self.Penal)**2*self.pPhase_t_Q[T,:]

        Perf_T = np.sum(Moy)
        Var_T = np.sum(Carr)-Perf_T**2
        
        Perf = Perf_t+Perf_T
        EcarType = np.sqrt(Var_t+Var_T)
        
       
     
        if self.LatestOpt == 'optim_Alterna_quantile_t_Q':

            PerfWord = 'Perf_Optim_t_Q'
            EcarTypeWord = 'EcarType_Optim_t_Q'
            
        if self.LatestOpt == 'E_Alterna_quantile_t_Q':

            PerfWord = 'Perf_Eq_t_Q   '
            EcarTypeWord = 'EcarType_Eq_t_Q   '           
            
        if self.LatestOpt == 'optimize_r_basic':

            PerfWord = 'Perf_r_bas_t_Q'
            EcarTypeWord = 'EcarType_r_bas_t_Q'

            
        if self.LatestOpt == 'optimize_r_approx':

            PerfWord = 'Perf_r_prx_t_Q'
            EcarTypeWord = 'EcarType_r_prx_t_Q'

            
        if self.LatestOpt == 'optimize_r_explicit':

            PerfWord = 'Perf_r_exp_t_Q'
            EcarTypeWord = 'EcarType_r_exp_t_Q'

            
        if self.LatestOpt == 'optimize_r_extract':

            PerfWord = 'Perf_r_ext_t_Q'
            EcarTypeWord = 'EcarType_r_ext_t_Q'
            
            
        if self.LatestOpt == 'optimize_linear':

            PerfWord = 'Perf_lin_t_Q  '
            EcarTypeWord = 'EcarType_lin_t_Q  '
            
        if self.LatestOpt == 'optimize_r_choice':

            PerfWord = 'Perf_r_cho_t_Q'
            EcarTypeWord = 'EcarType_r_cho_t_Q'
            
            
        if self.LatestOpt == 'optimize_explicit_Impactgain':

            PerfWord = 'Perf_Impga_t_Q'
            EcarTypeWord = 'EcarType_Impga_t_Q'
            
            
        if self.LatestOpt == 'optimize_Merdique':
            
            PerfWord = 'Perf_Merdique '
            EcarTypeWord = 'EcarType_Merdique '       
            
            
        if self.LatestOpt == 'optimize_Discrete':
            
            PerfWord = 'Perf_Discrete '
            EcarTypeWord = 'EcarType_Discrete '     
            
            
        if self.LatestOpt[0:15] == 'optimize_theory':
            formula = self.LatestOpt[16:]
            
            PerfWord = 'Perf_Theory_'+formula
            EcarTypeWord = 'EcarType_Theory'+formula


        self.dic_output_params[EcarTypeWord]=EcarType/self.M0
        self.dic_output_params[PerfWord]=Perf/self.M0
        
        
#        self.dic_output_params['Comp_Perf_VS_Alterna_q_t_Q']= (self.dic_output_params['Perf_Alterna_q_t_Q']-self.dic_output_params['Perf'])/ self.dic_output_params['Perf'] 
#        self.dic_output_params['Comp_EcarType_VS_Alterna_q_t_Q']= (self.dic_output_params['EcarType_Alterna_q_t_Q']-self.dic_output_params['EcarType'])/ self.dic_output_params['EcarType']
        


        pprint.pprint(self.dic_input_params)
        print(" ")
        pprint.pprint(self.dic_output_params)
        print(" ")        
        return
        
        
    def evolutionQ_t_Q(self,Ext = 0):  
        
        print("evolQ is computing")    

        import pandas as pd
        import numpy as np
        T = self.T        
        
        self.evolQ_t_Q = np.zeros((T+1,self.Qf+1+Ext))
        
        for t in range(T+1):
            for Q in range(min(t+1,self.Qf+1+Ext)):
                self.evolQ_t_Q[t,Q] = self.pPhase_t_Q[t,Q]
        
        self.df_evolQ_t_QDump = pd.DataFrame(self.evolQ_t_Q,index=range(T+1),columns=range(self.Qf+1+Ext) ) 


    def plot_evolQ_t_Q(self, k=0, Qf=None, T=None, fignum=1, ylim=None, fz=20):
        """

        """
        
        # import cal.plot as cplot
        import numpy as np
        import matplotlib.pyplot as plt
        from matplotlib import pylab
        
        import matplotlib
        matplotlib.rc('xtick', labelsize=fz) 
        matplotlib.rc('ytick', labelsize=fz)

        fig = plt.figure( fignum)
        plt.clf()
        plt.contourf(self.df_evolQ_t_QDump.index,self.df_evolQ_t_QDump.columns,self.df_evolQ_t_QDump.values.T,levels = np.linspace(0,1,15),
                     cmap=pylab.get_cmap('YlOrRd'))
        plt.xlabel('Time',fontsize=fz)
        plt.ylabel('Qt',fontsize=fz)
        plt.title('Density of Bought Quantity (Qt)',fontsize=1.5*fz)

        plt.tight_layout()
        if ylim is not None:
            plt.ylim(ylim)

        return fig


    def forward_EPerf(self):
        """
        Computes the forward Perf phase distribution (excluding the expected remaining Perf)
        """
        import numpy as np
        T=self.T
        self.fwdEPerf_t_Q = np.zeros((T+1,T+1))
        for t in range(T):
            for Q in range(t+1):
                q = self.quant_t_Q[t,Q]
                Perf = self.fwdEPerf_t_Q[t,Q]
                self.fwdEPerf_t_Q[t+1,Q] += q*(Perf + (Q+1-self.Qf)*self.sigma*self.mImpact(Q)/self.M0)+(1.-self.gImpact(Q))*self.g(q)/self.M0
                self.fwdEPerf_t_Q[t+1,Q+1] += (1-q)*Perf
    

    def evolution_EPerf_t_Q(self, NpasEU):
        """
        Works badly because evolution of the price is not taken into account
        """
        print("evolEU is computing")
        
        import numpy as np
        import pandas as pd
        T = self.T
        def nonZero(x):
            return (x!=0)*1
            
        access = nonZero(self.pPhase_t_Q)

        self.forward_EPerf()

        self.EPerf_t_Q = self.fwdEPerf_t_Q+self.bkwEPerf_t_Q
        
        FindMax = access*self.EPerf_t_Q
        EPerfmax = self.EPerf_t_Q[0,self.Q0]
        EPerfmin = self.EPerf_t_Q[0,self.Q0]

        for Q in range(T+1):
                
                Comp = FindMax[T,Q]
                if Comp > EPerfmax:
                    EPerfmax = Comp
                elif Comp < EPerfmin:
                    EPerfmin = Comp
            
        deltaEU = (EPerfmax-EPerfmin)/NpasEU
        print("min",EPerfmin,"max",EPerfmax)

        self.evolEPerf_t_Q = np.zeros((T+1,NpasEU+1))

        def index_w(EspU):
            return np.around((EspU-EPerfmin)/deltaEU)
    
        for t in range(T+1):
#            print("time is going up")
#            print(t)
                for Q in range(t+1):
                   
                   if self.pPhase_t_Q[t,Q]>0:
                       self.evolEPerf_t_Q[t,index_w(self.EPerf_t_Q[t,Q])] += self.pPhase_t_Q[t,Q]


        self.df_evolEPerf_t_QDump = pd.DataFrame(self.evolEPerf_t_Q,index=range(T+1),columns=range(NpasEU+1) )


    def plot_evolEPerf_t_Q(self, k=0, Qf=None, T=None, fignum=1, fz=20):
        """
        Works badly because evolution of the price is not taken into account
        """
        
        import numpy as np
        import matplotlib.pyplot as plt
        from matplotlib import pylab
        
        import matplotlib
        matplotlib.rc('xtick', labelsize=fz) 
        matplotlib.rc('ytick', labelsize=fz)

        #fig = cplot.Figure( key_name)
        fig = plt.figure(fignum)
        plt.clf()
        plt.contourf(self.df_evolEPerf_t_QDump.columns,self.df_evolEPerf_t_QDump.index,self.df_evolEPerf_t_QDump.values,levels = np.linspace(0,.2,50),cmap=pylab.get_cmap('YlOrRd')) # levels = np.linspace(0,1,50),
        plt.xlabel('EPerf',fontsize=fz)
        plt.ylabel('Time',fontsize=fz)
        # plt.ylim((0,5))
        plt.title('Density of the value function EPerf (capped at 0.2)',fontsize=1.5*fz)
        
        plt.tight_layout()
            
        return fig

    
    
    def optimize_r(self , mode_r , mode_g , sigma , T , P0, M0 , Q0 , Qf , Gain , RiskAvers , experiment_name=None , r_choice=1.):
        """        
        mode = 'explicit' , 'basic'    , 'approx' , 'extract'
        explicit and approx does not take into account market impact
        For explicit, K must be less than 15
        """
        import numpy as np
        T=self.T
        
        self.K = (self.RiskAvers*self.sigma**2*T)/(self.P0*self.Qf*1./T*self.sigma*self.Gain)
        
        if self.virgin:
            self.Init_dics(mode_g , sigma , T , P0, M0 , Q0 , Qf , Gain , RiskAvers , experiment_name=None)               
        
        
        print("r is optimizing")
        print(mode_r)
        if mode_r=='basic':        
    
          def Utility(r):
              U = 0.
              for t in range(T):
                  Q = self.Qf*(1.-(1.-t*1./T)**r)
                  q = min(1.,r*(self.Qf-Q)/(T-t))
            
                  U+= (1.-self.gImpact(Q))*self.g(q)/self.M0
                  U+= -q*(self.Qf-Q-1.)*self.sigma*self.mImpact(Q)/self.M0
                  U+= -self.RiskAvers/2.*self.sigma**2*(self.Qf-Q)**2/self.M0**2
                  U+= -self.RiskAvers/2.*self.sigma**2*(2*(Q-self.Qf)+1)*q/self.M0**2
              return U
        
          rmin = 1.
          rmax = T*1./self.Qf
          delta = (rmax-rmin)/self.Nmax3
          rOpt = rmin
          UOpt = Utility(rmin)
          rtest=rmin
        
          for k in range(self.Nmax3-1):
              rtest+=delta
              Candidat = Utility(rtest)
              if (Candidat > UOpt):
                  UOpt = Candidat
                  rOpt = rtest
          self.LatestOpt = 'optimize_r_basic'
          self.dic_output_params['r basic']=rOpt
        
        elif mode_r=='approx':
            rOpt = 1+7./18.*self.K
            self.LatestOpt = 'optimize_r_approx'
            self.dic_output_params['r approx']=rOpt
            self.dic_output_params['K']=self.K
            
        elif mode_r=='explicit':
            rOpt = self.explicit_horror(self.K)
            self.LatestOpt = 'optimize_r_explicit'
            self.dic_output_params['r explicit']=rOpt
            self.dic_output_params['K']=self.K
            
        elif mode_r=='extract':
            self.phase_filling_t_Q()
            def nonZero(x):
                return 1*(x!=0)
            Delete_1 = nonZero(self.quant_t_Q-1.)
            Inv = np.zeros((T+1,T+1))
            for t in range(T):
                for Q in range(min(t+1,self.Qf)):
                    Inv[t,Q]=(T-t)*1./(self.Qf-Q)
            Signif = nonZero(self.quant_t_Q*Inv*Delete_1)
            Phases_Meanintermediate = Signif*self.pPhase_t_Q
            Phases_Mean = Phases_Meanintermediate/np.sum(Phases_Meanintermediate)
            rOpt = np.sum(self.quant_t_Q*Inv*Phases_Mean)
            
            Var = np.sum((self.quant_t_Q*Inv)**2*Phases_Mean)-rOpt**2
            Ecartype = np.sqrt(Var)
            self.LatestOpt = 'optimize_r_extract'
            self.dic_output_params['r extract']=rOpt
            self.dic_output_params['r extract error']=Ecartype
            
        elif mode_r=='choice':
            rOpt = r_choice
            self.LatestOpt = 'optimize_r_choice'
            self.dic_output_params['r choice']=rOpt
        
        else:
            print(mode_r)
            raise RuntimeError ("Deso, je l ai pas definie, Noraj")
        self.rOpt = rOpt
            
        self.quant_t_Q=np.zeros((T+1,T+1))
        self.quant_t_Q +=-1     
        
        for t in range(T):
            for Q in range(t+1):
                self.quant_t_Q[t,Q]=max(0,min(1.,self.rOpt*(self.Qf-Q)*1./(T-t)))

        return



    def optimize_theory(self , mode_g , sigma , T , P0, M0 , Q0 , Qf , Gain ,
                        RiskAvers , formula = "border", experiment_name=None):
        """        
        mode = 'explicit' , 'basic'    , 'approx' , 'extract'
        explicit and approx does not take into account market impact
        For explicit, K must be less than 15
        """
        import numpy as np
        T=self.T
        
        if self.virgin:
            self.Init_dics(mode_g , sigma , T , P0, M0 , Q0 , Qf , Gain , RiskAvers , experiment_name=None)               
            
        self.quant_t_Q=np.zeros((T+1,T+1))
        self.quant_t_Q +=-1     
        
        for t in range(T):
            for Q in range(t+1):
                if Q >= Qf:
                    self.quant_t_Q[t,Q] = 0.
                elif T-t<=Qf-Q:
                    self.quant_t_Q[t,Q] = 1.
                elif T-t==2:
                    self.quant_t_Q[t,Q] = 0.5
                else:
                    q_0 = (self.Qf-Q)*1./(T-t)
                    if formula == "basic_shift":
                        late = -1.
                        q = 0.5+(q_0-0.5)*(T-t)/(T-t-late)
                    if formula == "border_better_keep":
                        if 0.5-np.abs(q_0-0.5)+1e-10 >= 0.25:
#                            if T-t <= 2:
#                                q = 0.5
#                            else:
                                late = -np.log(T-t)-2.4738
                                q = 0.5+(q_0-0.5)/((1+(np.log(T-t-1-late))*1./(T-t))*(1-1./max(T-t-late,2.)))
                        elif 0.5-np.abs(q_0-0.5)+1e-10 >= 0.15:
#                            if T-t <= 2:
#                                q = 0.5
#                            else:
                                late = -3.5723#3.7
                                q = 0.5+(q_0-0.5)/((1+(np.log(T-t-1-late))*1./(T-t))*(1-1./max(T-t-late,2.)))
                        else:
#                            if T-t <= 2:
#                                q = 0.5
#                            else:
                                late = -1.3445#1.4#1.35
                                nb_iter = 4
                                mode = "iter_right"
                                #mode = "right"
                                lbda = -0.5*solver_lambda(q_0, t, T, mode = mode, nb_iter = nb_iter, late = late)
                                q=0.5+lbda/(1.-1./(T-t-late))
#                        eps = (0.5-np.abs(q_0-0.5))/np.abs(q_0-0.5)
#                        nb_iter = 3
#                        mode = "iter"
#                        #mode = "exact"
#                        lbda = -solver_lambda(q_0, t, T, mode = mode, nb_iter = nb_iter)/2
#                        if eps>1.:
#                            q=0.5+lbda/(1.-1./(max(T-t,2.)))
#                        else:
#                            q=0.5+lbda
                    if formula == "border_better":
                        if 0.5-np.abs(q_0-0.5)+1e-10 >= 0.15:
                                late = -3.5723#3.7
                                q = 0.5+(q_0-0.5)/((1+(np.log(T-t-1-late))*1./(T-t))*(1-1./max(T-t-late,2.)))
                        else:
#                            if T-t <= 2:
#                                q = 0.5
#                            else:
                                late = -1.3445#1.4#1.35
                                nb_iter = 4
                                mode = "iter_right"
                                #mode = "right"
                                lbda = -0.5*solver_lambda(q_0, t, T, mode = mode, nb_iter = nb_iter, late = late)
                                q=0.5+lbda/(1.-1./(T-t-late))
#                        eps = (0.5-np.abs(q_0-0.5))/np.abs(q_0-0.5)
#                        nb_iter = 3
#                        mode = "iter"
#                        #mode = "exact"
#                        lbda = -solver_lambda(q_0, t, T, mode = mode, nb_iter = nb_iter)/2
#                        if eps>1.:
#                            q=0.5+lbda/(1.-1./(max(T-t,2.)))
#                        else:
#                            q=0.5+lbda
                    if formula == "border_approx":
#                        if T-t <= 2:
#                            q = 0.5
#                        else:
                            late = -1.3445#1.4#1.35
                            nb_iter = 4
                            mode = "iter_right"
                            #mode = "right"
                            lbda = -0.5*solver_lambda(q_0, t, T, mode = mode, nb_iter = nb_iter, late = late)
                            q=0.5+lbda/(1.-1./(T-t-late))#*(1-np.log((T-t-1-late)/(T-t-late))*T*1./(T-t))
#                        nb_iter = 3
#                        mode = "iter"
#                        #mode = "exact"
#                        lbda = -0.5*solver_lambda(q_0, t, T, mode = mode, nb_iter = nb_iter)
#                        q=0.5+lbda/(1.-1./(T-t-late))
                    elif formula == "free":
#                        if T-t <= 2:
#                            q = 0.5
#                        else:
                            late = 0
                            q = 0.5+(q_0-0.5)/((1+(np.log(T-t-1-late))*1./(T-t))*(1-1./max(T-t-late,2.)))
                    elif formula == "free_approx":    
#                        if T-t <= 2:
#                            q = 0.5
#                        else:
                            late = -3.5723#3.7
                            q = 0.5+(q_0-0.5)/((1+(np.log(T-t-1-late))*1./(T-t))*(1-1./max(T-t-late,2.)))
#                        late = -1.35
#                        eps = (0.5-np.abs(q_0-0.5))/np.abs(q_0-0.5)
#                        if T-t <= 2:
#                            q = 0.5
#                        elif eps>1.:
#                            nb_iter = 3
#                            mode = "iter"
#                            #mode = "exact"
#                            lbda = -solver_lambda(q_0, t, T, mode = mode, nb_iter = nb_iter)/2
#                            q=0.5+lbda/(1.-1./max(T-t,2.))
#                        else:
#                            nb_iter = 3
#                            mode = "iter_right"
#                            #mode = "right"
#                            lbda = -0.5*solver_lambda(q_0, t, T, mode = mode, nb_iter = nb_iter, late = late)
#                            q=0.5+lbda/(1.-1./(T-t-late))
                            #q=0.5+(q_0-0.5)/((1+(np.log(T-t-1))*1./(T-t))*(1-1./(T-t)))
                        #q=0.5+(q_0-0.5)/(1.+1./(T-t)*(np.log(T-t)-np.log(np.abs(q_0-0.5)/(0.5-np.abs(q_0-0.5)))))
                    elif formula == "border":
#                        if T-t <= 2:
#                            q = 0.5
#                        else:
                            late = 0.
                            nb_iter = 4
                            mode = "iter_right"
                            #mode = "right"
                            lbda = -0.5*solver_lambda(q_0, t, T, mode = mode, nb_iter = nb_iter, late = late)
                            q=0.5+lbda/(1.-1./(T-t-late))
                        #q = 0.5+(1-np.log((T-t))*1./(T-t))*(q_0-0.5)
                    elif formula == "new_test":
#                        if T-t <= 2:
#                            q = 0.5
#                        else:
                            late = -np.log(T-t)-2.4738
                            q = 0.5+(q_0-0.5)/((1+(np.log(T-t-1-late))*1./(T-t))*(1-1./max(T-t-late,2.)))
                            
                    self.quant_t_Q[t,Q]= max(0,min(1.,q))
        self.LatestOpt = 'optimize_theory_'+formula
        return



def solver_lambda(q_0, t, T, mode = "exact", nb_iter = 0, late = 0):
    import numpy as np
    from scipy.optimize import fsolve
    dt = T-t
    if dt <= 2:
        return 0.
    elif q_0 == 0.5:
        return 0.
    else:
        if mode == "exact":
            f = lambda x: x*(1+(np.log((dt-1)*(1-np.abs(x))/(np.abs(x))))/dt)-(1-2*q_0)
            x_0 = 1.-2*q_0
            sol = fsolve(f, x_0, full_output=True)
            #print(sol)
            lbda = sol[0]
            #print("test", f(lbda), "lambda", lbda, "x_0", x_0)
            if sol[2]:
                return lbda
            else:
                print("Failed solving.")
                return x_0
        elif mode == "iter":
            f = lambda x: (1-2*q_0)/(1+(np.log((dt-1)*(1-np.abs(x))/(np.abs(x))))/dt)
            x = 1.-2*q_0
            for i in range(nb_iter):
                x = f(x)
            return x
        elif mode == "right":
            def f(x):
                addon = 1./dt*(1+1./(1-np.abs(x)))
                addon = max(1e-5, min(1-1e-5,addon))
                part_1 = x*(1+np.log((dt-1-late)*(1-np.abs(x))/(np.abs(x)))/dt)
                part_2 = 1-2*q_0-np.sign(1.-2*q_0)*(1+late*(1-np.abs(x)))/dt
                return part_1-part_2
            x_0 = 1.-2*q_0#np.sign(1.-2*q_0)*(1.-1e-5)#
            sol = fsolve(f, x_0, full_output=True)
            #print(sol)
            lbda = sol[0]
            #print("test", f(lbda), "lambda", lbda, "x_0", x_0, "dt", dt, "p_0", q_0)
            if np.abs(f(lbda))<1e-6:#sol[2]:
                return lbda
            else:
                print("Failed solving.")
                raise("see what happens")
                return x_0
        elif mode == "iter_right":
            def f(x):
                addon = 1./dt*(1+1./(1-np.abs(x)))
                addon = max(0., min(1.,addon))
                #print("dt = ",dt , "x = ", x, "addon", addon)
                check(addon >= 0)
                check(np.log((dt-1-late)*(1-np.abs(x))/(np.abs(x)))/dt>=0)
                num = 1-2*q_0-np.sign(1.-2*q_0)*(1+late*(1-np.abs(x)))/dt
                div = 1+np.log((dt-1-late)*(1-np.abs(x))/(np.abs(x)))/dt
                return np.sign(1.-2*q_0)*np.abs(num/div)
            x = np.sign(1.-2*q_0)*(1-1.1/dt)
            for i in range(nb_iter):
                #x_sto = x
                x = f(x)
            #print("x", x_sto, "f(x)", x)
            return x
        else:
            raise("Not existing")
  
          
            
def check(to_check, info = None):
    if not to_check:
        print(info, " FAILED.")
        raise("check failure")
    return
            
            
        
    def param_lin_t_Q(self, a , mode_g , sigma , T , P0, M0 , Q0 , Qf , Gain , RiskAvers , experiment_name=None):
        import numpy as np
        self.T=T
        
        if self.virgin:
            self.Init_dics(mode_g , sigma , T , P0, M0 , Q0 , Qf , Gain , RiskAvers , experiment_name=None)
        
        self.quant_t_Q=np.zeros((T+1,T+1))
        self.quant_t_Q +=-1    
        
        for t in range(T):
            for Q in range(t+1):
                if Q>=self.Qf:
                    q = 0.
                elif Q<= self.Qf-(T-t):
                    q=1.
                else:
                    q = max(0.,min(1.,self.Qf*1./T*(1+a*(t*1./T-Q*1./self.Qf))))
                self.quant_t_Q[t,Q]= q
        
        self.dic_output_params['linear a']=a            
        self.LatestOpt = 'optimize_linear'                 
        

        return            
        

    def param_explicit_Impactgain_t_Q(self, mode_g , sigma , T , P0, M0 , Q0 , Qf , Gain , RiskAvers , experiment_name=None):
        import numpy as np
        self.T=T

        
        if self.virgin:
            self.Init_dics(mode_g , sigma , T , P0, M0 , Q0 , Qf , Gain , RiskAvers , experiment_name=None)
        
        self.quant_t_Q=np.zeros((T+1,T+1))
        self.quant_t_Q +=-1    
        
        def functional(t,Q):
            X = np.sqrt(1.-self.gImpact(1.*Q))
            Xf = np.sqrt(1.-self.gImpact(1.*self.Qf))
            return 4./self.Impactgain**2*(1./3*X**4-1./5*X**6-(1./3*Xf**3-1./5*Xf**5)*X)/(T-t)
#            return 2./3./self.Impactgain*(X**4-Xf**3*X)/(T-t)
        
        for t in range(T):
            for Q in range(t+1):
                if Q>=self.Qf:
                    q = 0.
                elif Q<= self.Qf-(T-t):
                    q=1.
                else:
                    q = max(0.,min(1.,functional(t,Q)))
                self.quant_t_Q[t,Q]= q
                 
        self.LatestOpt = 'optimize_explicit_Impactgain'               
        

        return


    def param_Merdique(self , mode_g , sigma , T , P0, M0 , Q0 , Qf , Gain , RiskAvers , experiment_name=None):
        import numpy as np
        self.T=T
        
        if self.virgin:
            self.Init_dics(mode_g , sigma , T , P0, M0 , Q0 , Qf , Gain , RiskAvers , experiment_name=None)
        
        self.quant_t_Q=np.zeros((T+1,T+1))
        self.quant_t_Q +=-1    
        
        for t in range(T):
            for Q in range(t+1):
                if Q>=self.Qf:
                    q = 0.
                elif Q<= self.Qf-(T-t):
                    q=1.
                else:
                    q = self.Qf*1./T
                self.quant_t_Q[t,Q]= q
                   
        self.LatestOpt = 'optimize_Merdique'                
        

        return      
        
        
    def param_App_Disc(self , mode_g , sigma , T , P0, M0 , Q0 , Qf , Gain , RiskAvers , 
                       prec , line , experiment_name=None):
        """
        Try the discrete approximation
        """
        import numpy as np
        self.T=T
        
        if self.virgin:
            self.Init_dics(mode_g , sigma , T , P0, M0 , Q0 , Qf , Gain , RiskAvers , experiment_name=None)
        
        self.quant_t_Q=np.zeros((T+1,T+1))
        self.quant_t_Q +=-1


        def fill(X):
            n = np.size(X)
            for i in range(n):
                X[i]=i
            return X

        self.Alpha = fill(np.zeros(T+1))
        self.Beta = fill(np.zeros(T+1))
        self.Gamma = fill(np.zeros(T+1))
        
        A = lambda x: 1.+1./2*np.log(x)
        B = lambda x: -2.-(x+2.)*np.log(x)
        C = lambda x: 2.+(x)*np.log(x)**2
        if prec <1 :
            A = lambda x: 0*x
        if prec <2 :
            B = lambda x: 0*x
        if prec <3:
            C = lambda x: 0*x

      
        self.Alpha = A(self.Alpha)
        self.Beta = B(self.Beta)
        self.Gamma = C(self.Gamma)
                
        for t in range(T):
            for Q in range(t+1):
                if Q>=self.Qf:
                    q = 0.
                elif Q<= self.Qf-(T-t):
                    q=1.
                elif T-t > line*(self.Qf - Q):
                    X=self.Qf - Q
                    to = T-t
                    q=X*1./to+(self.Alpha[X]/to+self.Beta[X]*np.log(to)/to**2+self.Gamma[X]/to**2)
                elif T-t < 1./line*(self.Qf - Q):
                    X=self.Qf - Q
                    to = T-t
                    q=X/to-(self.Alpha[X]/to+self.Beta[X]*np.log(to)/to**2+self.Gamma[X]/to**2)
                else:
                    q = (self.Qf-Q)*1./(T-t)
                self.quant_t_Q[t,Q]= q
                

        self.LatestOpt = 'optimize_Discrete'                
        

        return   

        
        
    def affiche(self):
        """
        to see the parameters
        """
        import pprint
        pprint.pprint(self.dic_input_params)
        print(" ")
        pprint.pprint(self.dic_output_params)
        print(" ")