import numpy as np
import matplotlib.pyplot as plt
import math

from fealpy.functionspace import FourierSpace
from fealpy.timeintegratoralg.timeline import UniformTimeLine

from ParabolicFourierSolver import ParabolicFourierSolver


init_value = {
        "bcc":( 
        np.array([
            [ 1,	 1,	 0],
            [-1,	 1,	 0],
            [-1,	-1,	 0],
            [ 1,	-1,	 0],
            [ 0,	 1,	 1],
            [ 0,	-1,	 1],
            [ 0,	-1,	-1],
            [ 0,	 1,	-1],
            [ 1,	 0,	 1],
            [-1,	 0,	 1],
            [-1,	 0,	-1],
            [ 1,	 0,	-1]], dtype=np.int),
            np.array(12*[0.3], dtype=np.float)
            ),
        "cam6fold":(
        np.array([
               [ 30,     0],
               [ 15,    26],
               [-15,    26],
               [-30,     0],
               [-15,   -26],
               [ 15,   -26]], dtype=np.int),
        np.array(6*[0.3], dtype=np.float)
        ),
        "LAM":(
	 np.array([[3, 0], [-3, 0]], dtype=np.int),
         np.array([0.058+0.2j, 0.058+0.2j], dtype=np.complex)
         ),
	"LAM_A":( 
        np.array([
            [0,	0, 0],
            [0,	1, 0],
            [0,-1, 0],
            [0, -2, 0],
            [0, 2, 0],
            [0, -5, 0],
            [0, 5, 0],
            [0, 4, 0],
            [0, -4, 0],
            [0,	-8, 0],
            [0, 8, 0],
            [0, -11, 0],
            [0, 11, 0],
            [0, 7, 0],
            [0, -7, 0],
            [0, -14, 0],
            [0, 14, 0],
            [0, 3, 0],
            [0, -3, 0],
            [0, 6, 0],
            [0, -6, 0],
            [0, 9, 0],
            [0, -9, 0],
            [0, 12, 0],
            [0, -12, 0],
            [0, 15, 0],
            [0, -15, 0],
            [0, 18, 0],
            [0, -18, 0],
            [0, 21, 0],
            [0, -21, 0],
            [0, -17, 0],
            [0, 17, 0],
            [0, -24, 0],
            [0, 24, 0],
            [0, -27, 0],
            [0, 27, 0],
            [0, -30, 0],
            [0, 30, 0]], dtype=np.int),
        np.array([0.3125000+ 0j, 
            -0.261905 -3.885002e-02j, -0.261905 +3.885002e-02j,
            0.1409348 +4.275210e-02j, 0.1409348 -4.275210e-02j,
            0.04673182+4.235525e-02j, 0.04673182-4.235525e-02j,
	    -0.04708859+3.146359e-02j, -0.04708859-3.146359e-02j,
	    0.01562500+3.772209e-02j,  0.01562500-3.772209e-02j,
            -0.001462647+2.977286e-02j,-0.001462647-2.977286e-02j,
            -0.01324711+2.210146e-02j, -0.01324711-2.210146e-02j,
            -0.01072663+2.006812e-02j, -0.01072663-2.006812e-02j,
            -0.01878014+8.882342e-03j, -0.01878014-8.882342e-03j,
	    -0.01306755+1.592286e-02j, -0.01306755-1.592286e-02j,
	    -0.004933303+1.969485e-02j, -0.004933303-1.969485e-02j,
	    0.00387973+1.950473e-02j,  0.00387973-1.950473e-02j,
	    0.01152416+1.553853e-02j,  0.01152416-1.553853e-02j, 
	    0.0164694+8.803119e-03j,  0.0164694-8.803119e-03j,
	    0.01784517+8.766769e-04j, 0.01784517-8.766769e-04j,
	    -0.01408331+1.044489e-02j,  -0.01408331-1.044489e-02j,
	    0.01562500+6.472087e-03j,  0.01562500-6.472087e-03j,
	    0.01060944+1.170571e-02j,   0.01060944-1.170571e-02j,
	    0.004210714+1.388087e-02j,	 0.004210714-1.388087e-02j], dtype=np.complex)
        ),
	"LAM_B":( 
        np.array([
            [0,	0, 0],
            [0,	1, 0],
            [0,-1, 0],
            [0, -2, 0],
            [0, 2, 0],
            [0, -5, 0],
            [0, 5, 0],
            [0, 4, 0],
            [0, -4, 0],
            [0,	-8, 0],
            [0, 8, 0],
            [0, -11, 0],
            [0, 11, 0],
            [0, 7, 0],
            [0, -7, 0],
            [0, -14, 0],
            [0, 14, 0],
            [0, 3, 0],
            [0, -3, 0],
            [0, 6, 0],
            [0, -6, 0],
            [0, 9, 0],
            [0, -9, 0],
            [0, 12, 0],
            [0, -12, 0],
            [0, 15, 0],
            [0, -15, 0],
            [0, 18, 0],
            [0, -18, 0],
            [0, 21, 0],
            [0, -21, 0],
            [0, -17, 0],
            [0, 17, 0],
            [0, -24, 0],
            [0, 24, 0],
            [0, -27, 0],
            [0, 27, 0],
            [0, -30, 0],
            [0, 30, 0]], dtype=np.int),
        np.array([3.750000e-01+ 0j, 
            -2.818696e-01+8.550420e-02j, -2.818696e-01-8.550420e-02j,
            9.417718e-02+6.292718e-02j, 9.417718e-02-6.292718e-02j,
	    -3.125000e-02+7.544417e-02j, -3.125000e-02-7.544417e-02j,
	    2.145326e-02+4.013623e-02j, 2.145326e-02-4.013623e-02j,
	    2.613510e-02+3.184571e-02j, 2.613510e-02-3.184571e-02j,
	    -7.759465e-03+3.900947e-02j, -7.759465e-03-3.900947e-02j,
	    -3.293895e-02+1.760624e-02j, -3.293895e-02-1.760624e-02j,	
	    -3.125000e-02+1.294417e-02j, -3.125000e-02-1.294417e-02j,
	    -8.421429e-03+2.776173e-02j, -8.421429e-03-2.776173e-02j,
	    2.606529e-02+5.184709e-03j,  2.606529e-02-5.184709e-03j,	
	    2.486595e-03+2.524683e-02j, 2.486595e-03-2.524683e-02j,
	    1.251699e-02+1.873301e-02j, 1.251699e-02-1.873301e-02j,
	    1.349471e-02+1.329113e-03j, 1.349471e-02-1.329113e-03j,
	    9.660291e-03+7.927996e-03j, 9.660291e-03-7.927996e-03j, 
            0.0 + 0.0j, 0.0+0.0j,
            0.0 + 0.0j, 0.0+0.0j,
            0.0 + 0.0j, 0.0+0.0j,
            0.0 + 0.0j, 0.0+0.0j,
            0.0 + 0.0j, 0.0+0.0j], dtype=np.complex)
        ),
	"LAM_C":( 
        np.array([
            [0,	0, 0],
            [0,	1, 0],
            [0,-1, 0],
            [0, -2, 0],
            [0, 2, 0],
            [0, -5, 0],
            [0, 5, 0],
            [0, 4, 0],
            [0, -4, 0],
            [0,	-8, 0],
            [0, 8, 0],
            [0, -11, 0],
            [0, 11, 0],
            [0, 7, 0],
            [0, -7, 0],
            [0, -14, 0],
            [0, 14, 0],
            [0, 3, 0],
            [0, -3, 0],
            [0, 6, 0],
            [0, -6, 0],
            [0, 9, 0],
            [0, -9, 0],
            [0, 12, 0],
            [0, -12, 0],
            [0, 15, 0],
            [0, -15, 0],
            [0, 18, 0],
            [0, -18, 0],
            [0, 21, 0],
            [0, -21, 0],
            [0, -17, 0],
            [0, 17, 0],
            [0, -24, 0],
            [0, 24, 0],
            [0, -27, 0],
            [0, 27, 0],
            [0, -30, 0],
            [0, 30, 0]], dtype=np.int),
        np.array([0.3125000+ 0j, 
            -0.261905 -3.885002e-02j, -0.261905 +3.885002e-02j,
            0.1409348 +4.275210e-02j, 0.1409348 -4.275210e-02j,
            0.04673182+4.235525e-02j, 0.04673182-4.235525e-02j,
	    -0.04708859+3.146359e-02j, -0.04708859-3.146359e-02j,
	    0.01562500+3.772209e-02j,  0.01562500-3.772209e-02j,
            -0.001462647+2.977286e-02j,-0.001462647-2.977286e-02j,
            -0.01324711+2.210146e-02j, -0.01324711-2.210146e-02j,
            -0.01072663+2.006812e-02j, -0.01072663-2.006812e-02j,
            -0.01878014+8.882342e-03j, -0.01878014-8.882342e-03j,
	    -0.01306755+1.592286e-02j, -0.01306755-1.592286e-02j,
	    -0.004933303+1.969485e-02j, -0.004933303-1.969485e-02j,
	    0.00387973+1.950473e-02j,  0.00387973-1.950473e-02j,
	    0.01152416+1.553853e-02j,  0.01152416-1.553853e-02j, 
	    0.0164694+8.803119e-03j,  0.0164694-8.803119e-03j,
	    0.01784517+8.766769e-04j, 0.01784517-8.766769e-04j,
	    -0.01408331+1.044489e-02j,  -0.01408331-1.044489e-02j,
	    0.01562500+6.472087e-03j,  0.01562500-6.472087e-03j,
	    0.01060944+1.170571e-02j,   0.01060944-1.170571e-02j,
	    0.004210714+1.388087e-02j,	 0.004210714-1.388087e-02j], dtype=np.complex)
        ),
        "C42C":( 
        np.array([
            [0,	0],
            [0,	1],
            [0,-1],
            [1,	0],
            [-1,0],
            [-1,1],
            [1,-1],
            [1,	1],
            [-1,-1],
            [0,	2],
            [0,-2],
            [2,	0],
            [-2,0],
            [-1,2],
            [1,-2],
            [-2,1],
            [2,-1],
            [1,	2],
            [-1,-2],
            [2,	1],
            [-2,-1],
            [-4,0],
            [4,	0],
            [0,	4],
            [0,-4],
            [-1,4],
            [1,-4],
            [4,	1],
            [-4,-1],
            [-4,1],
            [4,-1],
            [1,	4],
            [-1,-4],
            [3,	2],
            [-3,-2],
            [-2,3],
            [2,-3],
            [2,	3],
            [-2,-3],
            [-3,2],
            [3,-2]], dtype=np.int),
        np.array([0.14	+ 0j, 
            0.108189 -7.29663e-06j, 0.108189 +7.29663e-06j,
            0.108189 +1.53758e-06j, 0.108189 -1.53758e-06j,
            0.08244-6.80474e-06j, 0.08244+6.80474e-06j,
            0.0824395-4.42577e-06j, 0.0824395+4.42577e-06j,
            0.0424291-5.64406e-06j, 0.0424291+5.64406e-06j, 
            0.0424288+1.16854e-06j, 0.0424288-1.16854e-06j,
            0.0294126-4.4255e-06j, 0.0294126+4.4255e-06j,
            0.0294126-2.9508e-06j, 0.0294126+2.9508e-06j,
            0.0294122-3.50997e-06j, 0.0294122+3.50997e-06j,
            0.0294118-1.27548e-06j, 0.0294118+1.27548e-06j,
            -0.0152264+9.79735e-07j, -0.0152264-9.79735e-07j,
            -0.0152264+4.23116e-06j, -0.0152264-4.23116e-06j, 
            -0.0132502+3.87451e-06j, -0.0132502-3.87451e-06j,
            -0.0132501+4.84293e-08j, -0.0132501-4.84293e-08j,
            -0.0132499+1.67138e-06j, -0.0132499-1.67138e-06j, 
            -0.0132499+3.42747e-06j, -0.0132499-3.42747e-06j, 
            -0.0108379+9.19385e-07j, -0.0108379-9.19385e-07j,
            -0.0108377+2.42641e-06j, -0.0108377-2.42641e-06j,
            -0.0108376+1.8744e-06j, -0.0108376-1.8744e-06j,
            -0.0108375+1.74526e-06j,-0.0108375-1.74526e-06j], dtype=np.complex)
        ),
        "C42A":( 
            np.array([
            [0,	0],
            [0,	1],
            [0,-1],
            [-1,0],
            [1,	0],
            [-2,0],
            [2,	0],
            [0,	2],
            [0,-2],
            [-2,2],
            [2,-2],
            [2,	2],
            [-2,-2],
            [3,	1],
            [-3,-1],
            [-3,1],
            [3,-1],
            [1,	3],
            [-1,-3],
            [-1,3],
            [1,-3],
            [-1,2],
            [1,-2],
            [-2,1],
            [2,-1],
            [1,	2],
            [-1,-2],
            [2,	1],
            [-2,-1],
            [3,	0],
            [-3,0],
            [0,	3],
            [0,-3],
            [3,	2],
            [-3,-2],
            [-2,3],
            [2,-3],
            [2,	3],
            [-2,-3],
            [-3,2],
            [3,-2],
            [-3,3],
            [3,-3],
            [3,	3],
            [-3,-3],
            [-1,-1],
            [-1,1],
            [1,-1],
            [1,1],
            [4,-3],
            [4,-2],
            [4,-1],
            [4,0],
            [4,1],
            [4,2],
            [4,3],
            [4,4],
            [-3,4],
            [-2,4],
            [-1,4],
            [0,4],
            [1,4],
            [2,4],
            [3,4]], dtype=np.int),
           np.array([
            0.74+0j	, 
            -0.108419+6.99793e-06j, -0.108419-6.99793e-06j,
            -0.108419+2.03331e-06j, -0.108419-2.03331e-06j,
            -0.103226+4.69495e-06j, -0.103226-4.69495e-06j,
            -0.103226+1.29991e-05j, -0.103226-1.29991e-05j,
            -0.0414937+7.42252e-06j, -0.0414937-7.42252e-06j,
            -0.0414933+2.78787e-06j, -0.0414933-2.78787e-06j,
            0.0370139+5.85495e-07j, 0.0370139-5.85495e-07j, 
            0.0370132-4.82466e-06j, 0.0370132+4.82466e-06j,
            0.037013-6.22156e-06j, 0.037013+6.22156e-06j,
            0.037013-7.95719e-06j, 0.037013+7.95719e-06j,
            -0.0336456+6.20032e-06j, -0.0336456-6.20032e-06j, 
            -0.0336454+2.31633e-06j, -0.0336454-2.31633e-06j,
            -0.0336451+3.80009e-06j, -0.0336451-3.80009e-06j, 
            -0.0336449+1.74535e-06j, -0.0336449-1.74535e-06j,
            0.0193733+8.87915e-07j, 0.0193733-8.87915e-07j,
            0.0193731-4.28944e-06j, 0.0193731+4.28944e-06j, 
            0.0150872-1.38969e-06j, 0.0150872+1.38969e-06j,
            0.0150872-4.13871e-06j, 0.0150872+4.13871e-06j,
            0.0150869-2.1522e-06j , 0.0150869+2.1522e-06j ,
            0.0150866-2.01848e-06j, 0.0150866+2.01848e-06j,
            0.0130143-2.98794e-06j, 0.0130143+2.98794e-06j,
            0.0130142-1.88253e-06j, 0.0130142+1.88253e-06j,
            0+0j,0+0j,
            0+0j,0+0j,
            0+0j,0+0j,
            0+0j,0+0j,
            0+0j,0+0j,
            0+0j,0+0j,
            0+0j,0+0j,
            0+0j,0+0j,
            0+0j,0+0j,
            0+0j], dtype=np.complex)
           ),
       "C42B":( 
            np.array([
            [0,	0],
            [1,	1],
            [-1,-1],
            [-1,1],
            [1,-1],
            [2,	0],
            [-2,0],
            [0,	2],
            [0,-2],
            [2,	2],
            [-2,-2],
            [-2,2],
            [2,-2],
            [3,	1],
            [-3,-1],
            [-3,1],
            [3,-1],
            [1,	3],
            [-1,-3],
            [-1,3],
            [1,-3],
            [0,	3],
            [0,-3],
            [-3,0],
            [3,	0]], dtype=np.int),
           np.array([0.12+0j,
            -0.0828233+2.55012e-06j, -0.0828233-2.55012e-06j,
            -0.0828232+7.3354e-06j, -0.0828232-7.3354e-06j ,
            0.0598827+3.53369e-06j, 0.0598827-3.53369e-06j,
            0.0598822-7.23639e-06j, 0.0598822+7.23639e-06j,
            0.0381595-2.37293e-06j, 0.0381595+2.37293e-06j,
            0.0381593-6.67865e-06j, 0.0381593+6.67865e-06j,
            -0.0283762-6.53894e-07j,-0.0283762+6.53894e-07j,
            -0.028376+3.96855e-06j, -0.028376-3.96855e-06j,
            -0.0283757+4.53605e-06j,-0.0283757-4.53605e-06j,
            -0.0283756+5.98718e-06j, -0.0283756-5.98718e-06j,
            -0.0130638+2.83679e-06j, -0.0130638-2.83679e-06j,
            -0.0130637+5.26915e-07j, -0.0130637-5.26915e-07j
            ], dtype=np.complex)
            )
        }

def model_options(
        nspecies = 3,
        nblend = 2, 
        nABCblend = 1,
        nBCblend = 1,
        nblock = 5,
        ndeg = 100,
        fA = 0.5,
        fB1 = 0.3, #fB1 = N_B1/N_{ABC}
        fC1 = 0.2,
        fBC = 0.2, #fBC = N_BC/N_ABC
        fB2 = 0.1, #fB2 = N_B2/N_{BC}
        fC2 = 0.2,
        chiAB = 0.24,
        chiAC = 0.24,
        chiBC = 0.24,
        box = np.diag(2*[2*np.pi]),
        NS = 8,
        maxdt = 0.001,
        bA = 1,
        bB = 1,
        bC = 1,
        Maxit = 5000,
        tol = 1e-7):
        # the parameter for scft model
        options = {
                'nspecies': nspecies,
                'nblend': nblend,
                'nABCblend': nABCblend,
                'nBCblend': nBCblend,
                'nblock': nblock,
                'ndeg': ndeg,
                'fA': fA,
                'fB1': fB1,
                'fC1': fC1,
                'fBC': fBC,
                'fB2': fB2,
                'fC2': fC2,
                'chiAB': chiAB,
                'chiAC': chiAC,
                'chiBC': chiBC,
                'box': box,
                'dim': len(box),
                'NS' : NS,
                'maxdt': maxdt,
                'bA': bA,
                'bB': bB,
                'bC': bC,
                'Maxit':Maxit,
                'tol':tol
                }
        return options


class SCFTABC_BCBlendModel():
    def __init__(self, options=None):
        if options == None:
            options = pscftmodel_options()
        self.options = options
        dim = options['dim']
        box = options['box'] 
        self.space = FourierSpace(box,  options['NS'])

        fA = options['fA']
        fB1  = options['fB1']
        fC1  = options['fC1']
        fBC  = options['fBC']
        fB2  = options['fB2']*fBC
        fC2  = options['fC2']*fBC

        maxdt = options['maxdt']

        self.timelines = []
        self.timelines.append(UniformTimeLine(0, fA,   int(np.ceil(fA/maxdt))))
        self.timelines.append(UniformTimeLine(0, fB1,  int(np.ceil(fB1/maxdt))))
        self.timelines.append(UniformTimeLine(0, fC1,  int(np.ceil(fC1/maxdt))))
        self.timelines.append(UniformTimeLine(0, fB2,  int(np.ceil(fB2/maxdt))))
        self.timelines.append(UniformTimeLine(0, fC2,  int(np.ceil(fC2/maxdt))))



        self.pdesolvers = []
        for i in range(5):
            self.pdesolvers.append(
                    ParabolicFourierSolver(self.space, self.timelines[i])
                    )


        self.TNL = 0 # total number of time levels
        self.ABCNL = 0 # ABC number of time levels
        self.BCNL = 0  # BC number of time levels
        for i in range(5):
            NL = self.timelines[i].number_of_time_levels()
            self.TNL += self.timelines[i].number_of_time_levels()
        self.TNL -= options['nblock'] - 2

        for i in range(3):
            self.ABCNL += self.timelines[i].number_of_time_levels()
        self.ABCNL -= 2

        for i in range(2):
            self.BCNL += self.timelines[3+i].number_of_time_levels()
        self.BCNL -= 1
#????

        self.qf = self.space.function(dim=self.ABCNL) # forward  propagator of ABC
        self.cqf = self.space.function(dim=self.BCNL) # forward  propagator of BC
        self.qb = self.space.function(dim=self.ABCNL) # backward propagator of ABC
        self.cqb = self.space.function(dim=self.BCNL) # backward propagator of BC

        self.qf[0] = 1
        self.cqf[0] = 1
        self.qb[0] = 1
        self.cqb[0] = 1

        self.rho = self.space.function(dim=options['nspecies'] + 2)
        self.grad = self.space.function(dim=options['nspecies'] + 1)
        self.w = self.space.function(dim=options['nspecies'] + 1)
        self.Q = np.zeros(options['nblend'], dtype=np.float) # Q_ABC and Q_BC

    def init_field(self, rho):
        options = self.options
        chiABN = options['chiAB']*options['ndeg']
        chiBCN = options['chiBC']*options['ndeg']
        chiACN = options['chiAC']*options['ndeg']

        self.w[1] = chiABN*rho[1] + chiACN*rho[2]
        self.w[2] = chiABN*rho[0] + chiBCN*rho[2]
        self.w[3] = chiACN*rho[0] + chiBCN*rho[1]

    def compute(self):
        """
        目标函数，给定外场，计算哈密尔顿量及其梯度
        """
        # solver the forward and backward equation
        self.compute_propagator()
#        print("cqf:", self.cqf[-1])
#        print("qb", self.qb[-1])
        # compute single chain partition function Q
        self.compute_single_Q()
        print("Q:", self.Q)
        # compute density
        self.compute_density()
#        rho = self.rho
#        print('rhoA', rho[0])
#        print('rhoB', rho[1])
#        print('rhoC', rho[2])
        # compute energy function and its gradient
        #self.update_field()
        self.update_field_new()
        
#        w =self.w
        #print('w+', w[0])
#        print('wA', w[1])
#        print('WB', w[2])
#        print('WC', w[3])

        self.compute_wplus()
        w = self.w
        #print('w+', w[0])
        self.compute_energe()
        print('H',self.H)
        self.compute_gradient()

    def update_field(self, alpha=0.01):
        w = self.w
        rho = self.rho
        options = self.options
        chiABN = options['chiAB']*options['ndeg']
        chiBCN = options['chiBC']*options['ndeg']
        chiACN = options['chiAC']*options['ndeg']
        
        w[1] *= 1 - alpha
        w[1] += alpha*chiABN*rho[1]
        w[1] += alpha*chiACN*rho[2]
        w[1] += alpha*w[0]

        w[2] *= 1 - alpha
        w[2] += alpha*chiABN*rho[0]
        w[2] += alpha*chiBCN*rho[2]
        w[2] += alpha*w[0]

        w[3] *= 1 - alpha
        w[3] += alpha*chiACN*rho[0]
        w[3] += alpha*chiBCN*rho[1]
        w[3] += alpha*w[0]
        
    def update_field_new(self, alpha=0.01):
        w = self.w
        rho = self.rho
        options = self.options
        chiABN = options['chiAB']*options['ndeg']
        chiBCN = options['chiBC']*options['ndeg']
        chiACN = options['chiAC']*options['ndeg']
        
        w1 = chiABN*rho[1] + chiACN*rho[2] + w[0] - w[1]
        w1 = np.fft.fftn(w1)
        w1[0,0] = 0
        w[1] = np.fft.ifftn(np.fft.fftn(w[1])+alpha*w1).real
        
        w2 = chiABN*rho[0] + chiBCN*rho[2] + w[0] - w[2]
        w2 = np.fft.fftn(w2)
        w2[0,0] = 0
        w[2] = np.fft.ifftn(np.fft.fftn(w[2])+alpha*w2).real
        
        w3 = chiACN*rho[0] + chiBCN*rho[1] + w[0] - w[3]
        w3 = np.fft.fftn(w3)
        w3[0,0] = 0
        w[3] = np.fft.ifftn(np.fft.fftn(w[3])+alpha*w3).real

    def compute_wplus(self):
        w = self.w
        options = self.options
        chiAB = options['chiAB']
        chiBC = options['chiBC']
        chiAC = options['chiAC']

        Ndeg = options['ndeg']

        XA = chiBC*(chiAB + chiAC - chiBC)
        XB = chiAC*(chiBC + chiAB - chiAC)
        XC = chiAB*(chiAC + chiBC - chiAB)

        w[0] = XA*w[1] + XB*w[2] + XC*w[3]
        w[0]/= XA + XB + XC

        w[0] -= 2*Ndeg*chiAB*chiBC*chiAC/(XA + XB + XC)


    def compute_energe(self):
        options = self.options
        chiABN = options['chiAB']*options['ndeg']
        chiBCN = options['chiBC']*options['ndeg']
        chiACN = options['chiAC']*options['ndeg']
        w = self.w
        rho = self.rho
        nABC = options['nABCblend']
        nBC = options['nBCblend']
        fBC = options['fBC']
        E = chiABN*rho[0]*rho[1] 
        E += chiBCN*rho[1]*rho[2]
        E += chiACN*rho[0]*rho[2]
        E -= w[1]*rho[0]
        E -= w[2]*rho[1]
        E -= w[3]*rho[2]
        #E -= w[0]*(1 - rho.sum(axis=0))
        E = np.fft.ifftn(E)
        self.H = np.real(E.flat[0])
        self.H -= np.log(self.Q[0])*nABC/(nABC + fBC*nBC)
        self.H -= np.log(self.Q[1])*nBC/(nABC + fBC*nBC)

    def compute_gradient(self):
        w = self.w
        rho = self.rho
        options = self.options
        chiABN = options['chiAB']*options['ndeg']
        chiBCN = options['chiBC']*options['ndeg']
        chiACN = options['chiAC']*options['ndeg']
        self.grad[0] = rho[0] + rho[1] + rho[2] - 1
        self.grad[1] = w[1] - chiABN*rho[1] - chiACN*rho[2] - w[0]
        self.grad[2] = w[2] - chiABN*rho[0] - chiBCN*rho[2] - w[0]
        self.grad[3] = w[3] - chiACN*rho[0] - chiBCN*rho[1] - w[0]

    def compute_propagator(self):

        options = self.options
        w = self.w
        qf = self.qf
        cqf = self.cqf
        qb = self.qb
        cqb = self.cqb

        start = 0
        F = [w[1], w[2], w[3], w[2], w[3]]

        np.set_printoptions(precision=15, suppress=True) 
#         input("input")
        

        #print(self.qf.dtype)
        #print(self.qb.dtype)

        #for i in range(4):
        #    print(F[i].dtype)

        for i in range(3):
            NL = self.timelines[i].number_of_time_levels()
            #self.pdesolvers[i].initialize(self.qf[start:start + NL], F[i])
            #self.pdesolvers[i].solve(self.qf[start:start + NL])
            self.pdesolvers[i].BDF4(self.qf[start:start + NL], F[i])
            start += NL - 1
        start = 0
        for i in range(3, 5, 1):
            NL = self.timelines[i].number_of_time_levels()
            self.pdesolvers[i].BDF4(self.cqf[start:start + NL], F[i])
            start += NL - 1
        #print("qf", self.qf[-1])
#         input("input")
         

        start = 0
        for i in range(2, -1, -1):
            NL = self.timelines[i].number_of_time_levels()
            #self.pdesolvers[i].initialize(self.qb[start:start + NL], F[i])
            #self.pdesolvers[i].solve(self.qb[start:start + NL])
            self.pdesolvers[i].BDF4(self.qb[start:start + NL], F[i])
            start += NL - 1
        start = 0
        for i in range(4, 2, -1):
            NL = self.timelines[i].number_of_time_levels()
            self.pdesolvers[i].BDF4(self.cqb[start:start + NL], F[i])
            start += NL -1

        #print("qb", self.qb[-1])


#-----------------------------------------???????????????????????????
    def compute_single_Q(self, index=-1):
        q = self.qf[index]
        q = np.fft.ifftn(q)
        cq = self.cqf[index]
        cq = np.fft.ifftn(cq)
        self.Q[1] = np.real(cq.flat[0])
        self.Q[0] = np.real(q.flat[0])
        return self.Q

    def test_compute_single_Q(self, index, rdir):
        q = np.zeros(self.TNL)
        for i in range(self.TNL):
            q[i] = self.compute_single_Q(index=i)

        fig = plt.figure()
        axes = fig.gca()
        axes.plot(range(self.TNL), q)
        fig.savefig(rdir + 'Q_' + str(index) +'.png')
        plt.close()

    def compute_density(self):
        options = self.options
        q = self.qf*self.qb[-1::-1]
        cq = self.cqf*self.cqb[-1::-1]

        start = 0
        rho = []

        nABC = options['nABCblend']
        nBC = options['nBCblend']
        fBC = options['fBC']

        #-----------------?????????????????????the same as q
        for i in range(3):
            NL = self.timelines[i].number_of_time_levels()
            dt = self.timelines[i].current_time_step_length()
            rho.append(self.integral_time(q[start:start+NL], dt))
            start += NL - 1
        start = 0
        for i in range(3, 5, 1):
            NL = self.timelines[i].number_of_time_levels()
            dt = self.timelines[i].current_time_step_length()
            rho.append(self.integral_time(cq[start:start+NL], dt))
            start += NL - 1


        self.rho[0] = rho[0]*nABC/(nABC + fBC*nBC)
        self.rho[1] = rho[1]*nABC/(nABC + fBC*nBC)
        self.rho[2] = rho[2]*nABC/(nABC + fBC*nBC)

        self.rho[3] = rho[3]*nBC/(nABC  + fBC*nBC)
        self.rho[4] = rho[4]*nBC/(nABC  + fBC*nBC)
        #self.rho /= self.Q[0]

        self.rho[0] /= self.Q[0]
        self.rho[1] /= self.Q[0]
        self.rho[2] /= self.Q[0]
        self.rho[3] /= self.Q[1]
        self.rho[4] /= self.Q[1]

        self.rho[1] = self.rho[1] + self.rho[3]
        self.rho[2] = self.rho[2] + self.rho[4]


        #print("densityA", self.rho[0])
        #print("densityB", self.rho[1])
        #print("densityC", self.rho[2])

    def integral_time(self, q, dt):
        f = -0.625*(q[0] + q[-1]) + 1/6*(q[1] + q[-2]) - 1/24*(q[2] + q[-3])
        f += np.sum(q, axis=0)
        f *= dt
        return f

    def save_data(self, fname='rho.mat'):
        import scipy.io as sio

        rhoA = self.rho[0]
        rhoB = self.rho[1]
        rhoC = self.rho[2]
        data = {
                'rhoA':rhoA,
                'rhoB':rhoB,
                'rhoC':rhoC
                }
        sio.savemat(fname, data)

if __name__ == "__main__":
    import sys 
    rdir = sys.argv[1]
    rhoA = init_value['C42A']
    rhoB = init_value['C42B']
    rhoC = init_value['C42C']
    box = np.array([[4.1, 0], [0, 4.1]], dtype=np.float)
    fC  = 0.1
    fA = 0.5
    fB  = 0.5
    options = model_options(box=box, NS=64, fA=fA, fB=fB, fC=fC)
    model = SCFTA1BA2CLinearModel(options=options)
    rho = [ model.space.fourier_interpolation(rhoA), 
            model.space.fourier_interpolation(rhoB), 
            model.space.fourier_interpolation(rhoC) 
            ]
    model.init_field(rho)


    if True:
        for i in range(1):
            print("step:", i)
            model.compute()
            #model.test_compute_single_Q(i, rdir)
            ng = list(map(model.space.function_norm, model.grad))
            print("l2 norm of grad:", ng)
            #model.update_field()

            fig = plt.figure()
            for j in range(4):
                axes = fig.add_subplot(2, 2, j+1)
                im = axes.imshow(model.w[j])
                fig.colorbar(im, ax=axes)
            fig.savefig(rdir + 'w_' + str(i) +'.png')

            fig = plt.figure()
            for j in range(3):
                axes = fig.add_subplot(1, 3, j+1)
                im = axes.imshow(model.rho[j])
                fig.colorbar(im, ax=axes)
            fig.savefig(rdir + 'rho_' + str(i) +'.png')
            plt.close()
