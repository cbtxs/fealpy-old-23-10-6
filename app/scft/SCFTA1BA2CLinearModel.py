importnumpy as np
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
        nblend = 1,
        nblock = 4,
        ndeg = 100,
        fA1 = 0.25,
        fB = 0.25,
        fA2 = 0.25,
        fC = 0.25,
        chiAB = 0.80,
        chiAC = 0.80,
        chiBC = 0.80,
        box = np.diag(2*[2*np.pi]),
        NS = 256,
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
                'nblock': nblock,
                'ndeg': ndeg,
                'fA1': fA1,
                'fA2': fA2,
                'fB': fB,
                'fC': fC,
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


class SCFTA1BA2CLinearModel():
    def __init__(self, options=None):
        if options == None:
            options = pscftmodel_options()
        self.options = options
        dim = options['dim']
        box = options['box'] 
        self.space = FourierSpace(box,  options['NS'])

        fA1 = options['fA1']
        fB  = options['fB']
        fA2 = options['fA2']
        fC  = options['fC']
        maxdt = options['maxdt']

        self.timelines = []
        self.timelines.append(UniformTimeLine(0, fA1, int(np.ceil(fA1/maxdt))))
        self.timelines.append(UniformTimeLine(0, fB,  int(np.ceil(fB/maxdt))))
        self.timelines.append(UniformTimeLine(0, fA2, int(np.ceil(fA2/maxdt))))
        self.timelines.append(UniformTimeLine(0, fC,  int(np.ceil(fC/maxdt))))


        self.pdesolvers = []
        for i in range(4):
            self.pdesolvers.append(
                    ParabolicFourierSolver(self.space, self.timelines[i])
                    )

        self.TNL = 0 # total number of time levels
        for i in range(4):
            NL = self.timelines[i].number_of_time_levels()
            self.TNL += self.timelines[i].number_of_time_levels()
        self.TNL -= options['nblock'] - 1

        self.qf = self.space.function(dim=self.TNL) # forward  propagator 
        self.qb = self.space.function(dim=self.TNL) # backward propagator

        self.qf[0] = 1
        self.qb[0] = 1

        self.rho = self.space.function(dim=options['nspecies'])
        self.grad = self.space.function(dim=options['nspecies']+1)
        self.w = self.space.function(dim=options['nspecies']+1)
        self.Q = np.zeros(options['nblend'], dtype=np.float)

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
        print('a')
        # solver the forward and backward equation
        import time
        start1 =time.clock()
        self.compute_propagator()
        end =time.clock()
        print('Running timeq: %s Seconds'%(end-start1))
        start1 =time.clock()
        self.compute_single_Q()
        end =time.clock()
        print('Running timeQ: %s Seconds'%(end-start1))
# compute single chain partition function Q
        # compute density
        print("Q:", self.Q)

        start1 =time.clock()
        self.compute_density()
        end =time.clock()
        print('Running time phi: %s Seconds'%(end-start1))

        # compute energy function and its gradient
        start1 =time.clock()
        self.update_field()
        end =time.clock()
        print('Running time_field: %s Seconds'%(end-start1))

        start1 =time.clock()
        self.compute_wplus()
        end =time.clock()
        print('Running time w+: %s Seconds'%(end-start1))

        start1 =time.clock()
        self.compute_energe()
        end =time.clock()
        print('Running time H: %s Seconds'%(end-start1))

        start1 =time.clock()
        self.compute_gradient()
        end =time.clock()
        print('Running time grad: %s Seconds'%(end-start1))


    def update_field(self, alpha=0.01):
        """
        Parameters
        ----------


        References
        ----------

        Notes
        -----

        """
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

        XA = chiBC*(chiAB + chiAC - chiBC)
        XB = chiAC*(chiBC + chiAB - chiAC)
        XC = chiAB*(chiAC + chiBC - chiAB)

        w[0] = XA*w[1] + XB*w[2] + XC*w[3]
        w[0]/= XA + XB + XC


    def compute_energe(self):
        options = self.options
        chiABN = options['chiAB']*options['ndeg']
        chiBCN = options['chiBC']*options['ndeg']
        chiACN = options['chiAC']*options['ndeg']
        w = self.w
        rho = self.rho
        E = chiABN*rho[0]*rho[1] 
        E += chiBCN*rho[1]*rho[2]
        E += chiACN*rho[0]*rho[2]
        E -= w[1]*rho[0]
        E -= w[2]*rho[1]
        E -= w[3]*rho[2]
        #E -= w[0]*(1 - rho.sum(axis=0))
        E = np.fft.ifftn(E)
        self.H = np.real(E.flat[0])
        self.H -= np.log(self.Q[0])

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
        qb = self.qb

        start = 0
        F = [w[1], w[2], w[1], w[3]]

        np.set_printoptions(precision=15, suppress=True) 
#         input("input")

        #print(self.qf.dtype)
        #print(self.qb.dtype)

        #for i in range(4):
        #    print(F[i].dtype)

        for i in range(options['nblock']):
            NL = self.timelines[i].number_of_time_levels()
            #self.pdesolvers[i].initialize(self.qf[start:start + NL], F[i])
            #self.pdesolvers[i].solve(self.qf[start:start + NL])
            import time
            start1 =time.clock()
            self.pdesolvers[i].BDF4(self.qf[start:start + NL], F[i])
            end =time.clock()
            print('Running time: %s Seconds'%(end-start1))
            start += NL - 1
        print("qf", self.qf[-1])
#         input("input")

        start = 0
        for i in range(options['nblock']-1, -1,-1):
            NL = self.timelines[i].number_of_time_levels()
            #self.pdesolvers[i].initialize(self.qb[start:start + NL], F[i])
            #self.pdesolvers[i].solve(self.qb[start:start + NL])
            self.pdesolvers[i].BDF4(self.qb[start:start + NL], F[i])
            start += NL - 1
        #print("qb", self.qb[-1])


    def compute_single_Q(self, index=-1):
        q = self.qf[index]
        q = np.fft.ifftn(q)
        self.Q[0] = np.real(q.flat[0])
        return self.Q[0]

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

        start = 0
        rho = []
        for i in range(options['nblock']):
            NL = self.timelines[i].number_of_time_levels()
            dt = self.timelines[i].current_time_step_length()
            rho.append(self.integral_time(q[start:start+NL], dt))
            start += NL - 1
        self.rho[0] = rho[0] + rho[2]
        self.rho[1] = rho[1]
        self.rho[2] = rho[3]
        self.rho /= self.Q[0]

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
    fC  = 0.14
    fA2 = 0.29
    fB  = 0.118
    fA1 = 1-fA2-fB-fC
    options = model_options(box=box, NS=64, fA1=fA1, fB=fB, fA2=fA2, fC=fC)
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
