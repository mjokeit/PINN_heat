"""
@author: Maziar Raissi
"""

import sys
sys.path.insert(0, '../utilities/')

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from scipy.interpolate import griddata
from plotting import newfig, savefig
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.gridspec as gridspec
import time

np.random.seed(1234)
tf.set_random_seed(1234)

class PhysicsInformedNN:
    # Initialize the class
    def __init__(self, X, u, layers, lb, ub):
        
        self.lb = lb
        self.ub = ub
        
        self.x = X[:,0:1]
        self.t = X[:,1:2]
        self.u = u
        
        self.layers = layers
        
        # Initialize NNs
        self.weights, self.biases = self.initialize_NN(layers)
        
        # tf placeholders and graph
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                                     log_device_placement=True))
        
        # Initialize parameters
        self.lambda_c2 = tf.Variable([0], dtype=tf.float32)
        self.lambda_c1 = tf.Variable([0], dtype=tf.float32)
        self.lambda_c0 = tf.Variable([0], dtype=tf.float32)
        self.lambda_k = tf.Variable([0], dtype=tf.float32)
        
        self.x_tf = tf.placeholder(tf.float32, shape=[None, self.x.shape[1]])
        self.t_tf = tf.placeholder(tf.float32, shape=[None, self.t.shape[1]])
        self.u_tf = tf.placeholder(tf.float32, shape=[None, self.u.shape[1]])
                
        self.u_pred = self.net_u(self.x_tf, self.t_tf) / 800
        self.f_pred = self.net_f(self.x_tf, self.t_tf) / 60994270
        self.s_pred = self.net_s(self.x_tf, self.t_tf) / 60994270
        
        self.loss = tf.reduce_mean(tf.square(self.u_tf - self.u_pred))  + \
                    tf.reduce_mean(tf.square(self.f_pred - self.s_pred)) # 1e-10 good value!

        # self.loss = tf.log(tf.reduce_mean(tf.square(self.u_tf - self.u_pred)) + 1) + \
        #             tf.log(tf.reduce_mean(tf.square(self.f_pred - self.s_pred)) + 1)

        self.optimizer = tf.contrib.opt.ScipyOptimizerInterface(self.loss, 
                                                                method = 'L-BFGS-B', 
                                                                options = {'maxiter': 50000,
                                                                           'maxfun': 50000,
                                                                           'maxcor': 50,
                                                                           'maxls': 50,
                                                                           'ftol' : 1.0 * np.finfo(float).eps})
    
        self.optimizer_Adam = tf.train.AdamOptimizer(0.1)
        self.train_op_Adam = self.optimizer_Adam.minimize(self.loss)
        
        init = tf.global_variables_initializer()
        self.sess.run(init)

    def initialize_NN(self, layers):        
        weights = []
        biases = []
        num_layers = len(layers) 
        for l in range(0,num_layers-1):
            W = self.xavier_init(size=[layers[l], layers[l+1]])
            b = tf.Variable(tf.zeros([1,layers[l+1]], dtype=tf.float32), dtype=tf.float32)
            weights.append(W)
            biases.append(b)        
        return weights, biases
        
    def xavier_init(self, size):
        in_dim = size[0]
        out_dim = size[1]        
        xavier_stddev = np.sqrt(2/(in_dim + out_dim))
        return tf.Variable(tf.truncated_normal([in_dim, out_dim], stddev=xavier_stddev), dtype=tf.float32)
    
    def neural_net(self, X, weights, biases):
        num_layers = len(weights) + 1
        
        H = 2.0*(X - self.lb)/(self.ub - self.lb) - 1.0
        for l in range(0,num_layers-2):
            W = weights[l]
            b = biases[l]
            H = tf.tanh(tf.add(tf.matmul(H, W), b))
        W = weights[-1]
        b = biases[-1]
        Y = tf.add(tf.matmul(H, W), b)
        return Y
            
    def net_u(self, x, t):  
        u = self.neural_net(tf.concat([x,t],1), self.weights, self.biases)
        return u
    
    def net_f(self, x, t):
        lambda_c2 = tf.exp(self.lambda_c2)
        lambda_c1 = tf.exp(self.lambda_c1)
        lambda_c0 = tf.exp(self.lambda_c0)
        lambda_k = tf.exp(self.lambda_k)

        u = self.net_u(x,t)
        u_t = tf.gradients(u, t)[0]
        u_x = tf.gradients(u, x)[0]
        u_xx = tf.gradients(u_x, x)[0]

        # c = lambda_c2 * u**2 - lambda_c1*u + lambda_c0
        c = lambda_c0
        k = lambda_k

        # f = c * u_t - k*u_xx
        f = 5.849 * 10 ** 2 * u_t - 6.856 * u_xx
        
        return f

    def net_s(self, x, t):
        t_max = 0.5
        sigma = 0.02
        u_max = 800

        # computations for the rhs
        p = 0.25 * tf.cos(2 * np.pi * t / t_max) + 0.5
        p_t = tf.gradients(p, t)[0]

        u_sol = u_max * tf.exp(-(x - p) ** 2 / (2 * sigma ** 2))

        # k_sol = 1.29 * 10 ** -2 * u_sol + 6.856
        k_sol = 6.856
        # k_u_sol = 1.29 * 10 ** -2

        # c_sol = 4.55 * 10 ** -4 * u_sol ** 2 - 5.78 * 10 ** -3 * u_sol + 5.849 * 10 ** 2
        c_sol = 5.849 * 10 ** 2

        fac_sigma = 1/(sigma ** 2)

        # rhs = fac_sigma * k_sol * u_sol + u_sol * (x - p) * fac_sigma * (
        #         c_sol * p_t - (x - p) * fac_sigma * (k_sol + u_sol * k_u_sol))
        rhs = fac_sigma * k_sol * u_sol + u_sol * (x - p) * fac_sigma * (
                c_sol * p_t - (x - p) * fac_sigma * (k_sol ))

        s = rhs

        return s
    
    def callback(self, loss, lambda_c2, lambda_c1, lambda_c0, lambda_k):
        print('Loss: %e, c2: %.5f, c1: %.5f, c0: %.5f, k: %.5f' % (loss, lambda_c2, lambda_c1, lambda_c0, lambda_k))
        
        
    def train(self, nIter):

        saver = tf.train.Saver()
        # saver.restore(self.sess, 'models/'+model_string+'_model-iter')
        rest_iter = 0
        tf_dict = {self.x_tf: self.x, self.t_tf: self.t, self.u_tf: self.u}
        
        start_time = time.time()
        for it in range(nIter):
            self.sess.run(self.train_op_Adam, tf_dict)
            if it == nIter-1:
                saver.save(self.sess, 'models/'+model_string+'_model', global_step=nIter+rest_iter)

            # Print
            if it % 10 == 0:
                elapsed = time.time() - start_time
                loss_value = self.sess.run(self.loss, tf_dict)
                lambda_c2_value = self.sess.run(self.lambda_c2)
                lambda_c1_value = self.sess.run(self.lambda_c1)
                lambda_c0_value = self.sess.run(self.lambda_c0)
                lambda_k_value = self.sess.run(self.lambda_k)
                print('It: %d, Loss: %.3e, Lambda_c2: %.3f, Lambda_c1: %.6f, Lambda_c0: %.6f, Lambda_k: %.6f, Time: %.2f' %
                      (it, loss_value, lambda_c2_value, lambda_c1_value, lambda_c0_value, lambda_k_value, elapsed))
                start_time = time.time()
        
        self.optimizer.minimize(self.sess,
                                feed_dict = tf_dict,
                                fetches = [self.loss, self.lambda_c2, self.lambda_c1, self.lambda_c0, self.lambda_k],
                                loss_callback = self.callback)

        saver.save(self.sess, 'models/'+model_string+'_model', global_step=nIter+rest_iter)
        
        
    def predict(self, X_star):
        
        tf_dict = {self.x_tf: X_star[:,0:1], self.t_tf: X_star[:,1:2]}
        
        u_star = self.sess.run(self.u_pred, tf_dict)
        f_star = self.sess.run(self.f_pred, tf_dict)
        
        return u_star, f_star

    
if __name__ == "__main__":

    c2 = 4.55 * 10 ** -4
    c1 = 5.78 * 10 ** -3
    c0 = 5.849 * 10 ** 2
    k = 6.856


    N_u = 4000
    layers = [2, 20, 20, 20, 20, 1]

    model_string = 'heat1D_linear'
    data = scipy.io.loadmat('data/'+model_string+'.mat')
    
    t = data['ts'].flatten()[:,None]
    x = data['xs'].flatten()[:,None]
    Exact = np.real(data['usol']).T
    
    X, T = np.meshgrid(x,t)
    
    X_star = np.hstack((X.flatten()[:,None], T.flatten()[:,None]))
    u_star = Exact.flatten()[:,None]              

    # Doman bounds
    lb = X_star.min(0)
    ub = X_star.max(0)    
    
    ######################################################################
    ######################## Noiseles Data ###############################
    ######################################################################
    noise = 0.0            
             
    idx = np.random.choice(X_star.shape[0], N_u, replace=False)
    X_u_train = X_star[idx,:]
    u_train = u_star[idx,:]
    
    model = PhysicsInformedNN(X_u_train, u_train, layers, lb, ub)
    model.train(10000)
    
    u_pred, f_pred = model.predict(X_star)
            
    error_u = np.linalg.norm(u_star-u_pred,2)/np.linalg.norm(u_star,2)
    
    U_pred = griddata(X_star, u_pred.flatten(), (X, T), method='cubic')
        
    lambda_c2_value = model.sess.run(model.lambda_c2)
    lambda_c1_value = model.sess.run(model.lambda_c1)
    lambda_c0_value = model.sess.run(model.lambda_c0)
    lambda_k_value = model.sess.run(model.lambda_k)
    lambda_c2_value = np.exp(lambda_c2_value)
    lambda_c1_value = np.exp(lambda_c1_value)
    lambda_c0_value = np.exp(lambda_c0_value)
    lambda_k_value = np.exp(lambda_k_value)
    
    error_lambda_c2 = np.abs(lambda_c2_value - c2) / c2 * 100
    error_lambda_c1 = np.abs(lambda_c1_value - c1) / c1 * 100
    error_lambda_c0 = np.abs(lambda_c0_value - c0) / c0 * 100
    error_lambda_k = np.abs(lambda_k_value - k) / k * 100
    
    print('Error u: %e' % (error_u))    
    print('Error c2: %.5f%%' % (error_lambda_c2))
    print('Error c1: %.5f%%' % (error_lambda_c1))
    print('Error c0: %.5f%%' % (error_lambda_c0))
    print('Error k: %.5f%%' % (error_lambda_k))
    
    ######################################################################
    ########################### Noisy Data ###############################
    ######################################################################
    # noise = 0.01
    # u_train = u_train + noise*np.std(u_train)*np.random.randn(u_train.shape[0], u_train.shape[1])
    #
    # model = PhysicsInformedNN(X_u_train, u_train, layers, lb, ub)
    # model.train(10000)
    #
    # u_pred, f_pred = model.predict(X_star)
    #
    # lambda_1_value_noisy = model.sess.run(model.lambda_1)
    # lambda_2_value_noisy = model.sess.run(model.lambda_2)
    # lambda_2_value_noisy = np.exp(lambda_2_value_noisy)
    #
    # error_lambda_1_noisy = np.abs(lambda_1_value_noisy - 1.0)*100
    # error_lambda_2_noisy = np.abs(lambda_2_value_noisy - nu)/nu * 100
    #
    # print('Error lambda_1: %f%%' % (error_lambda_1_noisy))
    # print('Error lambda_2: %f%%' % (error_lambda_2_noisy))

 
    ######################################################################
    ############################# Plotting ###############################
    ######################################################################    
    
    fig, ax = newfig(1.0, 1.4)
    ax.axis('off')
    
    ####### Row 0: u(t,x) ##################    
    gs0 = gridspec.GridSpec(1, 2)
    gs0.update(top=1-0.06, bottom=1-1.0/3.0+0.06, left=0.15, right=0.85, wspace=0)
    ax = plt.subplot(gs0[:, :])
    
    h = ax.imshow(U_pred.T, interpolation='nearest', cmap='rainbow', 
                  extent=[t.min(), t.max(), x.min(), x.max()], 
                  origin='lower', aspect='auto')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(h, cax=cax)
    
    ax.plot(X_u_train[:,1], X_u_train[:,0], 'kx', label = 'Data (%d points)' % (u_train.shape[0]), markersize = 2, clip_on = False)
    
    line = np.linspace(x.min(), x.max(), 2)[:,None]
    ax.plot(t[25]*np.ones((2,1)), line, 'w-', linewidth = 1)
    ax.plot(t[50]*np.ones((2,1)), line, 'w-', linewidth = 1)
    ax.plot(t[75]*np.ones((2,1)), line, 'w-', linewidth = 1)
    
    ax.set_xlabel('$t$')
    ax.set_ylabel('$x$')
    ax.legend(loc='upper center', bbox_to_anchor=(1.0, -0.125), ncol=5, frameon=False)
    ax.set_title('$u(t,x)$', fontsize = 10)
    
    ####### Row 1: u(t,x) slices ##################    
    gs1 = gridspec.GridSpec(1, 3)
    gs1.update(top=1-1.0/3.0-0.1, bottom=1.0-2.0/3.0, left=0.1, right=0.9, wspace=0.5)
    
    ax = plt.subplot(gs1[0, 0])
    ax.plot(x,Exact[25,:], 'b-', linewidth = 2, label = 'Exact')       
    ax.plot(x,U_pred[25,:], 'r--', linewidth = 2, label = 'Prediction')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$u(t,x)$')    
    ax.set_title('$t = 0.25$', fontsize = 10)
    # ax.axis('square')
    ax.set_xlim([-0.1,1.1])
    ax.set_ylim([-80,880])
    
    ax = plt.subplot(gs1[0, 1])
    ax.plot(x,Exact[50,:], 'b-', linewidth = 2, label = 'Exact')       
    ax.plot(x,U_pred[50,:], 'r--', linewidth = 2, label = 'Prediction')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$u(t,x)$')
    # ax.axis('square')
    ax.set_xlim([-0.1,1.1])
    ax.set_ylim([-80,880])
    ax.set_title('$t = 0.50$', fontsize = 10)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.35), ncol=5, frameon=False)
    
    ax = plt.subplot(gs1[0, 2])
    ax.plot(x,Exact[75,:], 'b-', linewidth = 2, label = 'Exact')       
    ax.plot(x,U_pred[75,:], 'r--', linewidth = 2, label = 'Prediction')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$u(t,x)$')
    # ax.axis('square')
    ax.set_xlim([-0.1,1.1])
    ax.set_ylim([-80,880])
    ax.set_title('$t = 0.75$', fontsize = 10)
    
    ####### Row 3: Identified PDE ##################    
    gs2 = gridspec.GridSpec(1, 3)
    gs2.update(top=1.0-2.0/3.0, bottom=0, left=0.0, right=1.0, wspace=0.0)

    ax = plt.subplot(gs2[:, :])
    ax.axis('off')
    s1 = r'$\begin{tabular}{ |c|c| }  \hline Correct PDE & $ %.5f u_t - %.5f u_{xx} = s$ \\  \hline Identified PDE (clean data) & ' % (c0,k)
    s2 = r'$%.5f u_t - %.5f u_{xx} = s$ \\  \hline ' % (lambda_c0_value, lambda_k_value)
    # s3 = r'Identified PDE (1\% noise) & '
    # s4 = r'$u_t + %.5f u u_x - %.7f u_{xx} = 0$  \\  \hline ' % (lambda_1_value_noisy, lambda_2_value_noisy)
    s5 = r'\end{tabular}$'
    # s = s1+s2+s3+s4+s5
    s = s1 + s2 + s5
    ax.text(0.1,0.1,s)
        
    savefig('figures/'+model_string+'_identification')
    



