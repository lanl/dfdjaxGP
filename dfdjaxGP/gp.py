import numpy     as np
import jax.numpy as jnp

import jax
import numpyro

import numpyro.distributions as dist
from numpyro.infer import (
    MCMC,
    NUTS,
    init_to_median,
    init_to_sample
)

from scipy.optimize import minimize
from jax import grad, jit, vmap, random, jacfwd, jacrev
from typing import Callable

from jax.config import config
config.update("jax_enable_x64", True)




def matern52_1D(x: jnp.float64, y: jnp.float64, corr_len, marginal_variance) -> float:
    d = jnp.sqrt((x - y)**2)
    return marginal_variance * (1 + jnp.sqrt(5.)*d/corr_len + 5.*d**2/3./corr_len**2) *jnp.exp(-jnp.sqrt(5.) * d/corr_len)

def matern52_2D(x: jnp.float64, y: jnp.float64, corr_len, marginal_variance) -> float:
    d = jnp.sqrt((x1 - y1)**2 + (x2 - y2)**2)
    return marginal_variance * (1 + jnp.sqrt(5.)*d/corr_len + 5.*d**2/3./corr_len**2) *jnp.exp(-jnp.sqrt(5.) * d/corr_len)

def sq_exp_1D(x: jnp.float64, y: jnp.float64, corr_len, marginal_variance) -> float:
    return marginal_variance * jnp.exp(-jnp.sum((x/corr_len - y/corr_len) ** 2))

def sq_exp_2D(x1: jnp.float64, x2: jnp.float64, y1: jnp.float64, y2: jnp.float64, corr_len, marginal_variance) -> float:
    return marginal_variance * jnp.exp(-jnp.sum((x1/corr_len - y1/corr_len) ** 2) - jnp.sum((x2/corr_len - y2/corr_len) ** 2))




class JaxDerviativeGP:
    def __init__(self, X, y, covariance_function="sqexp"):
        self.n_dims = 1 if len(X.shape) == 1 else X.shape[1]
        
        if covariance_function == "sqexp":
            if self.n_dims == 1:
                self.cov_f = sq_exp_1D
            elif self.n_dims == 2:
                self.cov_f = sq_exp_2D
            else:
                print("No implementation for dimensions other than 1 or 2")
        elif covariance_function == "matern52":
            if self.n_dims == 1:
                self.cov_f = matern52_1D
            elif self.n_dims == 2:
                self.cov_f = matern52_2D
            else:
                print("No implementation for dimensions other than 1 or 2")
        else:
            print("Covariance function ", covariance_function, " not found!")
        
        # Starting with zero mean GP only. Will adjust later.
        self.mean       = 0.
        self.train_X    = X
        self.train_y    = y
        self.train_n    = y.size
                
        self.obs_w      = None
        self.obs_cov    = None
        self.obs_inv    = None
        self.samples    = None
        
        self.nugget     = 1.e-4
        self.corr_len   = 0.5
        self.marg_var   = 1.
        
    def get_cov_mat(self, x1, x2, corr_len, marg_var, covfn=None):
        if covfn==None:
            covfn = self.cov_f
        if self.n_dims == 1:
            return jax.vmap(lambda x1: jax.vmap(lambda y1: covfn(x1, y1, corr_len, marg_var))(x2))(x1)
        else:
            return jax.vmap(lambda x1: jax.vmap(lambda y1: covfn(x1[0], x1[1], y1[0], y1[1], corr_len, marg_var))(x2))(x1)
        
    def get_mixed_cov(self, x1, x2, d1, d2, corr_len, marg_var):
        if self.n_dims == 1:
            if d1 == 0:
                crossfn = self.cov_f
            else:
                eval_s  = "".join(["grad("]*d1) + "self.cov_f" + "".join([", argnums=0)"]*d1)
                crossfn = eval(eval_s)
            if d2 == 0:
                covfn   = crossfn
            else:
                eval_s  = "".join(["grad("]*d2) + "crossfn" + "".join([", argnums=1)"]*d2)
                covfn = eval(eval_s)
            return jax.vmap(lambda x1: jax.vmap(lambda y1: covfn(x1, y1, corr_len, marg_var))(x2))(x1)
        else:
            raise ValueError("Not implemented for higher than 1D input yet.")
        
    def bayes_train(self, n_warmup = 500, n_samples=1000, n_chains=4):
        def model(X, y):
            var    = numpyro.sample("marg_var", dist.LogNormal(0.0, 1.0))
            length = numpyro.sample("corr_len", dist.LogNormal(0.0, 1.0))

            k  = self.get_cov_mat(X, X, length, var)
            k += np.eye(X.shape[0])*1.e-4

            # sample Y according to the standard gaussian process formula
            numpyro.sample(
                "Y",
                dist.MultivariateNormal(loc=jnp.zeros(X.shape[0]), covariance_matrix=k),
                obs=y,
            )

        rng_key, rng_key_predict = random.split(random.PRNGKey(0))
        init_strategy = init_to_median(num_samples=10)
        kernel        = NUTS(model, init_strategy=init_strategy)
        mcmc = MCMC(
                kernel,
                num_warmup   = n_warmup,
                num_samples  = n_samples,
                num_chains   = n_chains,
                progress_bar = True,
            )
        mcmc.run(rng_key, self.train_X, self.train_y)
        mcmc.print_summary()
        self.samples = mcmc.get_samples() 
    
    def train(self):
        def neg_marginal_loglikelihood(params):
            corr_len = jnp.exp(params[0])
            marg_var = jnp.exp(params[1])
            obs_cov  = self.get_cov_mat(self.train_X, self.train_X, corr_len, marg_var)
            obs_cov  = obs_cov + np.eye(self.train_X.shape[0])*self.nugget
            
            _, logdet = jnp.linalg.slogdet(obs_cov)
            return 0.5 * logdet + 0.5 * self.train_y.T @ jnp.linalg.inv(obs_cov) @ self.train_y
        
        j_nmll  = jit(neg_marginal_loglikelihood)
        j_nmllg = jit(grad(neg_marginal_loglikelihood))
        
        opts    = [minimize(j_nmll, np.random.randn(2), method="Newton-CG", jac=j_nmllg) for ii in range(10)]
        opt_out = opts[np.argmin([x.fun for x in opts])]
        
        self.corr_len = np.exp(opt_out.x[0])
        self.marg_var = np.exp(opt_out.x[1])
        
        self.obs_cov = self.get_cov_mat(self.train_X, self.train_X, self.corr_len, self.marg_var)
        self.obs_cov = self.obs_cov + np.eye(self.train_n)*self.nugget
        self.obs_inv = jnp.linalg.inv(self.obs_cov)
        self.obs_w   = self.obs_inv @ self.train_y
        
    def predict(self, X, corr_len=None, marg_var=None, derivative=None, return_sample=False):
        if corr_len == None:
            corr_len = self.corr_len
        if marg_var == None:
            marg_var = self.marg_var
        
        if self.obs_w == None:
            print("Train first before predicting!")
            return 0
        if derivative == None:
            covfn     = self.cov_f
            crossfn   = self.cov_f
        elif self.n_dims == 1:
            if "y" in derivative:
                print("No y derivative for 1D data")
                return None
            if derivative == "df/dx":
                crossfn   = grad(self.cov_f, argnums=0)
                covfn     = grad(crossfn,    argnums=1)
            else: 
                d_order = int(derivative[-1])
                eval_s  = "".join(["grad("]*d_order) + "self.cov_f" + "".join([", argnums=0)"]*d_order)
                crossfn = eval(eval_s)
                eval_s  = "".join(["grad("]*d_order) + "crossfn" + "".join([", argnums=1)"]*d_order)
                covfn   = eval(eval_s)
        else:
            if "y" in derivative:
                arg1 = "1"
                arg2 = "3"
            elif "x" in derivative:
                arg1 = "0"
                arg2 = "2"
            else:
                print("Derivative needs to be an x or y derivative")
                return 0
            
            if derivative == "df/dx":
                crossfn = grad(self.cov_f, argnums=0)
                covfn   = grad(crossfn,    argnums=2)
            elif derivative == "df/dy":
                crossfn = grad(self.cov_f, argnums=1)
                covfn   = grad(crossfn,    argnums=3)
            else: 
                d_order = int(derivative[-1])
                eval_s  = "".join(["grad("]*d_order) + "self.cov_f" + "".join([", argnums=" + arg1 + ")"]*d_order)
                crossfn = eval(eval_s)
                eval_s  = "".join(["grad("]*d_order) + "crossfn" + "".join([", argnums=" + arg2 + ")"]*d_order)
                covfn   = eval(eval_s)
            
        cross_cov = self.get_cov_mat(X, self.train_X, corr_len, marg_var, covfn=crossfn)
        prior_cov = self.get_cov_mat(X, X, corr_len, marg_var, covfn=covfn)
        
        y_hat = cross_cov @ self.obs_w
        p_cov = prior_cov - cross_cov @ self.obs_inv @ cross_cov.T          
        upper = y_hat + 2*np.sqrt(np.diag(p_cov))
        lower = y_hat - 2*np.sqrt(np.diag(p_cov))
        
        return y_hat, p_cov, upper, lower

    def bpredict(self, X, corr_len, marg_var, derivative=None):
        obs_cov = self.get_cov_mat(self.train_X, self.train_X, corr_len, marg_var)
        obs_cov = obs_cov + np.eye(self.train_n)*self.nugget
        obs_inv = jnp.linalg.inv(obs_cov)
        obs_w   = obs_inv @ self.train_y
        
        if derivative == None:
            covfn     = self.cov_f
            crossfn   = self.cov_f
        elif self.n_dims == 1:
            if "y" in derivative:
                print("No y derivative for 1D data")
                return None
            if derivative == "df/dx":
                crossfn   = grad(self.cov_f, argnums=0)
                covfn     = grad(crossfn,    argnums=1)
            else: 
                d_order = int(derivative[-1])
                eval_s  = "".join(["grad("]*d_order) + "self.cov_f" + "".join([", argnums=0)"]*d_order)
                crossfn = eval(eval_s)
                eval_s  = "".join(["grad("]*d_order) + "crossfn" + "".join([", argnums=1)"]*d_order)
                covfn = eval(eval_s)
        else:
            if "y" in derivative:
                arg1 = "1"
                arg2 = "3"
            elif "x" in derivative:
                arg1 = "0"
                arg2 = "2"
            else:
                print("Derivative needs to be an x or y derivative")
                return 0
            
            if derivative == "df/dx":
                crossfn = grad(self.cov_f, argnums=0)
                covfn   = grad(crossfn,    argnums=2)
            elif derivative == "df/dy":
                crossfn = grad(self.cov_f, argnums=1)
                covfn   = grad(crossfn,    argnums=3)
            else: 
                d_order = int(derivative[-1])
                eval_s  = "".join(["grad("]*d_order) + "self.cov_f" + "".join([", argnums=" + arg1 + ")"]*d_order)
                crossfn = eval(eval_s)
                eval_s  = "".join(["grad("]*d_order) + "crossfn" + "".join([", argnums=" + arg2 + ")"]*d_order)
                covfn = eval(eval_s)
            
        cross_cov = self.get_cov_mat(X, self.train_X, corr_len, marg_var, covfn=crossfn)
        prior_cov = self.get_cov_mat(X, X, corr_len, marg_var, covfn=covfn)
        
        y_hat = cross_cov @ obs_w
        p_cov = prior_cov - cross_cov @ obs_inv @ cross_cov.T + np.eye(prior_cov.shape[0]) * 1.e-4
                                
        return y_hat + np.linalg.cholesky(p_cov) @ np.random.randn(p_cov.shape[0])

    def bayes_predict(self, X, derivative=None):
        print("This is currently really slow - ")
        print(" a minute or two for predicting 50 points from 4000 samples with 20 observations.")
        print("Just a heads up.")
        predictions = np.array([self.bpredict(X, 
                                             self.samples["corr_len"][ii], 
                                             self.samples["marg_var"][ii], 
                                             derivative=derivative) for ii in range(self.samples["corr_len"].size)])
        means = np.mean(predictions, axis=0)
        lower = np.percentile(predictions,  2.5, axis=0)
        upper = np.percentile(predictions, 97.5, axis=0)
        return predictions, means, lower, upper
    
    def deriv_properties(s):
        print("Function in progress")
        return (1 if "y" in s else 0, 1 if len(s)==1 else int(s[-1]))
