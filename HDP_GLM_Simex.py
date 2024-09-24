
import warnings

import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

from statsmodels.tools.sm_exceptions import ConvergenceWarning
from sklearn.preprocessing import MinMaxScaler

from load_data import DataLoader
from IR_Computations import func_IR_A, func_IR_B, noise_std_A, noise_std_B

warnings.simplefilter('ignore', ConvergenceWarning)

class VerticalGLMs:

	def __init__(self, dataname, family, n_partitions = 2):

		self.dataset = dataname
		self.rng = np.random.default_rng(10)
		self.family = family

		self.data = DataLoader(dataname)
		self.data.X_train_parts = np.array_split(self.data.X_train, n_partitions, axis=1)

		self.curr_params = self.initialize_params(family) 

		self.n_models = len(self.partial_models)
		self.n_obs = np.size(self.data.y_train)
		self.n_params = self.n_params_cum[-1]

	def initialize_params(self, family):
		params = []
		self.partial_models = []
		for partial_X_train in self.data.X_train_parts:
			partial_model = sm.GLM(self.data.y_train, partial_X_train, family=family)
			self.partial_models.append(partial_model)
			params.extend(partial_model.fit_regularized(alpha=0.001, L1_wt=0.0).params)

		self.n_params_cum = np.cumsum([0] + [np.shape(pm.exog)[1] for pm in self.partial_models])
		a, b, c = self.n_params_cum[0], self.n_params_cum[1], self.n_params_cum[2]
		params = np.array(params)
		#print(list(params))
		params[a:b] /= max(1, np.sqrt(np.sum(params[a:b]**2)))
		params[b:c] /= max(1, np.sqrt(np.sum(params[b:c]**2)))
		#print(list(params))
		return params

	def fit(self, epsilon=10, simex_adjustment = False, simex_reps = 10, max_epochs = 1000):

		self.simex_adjustment = simex_adjustment
		self.stopped = None
		self.params_history = []

		IR_A = func_IR_A(len(self.data.y_train))
		IR_B = func_IR_B(len(self.data.y_train))
		value_noise_std_A = noise_std_A(epsilon, 0.01, IR_A)
		value_noise_std_B = noise_std_B(epsilon, 0.01, IR_B)
		X_A = self.compute_X_A()
		X_B = self.compute_X_B()
		pm_A = self.partial_models[0]

		a, b, c = self.n_params_cum[0], self.n_params_cum[1], self.n_params_cum[2]
		self.curr_params = np.array([0.0 for _ in range(self.n_params)])
		for i in range(max_epochs):
			self.curr_epoch = i + 1
			lp_B = self.passive_party_B_lp()
			X_with_lp_B = np.hstack((pm_A.exog, np.array([lp_B]).T))
			model_A = sm.GLM(self.data.y_train, X_with_lp_B, family = self.family)
			score_factor = model_A.score_factor(np.array([*self.curr_params[a:b], 1]))
			#score_obs = score_factor[:, None] * self.data.X_train
			gradient = np.dot(score_factor, self.data.X_train)

			self.curr_params += gradient/self.n_obs - 0.001*self.curr_params
			self.curr_params[a:b] /= max(1, np.sqrt(np.sum(self.curr_params[a:b]**2)))
			self.curr_params[b:c] /= max(1, np.sqrt(np.sum(self.curr_params[b:c]**2)))
			self.params_history.append(self.curr_params.copy())

		model_final = sm.GLM(self.data.y_train, self.data.X_train, family = self.family)
		return model_final.predict(self.params_history[-1])

	def passive_party_B_lp(self):
		pm = self.partial_models[1]
		b, c = self.n_params_cum[1], self.n_params_cum[2]
		lp_B = pm.exog @ self.curr_params[b:c]
		return lp_B

	def compute_X_A(self): return self.partial_models[0].exog

	def compute_X_B(self): return self.partial_models[1].exog
