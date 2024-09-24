
from math import log

# logistic regression case
lambda_ = 0.001
b_teta = 0.25
b_y = 1.1
L = 1
k = 1
k_y = 1
e = 10
r = 1
eta = 1
T = e*r

def func_IR_A(batch_size):
	IR_A1 = 4*(b_teta**2)*(L**2)*(e**2)*T*(eta**2)/batch_size
	IR_A2 = 8*(b_teta*k + b_y*k_y)*b_teta*L*(e**2)*eta/batch_size
	IR_A3 = 4*((b_teta*k+b_y*k_y)**2)*e
	IR_A = (IR_A1 + IR_A2 + IR_A3)**(1/2)
	return IR_A

def func_IR_B(batch_size):
	IR_B1 = 4*(L**2)*(e**2)*T*(eta**2)/batch_size
	IR_B2 = 8*k*L*(e**2)*eta/batch_size
	IR_B3 = 4*(k**2)*e
	IR_B = (IR_B1 + IR_B2 + IR_B3)**(1/2)
	return IR_B

def noise_std_A(epsilon, delta, IR_A):
	c = (2*log(1.25/delta))**(1/2)
	std_A = c*IR_A/epsilon
	return std_A

def noise_std_B(epsilon, delta, IR_B):
	c = (2*log(1.25/delta))**(1/2)
	std_B = c*IR_B/epsilon
	return std_B