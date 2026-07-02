import math
import numpy as np
from scipy.stats import unitary_group,ortho_group
from scipy.linalg import eigh, expm
import matplotlib.pyplot as plt
from scipy.optimize import minimize

#Define Pauli basis
Id = np.array([[1,0],[0,1]])
X = np.array([[0,1], [1, 0]])
Y = np.array([[0, -1j], [1j, 0]])
Z = np.array([[1,0], [0, -1]])

basis = [Id, X, Y, Z]

IdId = np.einsum("ac,bd->abcd", Id, Id)

def get_V_core(J):
    exp_V = -1j * (np.pi/4 * np.kron(X, X) + np.pi/4 * np.kron(Y, Y) + J * np.kron(Z, Z))
    return expm(exp_V)

def uni_gate(J, u_plus, u_minus, v_plus, v_minus, phi=0.0): #universal q=2 self dual gate
    V_J = get_V_core(J)

    left_rot = np.kron(u_plus, u_minus)
    
    right_rot = np.kron(v_minus, v_plus) 
    
    U_matrix = np.exp(1j * phi) * (left_rot @ V_J @ right_rot)
    
    return U_matrix.reshape(2, 2, 2, 2)

def SDKI_main(h):
    J = 0
    v_plus = expm(1j * np.pi/4 * Y)
    v_minus = expm(1j * np.pi/4 * Y) @ expm(-1j * h * Z)
    
    u_minus = expm(-1j * np.pi/4 * X) @ expm(-1j * np.pi/4 * Y)
    u_plus = expm(-1j * h * Z) @ u_minus
    
    return uni_gate(J, u_plus, u_minus, v_plus, v_minus)

#define contractions of tensors or vectors
def inner_product(vec_1, vec_2):
    assert(vec_1.shape == vec_2.shape)
    num_indices = len(vec_1.shape)
    return np.tensordot(vec_1.conj(), vec_2, axis=num_indices)

def mult_tens_vec(tens,vec):
    #Apply tensor to vector
    assert(vec.shape + vec.shape == tens.shape)
    n_ind = len(vec.shape)
    vec_cont = np.einsum(tens, list(range(1,2*n_ind+1)), vec, list(range(n_ind+1,2*n_ind+1)))
    return vec_cont

#cast tensors to matrix
def tensor_to_matrix(tensor):
    #Given tensor T_abcd...efgh... returns matrix M_{abcd..., efgh...}
    num_row_indices = len(tensor.shape)//2
    q = tensor.shape[0]
    return tensor.reshape(2 * [q ** num_row_indices]) # just flattens the multidimensional tensor into 2D matrix

def tensors_to_vecs(tensor):
    #given tensor T_abcd...efgh returns a vector V_{abcd... efgh}
    num_row_indices = len(tensor.shape)
    q = tensor.shape[0]
    return tensor.reshape([q ** num_row_indices])

def const_M_plus(U_tensor):
    #first we flatten it for simpler calculation
    U = tensor_to_matrix(U_tensor)

    M_plus = np.zeros((4,4), dtype = complex)

    #constructing M_plus = 1/d tr_1[U.dagg * (a \cross Id) U]
    for j, a_beta in enumerate(basis):
        #prep a_beta \cross Id
        O = np.kron(a_beta , Id)
        #evolution in heisenberg picture
        O_evolved = U.conj().T @ O @ U

        O_reshape = O_evolved.reshape((2, 2, 2, 2))
        trace_1_o = np.trace(O_reshape, axis1=0, axis2=2)
        M_plus_inn = 0.5 * trace_1_o

        for i, a_alpha in enumerate(basis):
            main = 0.5 * np.trace(a_alpha @ M_plus_inn)
            M_plus[i, j] = main
    return M_plus

def corr(M_plus, alpha_idx, beta_idx, t_max):
    #calculation by applying 2t times 
    correlations = np.zeros(t_max + 1, dtype = float)
    for t in range(t_max + 1):
        M_2t = np.linalg.matrix_power(M_plus, 2 * t)
        corr_val = M_2t[alpha_idx, beta_idx]
        correlations[t] = np.real(corr_val)
    return correlations

U_tensor = SDKI_main(1.4)
M_rand = const_M_plus(U_tensor)

t_max = 20
time_steps = np.arange(t_max + 1)

correlation_XX = corr(M_rand, 1, 1, t_max)

print(np.round(np.linalg.eigvals(M_rand), 4))

plt.scatter(time_steps, correlation_XX)
plt.plot(time_steps, correlation_XX)
plt.grid()
plt.axhline(0, color = 'black', linewidth = 1, linestyle = '-')
plt.xlabel('Time Steps')
plt.ylabel('Correlation function')
plt.show()
