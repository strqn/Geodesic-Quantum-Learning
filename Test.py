#!/usr/bin/env python3
"""
Comprehensive Noise Robustness Experiment
==================================================
Compares GQE (Corrected Symplectic Geometric), QNG, Adam, SPSA, and SGD 
under various noise models: Noiseless, Depolarizing, Amplitude Damping, Phase Damping.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh
import time
import warnings
import os
warnings.filterwarnings('ignore')

from qiskit.circuit.library import RealAmplitudes
from qiskit.quantum_info import Statevector, SparsePauliOp
from qiskit_aer.noise import (NoiseModel, depolarizing_error,
                               amplitude_damping_error, phase_damping_error)
from qiskit_aer.primitives import Estimator as AerEstimator

# ============================================================================
# إعدادات الرسوم البيانية للبحث العلمي
# ============================================================================
plt.rcParams.update({
    'figure.dpi':      1200,
    'savefig.dpi':     1200,
    'font.size':       11,
    'axes.titlesize':  12,
    'axes.labelsize':  11,
    'legend.fontsize':  9,
})

COLORS  = {'GQE': '#E24A33', 'QNG': '#988ED5', 'Adam': '#2CA02C',
           'SPSA': '#348ABD', 'SGD': '#777777'}
MARKERS = {'GQE': 'o-', 'QNG': 's--', 'Adam': '^-.', 'SPSA': 'd:', 'SGD': 'v-.'}

# ============================================================================
# 1. Hamiltonian Formulation
# ============================================================================
def transverse_field_ising_hamiltonian(n_qubits=8, J=1.0, hx=1.0, hz=0.5):
    zz_terms = []
    for i in range(n_qubits - 1):
        pauli_str = ['I'] * n_qubits
        pauli_str[i] = 'Z'
        pauli_str[i + 1] = 'Z'
        zz_terms.append((''.join(pauli_str), -J))
    x_terms = []
    for i in range(n_qubits):
        pauli_str = ['I'] * n_qubits
        pauli_str[i] = 'X'
        x_terms.append((''.join(pauli_str), -hx))
    z_terms = []
    for i in range(n_qubits):
        pauli_str = ['I'] * n_qubits
        pauli_str[i] = 'Z'
        z_terms.append((''.join(pauli_str), -hz))
    return SparsePauliOp.from_list(zz_terms + x_terms + z_terms)

def exact_ground_state(hamiltonian):
    matrix = hamiltonian.to_matrix()
    eigvals, _ = eigh(matrix)
    return float(np.real(eigvals[0]))

# ============================================================================
# 2. Noise Models Configuration
# ============================================================================
def create_noise_model(noise_type='depolarizing', strength=0.01):
    if noise_type == 'noiseless':
        return None
        
    noise_model = NoiseModel()
    if noise_type == 'depolarizing':
        error_1q = depolarizing_error(strength / 2, 1)
        error_2q = depolarizing_error(strength, 2)
        noise_model.add_all_qubit_quantum_error(error_1q, ['u1', 'u2', 'u3', 'rx', 'ry', 'rz'])
        noise_model.add_all_qubit_quantum_error(error_2q, ['cx', 'cz'])

    elif noise_type == 'amplitude_damping':
        error_1q = amplitude_damping_error(strength)
        error_2q = error_1q.tensor(error_1q)
        noise_model.add_all_qubit_quantum_error(error_1q, ['u1', 'u2', 'u3', 'rx', 'ry', 'rz'])
        noise_model.add_all_qubit_quantum_error(error_2q, ['cx', 'cz'])

    elif noise_type == 'phase_damping':
        error_1q = phase_damping_error(strength)
        error_2q = error_1q.tensor(error_1q)
        noise_model.add_all_qubit_quantum_error(error_1q, ['u1', 'u2', 'u3', 'rx', 'ry', 'rz'])
        noise_model.add_all_qubit_quantum_error(error_2q, ['cx', 'cz'])

    return noise_model

def make_noisy_estimator(noise_model, shots=1024):
    estimator = AerEstimator()
    options = {"shots": shots}
    if noise_model is not None:
        options["backend_options"] = {"noise_model": noise_model}
    estimator.set_options(**options)
    return estimator

# ============================================================================
# 3. Optimizers
# ============================================================================
class BaseOptimizer:
    def __init__(self, hamiltonian, ansatz, estimator):
        self.hamiltonian = hamiltonian
        self.ansatz = ansatz
        self.estimator = estimator

    def expectation(self, parameters):
        try:
            bound = self.ansatz.assign_parameters(parameters)
            job = self.estimator.run([bound], [self.hamiltonian])
            return float(np.real(job.result().values[0]))
        except Exception:
            return 0.0

    def spsa_gradient(self, parameters):
        ck = 0.1
        delta = np.random.choice([-1, 1], size=len(parameters))
        p_plus = parameters + ck * delta
        p_minus = parameters - ck * delta
        try:
            e_plus = self.expectation(p_plus)
            e_minus = self.expectation(p_minus)
            return (e_plus - e_minus) / (2 * ck * delta)
        except Exception:
            return np.zeros_like(parameters)

# ---------------------------------------------------------
# خوارزمية GQE المصححة (Symplectic Geometric Quantum Euler)
# ---------------------------------------------------------
class GQE(BaseOptimizer):
    def __init__(self, hamiltonian, ansatz, eta=0.1, delta_t=0.5, beta=0.8, epsilon=1e-6, estimator=None):
        super().__init__(hamiltonian, ansatz, estimator)
        self.eta = eta
        self.delta_t = delta_t
        self.beta = beta  # معامل حفظ الزخم السيمبلكتي
        self.epsilon = epsilon
        self.velocity = None
        self.pauli_terms = hamiltonian.to_list()

    def expectation_and_variance(self, parameters):
        """يحسب الطاقة والتباين كبديل هندسي رخيص لمصفوفة فيشر"""
        try:
            bound = self.ansatz.assign_parameters(parameters)
            job = self.estimator.run([bound], [self.hamiltonian])
            energy = float(np.real(job.result().values[0]))
            
            var = 0.0
            for pauli_str, coeff in self.pauli_terms:
                coeff_real = float(np.real(coeff))
                if abs(coeff_real) > 1e-10:
                    pauli_op = SparsePauliOp.from_list([(pauli_str, 1.0)])
                    jobp = self.estimator.run([bound], [pauli_op])
                    pexp = float(np.real(jobp.result().values[0]))
                    var += (coeff_real * pexp) ** 2
            var = max(0.0, var - energy ** 2)
            return energy, var
        except Exception:
            return 0.0, 1.0

    def update_parameters(self, parameters):
        grad = self.spsa_gradient(parameters)
        energy, var = self.expectation_and_variance(parameters)
        
        if self.velocity is None:
            self.velocity = np.zeros_like(parameters)
            
        # 1. التكييف الهندسي (Geometric Preconditioning)
        # تصغير الخطوة في الاتجاهات ذات التباين/الانحناء العالي
        geometric_grad = grad / (np.sqrt(var) + self.epsilon)
        
        # 2. تحديث السرعة السيمبلكتي (Symplectic Momentum)
        self.velocity = self.beta * self.velocity - self.eta * geometric_grad
        
        # 3. تحديث الموقع
        new_parameters = parameters + self.delta_t * self.velocity
        
        return new_parameters, energy

class SPSAOptimizer(BaseOptimizer):
    def __init__(self, hamiltonian, ansatz, learning_rate=0.1, estimator=None):
        super().__init__(hamiltonian, ansatz, estimator)
        self.lr = learning_rate

    def update_parameters(self, parameters):
        grad = self.spsa_gradient(parameters)
        energy = self.expectation(parameters)
        return parameters - self.lr * grad, energy

class AdamOptimizer(BaseOptimizer):
    def __init__(self, hamiltonian, ansatz, learning_rate=0.05, beta1=0.9, beta2=0.999, epsilon=1e-8, estimator=None):
        super().__init__(hamiltonian, ansatz, estimator)
        self.lr = learning_rate
        self.beta1, self.beta2, self.epsilon = beta1, beta2, epsilon
        self.m, self.v, self.t = None, None, 0

    def update_parameters(self, parameters):
        self.t += 1
        grad = self.spsa_gradient(parameters)
        energy = self.expectation(parameters)
        
        if self.m is None:
            self.m, self.v = np.zeros_like(parameters), np.zeros_like(parameters)
        self.m = self.beta1 * self.m + (1 - self.beta1) * grad
        self.v = self.beta2 * self.v + (1 - self.beta2) * (grad ** 2)
        m_hat = self.m / (1 - self.beta1 ** self.t)
        v_hat = self.v / (1 - self.beta2 ** self.t)
        
        return parameters - self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon), energy

class SGDOptimizer(BaseOptimizer):
    def __init__(self, hamiltonian, ansatz, learning_rate=0.1, estimator=None):
        super().__init__(hamiltonian, ansatz, estimator)
        self.lr = learning_rate

    def update_parameters(self, parameters):
        grad = self.spsa_gradient(parameters)
        energy = self.expectation(parameters)
        return parameters - self.lr * grad, energy

class QNG(BaseOptimizer):
    def __init__(self, hamiltonian, ansatz, learning_rate=0.1, estimator=None, regularization=1e-6):
        super().__init__(hamiltonian, ansatz, estimator)
        self.lr = learning_rate
        self.reg = regularization

    def _fisher_matrix(self, parameters):
        M = len(parameters)
        delta = 1e-4
        psi_0 = Statevector.from_instruction(self.ansatz.assign_parameters(parameters))
        derivatives = []
        for i in range(M):
            params_d = parameters.copy()
            params_d[i] += delta
            psi_d = Statevector.from_instruction(self.ansatz.assign_parameters(params_d))
            derivatives.append((psi_d.data - psi_0.data) / delta)
            
        F = np.zeros((M, M), dtype=complex)
        for i in range(M):
            for j in range(M):
                F[i, j] = 4 * (np.vdot(derivatives[i], derivatives[j]) - 
                               np.vdot(derivatives[i], psi_0.data) * np.vdot(psi_0.data, derivatives[j]))
        return np.real(F)

    def update_parameters(self, parameters):
        grad = self.spsa_gradient(parameters) 
        energy = self.expectation(parameters)
        F = self._fisher_matrix(parameters)  
        F_reg = F + self.reg * np.eye(len(parameters))
        try:
            natural_grad = np.linalg.solve(F_reg, grad)
        except:
            natural_grad = grad
        return parameters - self.lr * natural_grad, energy

# ============================================================================
# 4. Experiment Runner
# ============================================================================
def run_single_experiment(opt_name, hamiltonian, n_qubits, estimator, n_steps=80, seeds=3):
    opt_classes = {'GQE': GQE, 'SPSA': SPSAOptimizer, 'QNG': QNG, 'Adam': AdamOptimizer, 'SGD': SGDOptimizer}
    OptClass = opt_classes[opt_name]
    
    all_energies = []
    for seed in range(seeds):
        np.random.seed(seed)
        ansatz = RealAmplitudes(n_qubits, entanglement='linear', reps=2)
        params = np.random.uniform(-0.1, 0.1, ansatz.num_parameters)
        
        optimizer = OptClass(hamiltonian, ansatz, estimator=estimator)
        
        energies = []
        for step in range(n_steps):
            params, energy = optimizer.update_parameters(params)
            energies.append(energy)
            
        all_energies.append(energies)
    
    return np.array(all_energies)

# ============================================================================
# 5. Main Execution & Plotting
# ============================================================================
def main():
    n_qubits = 6  # يمكنك تغييرها إلى 8 أو 10 حسب قدرة جهازك
    noise_configs = [
        ('Noiseless', 'noiseless', 0.0),
        ('Depolarizing', 'depolarizing', 0.01),
        ('Amplitude Damping', 'amplitude_damping', 0.01),
        ('Phase Damping', 'phase_damping', 0.01)
    ]
    optimizers = ['GQE', 'QNG', 'Adam', 'SPSA', 'SGD']
    
    hamiltonian = transverse_field_ising_hamiltonian(n_qubits)
    exact_energy = exact_ground_state(hamiltonian)
    
    results = {cfg[0]: {} for cfg in noise_configs}
    
    print(f"Starting Robustness Benchmark on {n_qubits} Qubits...")
    
    for title, n_type, strength in noise_configs:
        print(f"\n--- Environment: {title} ---")
        n_model = create_noise_model(n_type, strength)
        noisy_estimator = make_noisy_estimator(n_model, shots=1024)
        
        for opt in optimizers:
            print(f"  Running {opt}...")
            # لتقليل وقت التنفيذ للرؤية السريعة تم وضع n_steps=60 و seeds=3
            energies = run_single_experiment(opt, hamiltonian, n_qubits, noisy_estimator, n_steps=60, seeds=3)
            results[title][opt] = energies

    # الرسم البياني
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    axes = axes.flatten()
    
    for idx, (title, _, _) in enumerate(noise_configs):
        ax = axes[idx]
        for opt in optimizers:
            mean_e = np.mean(results[title][opt], axis=0)
            std_e = np.std(results[title][opt], axis=0)
            steps = np.arange(len(mean_e))
            
            ax.plot(steps, mean_e, MARKERS[opt], color=COLORS[opt], label=opt, linewidth=2, markersize=5, markevery=5)
            ax.fill_between(steps, mean_e - std_e, mean_e + std_e, color=COLORS[opt], alpha=0.15)
            
        ax.axhline(y=exact_energy, color='k', linestyle='--', label='Exact GS', linewidth=1.5)
        ax.set_title(f"Environment: {title}")
        ax.set_xlabel("Optimization Steps")
        ax.set_ylabel("Energy (Ha)")
        ax.grid(True, alpha=0.3)
        if idx == 0:
            ax.legend(loc='upper right', ncol=2)
            
    plt.tight_layout()
    os.makedirs('figures', exist_ok=True)
    plt.savefig('figures/corrected_noise_robustness.svg', format='svg', dpi=1200, bbox_inches='tight')
    print("\n✓ Saved figure to figures/corrected_noise_robustness.svg")
    plt.show()

if __name__ == "__main__":
    main()

