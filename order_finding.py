from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
from qiskit import Aer, execute
from qiskit.tools.visualization import circuit_drawer
import matplotlib.pyplot as plt
from cmath import pi
from math import log, floor, sqrt, gcd
import random
import sys

"""
Code writen by Samuel Crawford Loveys
This code is based on the paper:
Beauregard, S. Circuit for Shor's algorithm using 2n+3 qubits. Quantum
    Information and Computation, Vol. 3, No. 2 (2003) pp. 175-185

Code suplied here is provided for academic study
"""

class OrderFindingCircuit:
    qc = None
    result = None
    a = None

    # returns a list of each qubit in registers r0 and r1
    def _append_reg(self, r0, r1):
        reg_list = []
        for r in r0:
            reg_list.append(r)
        for r in r1:
            reg_list.append(r)
        return reg_list

    # applies QFT
    def fourier_transform(self, qr):
        size = len(qr)
        i = size-1
        while (i >= 0):
            self.qc.h(qr[i])
            j = 0
            while (j < i):
                self.qc.crz(pi/(2**(i-j)), qr[j], qr[i])
                j += 1
            i -= 1

    # inverse QFT
    def fourier_transform_daggar(self, qr):
        size = len(qr)
        i = 0
        while (i < size):
            j = i-1
            while (j >= 0):
                self.qc.crz(-pi/(2**(i-j)), qr[j], qr[i])
                j -= 1
            self.qc.h(qr[i])
            i += 1

    # inverse of phase addition
    def addition_transform_daggar(self, qr, classical_control, c=None):
        size = qr.size
        classical_control = 2**size - classical_control
        self.addition_transform(qr, classical_control, c=c)

    # phase addition to register qr
    def addition_transform(self, qr, classical_control, c=None):
        size = qr.size
        i = size-1
        while (i >= 0):
            j = 0
            bit = 1
            while (j <= i):
                if (bit & classical_control != 0):
                    if (c):
                        self.qc.crz(pi/(2**(i-j)), c, qr[i])
                    else:
                        self.qc.rz(pi/(2**(i-j)), qr[i])
                j += 1
                bit *= 2
            i -= 1

    # conditional (b+a) mod N to register qr where b is the initial state
    def add_mod(self, c0, c1, qr, an, a, N):
        four_reg = self._append_reg(qr, an[0])
        self.qc.cx(c0, c1)
        self.addition_transform(qr, a, c=c1)
        self.addition_transform_daggar(qr, N)
        self.fourier_transform_daggar(four_reg)
        self.qc.cx(an[0], an[1])
        self.fourier_transform(four_reg)
        self.addition_transform(qr, N, c=an[1])
        self.addition_transform_daggar(qr, a, c=c1)
        self.fourier_transform_daggar(four_reg)
        self.qc.x(an[0])
        self.qc.cx(an[0], an[1])
        self.qc.x(an[0])
        self.fourier_transform(qr)
        self.addition_transform(qr, a, c=c1)
        self.qc.cx(c0, c1)

    # inverse of add_mod
    def add_mod_daggar(self, c0, c1, qr, an, a, N):
        four_reg = self._append_reg(qr, an[0])
        self.qc.cx(c0, c1)
        self.addition_transform_daggar(qr, a, c=c1)
        self.fourier_transform_daggar(four_reg)
        self.qc.x(an[0])
        self.qc.cx(an[0], an[1])
        self.qc.x(an[0])
        self.fourier_transform(four_reg)
        self.addition_transform(qr, a, c=c1)
        self.addition_transform_daggar(qr, N, c=an[1])
        self.fourier_transform_daggar(four_reg)
        self.qc.cx(an[0], an[1])
        self.fourier_transform(four_reg)
        self.addition_transform(qr, N)
        self.addition_transform_daggar(qr, a, c=c1)
        self.qc.cx(c0, c1)

    # conditional (b+ax) mod N to b where x is given by qr
    def c_multiplier(self, c, qr, an, b, a, N):
        size = qr.size
        self.fourier_transform(b)
        i = 0;
        while (i < size):
            self.add_mod(c, qr[i], b, an, (2**i)*a, N)
            i += 1
        self.fourier_transform_daggar(qr)

    # inverse of c_multiplier
    def c_multiplier_daggar(self, c, qr, b, an, a, N):
        size = qr.size
        self.fourier_transform_daggar(qr)
        i = size-1;
        while (i >= 0):
            self.add_mod_daggar(c, qr[i], b, an, (2**i)*a, N)
            i -= 1
        self.fourier_transform(qr)

    # conditional swap form r0 to r1
    def c_swap_reg(self, c, r0, r1):
        size = r0.size # assuming r0 and r1 have the same size
        i = 0
        while (i < size):
            self.qc.cswap(c, r0[i], r1[i])
            i += 1

    # conditional (a*x) mod N on l1
    def c_U(self, c, l1, l2, an, a, N):
        self.c_multiplier(c, l1, l2, an, a, N)
        self.c_swap_reg(c, l1, l2)
        self.c_multiplier_daggar(c, l1, l2, an, a, N)

    # creates a circuit for simulating FFqRAM
    def __init__(self, N):
        size = floor(log(N, 2)) + 1

        # checksum 1
        if (N % 2 == 0):
            self.result = 2
            return

        # checksum 2
        for p in range(2, floor(sqrt(N))):
            q = 2
            pq = p**q
            while (pq <= N):
                if (pq == N):
                    self.result = p
                    return
                q += 1
                pq = p**q

        # gen our random input, passed in via compiling
        self.a = a = random.randint(1, N-1)

        # checksum 3
        common = gcd(a, N)
        if (common > 1):
            self.result = common
            return

        print("Compiling for a =", a, "and N =", N, "using", size*2+3, "qubits", flush=True)

        target = QuantumRegister(1, name="m") # target result
        l1 = QuantumRegister(size, name="l1") # loader
        l2 = QuantumRegister(size, name="l2")
        an = QuantumRegister(2, name="an") # ancillae
        cout = []
        i = 0
        while (i < size*2):
            cout.append(ClassicalRegister(1, name="r" + str(i) +"`"))
            i += 1
        self.qc = QuantumCircuit(target, l1, l2, an)
        for c in cout:
            self.qc.add_register(c)

        # init
        self.qc.x(l1[0])

        # x^j mod N
        i = 0
        while (i < size*2):
            self.qc.h(target)
            self.c_U(target, l1, l2, an, a**(2**i), N)

            # QFT^dag segment
            j = i-1
            while (j >= 0):
                self.qc.rz(pi/(2**(i-j)), target).c_if(cout[j], 1)
                j -= 1
            self.qc.h(target)

            # measure and reset
            self.qc.measure(target, cout[i])
            self.qc.x(target).c_if(cout[i], 1)
            i += 1


if __name__=='__main__':
    if (len(sys.argv) <= 1):
        print("must enter an integer")
        exit()
    N = int(sys.argv[1])
    OrderF = OrderFindingCircuit(N)
    if (OrderF.result):
        print("Early result found:", OrderF.result, int(N/OrderF.result))
        exit()
    #print(OrderF.qc) # requires python 3.7
    #OrderF.qc.draw(output='latex', plot_barriers=False, scale=0.5).show()

    simulation_epoch = 128
    print("Starting", simulation_epoch, "simulations...")
    backend_sim = Aer.get_backend('qasm_simulator')
    job_sim = execute(OrderF.qc, backend_sim, shots=simulation_epoch)
    sim_result = job_sim.result()
    counts = sim_result.get_counts(OrderF.qc)

    successes = 0
    for count in counts:
        # convert simulation output to integer values
        bits = count.split(" ")
        mul = 1
        order = 0
        for b in reversed(bits):
            order += int(b)*mul
            mul *= 2
        # check output is valid
        if (order % 2 == 1):
            continue
        # we now know that order is an even int and thus ordera is an int
        # the following should be implemented more efficiently in practice
        ordera = OrderF.a**int(order/2)
        result = gcd(ordera-1, N)
        if (result != None and result != 1 and result != N):
            if (successes <= 0):
                print("Factors of", N, "are", result, "and", int(N/result))
            successes += counts[count]
            continue
        result = gcd(ordera+1, N)
        if (result != None and result != 1 and result != N):
            if (successes <= 0):
                print("Factors of", N, "are", result, "and", int(N/result))
            successes += counts[count]

    if (successes <= 0):
        print("no factors found")
    else:
        print("successes =", successes)
    #print("Order Finding results: ", counts)
    #print("num resutls:", len(counts))
