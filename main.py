import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
import initial_conditions as inc
import bridge_locations as bloc

live_plotting = True
save_audio = False

# Set up the strings parameters
L = 0.62                                        # String Length
T_0 = 670                                       # String tension
rho = 5706                                      # String's material density
radius = 0.00059                                # String radius
cross_section = np.pi * radius ** 2             # Area of the cross section
rho_L = 0.0063                                  # Linear mass density
c = (T_0/rho_L) ** (1/2)
# Damping parameters
sigma_0 = 0 *  1e-3 / rho_L                     # Frequency independent damping parameter
sigma_1 = 5e-4 /rho_L                           # Frequency dependent damping parameter
# Stiffness parameters  
Q = 1.261 * 10 ** 11                            # Youngs modulus
I_0 = np.pi * radius ** 4 / 4                   # Area moment of inertia
kappa2 =  (Q * I_0 /rho_L)                      # Stiffness Parameter

# Set up the discretisation parameters
time_vals = 2                                                                               # Duration of simulation
SR = 44100                                                                                  # Sample rate
N = int(SR * time_vals)                                                                     # Number of samples to compute
k = 1/SR                                                                                    # Time step
h = np.sqrt((c ** 2 * k ** 2 + np.sqrt(c ** 4 * k ** 4 + 16 * kappa2 * k ** 2))/2)          # Initial approximation of the grid spacing
x_num = int(L/h)                                                                            # Set up the maximum number of grid points possible with the approximation
h = L/x_num                                                                                 # Set the grid spacing

# Set up the storage
U = np.zeros((3, x_num+1))                      # Instantiate a storage of 3 vectors for our previous, current and next time step deltas
total_energy_data = np.zeros(N)                 # Instantiate a storage to hold the total energy of the system for each sample
kinetic_energy_data = np.zeros(N)               # Instantiate a storage to hold the total energy of the system for each sample
potential_energy_data = np.zeros(N)             # Instantiate a storage to hold the total energy of the system for each sample
backboard_energy_data = np.zeros(N)             # Instantiate a storage to hold the total energy of the system for each sample
U_listen = np.zeros(N)                          # Instantiate a storage to hold the displacement of the string at a certain location for each sample
deltas = np.zeros((3, x_num-1))                 # Instantiate a storage of 3 vectors for our previous, current and next time step deltas
x = np.arange(0,L + 0.5 * h, h)                 # The location of the grid in the spatial domain for plotting

# Initial Conditions
inc.string_pluck(U, x_num, 0.55, 0.006)         # Set the initial values of the string relating to a pluck

# Define our FD matricies
# D_xx
D1 = np.diag(-2 * np.ones((x_num  - 1,)))                                           # Set up the tri-diagonal values of D_xx
D2 = np.diag(np.ones(x_num - 2,), k=-1)                                             # Set up the tri-diagonal values of D_xx
D3 = np.diag(np.ones(x_num - 2,), k=1)                                              # Set up the tri-diagonal values of D_xx
D_xx = (D1 + D2 + D3)/(h ** 2)                                                      # Add these matricies to form the full D_xx matrix
# D_xxxx
D_xxxx1 = np.diag(6 * np.ones(x_num  - 1,))                                         # Set up the quinque-diagonal values of D_xx
D_xxxx2 = -4 * np.diag(np.ones(x_num - 2,), k=-1)                                   # Set up the quinque-diagonal values of D_xx
D_xxxx3 = -4 * np.diag(np.ones(x_num - 2,), k=1)                                    # Set up the quinque-diagonal values of D_xx
D_xxxx4 = np.diag(np.ones(x_num - 3,), k=2)                                         # Set up the quinque-diagonal values of D_xx
D_xxxx5 = np.diag(np.ones(x_num - 3,), k=-2)                                        # Set up the quinque-diagonal values of D_xx
D_xxxx1[0, 0] = 5                                                                   # Alter the first value in the matrix to account for the simply supported boundary conditions
D_xxxx1[-1, -1] = 5                                                                 # Alter the first value in the matrix to account for the simply supported boundary conditions
D_xxxx = (D_xxxx1 + D_xxxx2 + D_xxxx3 + D_xxxx5 + D_xxxx4)/(h ** 4)                 # Add these matricies to form the full D_xxxx matrix
# D_x+
Dx_plus1 = np.diag(np.ones(x_num + 1, ))                                            # Set up the diagonal values of D_xx
Dx_plus2 = np.diag(np.ones(x_num, ),k=1)                                            # Set up the diagonal values of D_xx
Dx_plus = (Dx_plus2 - Dx_plus1) / h                                                 # Add these matricies to form the full D_x+ matrix
# ABC system
I = np.eye(x_num - 1)                                                               # Set up an identity matrix
B = 2 * I + k ** 2 * c ** 2 * D_xx - k ** 2 * kappa2 * D_xxxx                       # Calculate the B matrix from the model
C = I * (sigma_0*k - 1) - k * sigma_1 * D_xx                                        # Calculate the C matrix from the model
A = (1+(sigma_0*k)) * I - k * sigma_1 * D_xx                                        # Calculate the A matrix from the model
a_vec = np.diag(A, -1)                                                              # Set up a vector to hold the left diagonal of A
b_vec = np.diag(A)                                                                  # Set up a vector to hold the diagonal of A
c_vec = np.diag(A, 1)                                                               # Set up a vector to hold the right diagonal of A
A_1 = np.linalg.inv(A)                                                              # Calculate the inverse of A
Ones = np.ones(x_num - 1)                                                           # Vector of ones for energy analysis


# Set up the barriers
# Barrier location
BB_plot = bloc.sitar_bridge(x_num, L, 1, 0.015)                                      # Set up the barrier location for plotting
BB = BB_plot[1:-1].copy()                                                           # Set up a reduced barrier vector for calculations
# Barrier parameters
Kf = 10 ** 8                                                                        # Set the stiffness of the barrier
af = 1.5                                                                            # Set the exponent of the barrier
def phi(x):
    '''
    Takes a value and returns the associated value of phi
    '''
    return Kf / (af + 1) * np.fmax(x,0) ** (af + 1)
def phi_(x):
    '''
    Takes a value and returns the associated value of the first derivative of phi
    '''
    return Kf * np.fmax(x,0) ** af
def phi__(x):
    '''
    Takes a value and returns the associated value of the second derivative of phi
    '''
    return Kf * af * np.fmax(x,0) ** (af - 1)
def F(r, d):
    '''
    Returns the a vector of the distributed interaction force
    '''
    filter = np.abs(r) >= 1e-10                                             # Set a filter to ensure there is no small number division
    ans = np.zeros_like(r)                                                  # Instantiate a storage vector
    ans[filter] = (phi(r[filter] + d[filter])-phi(d[filter]))/r[filter]     # Calculate the force for non-small r values
    ans[~filter] = phi_(d[~filter])                                         # Calculate the force for small r values
    return ans


# Recursion function for linear system
def thomas_wave_recursion(U, a_vec, b_vec, c_vec, B, C):
    '''
    Solves the system A u^n+1 = B u^n + C u^n-1 using the thomas algorithm as defined in the appendix
    a_vec, b_vec and c_vec are the left, center and right diagonals of the matrix A
    '''
    d = B @ U[1] + C @ U[0]
    n = len(d)
    aa = a_vec.copy()
    bb = b_vec.copy()
    cc = c_vec.copy()
    dd = d.copy()
    w = np.zeros(n-1,float)
    g = np.zeros(n, float)
    
    w[0] = cc[0]/bb[0]
    g[0] = dd[0]/bb[0]

    for i in range(1,n-1):
        w[i] = cc[i]/(bb[i] - aa[i-1]*w[i-1])
    for i in range(1,n):
        g[i] = (dd[i] - aa[i-1]*g[i-1])/(bb[i] - aa[i-1]*w[i-1])
    U[2][n-1] = g[n-1]
    for i in range(n-1,0,-1):
        U[2][i-1] = g[i-1] - w[i-1]*U[2][i]

# Newton Raphson
def newtonv(x_n, f, f_x, y, deltas, tol):
    '''
    Computes the root of the function f using the newton raphson method as defined in the appendix
    '''
    err = np.inf                                                                # Set the initial error to be infinity to initialise the while loop
    while err > tol:                                                            # Whilst the current and previous approximation are sufficiently far apart
        x_n1 = x_n - np.linalg.solve(f_x(x_n, y, deltas), f(x_n, y, deltas))    # Compute the next guess for the root
        err = np.linalg.norm(x_n1 - x_n)                                        # Calculate the proximity of the guess to the previous guess
        x_n = x_n1.copy()                                                       # Update the current guess to be the previous guess
    return x_n

# G
def G(R, y, deltas):
    '''
    Solves the non-linear equation
    '''
    return R + y[2] - BB + deltas[0] + k ** 2 / (rho_L)  * A_1 @ F(R, deltas[0])

def G_(R, y, deltas):
    '''
    Solves the derivative of the non-linear equation
    '''
    filter = np.abs(R) >= 1e-10                                                                                 # Set a filter to ensure there is no small number division
    ans = np.zeros_like(R)                                                                                      # Instantiate a storage vector for the diagonal of the Jacobian
    d2 = R + deltas[0]                                                                                          # Set a vector for delta at the next time step
    ans[filter] = - (phi(d2[filter])-phi(deltas[0][filter]))/(R[filter] ** 2) + (phi_(d2[filter]))/R[filter]    # Calculate the force for non-small r values
    ans[~filter] = phi__(deltas[0][~filter])                                                                    # Calculate the force for small r values
    return np.eye(x_num - 1) + k ** 2 / (rho_L) * A_1 @ np.diag(ans)                                            # Construct the jacobain and complete the calculation


# Set up the plotting options
if live_plotting:
    plt.ion()
    i_vals = np.arange(N)
    fig = plt.figure(figsize=(10,6))
    fig.subplots_adjust(top=0.88,
bottom=0.11,
left=0.125,
right=0.9,
hspace=0.2,
wspace=0.35)
    # Set up the energy plot
    ax_e = fig.add_subplot(121)
    line_e, = ax_e.plot(i_vals, total_energy_data, 'b-')
    ax_e.set_title('Total Energy')
    ax_e.set_xlabel('Time step')
    ax_e.set_ylabel('Total Energy (J)')
    # Set up the displacement plot
    ax = fig.add_subplot(122)
    line1, = ax.plot(x, U[0])
    line2, = ax.plot(x, BB_plot)
    ax.set_ylim(-max(U[0]) * 1.1, max(U[0]) * 1.1)
    ax.set_title('String Displacement')
    ax.set_xlabel('x (m)')
    ax.set_ylabel('u (m)')

def stringKineticEnergy(U, deltas):
    '''
    Computes the kintetic energy of the system
    '''
    return h / 2 * np.linalg.norm((U[1] - U[0])/k) ** 2

def stringPotentialEnergy(U, deltas):
    '''
    Computes the potential energy of the system
    '''
    return h * c ** 2 / 2 * (Dx_plus @ U[1]).T @ (Dx_plus @ U[0])

def backboardEnergy(U, deltas):
    '''
    Computes the energy of the backboard interaction
    '''
    return h / 2 / rho_L * Ones.T @ (phi(deltas[1]) + phi(deltas[0]))

def totalEnergy(U, deltas):
    '''
    Computes the total energy of the system
    '''
    return stringKineticEnergy(U, deltas) + stringPotentialEnergy(U, deltas) + backboardEnergy(U, deltas)



R = -0.01 *  np.ones(x_num - 1)                     # Initialise the vector R with a reasonable first guess for the difference between deltas[2] and deltas[0]
deltas[0] = BB - U[0][1:-1].copy()                  # Set the value of delta at the previous time step
deltas[1] = BB - U[1][1:-1].copy()                  # Set the value of delta at the current time step
n = 0                                               # Initialise n, the number of samples currently calculated

while n < N:                                                                                # Run whilst the number of samples computed is less than the total number of samples to calculate
    y = U[:,1:-1].copy()                                                                    # Start by initialising y
    thomas_wave_recursion(y, a_vec, b_vec, c_vec, B, C)                                     # Sets the value of y[2] to be the value of y^n from the model
    R = newtonv(R, G, G_, y, deltas, 1e-10)                                                 # Find the value of R^n using newton raphson, an initial guess being the previous value
    deltas[2] = R + deltas[0]                                                               # Update the delta using R
    U[2][1:-1] = y[2].copy() + k ** 2 / (rho_L) * A_1 @ F(R, deltas[0])                     # Set the next time step string displacement
    total_energy_data[n] = totalEnergy(U, deltas)                                           # Calculate the total energy of the system
    # If we require a live plot
    if live_plotting:
        # Set the total energy at each time step
        total_energy_data[n] = totalEnergy(U, deltas)
        if n > 1:
            # Plot the total energy data on the right and the displacement on the right
            ax_e.set_xlim((0,n))
            line_e.set_xdata(i_vals[:n])
            line_e.set_ydata(total_energy_data[:n])
            ax_e.set_ylim(0, max(total_energy_data) * 1.1)
            line1.set_ydata(U[1])
            fig.canvas.draw() 
            fig.canvas.flush_events()
    U_listen[n] = U[0, int(x_num*0.2)]                                      # Set the displacement at the pickup
    U[0] = U[1].copy()                                                      # Update the time step by setting the current value to be the previous
    U[1] = U[2].copy()                                                      # Update the time step by setting the next value to be the current
    deltas[0] = deltas[1].copy()                                            # Update the time step by setting the current value to be the previous
    deltas[1] = deltas[2].copy()                                            # Update the time step by setting the next value to be the current
    n += 1


# Save the audio file generated at the pickup location
if save_audio:
    filename = "Sitar_Audio.wav"
    wavfile.write(filename, SR, U_listen)
