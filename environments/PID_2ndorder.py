import numpy as np
import matplotlib.pyplot as plt

class PIDController:
    def __init__(self, kp, ki, kd, setpoint=0):
        """
        Initialize the PID controller.
        
        :param kp: Proportional gain
        :param ki: Integral gain
        :param kd: Derivative gain
        :param setpoint: Desired setpoint
        """
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.setpoint = setpoint
        
        self.integral = 0
        self.previous_error = 0
    
    def update(self, measured_value, dt):
        """
        Update the PID controller.
        
        :param measured_value: The current measured value
        :param dt: Time interval between updates
        :return: Control output
        """
        error = self.setpoint - measured_value
        p_term = self.kp * error
        self.integral += error * dt
        i_term = self.ki * self.integral
        derivative = (error - self.previous_error) / dt if dt > 0 else 0
        d_term = self.kd * derivative
        self.previous_error = error
        return p_term + i_term + d_term

def second_order_system(u, y, dy, dt, omega_n=1.0, zeta=0.7):
    """
    Simulate a second-order system.
    
    :param u: Control input
    :param y: Current output (position)
    :param dy: Current velocity (first derivative of output)
    :param dt: Time step
    :param omega_n: Natural frequency
    :param zeta: Damping ratio
    :return: Updated output (position) and velocity
    """
    ddy = -2 * zeta * omega_n * dy - omega_n**2 * y + omega_n**2 * u
    dy_new = dy + ddy * dt
    y_new = y + dy_new * dt
    return y_new, dy_new

# PID parameters
kp = 150
ki = 75
kd = 10
setpoint = 1.0  # Desired setpoint

# Simulation parameters
dt = 0.01  # Time step
t_end = 3  # End time
t = np.arange(0, t_end, dt)

# Initialize the PID controller
pid = PIDController(kp, ki, kd, setpoint)

# Initial conditions for the second-order system
y = 0  # Initial position
dy = 0  # Initial velocity

# Store results for plotting
y_history = []
u_history = []

# Simulate the system
for time in t:
    # Get the PID controller output
    u = pid.update(y, dt)
    
    # Update the second-order system based on the control output
    y, dy = second_order_system(u, y, dy, dt, omega_n=1.1, zeta=0.3)
    
    # Record the values for plotting
    y_history.append(y)
    u_history.append(u)

# Plotting the response
plt.figure(figsize=(12, 6))

# Plot the system response
plt.subplot(2, 1, 1)
plt.plot(t, y_history, label="System Response")
plt.axhline(setpoint, color='r', linestyle='--', label="Setpoint")
plt.title("System Response (Transient and Steady-State)")
plt.ylabel("Position (y)")
plt.legend()
plt.grid(True)

# Plot the control input
plt.subplot(2, 1, 2)
plt.plot(t, u_history, label="Control Input (u)", color='g')
plt.title("Control Input")
plt.ylabel("Control Signal (u)")
plt.xlabel("Time (s)")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

