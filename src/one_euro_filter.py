import math
import time

class OneEuroFilter:
    def __init__(self, t0, x0, dx0=0.0, min_cutoff=1.0, beta=0.0, d_cutoff=1.0):
        """
        Initialize the 1 Euro Filter.
        min_cutoff: Minimum cutoff frequency (Hz) to filter slow movements.
        beta: Speed coefficient. Higher = less lag during fast movement.
        d_cutoff: Cutoff frequency for the derivative (Hz).
        """
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        
        self.x_prev = x0
        self.dx_prev = dx0
        self.t_prev = t0

    def smoothing_factor(self, t_e, cutoff):
        r = 2 * math.pi * cutoff * t_e
        return r / (r + 1)

    def exponential_smoothing(self, a, x, x_prev):
        return a * x + (1 - a) * x_prev

    def __call__(self, t, x):
        """
        Filter the signal.
        t: Current timestamp.
        x: Current value (float or numpy array).
        """
        if self.t_prev is None:
            self.t_prev = t
            self.x_prev = x
            return x
            
        t_e = t - self.t_prev
        
        # Avoid division by zero in rare cases
        if t_e <= 0.0:
            return self.x_prev

        # Calculate derivative (velocity)
        a_d = self.smoothing_factor(t_e, self.d_cutoff)
        dx = (x - self.x_prev) / t_e
        dx_hat = self.exponential_smoothing(a_d, dx, self.dx_prev)

        # Calculate adaptive cutoff frequency
        # cutoff = min_cutoff + beta * |velocity|
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        
        # Filter signal
        a = self.smoothing_factor(t_e, cutoff)
        x_hat = self.exponential_smoothing(a, x, self.x_prev)

        # Update state
        self.x_prev = x_hat
        self.dx_prev = dx_hat
        self.t_prev = t

        return x_hat
