import time
import numpy as np

def is_sqrt(n):
    if n > 0:
        x = 1 << (n.bit_length() + 1 >> 1)
        while True:
            y = (x + n // x) >> 1
            if y >= x:
                return x**2 == n
            x = y
    elif n == 0:
        return 0
    else:
        raise ValueError("square root not defined for negative numbers")


        
def seconds_to_minutes(seconds):
    return f"{str(seconds // 60)} minutes {str(np.round(seconds % 60))} seconds"
    
def print_time(text, stime):
    seconds = (time.time() - stime)
    print(text, seconds_to_minutes(seconds))
