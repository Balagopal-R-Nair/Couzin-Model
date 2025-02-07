import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.animation import FuncAnimation
import numpy as np



from couzin_functions import run_couzin_model, animate,snapshot, trajectory

R_r = 1           # Radius of repulsion zone
delta_ro = 2      # Width of orientation zone
delta_ra = 4      # Width of attraction zone

num_part = 60     # Number of agents
num_iter = 1000   # Number of time steps
alpha = 270       # vision angle
sigma = 0.1       # Width of noise

p=2               # Size of snapshot grid
q=5

t1 = 1            # Time frame to plot trajectory for
t2 = num_iter
i = 3             # Index of agent to plot trajectory for


e,x = run_couzin_model(num_part, R_r, delta_ro, delta_ra,num_iter,alpha,sigma)
snapshot(e,x,p,q,num_iter)
trajectory(x,num_part,num_iter,i,t1,t2)
animate(e,x,num_part,num_iter)


