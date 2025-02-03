# Functions for 2D Couzin Model with periodic boundaries + animation
# By Balagopal R Nair

import numpy as np
from tqdm import trange
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


# Initialising positions of all the agents. They initial position is restricted to the middle 1/3rd of the arena
def init_position(position,num_part,box_length):
    position[:,0] = np.random.uniform(box_length/3, 2*box_length/3, num_part)
    position[:,1] = np.random.uniform(box_length/3, 2*box_length/3, num_part) 
    

# Agents are given random initial directions and corresponding speeds    
def init_velocity(num_part,box_length,angles,velocity,position,v_abs):
    angles[:] = np.random.uniform(-np.pi, np.pi, num_part)
    
    velocity[:,0] = v_abs*np.cos(angles[:])
    velocity[:,1] = v_abs*np.sin(angles[:])


# This function applies periodic boundary conditions to any distance between two points   
def apply_periodic_boundary(dx, dy, box_length):
    dx = (dx - box_length * np.round(dx / box_length))
    dy = (dy - box_length * np.round(dy / box_length))

    return dx, dy  

# Gives a Boolean value table for whether a pair of agents are within each others zones of repulsion orientation and attraction.
# Outputs 3 matrices for each zone 
def get_neighbour_matrix(position, L, R_r, R_o, R_a):
    dx = np.subtract.outer(position[:, 0], position[:, 0])  # Pairwise distances
    dy = np.subtract.outer(position[:, 1], position[:, 1])
    apply_periodic_boundary(dx, dy, L)
    pair_dist = dx ** 2 + dy ** 2                           # Taking square to avoid negatives  
    
    # Checks for each zone
    neighbours_repulsion = pair_dist < R_r ** 2                            
    neighbours_orientation = (pair_dist < R_o ** 2) & (pair_dist > R_r ** 2) 
    neighbours_attraction = (pair_dist < R_a ** 2) & (pair_dist > R_o ** 2)  
    
    # Filling diagonals with False to prevent self counting
    np.fill_diagonal(neighbours_repulsion, False)
    
    return  neighbours_repulsion, neighbours_orientation, neighbours_attraction


# Updates velocity of each agent
def velocity_update(velocity,angles,position,v_abs,num_part,dt):
    
    velocity[:,0] =  v_abs*np.cos(angles)
    velocity[:,1] =  v_abs*np.sin(angles)

# Updates Position of each agent    
def position_update(velocity,angles,position,num_part,box_length,dt):
    position += velocity * dt
    position %= box_length # Accounting for periodic boundaries

    
def run_couzin_model(num_part, R_r, delta_ro, delta_ra, num_iter, alpha, sigma):
    
    ####################################################
    #         Initializing variables and arrays        #
    ####################################################  

    
    box_length = 100             # Length of square arena

    dt = 0.1                     #time step
    dtheta = np.pi/5             #turning rate
    alpha = alpha*np.pi/180      #vision angle, in radians

    R_o = delta_ro + R_r         # Radius of orientation zone
    R_a = delta_ra + R_o         # Radius of attraction zone

    
    v_abs = 1                    # Speed
    

    
    position = np.empty([num_part,2])               # This array only stores position data for current time step, x and y in each column
    anim_position = np.zeros([num_iter,num_part,2]) # This array stores position data for all time steps, to be used for animation
    
    angles =  np.empty([num_part,])                 # This array only stores angle data for current time step
    anim_angles = np.zeros([num_iter,num_part,])    # This array stores angle data for all time steps, to be used for animation
    
    velocity = np.empty([num_part,2])               # Velocities of current time step

    rep_angle = np.zeros([num_part,])               # Contains the change in angle of an agent due to neighbours in its repulsion,
    or_angle = np.zeros([num_part,])                # orientation and
    att_angle = np.zeros([num_part,])               # attraction zone

    rep_check = np.zeros([num_part,])               # Keeps a check of whether an agent should have a change in angle, useful when averaging effect of all zones
    or_check = np.zeros([num_part,])
    att_check = np.zeros([num_part,])

    ####################################################
    #              Starting simulation                 #
    ####################################################
  
    
    # Initialization
    init_position(position,num_part,box_length)
    init_velocity(num_part,box_length,angles,velocity,position,v_abs)
    
    # Running loop
    for i in trange(0,num_iter):
        neighbours_repulsion, neighbours_orientation, neighbours_attraction = get_neighbour_matrix(position, box_length, R_r, R_o, R_a)
        
        angles = (angles + np.pi) % (2 * np.pi) - np.pi     # Correcting angles to be between -pi to pi

        # Resetting all angle change and check matrices
        rep_angle.fill(0)
        or_angle.fill(0)
        att_angle.fill(0)

        rep_check.fill(0)
        or_check.fill(0)
        att_check.fill(0)    

    ############### ZOR ##################
        for j in range(0,num_part):
            rep_list = np.where(neighbours_repulsion[j])[0]  # Obtains list of indexes of agents that are in ZOR
            if len(rep_list) > 0:   
                dx = position[j, 0] - position[rep_list, 0]
                dy = position[j, 1] - position[rep_list, 1]
                apply_periodic_boundary(dx, dy, box_length)
                
                

                # Calculating angle of line joining the agent and its neighbour
                ang = np.arctan2(dy,dx)
                
                
                
                # Taking average of all the angles with neighbours
                # We need the agent to be repelled by these neighbours, hence the overall average vector should point towards the agent.
                
                x_avg = np.sum(np.cos(ang))/rep_list.size
                y_avg = np.sum(np.sin(ang))/rep_list.size
                
                avg_angle = np.arctan2(y_avg,x_avg)
                
                rep_angle[j] = avg_angle  # Storing change in angle
                rep_check[j] = 1          # Storing which agent has undergone change in angle due to repulsion zone



    ############## ZOO ################ 
        for j in range(0,num_part):
            if np.any(neighbours_repulsion[j]):                  # Checking whether current agent in consideration has a neighbour in ZOR, if yes move onto next agent                                           
                continue                                         # Only agents with no neighbours in ZOR can be affected by agents in its ZOO or ZOA
                
            or_list = np.where(neighbours_orientation[j])[0]     # Obtains list of indexes of agents that are in ZOO
            if len(or_list) > 0:
                dx = position[j, 0] - position[or_list, 0]
                dy = position[j, 1] - position[or_list, 1]
                apply_periodic_boundary(dx, dy, box_length) 
                
                # Calculating angle of line joining the agent and its neighbour
                ang = np.arctan2(dy,dx)

                for k in range(len(or_list)):   
                    if abs(ang[k] - angles[(or_list[k])]) <= alpha/2:  # Checking if the neighbour in consideration is within the visible range of the agent
                        
                    
                    
                        # Taking average of orientation of all the neighbours in the ZOO
                        x_avg = (np.sum(np.cos(angles[or_list])))/or_list.size
                        y_avg = (np.sum(np.sin(angles[or_list])))/or_list.size
                        avg_angle = np.arctan2(y_avg,x_avg)
                        
                        or_angle[j] = avg_angle                 # Storing change in angle
                        or_check[j] = 1                         # Storing which agent has undergone change in angle due to orientation zone


    ################### ZOA ###############
        for j in range(0,num_part):
            if np.any(neighbours_repulsion[j]):                 # Checking whether current agent in consideration has a neighbour in ZOR, if yes move onto next agent
                                                                # Only agents with no neighbours in ZOR can be affected by agents in its ZOO or ZOA
                continue
            att_list = np.where(neighbours_attraction[j])[0]    # Obtains list of indexes of agents that are in ZOA
            if len(att_list) > 0:
                dx = position[j, 0] - position[att_list, 0]
                dy = position[j, 1] - position[att_list, 1]
                apply_periodic_boundary(dx, dy, box_length)
                
                # Calculating angle of line joining the agent and its neighbour
                ang = np.arctan2(dy,dx)
                
                for k in range(len(att_list)): 
                    if abs(ang[k] - angles[(att_list[k])]) <= alpha/2: # Checking if the neighbour in consideration is within the visible range of the agent
                        
                        # Taking average of all the angles with neighbours
                        # We need the agent to be attracted by these neighbours, hence the overall average vector should point away from the agent.
                        # Hence we take -ve 
                        x_avg = -(np.sum(np.cos(ang)))/att_list.size    
                        y_avg = -(np.sum(np.sin(ang)))/att_list.size
                        avg_angle = np.arctan2(y_avg,x_avg)
                        att_angle[j] = avg_angle                # Storing change in angle
                        att_check[j] = 1                        # Storing which agent has undergone change in angle due to orientation zone


    ############# Angle update ######################
        for j in range(num_part):
            count = rep_check[j] + or_check[j] + att_check[j]   # Checks whether current agent in consideration has any neighbours in any zone
            if count > 0:
                
                # Taking average of all angle changes
                x_comp = (rep_check[j] * np.cos(rep_angle[j]) + or_check[j] * np.cos(or_angle[j]) + att_check[j] * np.cos(att_angle[j]))/count
                y_comp = (rep_check[j] * np.sin(rep_angle[j]) + or_check[j] * np.sin(or_angle[j]) + att_check[j] * np.sin(att_angle[j]))/count


                mag = np.sqrt(x_comp**2 + y_comp**2)
                x_comp /= mag
                y_comp /= mag

                theta_new = np.arctan2(y_comp, x_comp)

                delta_theta = theta_new - angles[j]         # Change in angle from current angle
                delta_theta = (delta_theta + np.pi) % (2 * np.pi) - np.pi
                
                
                # If change in angle is greater than turning rate, then the agent rotates with turning rate towards direction of the change
                if np.abs(delta_theta) > dtheta*dt:
                    angles[j] += np.sign(delta_theta)*dtheta*dt + np.random.normal(0,sigma)  # Noise is added to angle too
                else:
                    angles[j] += delta_theta + np.random.normal(0,sigma)

        velocity_update(velocity,angles,position,v_abs,num_part,dt)
        position_update(velocity,angles,position,num_part,box_length,dt)
        
        #Storing position and angle data for animation
        anim_position[i,:,:] = position
        anim_angles[i,:] = angles
    
    
        
    return anim_angles, anim_position

# Animation of the simulation. The animation is saved as a file named "couzin_model.mp4".
def animate(e,x,num_part,num_iter):
    print("Animating...")
    fig, ax = plt.subplots()
    ax.set_xlim(0, 100)  
    ax.set_ylim(0, 100)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    
    
    scatter = ax.scatter([], [], s=25) 
    arrows = [ax.annotate("", xy=(0, 0), xytext=(0, 0),
                          arrowprops=dict(arrowstyle="->", color='red', lw=1)) for _ in range(num_part)]
    
    def init():
        scatter.set_offsets(np.zeros((num_part, 2))) 
        for arrow in arrows:
           arrow.set_position((0, 0))
           arrow.xy = (0, 0)
        return scatter, *arrows
    
    
    def update(frame): 
        position = x[frame]
        angles = e[frame]
        scatter.set_offsets(position)
        
        for i, arrow in enumerate(arrows):
           x_start, y_start = position[i]
           x_end = x_start + np.cos(angles[i])*4
           y_end = y_start + np.sin(angles[i])*4
           arrow.set_position((x_start, y_start))
           arrow.xy = (x_end, y_end)
        
        return scatter,
    
    
    ani = FuncAnimation(fig, update, frames=num_iter, init_func=init, blit=False, interval=1)
    
    ani.save("couzin_model.mp4", writer="ffmpeg", fps=60, dpi=100)
    plt.close()
    print("Done")

# Outputs a snapshot of the simulation every num_iter/(p*q) steps
def snapshot(e,x,p,q,num_iter):
    
    fig, ax = plt.subplots(p, q, figsize=(20, 10))
    ax=ax.flatten()
    t = np.round(num_iter//(p*q))
    for i in range(p*q):
        ax[i].scatter(x[t*i,:,0],x[t*i,:,1],s=15)
        dx = np.cos(e[t*i,:])
        dy = np.sin(e[t*i,:])
        ax[i].quiver(x[t*i,:,0],x[t*i,:,1],dx,dy,color = 'red')
        ax[i].set_xlim(0,100)
        ax[i].set_ylim(0,100)
        ax[i].set(title=f'T = {t*i}')
        ax[i].tick_params(axis='x', labelsize=8 )
    plt.show()
    
 
# Outputs trajectory of a selected agent, for specified time duration
def trajectory(x,num_part,num_iter,i,t_start = None,t_end = None):
    # If no start and end time is specified, entire trajectory is plotted
    if t_start == None:
        t_start = 1
    if t_end == None:
        t_end = num_iter
    i = int(i - 1)
    x_pos,y_pos = x[t_start:t_end,i,0],x[t_start:t_end,i,1]    # Getting position data of selected agent 
    
    
    # Following is done to ensure no sudden jumps in trajectory due to periodic boundaries
    abs_d_xpos = np.abs(np.diff(x_pos))             
    abs_d_ypos = np.abs(np.diff(y_pos))
    mask_x = np.hstack([ abs_d_xpos > abs_d_xpos.mean()+3*abs_d_xpos.std(), [False]])
    mask_y = np.hstack([ abs_d_ypos > abs_d_ypos.mean()+3*abs_d_ypos.std(), [False]])
    masked_data_x = np.ma.MaskedArray(x_pos, mask_x)
    masked_data_y = np.ma.MaskedArray(y_pos, mask_y)
    plt.plot(masked_data_x, masked_data_y, zorder = 1)   
    
    
    # Plotting start and end point of the trajectory
    plt.scatter(x[t_start - 1,i,0],x[t_start - 1,i,1],color = 'green', zorder = 2)
    plt.scatter(x[t_end - 1,i,0],x[t_end - 1,i,1], color = 'red', zorder = 3)   
    
    plt.xlim(0,100)
    plt.ylim(0,100)
    
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.show()