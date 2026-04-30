<<<<<<< HEAD
# plot.py
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.widgets import Slider
import numpy as np
import config

def plot_simulation(ego_history, targets_history, title="Frenet-Based Simulation"):
    fig, ax = plt.subplots(figsize=(12, 6))
    plt.subplots_adjust(bottom=0.25) 
    
    max_steps = len(ego_history) - 1
    
    # Draw Road
    road_length = max(ego_history[:, 0].max(), max(t[:, 0].max() for t in targets_history)) + 20
    ax.fill_between([-10, road_length], -5.25, 5.25, color='gray', alpha=0.3)
    for line in [-1.75, 1.75]:
        ax.axhline(y=line, color='white', linestyle='--', linewidth=2)
    ax.axhline(y=5.25, color='white', linestyle='-', linewidth=3)
    ax.axhline(y=-5.25, color='white', linestyle='-', linewidth=3)

    # Plot Faint Trails
    # Ego history [s, e_y, e_psi, v] -> plot s vs e_y
    ax.plot(ego_history[:, 0], ego_history[:, 1], 'b--', alpha=0.3)
    colors = ['r', 'g']
    for i, t_hist in enumerate(targets_history):
        # Target history [x, y, vx, vy] -> plot x vs y
        ax.plot(t_hist[:, 0], t_hist[:, 1], f'{colors[i]}--', alpha=0.3)

    # Dynamic Elements
    ego_dot, = ax.plot(ego_history[0, 0], ego_history[0, 1], 'bo', markersize=8, label='EV')
    
    target_dots = []
    target_ellipses = []
    for i, t_hist in enumerate(targets_history):
        dot, = ax.plot(t_hist[0, 0], t_hist[0, 1], f'{colors[i]}o', markersize=8, label=f'TV {i+1}')
        ellipse = patches.Ellipse((t_hist[0, 0], t_hist[0, 1]), 
                                  width=config.COLLISION_A*2, height=config.COLLISION_B*2, 
                                  color=colors[i], fill=False, linestyle='-')
        ax.add_patch(ellipse)
        target_dots.append(dot)
        target_ellipses.append(ellipse)

    ax.set_title(title)
    ax.set_xlabel('Longitudinal Position / Arc Length s [m]')
    ax.set_ylabel('Lateral Position e_y [m]')
    ax.set_yticks(config.LANES)
    ax.set_yticklabels(['Right', 'Center', 'Left'])
    ax.set_aspect('equal')
    ax.legend(loc='upper right')

    # Slider
    ax_slider = plt.axes([0.2, 0.1, 0.65, 0.03]) 
    time_slider = Slider(ax_slider, 'Time Step', 0, max_steps, valinit=0, valstep=1)

    def update(val):
        step = int(time_slider.val)
        ego_dot.set_data([ego_history[step, 0]], [ego_history[step, 1]])
        
        for i, t_hist in enumerate(targets_history):
            target_dots[i].set_data([t_hist[step, 0]], [t_hist[step, 1]])
            target_ellipses[i].center = (t_hist[step, 0], t_hist[step, 1])
        fig.canvas.draw_idle()

    time_slider.on_changed(update)
    plt.show()
    return time_slider


=======
# plot.py
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.widgets import Slider
import numpy as np
import config

def plot_simulation(ego_history, targets_history, title="Frenet-Based Simulation"):
    fig, ax = plt.subplots(figsize=(12, 6))
    plt.subplots_adjust(bottom=0.25) 
    
    max_steps = len(ego_history) - 1
    
    # Draw Road
    road_length = max(ego_history[:, 0].max(), max(t[:, 0].max() for t in targets_history)) + 20
    ax.fill_between([-10, road_length], -5.25, 5.25, color='gray', alpha=0.3)
    for line in [-1.75, 1.75]:
        ax.axhline(y=line, color='white', linestyle='--', linewidth=2)
    ax.axhline(y=5.25, color='white', linestyle='-', linewidth=3)
    ax.axhline(y=-5.25, color='white', linestyle='-', linewidth=3)

    # Plot Faint Trails
    # Ego history [s, e_y, e_psi, v] -> plot s vs e_y
    ax.plot(ego_history[:, 0], ego_history[:, 1], 'b--', alpha=0.3)
    colors = ['r', 'g']
    for i, t_hist in enumerate(targets_history):
        # Target history [x, y, vx, vy] -> plot x vs y
        ax.plot(t_hist[:, 0], t_hist[:, 1], f'{colors[i]}--', alpha=0.3)

    # Dynamic Elements
    ego_dot, = ax.plot(ego_history[0, 0], ego_history[0, 1], 'bo', markersize=8, label='EV')
    
    target_dots = []
    target_ellipses = []
    for i, t_hist in enumerate(targets_history):
        dot, = ax.plot(t_hist[0, 0], t_hist[0, 1], f'{colors[i]}o', markersize=8, label=f'TV {i+1}')
        ellipse = patches.Ellipse((t_hist[0, 0], t_hist[0, 1]), 
                                  width=config.COLLISION_A*2, height=config.COLLISION_B*2, 
                                  color=colors[i], fill=False, linestyle='-')
        ax.add_patch(ellipse)
        target_dots.append(dot)
        target_ellipses.append(ellipse)

    ax.set_title(title)
    ax.set_xlabel('Longitudinal Position / Arc Length s [m]')
    ax.set_ylabel('Lateral Position e_y [m]')
    ax.set_yticks(config.LANES)
    ax.set_yticklabels(['Right', 'Center', 'Left'])
    ax.set_aspect('equal')
    ax.legend(loc='upper right')

    # Slider
    ax_slider = plt.axes([0.2, 0.1, 0.65, 0.03]) 
    time_slider = Slider(ax_slider, 'Time Step', 0, max_steps, valinit=0, valstep=1)

    def update(val):
        step = int(time_slider.val)
        ego_dot.set_data([ego_history[step, 0]], [ego_history[step, 1]])
        
        for i, t_hist in enumerate(targets_history):
            target_dots[i].set_data([t_hist[step, 0]], [t_hist[step, 1]])
            target_ellipses[i].center = (t_hist[step, 0], t_hist[step, 1])
        fig.canvas.draw_idle()

    time_slider.on_changed(update)
    plt.show()
    return time_slider


>>>>>>> a0716a10d0730f25f94851fde7115e1102c0813c
