# plot.py
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.widgets import Slider
import numpy as np
import cartersian_coordinates.config as config

def plot_simulation(ego_history, targets_history, title="Interactive Simulation"):
    """
    Plots the trajectories with an interactive time slider.
    """
    # 1. Setup Figure and Axes
    # Make room at the bottom for the slider
    fig, ax = plt.subplots(figsize=(12, 6))
    plt.subplots_adjust(bottom=0.25) 
    
    max_steps = len(ego_history) - 1
    
    # 2. Draw Static Road Environment
    road_length = max(ego_history[:, 0].max(), max(t[:, 0].max() for t in targets_history)) + 20
    ax.fill_between([-10, road_length], -5.25, 5.25, color='gray', alpha=0.3)
    
    # Draw lane lines
    for line in [-1.75, 1.75]:
        ax.axhline(y=line, color='white', linestyle='--', linewidth=2)
    ax.axhline(y=5.25, color='white', linestyle='-', linewidth=3)
    ax.axhline(y=-5.25, color='white', linestyle='-', linewidth=3)

    # 3. Plot Faint Trajectory Trails (Background)
    ax.plot(ego_history[:, 0], ego_history[:, 1], 'b--', alpha=0.3)
    colors = ['r', 'g', 'c', 'm']
    for i, t_history in enumerate(targets_history):
        ax.plot(t_history[:, 0], t_history[:, 1], f'{colors[i % len(colors)]}--', alpha=0.3)

    # 4. Initialize Dynamic Plot Elements (Vehicles at t=0)
    # These are the objects we will move when the slider changes
    ego_dot, = ax.plot(ego_history[0, 0], ego_history[0, 1], 'bo', markersize=10, label='Ego')
    ego_circle = patches.Circle((ego_history[0, 0], ego_history[0, 1]), config.COLLISION_RADIUS, 
                                color='blue', fill=False, linestyle='-')
    ax.add_patch(ego_circle)

    target_dots = []
    target_circles = []
    for i, t_history in enumerate(targets_history):
        color = colors[i % len(colors)]
        dot, = ax.plot(t_history[0, 0], t_history[0, 1], f'{color}o', markersize=10, label=f'Target {i+1}')
        circle = patches.Circle((t_history[0, 0], t_history[0, 1]), config.COLLISION_RADIUS, 
                                color=color, fill=False, linestyle='-')
        ax.add_patch(circle)
        
        target_dots.append(dot)
        target_circles.append(circle)

    # Aesthetics
    ax.set_title(title, fontsize=14)
    ax.set_xlabel('Longitudinal Position (x) [m]')
    ax.set_ylabel('Lateral Position (y) [m]')
    ax.set_yticks(config.LANES)
    ax.set_yticklabels(['Right Lane', 'Center Lane', 'Left Lane'])
    ax.set_aspect('equal', adjustable='box')
    ax.legend(loc='upper right')

    # 5. Create the Slider
    # [left, bottom, width, height]
    ax_slider = plt.axes([0.2, 0.1, 0.65, 0.03]) 
    time_slider = Slider(
        ax=ax_slider,
        label='Time Step',
        valmin=0,
        valmax=max_steps,
        valinit=0,
        valstep=1 # Snap to integer time steps
    )

    # 6. Define the Update Function
    def update(val):
        step = int(time_slider.val)
        
        # Update Ego position and safety circle
        ego_dot.set_data([ego_history[step, 0]], [ego_history[step, 1]])
        ego_circle.center = (ego_history[step, 0], ego_history[step, 1])
        
        # Update Target positions and safety circles
        for i, t_history in enumerate(targets_history):
            target_dots[i].set_data([t_history[step, 0]], [t_history[step, 1]])
            target_circles[i].center = (t_history[step, 0], t_history[step, 1])
            
        fig.canvas.draw_idle()

    # Register the update function with the slider
    time_slider.on_changed(update)

    plt.show()

    # We return the slider to prevent Python's garbage collector from destroying it!
    return time_slider