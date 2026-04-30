clear; clc;
T=1000;
dt=0.1;
e=0.2;
x=1;
y=1;
theta=0;
x_traj=[];
y_traj=[];
figure;
hold on; grid on;
xlim([-1 1]);
ylim([0 1.4]);
title('Unicycle Simulation');

for i=1:T
    
    v=-x*cos(theta)-y*sin(theta)-e;
    w=(1/e)*(x*sin(theta)-y*cos(theta));
    x=x+v*cos(theta)*dt;
    y=y+v*sin(theta)*dt;
    theta=theta+w*dt;
    x_traj=[x_traj x];
    y_traj=[y_traj y];
    

    plot(x_traj,y_traj,'b')
    plot(-x_traj,y_traj,'g')
    pause(0.01)
end