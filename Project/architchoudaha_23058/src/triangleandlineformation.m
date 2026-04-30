clc; clear; close all;
n=6;
triangle=[0; 5-5i*sqrt(3); 10-10i*sqrt(3); -10i*sqrt(3); -10-10i*sqrt(3); -5-5i*sqrt(3)];
line=[0; -10; -20; -30; -40; -50];
dt=0.01;
T=20;
steps=T/dt;
U=circshift(eye(n),1);
L=U-eye(n);
trajectory_triangle=zeros(n,steps);
trajectory_line=zeros(n,steps);
z_triangle=triangle;
z_line=line;

for t=1:steps
    z_triangle=z_triangle+dt*L*z_triangle;
    z_line=z_line+dt*L*z_line;
    trajectory_triangle(:,t)=z_triangle;
    trajectory_line(:,t)=z_line;
end

figure;
subplot(1,2,1);
hold on; grid on; axis equal;
for i=1:n
    plot(real(trajectory_triangle(i,:)),imag(trajectory_triangle(i,:)),'LineWidth',1.5);
end
title('Triangular Formation (6 Robots)');
xlabel('Real Axis'); ylabel('Imaginary Axis');

subplot(1,2,2);
hold on; grid on; axis equal;
for i=1:n
    plot(real(trajectory_line(i,:)),imag(trajectory_line(i,:)),'LineWidth',1.5);
end
title('Line Formation (6 Robots)');
xlabel('Real Axis'); ylabel('Imaginary Axis');