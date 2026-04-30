clear; clc;
n=4;
z= [1;1j;-1;-1j];
dt= 0.01;
T = 20;
steps=T/dt;
epsilons = [0.05, 0.1, 0.5];
U = [0 1 0 0;
     0 0 1 0;
     0 0 0 1;
     1 0 0 0];
L=U-eye(n);
figure; hold on; grid on;
for eps = epsilons
    z_temp=z;
    trajectory =zeros(n, steps);
    for t = 1:steps
        z_temp= z_temp + dt*eps*L*z_temp;
        trajectory(:, t)= z_temp;
    end
    for i = 1:n
        plot(real(trajectory(i, :)), imag(trajectory(i, :)), 'LineWidth', 1);
    end
end
axis equal;
title('Cyclic Pursuit with Varying Epsilon');
xlabel('Real'); ylabel('Imaginary');
legend('\epsilon = 0.05','\epsilon = 0.1','\epsilon = 0.5');
