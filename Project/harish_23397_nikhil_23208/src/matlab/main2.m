clc;
clear;
close all;

g  = 9.81;
r  = 0.05;
MW = 0.13;
MP = 1.24;
l  = 0.213;
IW = 0.0002899;
IP = 0.05626;
km = 0.2774;
ke = 0.509;
R  = 7.101;

beta = 2*MW + 2*IW/(r^2) + MP;
alpha = IP*beta + MP*l^2*(MW + IW/(r^2));

A = [0 1 0 0;
     0 (2*km*ke*(MP*l*r - IP - MP*l^2))/(R*r^2*alpha) (MP*g*l)/alpha 0;
     0 0 0 1;
     0 (2*km*ke*(r*beta - MP*l))/(R*r^2*alpha) (MP*g*l*beta)/alpha 0];

B = [0;
     (2*km*(-MP*l*r + IP + MP*l^2))/(R*r*alpha);
     0;
     (2*km*(-r*beta + MP*l))/(R*r*alpha)];

C = [0 0 1 0];
D = 0;

sys = ss(A,B,C,D);

G = tf(sys);

%% PID 
Kp = 194.534;
Ki = 1432.586;
Kd = 5.551;

PID = pid(Kp, Ki, Kd);

CL_PID = feedback(PID*G,1);

%% Plot PID
figure;
step(CL_PID);
title('PID Response');
grid on;

info_PID = stepinfo(CL_PID);
disp('PID Performance:');
disp(info_PID);