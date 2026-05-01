clear
clc
close all

t = 0:0.05:15;

theta = zeros(size(t));
x_move = zeros(size(t));

for i = 1:length(t)

    if t(i) < 4
        
        theta(i) = 0.15;
        x_move(i) = 0.5 * t(i);
        
    elseif t(i) < 8
        
        theta(i) = 0.15*exp(-0.5*(t(i)-4)).*cos(3*(t(i)-4));
        x_move(i) = 2 + 0.3*(t(i)-4);
        
    elseif t(i) < 11
        
        theta(i) = -0.15;
        x_move(i) = 3.2 - 0.4*(t(i)-8);
        
    else
        
        theta(i) = 0.15*exp(-0.8*(t(i)-11)).*cos(4*(t(i)-11));
        x_move(i) = 2;
        
    end
end

L = 1;
wheel_r = 0.15;
wheel_distance = 0.6;

figure

for i = 1:length(t)

    clf
    
    w1 = x_move(i) - wheel_distance/2;
    w2 = x_move(i) + wheel_distance/2;
    
    xb = x_move(i) + [0 L*sin(theta(i))];
    yb = [wheel_r L*cos(theta(i))];
    
    rectangle('Position',[w1-wheel_r,0,2*wheel_r,2*wheel_r],...
        'Curvature',[1 1],'FaceColor','k')
    hold on
    
    rectangle('Position',[w2-wheel_r,0,2*wheel_r,2*wheel_r],...
        'Curvature',[1 1],'FaceColor','k')
    
    plot(xb,yb,'r','LineWidth',4)
    
    plot(xb(2),yb(2),'bo','MarkerSize',12,'MarkerFaceColor','b')
    
    axis([-2 4 0 1.5])
    grid on
    
    title('REALISTIC: Forward + Backward + Balance')
    
    drawnow

end