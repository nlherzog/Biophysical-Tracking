function [next_step,step_next_size] = next_step(x,y)

step_next_size = zeros(1,floor((length(x)-2)));
next_step = zeros(1,floor((length(x)-2)));

for i = 2:length(x)-1
    next_step(i) = ( (x(i+1)-x(i))*(x(i)-x(i-1)) + (y(i+1)-y(i))*(y(i)-y(i-1)) ) / sqrt( (x(i)-x(i-1))^2 + (y(i)-y(i-1))^2 );
    step_next_size(i) = sqrt( (x(i)-x(i-1))^2 + (y(i)-y(i-1))^2 );
end

end