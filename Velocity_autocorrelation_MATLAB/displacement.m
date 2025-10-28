function displ = displacement(x,y,step)

displ = zeros(1,length(x));


for i = 1:step:length(x)-step
   displ(i) = sqrt( ( (x(i+step)-x(i))^2 + (y(i+step)-y(i))^2 ) );
end





end