%Density data

L = 400; number = 15000;
beta = 0.5; a=0.1;
Input = zeros(number , L);
Output_q = zeros(number, L);
Output_th = zeros(number,L);

for j= 1 :number


s = -2*rand(L,1)-2*rand; s(1) = -1; s = Flatten_potential(s,L);
A = zeros( L,1) ; B = zeros(L,1); rho =zeros(L,1);psi = zeros(L,1);

K = sqrt(-s);

A(1) = 1 ;  B(1) = 1;

for i = 2 : L 
    x = (i-1)*a; k1 = K(i-1); k2 = K(i); A1 = A(i-1); B1 =B(i-1);
    A(i)  = (sin(k1*x)*(-B1*k1*cos(k2*x)+A1*k2*sin(k2*x)) +cos(k1*x)*(A1*k1*cos(k2*x)+B1*k2*sin(k2*x)))/k2;
    B(i) = (cos(k2*x)*(cos(k1*x)*(B1*k2 - A1*k1*tan(k2*x))+sin(k1*x)*(A1*k2+B1*k1*tan(k2*x))))/k2;
end

for k = 1 : L 
    psi(k) = A(k)*sin(K(k)*k*a)+B(k)*cos(K(k)*k*a);
    rho(k) = psi(k)^2;
    
end
Input(j,:) = s;
Output_q(j,:) = rho;
Output_th(j,:) = exp(-beta*s);
end