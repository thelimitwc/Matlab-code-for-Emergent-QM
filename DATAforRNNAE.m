N = 15000; n = 6; L =400; len=60;
load('bestresult');

Output_h = zeros(N,n*len);
Input_v = zeros(N,len);
h0 = ones(n,1);
for k = 1: N
s = -2*rand(L,1)-2*rand; s(1) = -1; s = Flatten_potential(s,L);
[out,h]=RNN_output(V,W,s,len,h0,3);
Input_v(k,:) = s(1:len);
Output_h(k,1:n) =h0; 
for i = 2 :len
Output_h(k,1+(i-1)*n:i*n) = h{i-1};
end


end
