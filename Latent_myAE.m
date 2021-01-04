L=400; nz =2 ; n =6;
for k = 1: L
    pp(k) = k;
end
s = -ones(L,1);
ZZ = zeros(L,nz);

Z0= ones(nz,1);
[out,h]=RNN_output(V,W,s,L,h0,3);

ZZ(1,:) = WR{1}*Z0 +s(1)* WR{2}*Z0; 

for k = 2 :L
    ZZ(k,:) = WR{1}*ZZ(k-1,:)'+s(k)*WR{2}*ZZ(k-1,:)'; 
    
end


plot(pp,out,pp,ZZ(:,1),pp,ZZ(:,2)); 
drawnow;