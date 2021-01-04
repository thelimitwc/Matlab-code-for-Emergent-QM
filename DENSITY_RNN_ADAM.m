Data_rho = Output_q;
Data_v = Input;

n = 6;%hidden units
len = 55; start = 5;
Len_data = 400;
l=0;
echo = 1;
loss= 0;loss_p = 0;

V = zeros(1,n); V(1)=1;   

lossf = 1;




K= 3;
gamma = 0;
for i = 1 :K
    W{i} = rand(n)*0.001/n;
end
W{1} = W{1} + diag(ones(n,1));

pp = zeros(Len_data,1);
for k = 1: Len_data
    pp(k) = k;
end


r=0.0001;
B=50;


DeltaV =zeros(1,n);
DeltaU = zeros(1,2);


%ADAM
beta1=0.9;
beta2=0.999;
epsilon=0.1;


     for i= 1: K
         DW_m1{i} = zeros(n);
     end




     for i= 1: K
         DW_m2{i} = zeros(n);
     end
     


for i = 1: 1000000000

for j = 1 :K
    DW{j} = zeros(n);
end
batch = randi(10000,B,1);
for k=1:B
    h0 = ones(n,1);
    
    
    
dW = RNN_delta(V,W,Data_v(batch(k),:),Data_rho(batch(k),:)',len,n,h0,K,start);
for j=1:K
    DW{j} = DW{j}+dW{j}/B;
end
end

lr_ = r *sqrt(1 - beta2^i)/(1 - beta1^i); 


for k = 1:K 
    
 DW_m1{k} = beta1*DW_m1{k}  + (1 - beta1)*DW{k};
 DW_m2{k} = beta2*DW_m2{k} + (1 -beta2)*DW{k}.*DW{k};
 W{k} = W{k} - lr_* DW_m1{k}./(DW_m2{k}.^(1/2)+epsilon);
 
    
end





if mod(i,200)==0
    disp(echo);loss(echo)=0; loss_p(echo)=0;
    for k = 1 : 500
        Q = randi(5000)+10000;
        h0 = ones(n,1);
        [out,h]=RNN_output(V,W,Data_v(Q,:),len+100,h0,K);
        rho = Data_rho(Q,1:len+100)';
        loss(echo) = loss(echo) + sum((out(start:len) - rho(start:len)).^2)/(500*51);
   end

   disp(loss(echo));
   if loss(echo) < lossf
       lossf = loss(echo);
       save('bestresult','V','W');
   end
  
    [out,h]=RNN_output(V,W,Data_v(Q,:),Len_data,h0,K);
    rho = Data_rho(Q,1:Len_data)'; 
    plot(pp,rho,pp,out);

   
    drawnow
    echo = echo + 1;
end

end