n=6; m=2 ; N=50; INI=0.1/N;
len = 60;echo=1;


[W1,b1] = myintialize(N,m,INI);
[W2,b2] = myintialize(n,N,INI);

r = 0.001;
K = 2;B=5;
echo = 1;
for k = 1: K
    WR{k} =0.01*randn(m,m);
end
WR{1} = WR{1} + eye(m);


%ADAM
beta1=0.9;
beta2=0.999;
epsilon=1e-08;
[DW1_m1,Db1_m1] = myintialize(N,m,0);
[DW2_m1,Db2_m1] = myintialize(n,N,0);

     for i= 1: K
         DWR_m1{i} = zeros(m);
     end


[DW1_m2,Db1_m2] = myintialize(N,m,0);
[DW2_m2,Db2_m2] = myintialize(n,N,0);

     for i= 1: K
         DWR_m2{i} = zeros(m);
     end
     
     
     
     
     
     
     

for p= 1 : 100000000
[DW1,Db1] = myintialize(N,m,0);
[DW2,Db2] = myintialize(n,N,0);

     for i= 1: K
         DWR{i} = zeros(m);
     end






 for j = 1 :B   
     
     q = randi(10000);
     v = Input_v(q,:);
     Y = Output_h(q,:)';
     
     
     [dW1,db1,dW2,db2,dWR,loss] = EncoderRNN_delta(Y,W1,b1,W2,b2,WR,v,K,len,n,m,N);

     
DW1 = DW1 + dW1/B; 
DW2 = DW2 + dW2/B;

Db1 = Db1 + db1/B;
Db2 = Db2 + db2/B;

for k= 1: K
       DWR{k} = DWR{k}+dWR{k}/B;
end 
     
     
     
     
 end
 
 
lr_ = r *sqrt(1 - beta2^p)/(1 - beta1^p); 
 
 DW1_m1 = beta1*DW1_m1  + (1 - beta1)*DW1;
 DW1_m2 = beta2*DW1_m2 + (1 -beta2)*DW1.*DW1;
 W1 = W1 - lr_* DW1_m1./(DW1_m2.^(1/2)+epsilon);


 DW2_m1 = beta1*DW2_m1  + (1 - beta1)*DW2;
 DW2_m2 = beta2*DW2_m2 + (1 -beta2)*DW2.*DW2;
 W2 = W2 - lr_* DW2_m1./(DW2_m2.^(1/2)+epsilon);

 Db1_m1 = beta1*Db1_m1  + (1 - beta1)*Db1;
 Db1_m2 = beta2*Db1_m2 + (1 -beta2)*Db1.*Db1;
 b1 = b1 - lr_* Db1_m1./(Db1_m2.^(1/2)+epsilon);
 
 
 Db2_m1 = beta1*Db2_m1  + (1 - beta1)*Db2;
 Db2_m2 = beta2*Db2_m2 + (1 -beta2)*Db2.*Db2;
 b2 = b2 - lr_* Db2_m1./(Db2_m2.^(1/2)+epsilon);
  
 
 


for k = 1:K 
    
 DWR_m1{k} = beta1*DWR_m1{k}  + (1 - beta1)*DWR{k};
 DWR_m2{k} = beta2*DWR_m2{k} + (1 -beta2)*DWR{k}.*DWR{k};
 WR{k} = WR{k} - lr_* DWR_m1{k}./(DWR_m2{k}.^(1/2)+epsilon);
 
    
end




if ( mod(p,1000/B)==0)
    Latent_myAE; loss(echo)=0;
    
    for k = 1 : 500 
 
    Q = randi(5000)+10000;
    
     v = Input_v(Q,:);
     Y = Output_h(Q,:)';
    
     
     
     loss(echo) = loss(echo) + EncoderRNN_loss(Y,W1,b1,W2,b2,WR,v,K,len,n,m,N)/500;
     end
    disp(echo);disp(loss(echo));echo = echo + 1; 
end
end
 
 
 
 
 
 
 
 
 
