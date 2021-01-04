function  [dW1,db1,dW2,db2,dWR,loss] = EncoderRNN_delta(Y,W1,b1,W2,b2,WR,v,K,len,n,m,N)
loss =0;
for k= 1 :len
    %h{k} = zeros(n,1); 
    z{k}=zeros(m,1);
    s{k}=zeros(N,1);
    y{k}=zeros(n,1); 
    %dz{k} =zeros(m,1);
end

z{1} = ones(m,1); 
for k = 2 :len
    for i = 1 : K
        z{k}= z{k} + WR{i}*z{k-1}*v(k-1)^(i-1);
    end
end

for k = 1:len 
    y{k} = Y(1+(k-1)*n : k*n);
    s{k}=Ramp(W1*z{k}+b1);
    h{k}=W2*s{k}+b2;
end






dW2 = zeros(n,N); db2 = zeros(n,1);dW1=zeros(N,m);db1 = zeros(N,1);

for k = 1 : len
    delta_h = h{k}-y{k};
    dW2 = dW2  + delta_h*s{k}'; db2 = db2 + delta_h; ds{k} = W2'*(delta_h);
    delta_s = (ds{k}.*(s{k}>0));
    dW1 = dW1 + delta_s*z{k}'; db1 = db1 + delta_s; loss = loss + sum(delta_h.^2);
    dz{k} = W1'*delta_s;
end



for k = len-1 : - 1 : 1
   for i = 1 :K 
       dz{k} = dz{k} + WR{i}' * dz{k+1}*v(k+1)^(i-1);
   end
end

for i = 1:K
    dWR{i} = zeros(m);
end


for k = 2 :len
    M = dz{k} *z{k-1}';
    for i = 1 : K
    dWR{i} = dWR{i} +  M*v(k)^(i-1);
    end
end












end

