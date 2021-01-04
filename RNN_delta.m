function dW = RNN_delta( V,W,input,outreal,len,n,h0,K,Start)
[outnet,h]=RNN_output(V,W,input,len,h0,K);


for k = Start : len
    dh{k} = (outnet(k) - outreal(k))*V';
end


for k = len-1 : -1:Start
    for i = 1 :K
    dh{k} =dh{k} + W{i}' * dh{k+1}*input(k+1)^(i-1);
    end
end
for i = 1:K
    dW{i} = zeros(n);
end
for k = Start+1 :len
    M = dh{k} *h{k-1}';
    for i = 1 : K
    dW{i} = dW{i} +  M*input(k)^(i-1);
    end
   
    
end

if Start ~= 1
     M = dh{Start} *h{Start-1}';
    for i = 1 : K
    dW{i} = dW{i} +  M*input(Start)^(i-1);
    end
else
 M =  dh{1} *h0';
    for i= 1 : K
    dW{i} = dW{i} + M*input(1)^(i-1);
    end
end


