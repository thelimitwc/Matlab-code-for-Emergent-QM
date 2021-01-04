function B=Flatten_potential(A,L)
B = A;
for i= 1 : randi(20)
for k= 2 : L
    B(k) = 0.5*(B(k-1)+B(k));
end
end



end

