rng('default')

[u, v] = meshgrid(10:0.5:20, 10:0.5:20);
u = reshape(u, [], 1);
v = reshape(v, [], 1);
% w = reshape(w, [], 1);

z = 0.5 - 10 * u - (3.2 * v);
r1 = randn(size(z)) * 5;
% z = z + r1;

T = table(u, v, z, 'VariableNames', {'x', 'y', 'z'});
writetable(T,'output.csv');