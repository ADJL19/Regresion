rng('default')
[u, v] = meshgrid(10:0.5:20, 10:0.5:20);
z = 0.5 - 10 * u - (3.2 * v);
r1 = randn(size(z)) * 2.5;
z = z + r1;
figure
surf(u, v, z)