t = sym("t");
t2 = sym("t2");
t1 = sym("t1");
a0 = sym("a0");
a1 = sym("a1");
a2 = sym("a2");
a3 = sym("a3");

% result = int(int(exp(a0*(t-t2))*dirac(t2)*exp(a1*(t2-t1))*dirac(t1)*exp(t1*a2), t1, 0, t2), t2, 0, t)

result2 = int(dirac(t1)*exp(t1), t1, 1, 2)