Yxs = 0.4;   % g/g
beta = 0.2;  % h^(-1)
Pm = 50;     % g/L
Ki = 22;     % g/L
alpha = 2.2; % g/g
mum = 0.48;  % h^(-1)
Km = 1.2;    % g/L
Sf = 20;     % g/L
D0 = 0.202;  % 1/h
deltaD = 0.025 * D0;   % same as in your code
tfinal = 300;          % simulation time

% Get steady state using your existing nonlinear model:
x0_guess = [0, 2, 0];
mu = @(S,p) (mum * (1 - (p / Pm)) * S) / (Km + S + (S.^2) / Ki);
f = @(x,u) [
    -u * x(1) + mu(x(2), x(3)) * x(1);
    u * (Sf - x(2)) - (1 / Yxs) * mu(x(2), x(3)) * x(1);
    -u * x(3) + (alpha * (mu(x(2), x(3))) + beta) * x(1)
];
opts = optimoptions("fsolve","Display","off");
x_ss = fsolve(@(x) f(x, D0), x0_guess, opts);
X_ss = x_ss(1); S_ss = x_ss(2); P_ss = x_ss(3);


% Initial conditions = steady state
X0 = X_ss;
S0 = S_ss;
P0 = P_ss;
xss=transpose(x_ss);
%P_sp=P_ss*1.07;
x0=xss;
