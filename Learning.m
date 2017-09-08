function [v, w, v0, w0, epoch, error] = Learning(input, J, output, alpha, ERROR, nguyen, momentum)

[~,I] = size(input);
[P,K] = size(output);

Zin = zeros(1,J);
e_Zin = zeros(1,J);
Z = zeros(1,J);
Yin = zeros(1,K);
e_Yin= zeros(1,K);
Y = zeros(1,K);
dfy = zeros(1,K);
do_k = zeros(1,K);    
delta_w0 = zeros(1,K);
delta_w = zeros(J,K);
dfz = zeros(1,J);
do_in = zeros(1,J);
do_j = zeros(1,J);
delta_v0 = zeros(1,J);
delta_v = zeros(I,J);
delta_v_prv = zeros(I,J);
delta_v0_prv = zeros(1,J);
delta_w_prv = zeros(J,K);
delta_w0_prv = zeros(1,K);

% Inisialisasi bobot
v = -0.5 + (0.5-(-0.5)).*rand(I,J);
v0 = -0.5 + (0.5-(-0.5)).*rand(1,J);
w = -0.5 + (0.5-(-0.5)).*rand(J,K);
w0 = -0.5 + (0.5-(-0.5)).*rand(1,K);

%Nguyen
if nguyen==1
    vj = zeros(1,J);
    beta = 0.7*(J^(1/I));
    
    for j=1:J
        for i=1:I
            vj(j) = vj(j) + v(i,j);
        end
        
        for i=1:I
            v(i,j) = beta*v(i,j)/vj(j);
        end
    end
    
    v0 = -beta + (beta-(-beta)).*rand(1,J);
end

error = 2;
epoch = 0;

while error>ERROR && epoch<100000;
    
    error = 0;
    
    for p=1:P
        % Feedforward
        % Hidden Layer
        
        for j = 1:J
            Zin(j) = v0(j);

            for i = 1:I;
                Zin(j) = Zin(j)+(input(p,i)*v(i,j));
            end

            e_Zin(j) = exp(-Zin(j));
            Z(j) = 1./(1+e_Zin(j));
        end

        % Output Layer
        for k = 1:K
            Yin(k) = w0(k);

            for j = 1:J
                Yin(k) = Yin(k)+(Z(j)*w(j,k));
            end

            e_Yin(k) = exp(-Yin(k));
            Y(k) = 1./(1+e_Yin(k));
        end

        % Backpropagation
        % Menghitung Error dan Nilai w baru
        for k = 1:K
            error = error + ((0.5*(output(p,k)-Y(k)))^2);
            
            dfy(k) = e_Yin(k)/((1+e_Yin(k))^2);
            do_k(k) = (output(p,k)-Y(k))*dfy(k);
            
            delta_w0(k) = alpha*do_k(k);
            for j = 1:J
                delta_w(j,k) = alpha*Z(j)*do_k(k);
            end
        end

        % Menghitung Nilai v baru
        do_in(j) = 0;
            
        for j = 1:J
            do_in(j) = 0;
            for k = 1:K
                do_in(j) = do_in(j) + do_k(k)*w(j,k);
            end
            dfz(j) = e_Zin(j)/((1+e_Zin(j))^2);
            do_j(j) = do_in(j)*dfz(j);
            
            delta_v0(j) = alpha*do_j(j);
            for i = 1:I
                delta_v(i,j) = alpha*do_j(j)*input(p,i);
            end
        end

        % Update Bobot
        w = w + delta_w + momentum*delta_w_prv;
        w0 = w0 + delta_w0 + momentum*delta_w0_prv;
        v = v + delta_v + momentum*delta_v_prv;
        v0 = v0 + delta_v0 + momentum*delta_v0_prv;
        
        delta_w_prv = delta_w;
        delta_w0_prv = delta_w0;
        delta_v_prv = delta_v;
        delta_v0_prv = delta_v0;
    end
    
    clc
    epoch = epoch + 1
    e(epoch)=error;
	error
end

figure('Renderer', 'Painters');
plot(1:epoch, e);
