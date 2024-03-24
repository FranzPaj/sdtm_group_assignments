clear all, close all, clc

addpath ../Pc2D_Foster
addpath(genpath('../Utils'))

Accuracy        = 0.1; % Desired MC Accuracy (0.01=1%)
MC_Confidence   = 0.95; % Desired Monte Carlo Confidence Level
GM              = 3.986004418e14;% Earth gravitational constant mu = GM (EGM-96) [m^3/s^2] 
LowVelThreshold = 0.05; % 5% of Orbital Period
expSolution     = 4.20E-01;

filename = 'data_files/obj_dict.txt';
M = readmatrix(filename);
num_obj = size(M,1);

%%

for i = 1:num_obj

    noradid1 = int2str(int32(M(i,45)));
    A1 = M(i,46);
    r1 = M(i,47:49);
    v1 = M(i,50:52); 
    P1 = M(i,53:88);
    P1 = reshape(P1,[6,6]);
    radius1 = sqrt(A1 / pi);
    
    noradid2 = int2str(int32(M(i,1)));
    A2 = M(i,2);
    r2 = M(i,3:5);
    v2 = M(i,6:8); 
    P2 = M(i,9:44);
    P2 = reshape(P1,[6,6]);
    radius2 = sqrt(A2 / pi);
    
    HBR = radius1 + radius2;

    disp('-------------------------------------------')
    disp(join(['RSO: ',noradid2]))
    
    
    % Save data and plots in right folders
    out_dir = join(['./outputs/',noradid2]);
    if not(isfolder(out_dir))
           mkdir(out_dir)
    end
    
    plot_dir = join(['./plots/',noradid2]);
    if not(isfolder(plot_dir))
           mkdir(plot_dir)
    end
    
    %%
    
    % Change to NASA convention - weird mixed measurement units
    % r_J2K and v_J2K in km, C_J2K in m
    r1_J2K = r1 / 1000;
    v1_J2K = v1 / 1000;
    C1_J2K = P1;
    r2_J2K = r2 / 1000;
    v2_J2K = v2 / 1000;
    C2_J2K = P2;
    
    % Calculate 2D PC            
    Pc2D = Pc2D_Foster(r1_J2K*1000,v1_J2K*1000,C1_J2K,r2_J2K*1000,v2_J2K*1000,C2_J2K,HBR,1e-8,'circle');
    
    % If PC too small -> it's going to break
    % Pass to the next object
    if Pc2D < 1e-6
        filename = join([out_dir, '/Pc_foster.dat']);
        writematrix(Pc2D,filename) 
        warning(['object ignored as Pc2D < 1e-6'])
        continue
    end 


    % Estimate Number of Samples
    Nsample_kep = EstimateRequiredSamples(Pc2D,Accuracy*0.75,MC_Confidence);
    Nsample_kep = max(1e2,Nsample_kep);
    Nsample_kep = min(1e10,Nsample_kep);
    
    % Determine Batch Size min 1000, max 5000
    p = gcp;
    Nsample_batch = max([min([ceil(Nsample_kep/p.NumWorkers/1000)*1000 5000]) 1000]);
    
    % Get Primary Objecty Orbital Period
    a = -GM/2/(norm(v1_J2K*1000)^2/2-GM/norm(r1_J2K*1000));
    OrbitalPeriod = 2*pi()*sqrt(a^3/GM);
    
    % Get Bounds
    [tau0,tau1] = conj_bounds_Coppola(1e-16, HBR, (r2_J2K'-r1_J2K')*1000, (v2_J2K'-v1_J2K')*1000, C2_J2K+C1_J2K);
    
    % Check if event has low relative velocity
    if (tau1-tau0) > LowVelThreshold*OrbitalPeriod
        warning(['Event is classified as a low speed event: Coppola encounter time exceeds ' num2str(LowVelThreshold*100,'%.1f') '% of the Primary Object''s orbital period']);
        warning(['Input encounter time will be modified to be no more than ' num2str(LowVelThreshold*100,'%.1f') '% of the orbital period'])
        tau1    = LowVelThreshold*OrbitalPeriod*tau1/(tau1-tau0);
        tau0    = tau1-LowVelThreshold*OrbitalPeriod;
        fprintf(['Conjunction duration bounds (s) modified to: Dur = ' num2str(tau1-tau0,'%0.4f') ' (' num2str(tau0,'%0.4f') ' < t-TCA < ' num2str(tau1,'%0.4f') ')\n\n'])
    end
    tfc_bins.N   = 100;
    tfc_bins.x1  = tau0;
    tfc_bins.x2  = tau1;
    tfc_bins.wid = (tfc_bins.x2-tfc_bins.x1);
    tfc_bins.del = tfc_bins.wid/tfc_bins.N;
    tfc_bins.xhi = tfc_bins.x1+(1:tfc_bins.N)*tfc_bins.del;
    tfc_bins.xlo = tfc_bins.xhi-tfc_bins.del;
    tfc_bins.xmd = tfc_bins.xhi-tfc_bins.del/2;
    
    % Run MC Code
    X1TCA   = [r1_J2K v1_J2K]; % Cartesian state at TCA (km)
    [KEP] = Cart2Kep([r1_J2K v1_J2K],'Mean','Rad');
    if abs(KEP(3))>.95*pi && abs(KEP(3))<1.05*pi 
        fr      = -1;
    else
        fr      = 1;
    end
    [~,n,af,ag,chi,psi,lM,F] = convert_cartesian_to_equinoctial(r1_J2K,v1_J2K,fr);
    E1TCA   = [n af ag chi psi lM fr];
    Jctoe1 = jacobian_equinoctial_to_cartesian(E1TCA,X1TCA,fr);
    Jctoe1 = inv(Jctoe1);
    PEq1TCA = Jctoe1 * (C1_J2K/1e6) * Jctoe1'; % Equinoctial covariance at TCA
    
    X2TCA   = [r2_J2K v2_J2K]; % Cartesian state at TCA (km)
    [KEP] = Cart2Kep([r2_J2K v2_J2K],'Mean','Rad');
    if abs(KEP(3))>.95*pi && abs(KEP(3))<1.05*pi 
        fr      = -1;
    else
        fr      = 1;
    end
    [~,n,af,ag,chi,psi,lM,F] = convert_cartesian_to_equinoctial(r2_J2K,v2_J2K,fr);
    E2TCA   = [n af ag chi psi lM fr];
    Jctoe2 = jacobian_equinoctial_to_cartesian(E2TCA,X2TCA,fr);
    Jctoe2 = inv(Jctoe2);
    PEq2TCA = Jctoe2 * (C2_J2K/1e6) * Jctoe2'; % Equinoctial covariance at TCA
       
    
    
    % Equinoctial Sampling
    [~, ~, ~,...
         actSolution, Uc_EQN_MC_att, Nc_EQN_MCtfc_att, ...
         ~  , ~  , ~] = Pc_MC_Kep2body_parallel(...
            Nsample_kep, tfc_bins,...
            E1TCA', [], PEq1TCA,...
            E2TCA', [], PEq2TCA,...
            HBR, GM, 'k2bpri',...
            MC_Confidence, Nsample_batch, plot_dir);
    
    
    filename = join([out_dir, '/Pc_foster.dat']);
    writematrix(Pc2D,filename)  
    
    filename = join([out_dir, '/Pc_MonteCarlo.dat']);
    writematrix(actSolution,filename)  

    filename = join([out_dir, '/NumSamples_MonteCarlo.dat']);
    writematrix(Nsample_kep,filename)  

end





    
