import h5py
import numpy as np
import matplotlib.pyplot as plt
import sys
from collections import Counter
import seaborn as sns
import matplotlib.gridspec as gridspec
import pandas as pd

#Plots
plot_Rvalues                = False
R_tables                    = True

plot_ModelIndicator         = False
plot_ModelIndicator_trials  = False

plot_Wvalues                = False
plot_weighdist              = False
plot_weighthyper            = False
plot_alphaGain_pg           = False
plot_alphaGain_pg_hyper     = False
plot_beta_pg                = False

plot_alphaGain_p            = False
plot_alphaGain_p_hyper      = False
plot_alphaLoss_p            = False
plot_alphaLoss_p_hyper      = False
plot_lambda_p               = False
plot_lambda_p_hyper         = False
plot_beta_p                 = False

fig_nr = 1


#Prior 1 on PT_original (to retrieve parameters)
# file_name = 'JAGS_test_PT_PG_v4_PT_prior_1_allData_burn_100_samps_50_chains_4_04-30-2020 12-49'

#Flat prior on PT_original and PT_gain
file_name = 'JAGS_test_PT_PG_v4_FlatZprior_allData_burn_10000_samps_5000_chains_4_04-30-2020 05-23'

with h5py.File(f'samples_stats/{file_name}.mat', 'r') as file:
    print(list(file.keys()))

    print(list(file['samples'].keys()))

    z = file['samples'].get('z').value 
    
    ##PT_gain
    #Priors
    beta_p       = file['samples'].get('beta_p').value
    alphaGain_pg = file['samples'].get('alphaGain_pg').value
    beta_pg      = file['samples'].get('beta_pg').value
    w_pg         = file['samples'].get('w_pg').value

    #Hyperpriors
    mu_log_alphaGain_pg     = file['samples'].get('mu_log_alphaGain_pg').value
    tau_log_alphaGain_pg    = file['samples'].get('tau_log_alphaGain_pg').value
    sigma_log_alphaGain_pg  = file['samples'].get('sigma_log_alphaGain_pg').value
    mu_log_beta_pg          = file['samples'].get('mu_log_beta_pg').value
    tau_log_beta_pg         = file['samples'].get('tau_log_beta_pg').value
    sigma_log_beta_pg       = file['samples'].get('sigma_log_beta_pg').value
    weight_a_pg             = file['samples'].get('weight_a_pg').value
    weight_b_pg             = file['samples'].get('weight_b_pg').value

    ##PT_original
    #Priors
    alphaGain_p  = file['samples'].get('alphaGain_p').value
    alphaLoss_p  = file['samples'].get('alphaLoss_p').value
    lambda_p     = file['samples'].get('lambda_p').value

    #Hyperpriors
    mu_log_alphaGain_p      = file['samples'].get('mu_log_alphaGain_p').value
    tau_log_alphaGain_p     = file['samples'].get('tau_log_alphaGain_p').value
    sigma_log_alphaGain_p   = file['samples'].get('sigma_log_alphaGain_p').value
    mu_log_alphaLoss_p      = file['samples'].get('mu_log_alphaLoss_p').value
    tau_log_alphaLoss_p     = file['samples'].get('tau_log_alphaLoss_p').value
    sigma_log_alphaLoss_p   = file['samples'].get('sigma_log_alphaLoss_p').value
    mu_log_lambda_p         = file['samples'].get('mu_log_lambda_p').value
    tau_log_lambda_p        = file['samples'].get('tau_log_lambda_p').value
    sigma_log_lambda_p      = file['samples'].get('sigma_log_lambda_p').value
    mu_log_beta_p           = file['samples'].get('mu_log_beta_p').value
    tau_log_beta_p          = file['samples'].get('tau_log_beta_p').value
    sigma_log_beta_p        = file['samples'].get('sigma_log_beta_p').value

    #R_hat values
    print(list(file['stats']['Rhat'].keys()))
    R_z = file['stats']['Rhat'].get('z').value

    R_alphaGain_p  = file['stats']['Rhat'].get('alphaGain_p').value
    R_alphaLoss_p  = file['stats']['Rhat'].get('alphaLoss_p').value
    R_lambda_p     = file['stats']['Rhat'].get('lambda_p').value
    R_beta_p       = file['stats']['Rhat'].get('beta_p').value
    R_alphaGain_pg = file['stats']['Rhat'].get('alphaGain_pg').value
    R_beta_pg      = file['stats']['Rhat'].get('beta_pg').value
    R_w_pg         = file['stats']['Rhat'].get('w_pg').value

pt_model = [2,4,6,8]
pg_model = [1,3,5,7]
# pt_model = [1,3,5,7]
# pg_model = [2,4,6,8]

n_subjects = np.shape(w_pg)[0]
n_samples = np.shape(w_pg)[1]
n_trials = np.shape(w_pg)[2]

print("Number of subjects:", n_subjects)
print("Number of samples:", n_samples)
print("Number of trials:", n_trials)

## R-values
if plot_Rvalues:
    #Model indicator
    max_r_value = 1.1
    plt.figure(fig_nr)
    fig_nr += 1
    plt.title("R_hat values for z")
    plt.plot(range(n_subjects), R_z[0], 'o')
    plt.xlim(-1,18)
    plt.axhline(y=max_r_value, color='r', linestyle='--')
    
    #PT_gain
    plt.figure(fig_nr)
    fig_nr += 1
    plt.suptitle("Rhat values for PT_gain")
    plt.subplot(311)
    plt.title("R_alphaGain_pg")
    plt.plot(range(n_subjects),R_alphaGain_pg[0], 'o')
    plt.xlim(-1,18)
    plt.axhline(y=max_r_value, color='r', linestyle='--')
    plt.subplot(312)
    plt.title("R_beta_pg")
    plt.plot(range(n_subjects),R_beta_pg[0], 'o')
    plt.xlim(-1,18)
    plt.axhline(y=max_r_value, color='r', linestyle='--')
    plt.subplot(313)
    plt.title("R_w_pg")
    plt.plot(range(n_subjects),R_w_pg[0], 'o')
    plt.xlim(-1,18)
    plt.axhline(y=max_r_value, color='r', linestyle='--')

    #PT_original
    plt.figure(fig_nr)
    fig_nr += 1
    plt.suptitle("Rhat values for PT_original")
    plt.subplot(311)
    plt.title("R_alphaGain_p")
    plt.plot(range(n_subjects),R_alphaGain_p[0], 'o')
    plt.xlim(-1,18)
    plt.axhline(y=max_r_value, color='r', linestyle='--')
    plt.subplot(312)
    plt.title("R_alphaLoss_p")
    plt.plot(range(n_subjects),R_alphaLoss_p[0], 'o')
    plt.xlim(-1,18)
    plt.axhline(y=max_r_value, color='r', linestyle='--')
    plt.subplot(313)
    plt.title("R_beta_p")
    plt.plot(range(n_subjects),R_beta_p[0], 'o')
    plt.xlim(-1,18)
    plt.axhline(y=max_r_value, color='r', linestyle='--')
    plt.show()

if R_tables:
    subjects = [a + 1 for a in range(18)] 
    df = pd.DataFrame({'Subject': subjects, 'Z': R_z[0], 'alpha_gain': R_alphaGain_p[0], 'alpha_loss': R_alphaLoss_p[0], 'Lambda': R_lambda_p[0], 'beta_add_pt': R_beta_p[0],'beta_mul_pt': R_beta_p[1], \
        'alpha': R_alphaGain_pg[0], 'Weights': R_w_pg[0], 'beta_add_pg': R_beta_pg[0],'beta_mul_pg': R_beta_pg[1]})
    df = df.round(4)
    print(df.to_latex(index=False))

########### Modelindicator ###############
if plot_ModelIndicator:
    #Looking at the distribution of model indicator for all samples*trials
    z_sample_choices_extended_subjects = np.empty([n_samples*n_trials , n_subjects])
    z_sample_choices_extended_subjects_helper = []
    z_sample_choices_subjects = []
    z_sample_choices_extended = []
    z_sample_choices = []

    PT = 0
    PG = 0
    for i in range(n_subjects): 
        tmp =  []
        PT_tmp = 0
        PG_tmp = 0
        for j in range(n_samples):
            for l in range(n_trials):
                #Append for extended
                tmp.append(z[i,j,l])

                z_sample_choices_extended.append(z[i,j,l])

                #Condence choices
                if z[i,j,l] in pt_model:
                    PT += 1
                    PT_tmp += 1
                elif z[i,j,l] in pg_model:
                    PG += 1
                    PG_tmp += 1
                else:
                    print("\n\n\nWARNING: Something is wrong!!!\n\n\n")
        
        z_sample_choices_extended_subjects[:,i] = tmp
        z_sample_choices_extended_subjects_helper.append(Counter(tmp))
        z_sample_choices_subjects.append([PT_tmp, PG_tmp])
    z_sample_choices.append([PT,PG])

    print(z_sample_choices_subjects)

    x = ['PT', 'PG']
    plt.figure(fig_nr)
    fig_nr += 1
    plt.suptitle("Model indicator for each Subject")
    for j in range(n_subjects):
        plt.subplot(6,3,j+1)
        plt.title("subject %.0f - Rhat = %.2f" %(j+1, R_z[0][j]))
        plt.bar(x, z_sample_choices_subjects[j])
        plt.ylim([0,20000])
        plt.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9, wspace=0.2, hspace=0.5)

    plt.figure(fig_nr)
    fig_nr += 1
    plt.suptitle("Model indicator for each Subject (extended)")
    for j in range(n_subjects):
        plt.subplot(6,3,j+1)
        plt.title(f"subject {j+1}")
        # print(f"\nSubject {j+1} has the following: {z_sample_choices_extended_subjects_helper[j]}")
        plt.hist(z_sample_choices_extended_subjects[:,j], bins = 8)
        plt.xlim([1,8])
        plt.ylim([0,5500])
        plt.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9, wspace=0.2, hspace=0.5)

    plt.figure(fig_nr)
    fig_nr += 1
    plt.title("Modelindicator")
    plt.bar(x,z_sample_choices[0])

    plt.figure(fig_nr)
    fig_nr += 1
    plt.title("Modelindicator (extended)")
    plt.hist(z_sample_choices_extended, bins= 8)
    plt.show()

if plot_ModelIndicator_trials:
    #Looking at the distribution of model indicator for all samples for each trial
    z_sample_choices_trials = []
    tmp2 = []
    for i in range(n_subjects):
        tmp1 = []
        for l in range(n_trials):
            PT_tmp = 0
            PG_tmp = 0
            for j in range(n_samples):

                #Condence choices
                if z[i,j,l] in pt_model:
                    PT_tmp += 1
                elif z[i,j,l] in pg_model:
                    PG_tmp += 1
                else:
                    print("\n\n\nWARNING: Something is wrong!!!\n\n\n")
            tmp1.append([PT_tmp, PG_tmp])
        tmp2.append(tmp1)
    z_sample_choices_trials.append(tmp2)

    x = ['PT', 'PG']
    for i in range(n_subjects):
        print(i)
        plt.figure(fig_nr)
        fig_nr += 1
        plt.suptitle(f"Subject {i+1}")
        for l in range(n_trials):
            plt.subplot(4,1,l+1)
            plt.title(f"Trial {l+1}")
            plt.bar(x,z_sample_choices_trials[0][i][l])
            plt.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9, wspace=0.2, hspace=0.5)
    plt.show()

########### PT_gain ###############
##Weights
if plot_Wvalues:
    ## plotting all the weights
    for i in range(n_subjects):
        plt.figure(fig_nr)
        fig_nr += 1
        plt.subplot(411)
        plt.title(f"Weights for subject {i+1}")
        plt.plot(range(n_samples), w_pg[i,:,0], color = "red")
        plt.subplot(412)
        plt.plot(range(n_samples), w_pg[i,:,1], color = "red")
        plt.subplot(413)
        plt.plot(range(n_samples), w_pg[i,:,2], color = "red")
        plt.subplot(414)
        plt.plot(range(n_samples), w_pg[i,:,3], color = "red")
    plt.show()

## W-distributions
if plot_weighdist:
    w_pg_all = []
    w_pg_means = []
    w_pg_ = []
    for i in range(n_subjects):
        tmp = []
        for j in range(n_samples):
            for l in range(n_trials):
                if z[i,j,l] in pg_model:
                    w_pg_all.append(w_pg[i,j,l])
                    tmp.append(w_pg[i,j,l])
        w_pg_means.append(np.mean(tmp))
        w_pg_.append(tmp)

    plt.figure(fig_nr)
    fig_nr += 1
    plt.suptitle("Subjective weights for Weighted gain")
    for j in range(n_subjects):
        plt.subplot(6,3,j+1)
        plt.title(f"Subject {j+1} - Chosen: {len(w_pg_[j])/(n_samples*n_trials)*100:.3f} %")
        plt.hist(w_pg_[j])
        plt.axvline(np.mean(w_pg_[j]),color='red')
        plt.xlim([0,1])
        plt.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9, wspace=0.2, hspace=0.5)

    plt.figure(fig_nr)
    fig_nr += 1
    plt.title(f"Total weightdistribution for Weighted gain - Chosen: {len(w_pg_all)/(n_samples*n_trials*n_subjects)*100:.3f} %")
    # plt.hist(w_pg_all, bins = 1000)
    sns.kdeplot(w_pg_all)
    plt.xlim([0,1])
    [plt.axvline(_w, linewidth=1, color='red') for _w in w_pg_means]
    plt.show()

if plot_weighthyper:
    w_a_ = []
    w_b_ = []
    w_h_ = []

    for l in range(n_trials):
        for j in range(n_samples):
            w_a_.append(weight_a_pg[j,l])
            w_b_.append(weight_b_pg[j,l])
            w_h_.append(weight_a_pg[j,l] - weight_b_pg[j,l])

    plt.figure(fig_nr)
    fig_nr += 1
    plt.suptitle("Distributions for hyperpriors on weights")
    plt.subplot(211)
    plt.title(f"Distribution for dfference in each sample in a and b (a-b)")
    sns.kdeplot(w_h_)
    plt.subplot(223)
    plt.title(f"Distribution for a")
    # plt.hist(w_pg_all, bins = 1000)
    sns.kdeplot(w_a_)
    plt.subplot(224)
    plt.title(f"Distribution for b")
    # plt.hist(w_pg_all, bins = 1000)
    sns.kdeplot(w_b_)
    plt.show()

##alphaGain_pg
if plot_alphaGain_pg:
    alphaGain_pg_all = []
    alphaGain_pg_means = []
    alphaGain_pg_ = []
    for i in range(n_subjects):
        tmp = []
        for j in range(n_samples):
            for l in range(n_trials):
                if z[i,j,l] in pg_model:
                    alphaGain_pg_all.append(alphaGain_pg[i,j,l])
                    tmp.append(alphaGain_pg[i,j,l])
        alphaGain_pg_means.append(np.mean(tmp))
        alphaGain_pg_.append(tmp)

    plt.figure(fig_nr)
    fig_nr += 1
    plt.suptitle("Distribution for alphaGain_pg")
    for j in range(n_subjects):
        plt.subplot(6,3,j+1)
        plt.title(f"Subject {j+1} - Chosen: {len(alphaGain_pg_[j])/(n_samples*n_trials)*100:.3f} %")
        plt.hist(alphaGain_pg_[j])
        plt.axvline(np.mean(alphaGain_pg_[j]),color='red')
        plt.xlim([0,1])
        plt.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9, wspace=0.2, hspace=0.5)

    plt.figure(fig_nr)
    fig_nr += 1
    plt.title(f"Total distribution for alphaGain_pg_all - Chosen: {len(alphaGain_pg_all)/(n_samples*n_trials*n_subjects)*100:.3f} %")
    sns.kdeplot(alphaGain_pg_all)
    plt.xlim([0,1])
    [plt.axvline(_w, linewidth=1, color='red') for _w in alphaGain_pg_means]
    plt.show()

if plot_alphaGain_pg_hyper:
    alphaGain_pg_mu = []
    alphaGain_pg_tau = []
    alphaGain_pg_sigma = []
    for l in range(n_trials):
        for j in range(n_samples):
            alphaGain_pg_mu.append(np.exp(mu_log_alphaGain_pg[j,l]))
            alphaGain_pg_tau.append(tau_log_alphaGain_pg[j,l])
            alphaGain_pg_sigma.append(sigma_log_alphaGain_pg[j,l])

    plt.figure(fig_nr)
    fig_nr += 1
    plt.suptitle("Distributions for hyperpriors on AlphaGain")
    plt.subplot(211)
    plt.title(f"Distribution for $\mu$")
    sns.kdeplot(alphaGain_pg_mu)
    plt.subplot(223)
    plt.title(f"Distribution for $\\tau$")
    sns.kdeplot(alphaGain_pg_tau)
    plt.subplot(224)
    plt.title(f"Distribution for $\sigma$")
    sns.kdeplot(alphaGain_pg_sigma)
    plt.show()
    

##beta_pg
if plot_beta_pg:
    beta_pg_all_add = []
    beta_pg_means_add = []
    beta_pg_add_ = []
    beta_pg_all_mul = []
    beta_pg_means_mul = []
    beta_pg_mul_ = []
    for i in range(n_subjects):
        tmp_add = []
        tmp_mul = []
        for j in range(n_samples):
            for l in range(n_trials):
                if z[i,j,l] in pg_model:
                    beta_pg_all_add.append(beta_pg[0,i,j,l])
                    tmp_add.append(beta_pg[0,i,j,l])
                    beta_pg_all_mul.append(beta_pg[1,i,j,l])
                    tmp_mul.append(beta_pg[1,i,j,l])
        beta_pg_means_add.append(np.mean(tmp_add))
        beta_pg_add_.append(tmp_add)
        beta_pg_means_mul.append(np.mean(tmp_mul))
        beta_pg_mul_.append(tmp_mul)

    fig = plt.figure(fig_nr)
    fig_nr += 1
    outer = gridspec.GridSpec(6, 3, wspace=0.5, hspace=0.8)

    for j in range(n_subjects):
        inner = gridspec.GridSpecFromSubplotSpec(1, 2,
                        subplot_spec=outer[j], wspace=0.4, hspace=0.1)
        
        ax = plt.Subplot(fig, inner[0])
        ax.set_title(f"Additive")
        t = ax.hist(beta_pg_add_[j])
        fig.add_subplot(ax)

        ax = plt.Subplot(fig, inner[1])
        t = ax.hist(beta_pg_mul_[j])
        ax.set_title(f"Multiplicative")
        fig.add_subplot(ax)
    fig.suptitle('Distribution of beta_pg')

    plt.figure(fig_nr)
    fig_nr += 1
    plt.suptitle(f"Total distribution for beta_pg - Chosen: {len(beta_pg_all_add)/(n_samples*n_trials*n_subjects)*100:.3f} %")
    plt.subplot(121)
    plt.title("Additive")
    # plt.hist(beta_pg_all_add, bins = 20)
    sns.kdeplot(beta_pg_all_add)
    plt.xlim([0,1000])
    # [plt.axvline(_w, linewidth=1, color='red') for _w in beta_pg_means_add]
    plt.subplot(122)
    plt.title("Multiplikative")
    # plt.hist(beta_pg_all_mul, bins = 20)
    sns.kdeplot(beta_pg_all_mul)
    plt.xlim([0,1000])
    # [plt.axvline(_w, linewidth=1, color='red') for _w in beta_pg_means_mul]

    plt.show()

########### PT_original ###############
##alphaGain_p
if plot_alphaGain_p:
    alphaGain_p_all = []
    alphaGain_p_means = []
    alphaGain_p_ = []
    for i in range(n_subjects):
        tmp = []
        for j in range(n_samples):
            for l in range(n_trials):
                if z[i,j,l] in pt_model:
                    alphaGain_p_all.append(alphaGain_p[i,j,l])
                    tmp.append(alphaGain_p[i,j,l])
        alphaGain_p_means.append(np.mean(tmp))
        alphaGain_p_.append(tmp)

    print(alphaGain_p_means)
    plt.figure(fig_nr)
    fig_nr += 1
    plt.suptitle("Distribution for alphaGain_p")
    for j in range(n_subjects):
        plt.subplot(6,3,j+1)
        plt.title(f"Subject {j+1} - Chosen: {len(alphaGain_p_[j])/(n_samples*n_trials)*100:.3f} %")
        plt.hist(alphaGain_p_[j])
        plt.axvline(np.mean(alphaGain_p_[j]),color='red')
        plt.xlim([0,1])
        plt.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9, wspace=0.2, hspace=0.5)

    plt.figure(fig_nr)
    fig_nr += 1
    plt.title(f"Total distribution for alphaGain_p_all - Chosen: {len(alphaGain_p_all)/(n_samples*n_trials*n_subjects)*100:.3f} %")
    sns.kdeplot(alphaGain_p_all)
    plt.xlim([0,1])
    [plt.axvline(_w, linewidth=1, color='red') for _w in alphaGain_p_means]
    # plt.show()

if plot_alphaGain_p_hyper:
    alphaGain_p_mu = []
    alphaGain_p_tau = []
    alphaGain_p_sigma = []
    for l in range(n_trials):
        for j in range(n_samples):
            alphaGain_p_mu.append(mu_log_alphaGain_p[j,l])
            alphaGain_p_tau.append(tau_log_alphaGain_p[j,l])
            alphaGain_p_sigma.append(sigma_log_alphaGain_p[j,l])

    plt.figure(fig_nr)
    fig_nr += 1
    plt.suptitle("Distributions for hyperpriors on AlphaGain")
    plt.subplot(211)
    plt.title(f"Distribution for mu")
    sns.kdeplot(alphaGain_p_mu)
    plt.subplot(223)
    plt.title(f"Distribution for tau")
    sns.kdeplot(alphaGain_p_tau)
    plt.subplot(224)
    plt.title(f"Distribution for sigma")
    sns.kdeplot(alphaGain_p_sigma)
    plt.show()
    

##alphaLoss_p
if plot_alphaLoss_p:
    alphaLoss_p_all = []
    alphaLoss_p_means = []
    alphaLoss_p_ = []
    for i in range(n_subjects):
        tmp = []
        for j in range(n_samples):
            for l in range(n_trials):
                if z[i,j,l] in pt_model:
                    alphaLoss_p_all.append(alphaLoss_p[i,j,l])
                    tmp.append(alphaLoss_p[i,j,l])
        alphaLoss_p_means.append(np.mean(tmp))
        alphaLoss_p_.append(tmp)

    plt.figure(fig_nr)
    fig_nr += 1
    plt.suptitle("Distribution for alphaLoss_p")
    for j in range(n_subjects):
        plt.subplot(6,3,j+1)
        plt.title(f"Subject {j+1} - Chosen: {len(alphaLoss_p_[j])/(n_samples*n_trials)*100:.3f} %")
        plt.hist(alphaLoss_p_[j], bins = 10)
        plt.axvline(np.mean(alphaLoss_p_[j]),color='red')
        plt.xlim([0,1])
        plt.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9, wspace=0.2, hspace=0.5)

    plt.figure(fig_nr)
    fig_nr += 1
    plt.title(f"Total distribution for alphaLoss_p_all - - Chosen: {len(alphaLoss_p_all)/(n_samples*n_trials*n_subjects)*100:.3f} %")
    sns.kdeplot(alphaLoss_p_all)
    plt.xlim([0,1])
    [plt.axvline(_w, linewidth=1, color='red') for _w in alphaLoss_p_means]
    plt.show()

if plot_alphaLoss_p_hyper:
    alphaLoss_p_mu = []
    alphaLoss_p_tau = []
    alphaLoss_p_sigma = []
    for l in range(n_trials):
        for j in range(n_samples):
            alphaLoss_p_mu.append(mu_log_alphaLoss_p[j,l])
            alphaLoss_p_tau.append(tau_log_alphaLoss_p[j,l])
            alphaLoss_p_sigma.append(sigma_log_alphaLoss_p[j,l])

    plt.figure(fig_nr)
    fig_nr += 1
    plt.suptitle("Distributions for hyperpriors on alphaLoss")
    plt.subplot(211)
    plt.title(f"Distribution for mu")
    sns.kdeplot(alphaLoss_p_mu)
    plt.subplot(223)
    plt.title(f"Distribution for tau")
    sns.kdeplot(alphaLoss_p_tau)
    plt.subplot(224)
    plt.title(f"Distribution for sigma")
    sns.kdeplot(alphaLoss_p_sigma)
    plt.show()

##lambda_p
if plot_lambda_p:
    lambda_p_all = []
    lambda_p_means = []
    lambda_p_ = []
    for i in range(n_subjects):
        tmp = []
        for j in range(n_samples):
            for l in range(n_trials):
                if z[i,j,l] in pt_model:
                    lambda_p_all.append(lambda_p[i,j,l])
                    tmp.append(lambda_p[i,j,l])
        lambda_p_means.append(np.mean(tmp))
        lambda_p_.append(tmp)

    plt.figure(fig_nr)
    fig_nr += 1
    plt.suptitle("Distribution for lambda_p")
    for j in range(n_subjects):
        plt.subplot(6,3,j+1)
        plt.title(f"Subject {j+1} - Chosen: {len(lambda_p_[j])/(n_samples*n_trials)*100:.3f} %")
        plt.hist(lambda_p_[j], bins = 10)
        plt.axvline(np.mean(lambda_p_[j]),color='red')
        plt.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9, wspace=0.2, hspace=0.5)

    plt.figure(fig_nr)
    fig_nr += 1
    plt.title(f"Total distribution for lambda_p_all- Chosen: {len(lambda_p_all)/(n_samples*n_trials*n_subjects)*100:.3f} %")
    sns.kdeplot(lambda_p_all)
    [plt.axvline(_w, linewidth=1, color='red') for _w in lambda_p_means]
    plt.show()

if plot_lambda_p_hyper:
    lambda_p_mu = []
    lambda_p_tau = []
    lambda_p_sigma = []
    for l in range(n_trials):
        for j in range(n_samples):
            lambda_p_mu.append(mu_log_lambda_p[j,l])
            lambda_p_tau.append(tau_log_lambda_p[j,l])
            lambda_p_sigma.append(sigma_log_lambda_p[j,l])

    plt.figure(fig_nr)
    fig_nr += 1
    plt.suptitle("Distributions for hyperpriors on lambda")
    plt.subplot(211)
    plt.title(f"Distribution for mu")
    sns.kdeplot(lambda_p_mu)
    plt.subplot(223)
    plt.title(f"Distribution for tau")
    sns.kdeplot(lambda_p_tau)
    plt.subplot(224)
    plt.title(f"Distribution for sigma")
    sns.kdeplot(lambda_p_sigma)
    plt.show()

##beta_p
if plot_beta_p:
    beta_p_all_add = []
    beta_p_means_add = []
    beta_p_add_ = []
    beta_p_all_mul = []
    beta_p_means_mul = []
    beta_p_mul_ = []
    for i in range(n_subjects):
        tmp_add = []
        tmp_mul = []
        for j in range(n_samples):
            for l in range(n_trials):
                if z[i,j,l] in pt_model:
                    beta_p_all_add.append(beta_p[0,i,j,l])
                    tmp_add.append(beta_p[0,i,j,l])
                    beta_p_all_mul.append(beta_p[1,i,j,l])
                    tmp_mul.append(beta_p[1,i,j,l])
        beta_p_means_add.append(np.mean(tmp_add))
        beta_p_add_.append(tmp_add)
        beta_p_means_mul.append(np.mean(tmp_mul))
        beta_p_mul_.append(tmp_mul)

    fig = plt.figure(fig_nr)
    fig_nr += 1
    outer = gridspec.GridSpec(6, 3, wspace=0.5, hspace=0.8)

    for j in range(n_subjects):
        inner = gridspec.GridSpecFromSubplotSpec(1, 2,
                        subplot_spec=outer[j], wspace=0.4, hspace=0.1)
        

        ax = plt.Subplot(fig, inner[0])
        ax.set_title(f"Additive")
        t = ax.hist(beta_p_add_[j])
        fig.add_subplot(ax)

        ax = plt.Subplot(fig, inner[1])
        t = ax.hist(beta_p_mul_[j])
        ax.set_title(f"Multiplicative")
        fig.add_subplot(ax)
    fig.suptitle('Distribution of beta_p')

    plt.figure(fig_nr)
    fig_nr += 1
    plt.suptitle(f"Total distribution for beta_p - Chosen: {len(beta_p_all_add)/(n_samples*n_trials*n_subjects)*100:.3f} %")
    plt.subplot(121)
    plt.title("Additive")
    # plt.hist(beta_p_all_add, bins = 20)
    sns.kdeplot(beta_p_all_add)
    plt.xlim([0,35])
    plt.subplot(122)
    plt.title("Multiplikative")
    # plt.hist(beta_p_all_mul, bins = 20)
    sns.kdeplot(beta_p_all_mul)
    plt.xlim([0,35])
    plt.show()


