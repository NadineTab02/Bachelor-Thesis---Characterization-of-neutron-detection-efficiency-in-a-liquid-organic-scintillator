import numpy as np
import matplotlib.pyplot as plt
import math
import matplotlib.font_manager as fm

plt.rc('font', size=12) 
font_path = fm.findfont(fm.FontProperties(family='Arial'))
plt.rcParams['font.family'] = 'Arial'


################### experimental data ######################################3
qtot_big = [] #your data from experiment here
tof_big = [] #your data from experiment here

indexes_neutrongamma = np.where(np.logical_and(tof_big>=26, tof_big<=70)) #finds indexes for the intervals of interest
indexes_background = np.where(np.logical_and(tof_big>-200, tof_big<-100))
indexes_background_peak = np.where(np.logical_and(tof_big>-60, tof_big<-48))

binspick_L = np.arange(0, 4, 0.01) #creates an array of energy bins, ranging from 0 to 4 MeVee
binspick_tof = np.arange(-200, 100, 1) #creates array of time of flight bins

scale = (70-26)/(100) #scaling background factor

### Light yield calibrate
offset = 0.1271095285422718
gradient = 0.00033306607893887375

L_meas = qtot_big * gradient + offset #MeVee. Converts energy values into the units of MeVee

light_neutrongamma_list = [L_meas[index] for index in indexes_neutrongamma] #extracts energy values in the right intervals
light_background_list = [L_meas[index] for index in indexes_background]

num, bins_tof = np.histogram(tof_big, bins=binspick_tof)

num6, bins = np.histogram(light_neutrongamma_list, bins=binspick_L)  #histogram for gamma-neutron coincidence

num7, bins = np.histogram(light_background_list, bins=binspick_L) #histogram for background radiation

num8 = num6 - (num7*scale) #neutron-gamma coincidence with background radiation subtracted
num8[num8 < 0]=0 #make every negative count numbe zero

######################## simulation ###############################

threshold = 0.22
file_name = r" " #data from each simulation. AmBe
file_name_cf = r" " #Cf
file_name_pu = r" " #Pu

data = np.genfromtxt(file_name, dtype=float, delimiter=',', skip_header=7, usecols=(0))
data_cf = np.genfromtxt(file_name_cf, dtype=float, delimiter=',', skip_header=7, usecols=(0))
data_pu = np.genfromtxt(file_name_pu, dtype=float, delimiter=',', skip_header=7, usecols=(0))

#rebin to 400 bins instead of 4000
data_rebin = np.array([])
for i in range(0,len(data),10):
    group = data[i:i+10]
    group_sum = sum(group)
    data_rebin = np.hstack((data_rebin, group_sum))

data_rebin_cf = np.array([])
for i in range(0,len(data_cf),10):
    group = data_cf[i:i+10]
    group_sum = sum(group)
    data_rebin_cf = np.hstack((data_rebin_cf, group_sum))

data_rebin_pu = np.array([])
for i in range(0,len(data_pu),10):
    group = data_pu[i:i+10]
    group_sum = sum(group)
    data_rebin_pu = np.hstack((data_rebin_pu, group_sum))


file1 = open(file_name, 'r') #open simulation file
count = 0

while count < 10:
    count += 1
    line = file1.readline()

    if "axis fixed" in line:  # reads line nr 6 in csv file
        words = line.split()
        n_bins_x = int(words[2])  # in this case, we have 4000 nr bins
        x_min = float(words[3])  # lowest bin is 0 MeV 
        x_max = float(words[4])  # highest bin is 4 MeV
        bin_width = 0.01 
        x_axis = np.linspace(x_min + 0.5 * bin_width, x_max + 0.5 * bin_width, 400)
        break
    print("Line{}: {}".format(count, line.strip()))

new_x_axis = np.delete(x_axis, np.where(x_axis < threshold)) #new x-axis without bins under the threshold value

start_data = int(threshold/0.01) #first index over the threshold

scale_factor = sum(num8)/sum(data_rebin[start_data:-1]) #scale factor between expeiemental and simulated counts

new_data = scale_factor*data_rebin[start_data:-1] #counts in the simulated data set normalised to the expeimental counts
sum_data_before_conv = sum(new_data)

scale_factor_ambe=sum(num8)/sum(data_rebin[start_data:-1]) #rescales the counts for ambe simulation
new_data_ambe = scale_factor_ambe*data_rebin[start_data:-1] 
new_data_cf = data_rebin_cf[start_data:-1]
new_data_pu = data_rebin_pu[start_data:-1]

sum_data_before_conv_cf=sum(new_data_cf)
sum_data_before_conv_ambe = sum(new_data_ambe)
sum_data_before_conv_pu = sum(new_data_pu)

######################### convolution ##########################

a = 0.053 #constants from calibration
b = 0.015
c = 0.767

gaussian_matrix = np.empty((0, len(new_x_axis))) #creates an empty matrix

#computes the Gaussian vectors for each value of L
for L in new_x_axis:
    #initializes the Gaussian vector
    gaussian_vector = np.empty((0,))

    #computes the Gaussian values for each value of x
    for x in new_x_axis:
        sigma = (a + b * np.sqrt(L + c * (L ** 2))) / (2 * np.sqrt(2 * np.log(2)))
        gaussian = 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-((x - L) ** 2) / (2 * sigma ** 2))

        #append the new Gaussian value to the Gaussian vector
        gaussian_vector = np.append(gaussian_vector, gaussian)

    #append the Gaussian vector to the matrix
    gaussian_matrix = np.vstack([gaussian_matrix, gaussian_vector])

gaussian_matrix = gaussian_matrix.T

new_data_cf = gaussian_matrix @ new_data_cf #matrix multiplication
new_data_ambe = gaussian_matrix @ new_data_ambe
new_data_pu = gaussian_matrix @ new_data_pu

sum_data_after_conv_cf=sum(new_data_cf)
sum_data_after_conv_ambe = sum(new_data_ambe)
sum_data_after_conv_pu = sum(new_data_pu)

scale_factor_convolution_cf = sum_data_before_conv_cf/sum_data_after_conv_cf
scale_factor_convolution_ambe = sum_data_before_conv_ambe/sum_data_after_conv_ambe
scale_factor_convolution_pu = sum_data_before_conv_pu/sum_data_after_conv_pu

new_data_cf = new_data_cf*scale_factor_convolution_cf #rescales the data sets after convolution
new_data_ambe = new_data_ambe*scale_factor_convolution_ambe
new_data_pu = new_data_pu*scale_factor_convolution_pu

scale_factor_integral_ambe = sum(new_data_cf)/sum(new_data_ambe) #normalizes all data sets to each other
scale_factor_integral_pu = sum(new_data_cf)/sum(new_data_pu)
new_data_ambe_comp = new_data_ambe*scale_factor_integral_ambe
new_data_pu = new_data_pu*scale_factor_integral_pu

################## Error in experiment ######################

error_bars = np.array([]) 
error_bars_sim = np.array([]) #calculates error for each energy bin in the experimental data
for i in num6:
    error = np.sqrt(i)
    error_bars = np.hstack((error_bars, error))

################## Plots #######################

fig1 = plt.figure()
fig2 = plt.figure()
fig3 = plt.figure()
fig4 = plt.figure()
fig5 = plt.figure()
fig6 = plt.figure()

ax1 = fig1.add_subplot(111)
ax1.plot(bins_tof[:-1], num)
ax1.set_yscale('log')
ax1.set_xlabel('Time of Flight [ns]')
ax1.set_ylabel('Events')
ax1.set_title('Time of Flight spectrum for Am-Be')
ax1.plot(-100.4, 4485.87, "|", color="green", label='Interval background radiation')
ax1.plot(-200.5, 4485.87, "|", color="green")
ax1.plot(26.1, 5000.68, "|", linewidth=10, color="orange", label='Interval neutron-gamma coincidence')
ax1.plot(70.1, 5000.57, "|", color="orange")
ax1.plot(-12.4, 5084.57, "|", color="purple", label='Interval gamma-gamma coincidence')
ax1.plot(12.3, 5084.57, "|", color="purple")
ax1.legend()

ax2 = fig2.add_subplot(111)
ax2.plot(bins[:-1], num8, label='Experimental data', color='r')
ax2.plot(new_x_axis, new_data_ambe, label='Simulated data')
ax2.fill_between(bins[:-1], (num8-error_bars), (num8+error_bars), color='blue', alpha=0.1, label='Error')
ax2.set_yscale('log')
ax2.set_xlabel('Light Output [MeVee]')
ax2.set_ylabel('Counts')
ax2.set_title('Light Output, log scale')
ax2.legend()

ax3 = fig3.add_subplot(111)
ax3.plot(bins[:-1], num8, label='Experimental data', color='r')
ax3.plot(new_x_axis, new_data_ambe, label='Simulated data')
ax3.fill_between(bins[:-1], (num8-error_bars), (num8+error_bars), color='blue', alpha=0.1, label='Error')
ax3.legend()
ax3.set_xlabel('Light Output [MeVee]')
ax3.set_title('Light Output')
ax3.set_ylabel('Counts')

ax4 = fig4.add_subplot(111)
ax4.plot(bins[:-1], num8, label='Experimental data', color='r')
ax4.plot(new_x_axis, new_data, label='Sim data')
ax4.set_title('Without convolution')

ax5 = fig5.add_subplot(111)
ax5.plot(new_x_axis, new_data_cf, label='Simulated data for Cf')
ax5.plot(new_x_axis, new_data_ambe_comp, label='Simulated data for AmBe')
ax5.plot(new_x_axis, new_data_pu, label='Simulated data for Pu')
ax5.set_yscale('log')
ax5.set_xlabel('Light Output [MeVee]')
ax5.set_ylabel('Counts')
ax5.set_title('Light Output, log scale')
ax5.legend()

ax6 = fig6.add_subplot(111)
ax6.plot(new_x_axis, new_data_cf, label='Simulated data for Cf')
ax6.plot(new_x_axis, new_data_ambe_comp, label='Simulated data for AmBe')
ax6.plot(new_x_axis, new_data_pu, label='Simulated data for Pu')
ax6.set_xlabel('Light Output [MeVee]')
ax6.set_ylabel('Counts')
ax6.set_title('Light Output')
ax6.legend()


################## Efficiency ##########################

theta = math.radians(180-162.543) #angle of cone simulated
alpha = math.radians(180-178.543) #angle of cone hitting the detector 
Omega = 2*np.pi*(1-np.cos(theta)) #Solid angle of simulation
Omega_det = 2*np.pi*(1-np.cos(alpha)) #Solid angle of detector
gradient=Omega/Omega_det

threshold_values=[0.1, 0.2, 0.3, 0.4, 0.7, 1, 1.3, 1.6, 1.9, 1.2, 1.6, 2, 2.4, 2.8, 3.2, 3.8, 4]

efficiency_cf = np.array([])
efficiency_pu = np.array([])
int_efficiency_cf = np.array([])
int_efficiency_pu = np.array([])

for i in threshold_values:
#for each threshold, calculates an absolute and intrinsic efficiency for cu-252 and pu-240

 threshold_val = i
 start_data=int(threshold_val/0.01)
 
 scaled_emitted_data = (10**8)*(4*np.pi)/Omega

 efficiency_cf = np.hstack((efficiency_cf, sum(data_rebin_cf[start_data:-1]) / (scaled_emitted_data)))
 efficiency_pu = np.hstack((efficiency_pu, sum(data_rebin_pu[start_data:-1]) / (scaled_emitted_data)))

 int_efficiency_cf = np.hstack((int_efficiency_cf, (sum(data_rebin_cf[start_data:-1]) / (10**8))*gradient))
 int_efficiency_pu = np.hstack((int_efficiency_pu, (sum(data_rebin_pu[start_data:-1]) / (10**8))*gradient))

fig7 = plt.figure()
ax7 = fig7.add_subplot(111)
ax7.plot(threshold_values, efficiency_cf, label='Absolute efficiency for Cf-252', marker='*')
ax7.plot(threshold_values, efficiency_pu, label='Absolute efficiency for Pu-240', marker='*')
ax7.set_xlabel('Threshold [MeVee]')
ax7.set_ylabel('Efficiency')
ax7.set_title('Absolute efficiency for Cf-252 and Pu-240')
ax7.legend()

fig8 = plt.figure()
ax8 = fig8.add_subplot(111)
ax8.plot(threshold_values, int_efficiency_cf, label='Intrinsic efficiency for Cf-252', marker='*')
ax8.plot(threshold_values, int_efficiency_pu, label='Intrinsic efficiency for Pu-240', marker='*')
ax8.set_xlabel('Threshold [MeVee]')
ax8.set_ylabel('Efficiency')
ax8.set_title('Intrinsic efficiency for Cf-252 and Pu-240')
ax8.legend()

################## Quantative analysis #####################


Normalization_AmBe = 1/sum(new_data_ambe) #normalization factor to make integral = 1
new_data_ambe_norm = new_data_ambe*Normalization_AmBe #probability distribution

Normalization_Cf = 1/sum(new_data_cf)
new_data_cf_norm = new_data_cf * Normalization_Cf

idx1 = np.where(new_x_axis > 1)[0] #creates interval of indexes
idx2 = np.where(new_x_axis > 2.5)[0]

idx_min=min(idx1) #first index in first interval
idx_max=min(idx2) #last index in first interval

idx1_2 = np.where(new_x_axis > 2.5)[0] 
idx2_2 = np.where(new_x_axis > 4)[0]

idx_min_2=min(idx1_2) #first index in second interval
idx_max_2=min(idx2_2) #last index in second interval

counts=[10**8, 10**7, 10**6, 10**5, 10**4, 10**3, 560, 550, 10**2, 10**1] #total counts to test for

p1_ambe = sum(new_data_ambe_norm[idx_min: idx_max]) #sum of all counts
p2_ambe = sum(new_data_ambe_norm[idx_min_2: idx_max_2])

p1_cf = sum(new_data_cf_norm[idx_min: idx_max])
p2_cf= sum(new_data_cf_norm[idx_min_2: idx_max_2])

x_ambe=p1_ambe/p2_ambe #expected value
x_cf=p1_cf/p2_cf

mean_difference = abs(x_ambe-x_cf) #difference between expected values

for N in counts:
    #tests if the distance between the Gaussians for Am-Be and Cf is big enough for different counts

    x_ambe_1 = N*p1_ambe
    x_cf_1 = N*p1_cf
    x_ambe_2 = N*p2_ambe
    x_cf_2 = N*p2_cf

    std_dev_ambe_1=np.sqrt(x_ambe_1) #standard deviation for interval 1, ambe
    std_dev_cf_1=np.sqrt(x_cf_1)

    std_dev_ambe_2=np.sqrt(x_ambe_2) #interval 2
    std_dev_cf_2=np.sqrt(x_cf_2)

    tot_std_dev_ambe = x_ambe*np.sqrt((std_dev_ambe_1/x_ambe_1)**2 + (std_dev_ambe_2/x_ambe_2)**2) #error propagation for ambe
    tot_std_dev_cf = x_cf*np.sqrt((std_dev_cf_1/x_cf_1)**2 + (std_dev_cf_2/x_cf_2)**2)

    FWHM_ambe = tot_std_dev_ambe*2*np.sqrt(2*np.log(2)) #FWHM for ambe
    FWHM_cf = tot_std_dev_cf*2*np.sqrt(2*np.log(2))

    distance_FWHM = mean_difference - ((FWHM_ambe / 2) + (FWHM_cf / 2))

    print('counts: ', N, 'std_ambe_1: ', std_dev_ambe_1, 'std_ambe_2: ', std_dev_ambe_2, 'std_cf_1: ', std_dev_cf_1, 'std_cf_2: ', std_dev_cf_2, 'dist:',  distance_FWHM)
    print('total_err_ambe:', tot_std_dev_ambe, 'total_err_cf:', tot_std_dev_cf)

    if distance_FWHM >= 0:
        continue
    else:
        print(N, 'too few counts')

plt.show()