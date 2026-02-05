# -*- coding: utf-8 -*-
"""
EDEMpy script for the analysis of a binary mixture 

Summary:
This EDEMpy script calculates the Lacey Mixing Index and Relative Standard 
Deviation (RSD) values for number and mass fractions of a binary mixture. 
It reads user-defined parameters from Lacey_RSD_settings_v3.txt and generates 
4 csvs and 4 pngs as output files.

Usage:
1. Place Lacey_RSD_settings.txt and this python file in the same directory 
as your EDEM simulation.
2. Open Lacey_RSD_settings.txt, specify parameters.
3. Run this script using Python or from the EDEM Analyst.

Input Parameters:
- Geometry Name: If specified, the bins will follow this geometry's rotation.
- Min XYZ Coords/ Max XYZ Coords: minimum/maximum coordinates of the region in
which the analysis will be performed.
- Number XYZ bins: number of bins in each direction
- Number/Mass Cut Off: Analysis thresholds. A bin will not be used to calculte
any indices if the total number/mass of particles in it is smaller than the cut off.
- Start/End Time: The time range for the analysis.

Output Files:
- {Index Name}.csv: Lacey Mixing Index and RSD values.
- {Index Name}.png: Lacey Mixing Index and RSD values plotted over time.

References:
[1] Gu, Zongyu, and J. J. J. Chen. "A probabilistic analysis of some selected 
mixing indices", Chemical Engineering Research and Design 93 (2015): 293-303.
https://doi.org/10.1016/j.cherd.2014.04.014

[2] Alian, Meysam, Farhad Ein-Mozaffari, and Simant R. Upreti. "Analysis of the
mixing of solid particles in a plowshare mixer via discrete element method (DEM)"
Powder Technology 274 (2015): 77-87.
https://doi.org/10.1016/j.powtec.2015.01.012

2022/02 - Stefan Pantaleev

2023/10 Renan Calmon --- renan@altair.com
For any questions or suggestions, please contact the address above
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from math import floor

def calculate_quantity_in_bins(positions: list, masses:list, domain_min: np.ndarray,
                               domain_max: np.ndarray, num_bins: np.ndarray):
    
    if len(positions) > 2:
        raise ValueError("Lacey index is only valid for a binary mixture. \
                         More than 2 particle types were detected.")
    elif len(positions) < 2:
        raise ValueError("Less than 2 particle types were detected.")
    elif ((len(positions[0]) == 0) or (len(positions[1]) == 0)):
        raise ValueError("Only one particle type is present in the simulation.")

    # calculate bin index. If coord is outside returns "-1"
    def grid_index_full(coord, origin, bin_size, num_bins):
        ix = int(floor((coord[0] - origin[0]) / bin_size[0]))
        if ix < 0 or ix >= num_bins[0]:
            return -1
        iy = int(floor((coord[1] - origin[1]) / bin_size[1]))
        if iy < 0 or iy >= num_bins[1]:
            return -1
        iz = int(floor((coord[2] - origin[2]) / bin_size[2]))
        if iz < 0 or iz >= num_bins[2]:
            return -1
        return ix + iy * num_bins[0] + iz * num_bins[0] * num_bins[1]
    
    total_bins = num_bins[0] * num_bins[1] * num_bins[2]
    lengths = domain_max - domain_min
    bin_size = lengths / num_bins

    # quantities is a num_bins x 2 matrix that contains the total mass/number
    # of each particle type inside each bin
    quantities_mass = np.zeros((total_bins, 2))
    quantities_number = np.zeros((total_bins, 2))
    
    for i in range(2):
        # Get position and mass for this particle type
        this_position = positions[i]
        this_mass = masses[i]
 
        # get bin index per particle
        particles_in_bin = np.array([grid_index_full(position, domain_min, bin_size, num_bins) for position in this_position])

        # remove positions out of the domain (marked as bin_index = total_cell+1)
        inside_indices = np.flatnonzero(particles_in_bin != -1)

        particles_in_bin = particles_in_bin[inside_indices]
        
        # count particles/mass per cell
        quantities_number[:, i] = np.bincount(particles_in_bin, minlength=total_bins)
        quantities_mass[:, i] = np.bincount(particles_in_bin, minlength=total_bins,
                                            weights=this_mass[inside_indices])

    return quantities_number, quantities_mass

def calculate_lacey_index_and_rsd(overall_proportion: float, quantities: np.ndarray,
                                  cut_off: float, quantity: str):
    
    S0 = overall_proportion * (1 - overall_proportion)
    
    new_quantities = quantities[np.sum(quantities, axis= 1)> cut_off]
    
    num_bins = new_quantities.shape[0]
    
    sum_quantities = np.sum(new_quantities, axis=1)
    
    # calculate concentration when values in the cell
    concentrations = np.where(sum_quantities > 0, new_quantities[:,0] / sum_quantities, 0.0)

    if concentrations.size == 0:
        return 0, 0
    
    rsd = np.std(concentrations) / np.mean(concentrations)

    S = np.sum(np.power(concentrations - overall_proportion, 2)) / (num_bins)
    
    Sr = S0 / np.mean(sum_quantities)
    
    if quantity.lower() == "number":
        M = (S0 - S) / (S0 - Sr)
    elif quantity.lower() == "mass":
        M = 1 - (S/S0)
    
    return M, rsd

def find_nearest(array: np.ndarray, value: float) -> int:
    array=np.array(array)
    index = (np.abs(array-value)).argmin()
    return int(index)
       
def create_plot(time_values: np.ndarray, y_values: np.ndarray, ylabel: str, 
                filename: str) -> None:
    
    dpi = 150
    fig, ax = plt.subplots(dpi = dpi)
    
    ax.set_xlabel("Time (s)", fontsize = 13, fontname = "Arial")
    ax.set_ylabel(ylabel, fontsize = 13, fontname = "Arial")
    ax.grid()
    ax.grid(which="major", axis="both", alpha= 0.3)
    ax.tick_params(which="both", direction="in")
    ax.minorticks_on()
    ax.tick_params(which="minor", direction="in", color= "k")
    
    ax.plot(time_values, y_values, linewidth= 2)
    
    fig.savefig(f"{filename}.png")
    plt.close(fig)
    
def create_plots(values: np.ndarray, name: str, positions: list, 
                 domain_min: np.ndarray, domain_max: np.ndarray, num_bins: np.ndarray):
    
    def grid(minCoords, maxCoords, Bins):
        
        lengths = maxCoords - minCoords
        resolution = np.array(Bins)
        divisionSize = lengths / resolution
        xCoords = np.arange(minCoords[0] + ((divisionSize[0]) / 2), maxCoords[0], divisionSize[0])
        yCoords = np.arange(minCoords[1] + ((divisionSize[1]) / 2), maxCoords[1], divisionSize[1])
        zCoords = np.arange(minCoords[2] + ((divisionSize[2]) / 2), maxCoords[2], divisionSize[2])
        coords = []
        
        for i in range(0, len(xCoords)):
            for j in range(0, len(yCoords)):
                for k in range(0, len(zCoords)):
                    coords.append([xCoords[i], yCoords[j], zCoords[k]])
        
        return np.array(coords), divisionSize
    
    fig_1 = plt.figure()
    ax = fig_1.add_subplot(111, projection='3d')
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.grid(False)
    
    pos1 = positions[0]
    pos2 = positions[1]
    
    ax.scatter(pos1[:,0], pos1[:,1], pos1[:,2], s= 0.01, c= "b")
    ax.scatter(pos2[:,0], pos2[:,1], pos2[:,2], s= 0.01, c= "r")
    
    b_coords, div_size = grid(domain_min, domain_max, num_bins)
    
    x= np.unique(b_coords[:,0])
    y= np.unique(b_coords[:,1])
    z= np.unique(b_coords[:,2])
    
    x = np.append(x-div_size[0]/2, np.amax(x)+div_size[0]/2)
    y = np.append(y-div_size[1]/2, np.amax(y)+div_size[1]/2)
    z = np.append(z-div_size[2]/2, np.amax(z)+div_size[1]/2)
    
    for i in  range(len(x)):
        for j in range(len(y)):
            for k in range(len(z)):
                mult_x = np.ones_like(x)
                mult_y = np.ones_like(y)
                mult_z = np.ones_like(z)
                
                ax.plot(x[i]*mult_y, y, mult_y*z[k], c='black', linewidth=0.03)
                ax.plot(x, y[j]*mult_x, mult_x*z[k], c='black', linewidth=0.03)
                ax.plot(x[i]*mult_z, y[j]*mult_z, z, c='black', linewidth=0.03)
                
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_xlim([np.amin(b_coords), np.amax(b_coords)])
    ax.set_ylim([np.amin(b_coords), np.amax(b_coords)])
    ax.set_zlim([np.amin(b_coords), np.amax(b_coords)])
    plt.close(fig_1)
    fig_1.savefig(str(name)+'_Bins.png', dpi= 150)
    
    prop_names = ["Lacey Index by Mass", "Lacey Index by Number", "RSD by Mass", "RSD by Number"]
    
    for i, prop_name in enumerate(prop_names):
        
        fig2, ax2 = plt.subplots(dpi = 150)
        ax2.set_xlabel("Time (s)", fontsize = 13, fontname = "Arial")
        ax2.set_ylabel(prop_name, fontsize = 13, fontname = "Arial")
        ax2.grid()
        ax2.grid(which="major", axis="both", alpha= 0.3)
        ax2.tick_params(which="both", direction="in")
        ax2.minorticks_on()
        ax2.tick_params(which="minor", direction="in", color= "k")
        plt.title(f"{prop_name} evolution for {name}")
        
        ax2.plot(values[:,0], values[:,i+1], linewidth= 2)
        fig2.savefig(f"{name}_{prop_name}_vs_Time.png", dpi= 150)
        plt.close(fig2)
    
if __name__ == "__main__":
    
    from edempy import Deck
    import os
    import csv
    
    # Get current file path
    directory = os.getcwd()
    
    # Set preference file name    
    pref_file_name = "Lacey_settings.txt"
    
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".dem"):
                
                name= file[:-4]
                print("-------------------------------------------------------")
                print("Loading: " + str(name) + ".dem")
                print("-------------------------------------------------------")
                
                if os.path.exists(os.path.join(root,pref_file_name)):
                    try:
                        with open(os.path.join(root,pref_file_name), "r") as f:
                            preferences= f.read().splitlines()
                            minCoords= np.array([float(i) for i in str(preferences[1]).split(',')])
                            maxCoords= np.array([float(i) for i in str(preferences[3]).split(',')])
                            bins= np.array([int(i) for i in str(preferences[5]).split(',')])
                            mass_cut_off= float(preferences[7])
                            number_cut_off = float(preferences[9])
                            start_time = float(preferences[11])
                            end_time = float(preferences[13])
                            plot= str(preferences[15])
                            settings= True
                    except:
                        settings=False
                        sim_end=0
                else:
                    settings=False
                    sim_end=0
                    
                if settings==True:
                    
                    print("-------------------------------------------------------")
                    print("Processing: " + str(name) + ".dem")
                    print("-------------------------------------------------------")
                    
                    with Deck(os.path.join(root,file)) as deck:
                        
                        # Find timestep start, timestep end and num of timesteps
                        start_tstep = find_nearest(np.array(deck.timestepValues), start_time)
                        end_tstep = find_nearest(np.array(deck.timestepValues), end_time) + 1
                        num_steps = end_tstep - start_tstep

                        values = np.zeros((num_steps, 5)) # time, lacey_mass, lacey_number, rsd_mass, rsd_number
                        
                        for tstep in range(start_tstep, end_tstep):
                            
                            step_local = int(tstep-start_tstep)
                            time = float(deck.timestepValues[tstep])
                            
                            print(f"Processing timestep: {time} s")
                            
                            try:
                                
                                positions = []
                                masses = []
                
                                for ptype in deck.timestep[tstep].h5ParticleTypes:
                                    positions.append(deck.timestep[tstep].particle[ptype].getPositions())
                                    masses.append(deck.timestep[tstep].particle[ptype].getMass())

                                number_per_bin, mass_per_bin = calculate_quantity_in_bins(positions, masses, minCoords, maxCoords, bins)
                                overall_number = np.sum(number_per_bin[:,0]) / np.sum(number_per_bin)
                                overall_mass = np.sum(mass_per_bin[:,0]) / np.sum(mass_per_bin)
                
                                lacey_by_mass, rsd_by_mass = calculate_lacey_index_and_rsd(overall_mass, mass_per_bin, mass_cut_off, "mass")
                                lacey_by_number, rsd_by_number = calculate_lacey_index_and_rsd(overall_number, number_per_bin, number_cut_off, "number")
                
                                values[step_local,:] = [time, lacey_by_mass, lacey_by_number, rsd_by_mass, rsd_by_number]
                                
                            except:
                                values[step_local,:] = [time, 0.0, 0.0, 0.0, 0.0]
                                #print("Warning: " + error)

                    # Write single CSV file with all values            
                    csv_file = f"{name}_Report.csv"
                    with open(csv_file, "w", newline="") as f:
                        writer = csv.writer(f)
                        fieldnames = ["Time (s)", "Lacey Index by Mass", "Lacey Index by Number", "RSD by Mass", "RSD by Number"]
                        writer.writerow(fieldnames)
                        writer.writerows(values)

                    if plot.lower() == "yes":
                        create_plots(values, name, positions, minCoords, maxCoords, bins)
   
                    print("-------------------------------------------------------")
                    print("Processing complete")
                    print("-------------------------------------------------------")
                
                else:
                    
                    print("----------------------------------------------------------------------")
                    print(str(name)+".dem"+" : Settings file not found. Moving to next simulation")
                    print("----------------------------------------------------------------------")

    
    print("-------------------------------------------------------")
    print("Analysis is complete!")
    print("-------------------------------------------------------")