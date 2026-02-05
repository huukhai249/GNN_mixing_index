
#Importing libraries
from edempy import Deck
import numpy as np
import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt

import math
import os
import os.path
import csv


def Calculate_GNN_Mixing_Index(fraction_nB_nnb: np.ndarray, fraction_nA_nnb: np.ndarray, 
                               NA: int,NB:int) -> float:
    sum_A=np.sum(fraction_nB_nnb)
    sum_B=np.sum(fraction_nA_nnb)
    GNN_Mixing_Index= sum_A/NB + sum_B/NA
    return GNN_Mixing_Index

# Finf N_nb nearest particles based on Euclidean distance
def find_k_nearest_particle_ids(core: np.ndarray,
                               listAllParticle: list,
                               k: int) -> list:
    positions = np.array([p["position"] for p in listAllParticle])
    core_position = np.asarray(core)

    diff = positions - core_position
    dist_sq = np.sum(diff**2, axis=1)

    nearest_idx = np.argpartition(dist_sq, k)[:k]

    return [listAllParticle[i]["id"] for i in nearest_idx]


# Find current timestep respect to time value
def find_nearest(array: np.ndarray, value: float) -> int:
    array=np.array(array)
    index = (np.abs(array-value)).argmin()
    return int(index)

def selected_candidates_within_cutoff(coreParticlePosition: np.ndarray, listAllParticle: np.ndarray, cutoff_distance: float):
    selected_candidates = []
    for particle in listAllParticle:
        distance = math.sqrt((coreParticlePosition[0] - particle["position"][0])**2 +
                             (coreParticlePosition[1] - particle["position"][1])**2 +
                             (coreParticlePosition[2] - particle["position"][2])**2)
        if distance <= cutoff_distance and distance != 0:
            selected_candidates.append(particle)
    return selected_candidates


def find_fraction_ptypel_N_nb_nearest_particles(coreParticleType, coreParticlePosition , listAllParticle: np.ndarray, N_nb: int):
    listIds = find_k_nearest_particle_ids(coreParticlePosition, listAllParticle, N_nb)
    if coreParticleType == '0':
        nParTypeB = 0
        for particle in listAllParticle:
            if particle["id"] in listIds and particle["type"] == "B":
                nParTypeB += 1
        fraction_nB_nnb = nParTypeB / N_nb
        return fraction_nB_nnb
    else:
        nParTypeA = 0
        for particle in listAllParticle:
            if particle["id"] in listIds and particle["type"] == "A":
                nParTypeA += 1
        fraction_nA_nnb = nParTypeA / N_nb
        return fraction_nA_nnb

#Reading in simulation data
for root, dirs, files in os.walk(os.curdir):
    for file in files:
        if file.endswith(".dem"):
            name=file.replace(".dem","")
        
            print ("-------------------------------------------------------")
            print ("Loading: "+str(name)+".dem")


            print ("-------------------------------------------------------")
            
            deck=Deck(os.path.join(root,file))

            last_timestep = deck.numTimesteps - 1
            
            #Reading in preferences
    
            if os.path.exists(os.path.join(root,'GNNMixingIdx_settings.txt')): 
                with open(os.path.join(root,'GNNMixingIdx_settings.txt'), 'r') as file:
                    preferences=file.readlines()
                    start_time=float(preferences[1])
                    print("start_time:",start_time)
                    end_time=float(preferences[3])
                    print("end_time:",end_time)
                    N_nb=int(preferences[5])
                    print("N_nb:",N_nb)
                    plots=str(preferences[7])
                    print("plots:",plots)
                    cutoff_distance = float(preferences[9])*6
                    print("cutoff_distance:",cutoff_distance)
                settings=True
            else:
                settings=False
            
            #Check if simulation is run to the end
            t_tstep = find_nearest(np.array(deck.timestepValues), end_time)            
            fraction_nA_nnb = []
            fraction_nB_nnb = []
            for ptype in deck.timestep[t_tstep].h5ParticleTypes: 
                listAllParticle = []
                if(deck.timestep[t_tstep].particle["0"].numParticles >0):
                    for i in range(deck.timestep[t_tstep].particle["0"].numParticles):
                        ParticleA = {
                            "type": "A",
                            "id": deck.timestep[t_tstep].particle["0"].getIds()[i],
                            "position": deck.timestep[t_tstep].particle["0"].getPositions()[i]
                        }
                        listAllParticle.append(ParticleA)
                if(deck.timestep[t_tstep].particle["1"].numParticles >0):                   
                    for j in range(deck.timestep[t_tstep].particle["1"].numParticles):
                        ParticleB = {
                            "type": "B",
                            "id": deck.timestep[t_tstep].particle["1"].getIds()[j],
                            "position": deck.timestep[t_tstep].particle["1"].getPositions()[j]
                            }
                        listAllParticle.append(ParticleB)                

                if(ptype == '0'):      
                    nParTypA = deck.timestep[t_tstep].particle[ptype].numParticles
                    for i in range(nParTypA):        
                        coreParticlePosition = deck.timestep[t_tstep].particle[ptype].getPositions()[i]
                        selected_candidates = selected_candidates_within_cutoff(coreParticlePosition, listAllParticle, cutoff_distance)
                        nNeigbor = min(len(selected_candidates), N_nb)
                        fraction_nB_nnb.append(find_fraction_ptypel_N_nb_nearest_particles(ptype, coreParticlePosition, selected_candidates, nNeigbor))
                else:
                    nPartypB = deck.timestep[t_tstep].particle[ptype].numParticles
                    for j in range(nPartypB):        
                        coreParticlePosition = deck.timestep[t_tstep].particle[ptype].getPositions()[j]
                        selected_candidates = selected_candidates_within_cutoff(coreParticlePosition, listAllParticle, cutoff_distance)
                        nNeigbor = min(len(selected_candidates), N_nb)
                        fraction_nA_nnb.append(find_fraction_ptypel_N_nb_nearest_particles(ptype, coreParticlePosition, selected_candidates, nNeigbor))
            print("Calculating GNN Mixing Index...\n")
            print("fraction_nB_nnb:", len(fraction_nB_nnb))
            print("fraction_nA_nnb:", len(fraction_nA_nnb))
                
                        
                    

                    
                   
                    
                
