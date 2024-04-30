from math import sqrt
from Bio import *
from Bio.PDB.PDBParser import PDBParser
from Bio.PDB import NeighborSearch
import numpy as np
import os
import random
import MDAnalysis as mda
from utils.IO_relate import *

def normalize_vec(vec,MIN,MAX):
    min_vec = np.min(vec, axis=0)
    max_vec = np.max(vec, axis=0)
    if max_vec == min_vec:
        normalized_vec = np.zeros(len(vec))
    else:
        normalized_vec = MIN + (MAX - MIN) * (vec - min_vec) / (max_vec - min_vec)
    return normalized_vec

# str(residue_serializedId[i] + protein_idx * 100000) + ' ')
def get_residue_idx_in_file(residue_serilizedId, protein_idx):
    return residue_serilizedId + protein_idx * 100000

def distance(vec1, vec2):
    return sqrt((vec1[0] - vec2[0]) ** 2 + (vec1[1] - vec2[1]) ** 2 + (vec1[2] - vec2[2]) ** 2)

def get_positional_encoding(index, length, d_model):
    pos = np.arange(length)[:, np.newaxis]
    i = np.arange(d_model)[np.newaxis, :]
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
    angle_rads = pos * angle_rates
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    pos_encoding = angle_rads[np.newaxis, index, :]
    return pos_encoding



# Featurize this pdb file
def process_from_pdb(pdb_path, dssp_path, chain_name, breakchains):
    # print(pdb_path)
    # print(dssp_path)
    # print(chain_name)
    # print(breakchains)
    # print(chain_name)
    # print(breakchains)
    breakchain_merge = []
    for chain in breakchains:
        for idx in chain:
            breakchain_merge.append(idx)

    
    pdb_parser = PDBParser(PERMISSIVE=True)
    pdb_structure = pdb_parser.get_structure('X', pdb_path)

    
    residue_id = [] # e.g. 1cy5_A+20
    residue_k_neighbors_id = []
    residue_k_neighbors_distance = []
    residue_channels = []
    residue_k_neighbors_AAtype = []
    residue_k_neighbors_rsa=[]
    residue_rsa=[]
    residue_ss8=[]
    residue_ss3=[]

    ns = NeighborSearch(list(pdb_structure.get_atoms()))

    # pdb_to_pqr_command = 'pdb2pqr --ff=CHARMM --whitespace ' + pdb_path + ' ' + pdb_path[0:-4] + '.pqr'
    # try:
    #     os.system(pdb_to_pqr_command)
    # except:
    #     print('[protein_parser] pdb2pqr error!')
    #     return

    # pqr_path = pdb_path[0:-4] + '.pqr'
    # print(pqr_path)

    try:
        u = mda.Universe(pdb_path)
    except:
        print('[protein_parser] MDAnalysis error!')
        return

    def get_id(residue):
        return pdb_path[-8:-4] + '_' + residue.get_parent().get_id() + '+' + str(residue.get_full_id()[3][1])

    # select every atom without HETATM
    mainchain = u.select_atoms("protein and not name H*")
    current_residue_index = 0
    
    ss8_series, ss3_series, rsa_series, df_resnum,idx_rsa_dict=extract_SS_ASA_fromDSSP(dssp_path, breakchains, chain_name, includeside = True)
    # print(len(ss8_series), len(ss3_series), len(rsa_series), len(df_resnum), len(idx_rsa_dict))
    # print(ss8_series, ss3_series, rsa_series, df_resnum, idx_rsa_dict)
    for current_chain in pdb_structure.get_chains():
        # residue_idx_in_chain = 0
        for current_residue in current_chain.get_residues():
            # print(current_residue.get_parent().get_id(), current_residue.get_full_id()[3][1])

            if current_residue.get_parent().get_id() != chain_name or (not(current_residue.get_full_id()[3][1] in breakchain_merge)):
                continue

            # Check if this residue is from main_chain
            if current_residue.get_id()[0] != ' ':
                continue
            for current_CA in current_residue.get_atoms():
                if current_CA.get_id() == 'CA':  # Get every CAlpha
                    current_residue_index += 1

                    # Calculate CAlpha_contains from neighbors, fill 4 channels (C, N, O, S)
                    current_channel = {}
                    current_channel['C'] = 0
                    current_channel['N'] = 0
                    current_channel['O'] = 0
                    current_channel['S'] = 0
                    # residue_k_neighbors_id.append([])
                    current_neighbours_id = [] # residue_k_neighbors_id[current_residue_index - 1]
                    # residue_k_neighbors_distance.append([])
                    current_neighbours_distance = [] # residue_k_neighbors_distance[current_residue_index - 1]
                    current_residue_k_neighbors_AAtype = []
                    current_neighbours_rsa=[]


                    # Get and interate neighbor from 20 angstrom
                    neighbor_atoms = ns.search(current_CA.get_coord(), 20)
                    for neighbor_atom in neighbor_atoms:
                        residue_of_current_neighbor_atom = neighbor_atom.get_parent()
                        if residue_of_current_neighbor_atom.get_parent().get_id() != chain_name or (not(residue_of_current_neighbor_atom.get_full_id()[3][1] in breakchain_merge)):
                            continue
                        # Do NOT calculate atoms inside this residue itself
                        if residue_of_current_neighbor_atom != current_residue:
                            neighbor_atom_id = neighbor_atom.get_id()
                            # Update channels of current residue
                            if (neighbor_atom_id[0] == 'C' or neighbor_atom_id[0] == 'N' or neighbor_atom_id[0] == 'O' or
                                    neighbor_atom_id[0] == 'S'):
                                current_channel[neighbor_atom_id[0]] += 1
                            # if neighbor_atom_id == 'CA':  # Update graph, inserting edge between residues pairs closer than 20 angstrom
                            #     neighbor_residue = neighbor_atom.get_parent()
                            #     if neighbor_residue.get_parent().get_id() == current_residue.get_parent().get_id():
                            #         current_residue_k_neighbors_AAtype.append(neighbor_residue.get_resname())
                            #         current_neighbours_id.append(get_id(neighbor_residue))
                            #         current_neighbours_distance.append(distance(neighbor_atom.get_coord(), current_CA.get_coord()))
                                
                    
                    neighbor_atoms = ns.search(current_CA.get_coord(), 40)
                    for neighbor_atom in neighbor_atoms:
                        residue_of_current_neighbor_atom = neighbor_atom.get_parent()
                        if residue_of_current_neighbor_atom.get_parent().get_id() != chain_name or (not(residue_of_current_neighbor_atom.get_full_id()[3][1] in breakchain_merge)):
                            continue
                        # Do NOT calculate atoms inside this residue itself
                        if residue_of_current_neighbor_atom != current_residue:
                            neighbor_atom_id = neighbor_atom.get_id()
                            # Update channels of current residue
                            # if (neighbor_atom_id[0] == 'C' or neighbor_atom_id[0] == 'N' or neighbor_atom_id[0] == 'O' or
                            #         neighbor_atom_id[0] == 'S'):
                            #     current_channel[neighbor_atom_id[0]] += 1
                            if neighbor_atom_id == 'CA':  # Update graph, inserting edge between residues pairs closer than 20 angstrom
                                neighbor_residue = neighbor_atom.get_parent()
                                if neighbor_residue.get_parent().get_id() == current_residue.get_parent().get_id():
                                    current_residue_k_neighbors_AAtype.append(neighbor_residue.get_resname())
                                    current_neighbours_id.append(get_id(neighbor_residue))
                                    current_neighbours_distance.append(distance(neighbor_atom.get_coord(), current_CA.get_coord()))
                                    # print(idx_rsa_dict)
                                    # print(breakchain_merge)
                                    current_neighbours_rsa.append(idx_rsa_dict[neighbor_residue.get_full_id()[3][1]])



                    residue_id.append(get_id(current_residue))
                    # sort by distance
                    sorted_tuples = sorted(zip(current_neighbours_distance, current_neighbours_id, current_residue_k_neighbors_AAtype,current_neighbours_rsa))
                    current_neighbours_distance, current_neighbours_id, current_residue_k_neighbors_AAtype ,current_neighbours_rsa = [x[0] for x in sorted_tuples], [x[1] for x in sorted_tuples], [x[2] for x in sorted_tuples],[x[3] for x in sorted_tuples]
                    # pick min 20
                    current_neighbours_distance, current_neighbours_id = current_neighbours_distance[:20], current_neighbours_id[:20]
                    current_residue_k_neighbors_AAtype = current_residue_k_neighbors_AAtype[:20]
                    current_neighbours_rsa = current_neighbours_rsa[:20]
                    
                    
                    residue_k_neighbors_id.append(current_neighbours_id)
                    residue_k_neighbors_distance.append(current_neighbours_distance)
                    residue_k_neighbors_AAtype.append(current_residue_k_neighbors_AAtype)
                    residue_channels.append(current_channel)
                    residue_k_neighbors_rsa.append(current_neighbours_rsa)
                    residue_rsa.append(idx_rsa_dict[current_residue.get_full_id()[3][1]])
                    residue_ss8.append(ss8_series[current_residue_index - 1])
                    residue_ss3.append(ss3_series[current_residue_index - 1])
                    # print(residue_id[current_residue_index - 1])
                    # print(residue_channels[current_residue_index - 1])
                    # print(residue_k_neighbors_id[current_residue_index - 1])
                    # print(residue_k_neighbors_distance[current_residue_index - 1])

    return residue_id, residue_k_neighbors_id, residue_k_neighbors_distance, residue_k_neighbors_AAtype, residue_channels,residue_k_neighbors_rsa,residue_rsa,residue_ss3,residue_ss8


if __name__ == '__main__':
    print("[protein_parser] Initializing...")
    # init_files()

    print("[protein_parser] Making cites and content...")
    pdb_names_list = []
    pdb_paths_list = []


    for parent, _, fileNames in os.walk("input"):
        for name in fileNames:
            if name.startswith('.'):
                continue
            if name.endswith(".pdb"):
                pdb_names_list.append(name)
                pdb_paths_list.append(os.path.join(parent, name))

    for pdb_path in pdb_paths_list:
        process_from_pdb(pdb_path)