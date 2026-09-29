# -*- coding: utf-8 -*-
"""
@author: rstrauss
Updated on Thurs Sept 10
"""
import numpy as np
from pathlib import Path
from scipy.spatial import cKDTree

PROJECT_ROOT = Path(__file__).resolve().parent
ZONATION_RANDOM_SEED = 42

def project_path(*parts):
    return PROJECT_ROOT.joinpath(*parts)

def Initialize_Hepatocytes_File():
    HepatocytesField = project_path('Field_Hepatocytes.txt').open('w')
    GSHField = project_path('Mesh_Volume_Calculations', 'Field_Zonation.txt').open('r+')
    GSHNodes = GSHField.readlines()
    
    for line in GSHNodes:
        GSHMass = line.split(',')

        # Column names taken from previous MALD implementation
        HepatocytesField.write( str(GSHMass[0].rstrip()) + ',' # node_id
                               + '0.0' + ',' # APAP
                               + '0.0' + ',' # NAPQI
                               + str(GSHMass[1].rstrip()) + ',' # GSH
                               + '1' + ',' # Healthy hepatocyte ratio
                               + '0' + ',' # Damaged hepatocyte ratio
                               + '0' + ',' # Necrosed hepatocyte ratio
                               + '0' + ',' # Regenerating hepatocyte ratio
                               + '12' + ',' # AST starting value - from Pinto
                               + '9' + ',' # ALT starting value - from Pinto
                               + '0' + ',' # Clotting factor
                               + str(GSHMass[2].rstrip()) + ',' # GSH production rate
                               + str(GSHMass[3].rstrip()) + ',' # GSH binding rate
                               + str(GSHMass[4].rstrip()) + ',' # regeneration rate
                               + str(GSHMass[5].rstrip()) + '\n') # random value
        
    HepatocytesField.close()
    GSHField.close()
    return

def Initialize_Zonation_File(): # Must call before Hepatocyte file init
    # Define filepaths
    node_file = project_path('Mesh_Volume_Calculations', 'Field_Nodes.csv')
    pv_wall_file = project_path('Mesh_Volume_Calculations', 'Portal_Vein_Walls.txt')
    cv_wall_file = project_path('Mesh_Volume_Calculations', 'Central_Vein_Wall.txt')
    zonation_file = project_path('Mesh_Volume_Calculations', 'Field_Zonation.txt')

    # LOAD FIELD NODES Assumed columns:
    # node_id, x, y, z, [other columns ignored]
    field_data = np.loadtxt(
        node_file,
        delimiter=',',
        skiprows=0, 
    )



    node_ids = field_data[:, 0].astype(int)
    field_nodes_cords = field_data[:, 1:4] # grab X,y,z

    # ------------------------------------------------------------
    # LOAD VESSEL WALL POINTS
    # ------------------------------------------------------------

    pv_wall_nodes_cords = np.loadtxt(pv_wall_file)
    cv_wall_nodes_cords = np.loadtxt(cv_wall_file)

    # ------------------------------------------------------------
    # VALIDATE INPUT
    # ------------------------------------------------------------

    #if field_nodes_cords.shape[1] != 3:
    #    raise ValueError(
    #        "Field_Nodes.csv must contain node_id followed by x,y,z."
    #    )
#
    #if pv_wall_nodes_cords.ndim != 2 or pv_wall_nodes_cords.shape[1] != 3:
    #    raise ValueError(
    #        "Portal vein wall file must contain X Y Z."
    #    )
#
    #if cv_wall_nodes_cords.ndim != 2 or cv_wall_nodes_cords.shape[1] != 3:
    #    raise ValueError(
    #        "Central vein wall file must contain X Y Z."
    #    )




    # ------------------------------------------------------------
    # LAUE COORDINATE
    #
    # xi = 0 -> portal
    # xi = 1 -> central
    # ------------------------------------------------------------

    pv_tree = cKDTree(pv_wall_nodes_cords)
    cv_tree = cKDTree(cv_wall_nodes_cords)

    dPV, _ = pv_tree.query(field_nodes_cords)
    dCV, _ = cv_tree.query(field_nodes_cords)

    denominator = dPV + dCV

    if np.any(denominator <= 0.0):
        raise ValueError(
            "Invalid Laue coordinate: dPV + dCV <= 0."
        )

    xi = dPV / denominator

    xi = np.clip(xi, 0.0, 1.0)

    # ------------------------------------------------------------
    # PINTO ENDPOINTS
    # ------------------------------------------------------------

    valueGSHInitPortal = 1.0e-14
    valueGSHInitCentral = 0.5e-14

    valueGSHProductRatePortal = 1.575e-14 / 8.64e4
    valueGSHProductRateCentral = 1.175e-14 / 8.64e4

    valueGSHBindingRatePortal = 1.8e18 / 8.64e4
    valueGSHBindingRateCentral = 1.4e18 / 8.64e4

    valueRegenRatePortal = 1.1 / 8.64e4
    valueRegenRateCentral = 0.9 / 8.64e4

    # ------------------------------------------------------------
    # PINTO QSP ZONATION USING LAUE COORDINATE
    # ------------------------------------------------------------

    GSHInitValue = (
        valueGSHInitPortal
        + xi * (
            valueGSHInitCentral
            - valueGSHInitPortal
        )
    )

    GSHProducRateValue = (
        valueGSHProductRatePortal
        + xi * (
            valueGSHProductRateCentral
            - valueGSHProductRatePortal
        )
    )

    GSHBindingRateValue = (
        valueGSHBindingRatePortal
        + xi * (
            valueGSHBindingRateCentral
            - valueGSHBindingRatePortal
        )
    )

    regenRateValue = (
        valueRegenRatePortal
        + xi * (
            valueRegenRateCentral
            - valueRegenRatePortal
        )
    )

    # ------------------------------------------------------------
    # WRITE FIELD_ZONATION
    # ------------------------------------------------------------

    rd = np.random.default_rng(ZONATION_RANDOM_SEED) # Write random values to file for reproducibility
    with zonation_file.open('w') as zonationField:
        for i in range(len(field_nodes_cords)):
            random_value = rd.random()
            zonationField.write(
                str(node_ids[i]) + ','
                + str(GSHInitValue[i]) + ','
                + str(GSHProducRateValue[i]) + ','
                + str(GSHBindingRateValue[i]) + ','
                + str(regenRateValue[i]) + ','
                + str(random_value)
                + '\n'
            )

    return


def Initialize_Status_File():
    statusFile = project_path('hepatocytes_status_Hepatocytes.txt').open('w')
    nodeField = project_path("Mesh_Volume_Calculations", "Field_Nodes.csv").open('r+')
    Nodes = nodeField.readlines()
    for line in Nodes:
        statusFile.write('0.0\n')
    statusFile.close()
    nodeField.close()
    return

def Initialize_All_Files(): # Init all files
    Initialize_Zonation_File()
    Initialize_Hepatocytes_File()
    #Initialize_Porosity_File()
    Initialize_Status_File()
    
    return