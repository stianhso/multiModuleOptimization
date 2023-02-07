import numpy as np
import scipy as sc
import matplotlib.pyplot as plt
import pandas as pd
import os
from scipy.integrate import odeint,solve_ivp
from lib.HydroDReader import getMatrices
import lib.WadamReader


class Module():
    """""""""""""""""""""""""""""""""
    Function: __init__()

    - Assigns variable values 
    - Creates lists containing local and global coordinates for module corners, connection points & pontoons
    - Assigns forces 

    """""""""""""""""""""""""""""""""

    def __init__(self, T: "wave period", modNrInX, modNrInY):

        #General Constants
        self.rho_w = 1025  # kg/m^3
        self.g = 9.81  # m/s

        #Wave Characteristics
        self.T = T  # Wave period
        self.omega = 2 * np.pi / T  # Circular wave frequency
        #self.Beta = Beta  # Wave angle

        # Module characteristics
        self.D = 1.13  # m - Draught
        self.d = 1.6  # m - Diameter of pontoon
        self.l = 8.4  # m - longitudinal length between pontoons
        self.b = 8.4  # m - breadth between pontoons
        self.Aw = np.pi * (self.d / 2) ** 2  # Waterplane area
        self.xspace = 13 # m - Spacing between modules in x-axis
        self.yspace = 13 # m - Spacing between modules in y-axis
        self.airgap = 1.5 #m - Distance from still water to module. Onsrud numbers: Floater height - Module draught = 2.63 - 1.13


        #Extracting hydrodynamic matrices from wadam
        self.WADAMFilename = "WADAMFiles/WADAM1.txt"  # Location and name of wadam file
        self.M, self.A, self.B, self.C, self.Motion, self.excitationForce = getMatrices(self.WADAMFilename, T)

        #Spring stiffnesses (variable)

        #
        #
        #

        #BUILDING MODULES AND MODULE NETWORK

        #GLOBAL POSITIONING
        # Global position of module's center of area in XY plane
        self.i = modNrInX
        self.j = modNrInY
        self.x = self.xspace * self.i
        self.y = self.yspace * self.j

        # Center of origin of module in XY-plane (arealsenter)
        self.globCenterOfOriginXY = (self.x, self.y)

        """ 
        Global top and bottom coordinate for each pontoon in XZ-plane, in addition to COG
        These are used to plot the XZ-figure in Visualizer.py
        NOTE: COG burde kanskje dobbelsjekkes
        """

        self.globPosPontoonsXZ = [(self.x - self.l / 2, - self.D),
                                  (self.x - self.l / 2, + self.airgap),
                                  (self.x + 0, self.airgap),
                                  (self.x + self.l / 2, self.airgap),
                                  (self.x + self.l / 2, - self.D)]

        """
        Global positions of "panel" corners in XY-plane
        These are used to plot the XY-figure in Visualizer.py
        """
        self.globPosCornersXY = [(self.x - 6, self.y + 6),
                                 (self.x + 6, self.y + 6),
                                 (self.x + 6, self.y - 6),
                                 (self.x - 6, self.y - 6)]

        # Global position of Connection points on each side
        self.globPosConnections = np.array([(self.x, self.y + 6, self.airgap),
                                                   (self.x + 6, self.y, self.airgap),
                                                   (self.x, self.y - 6, self.airgap),
                                                   (self.x - 6, self.y, self.airgap)])


        #LOCAL POSITIONING
        # Local position for each pontoon in XZ-plane (Vegard = LocalPositions)
        self.locPosPontoonsXZ = [(-self.l / 2, -self.D), # (pontoon 1)
                               (-self.l / 2, self.airgap),          # (pontoon 2)
                               (0, self.airgap),                    #
                               (self.l / 2, self.airgap),           # (pontoon 3)
                               (self.l / 2, -self.D)]               # (pontoon 4)

        # Local positions of corners in XY-plane
        self.locPosCornersXY = [(-6, 6),
                                (6, 6),
                                (6, -6),
                                (-6, -6)]



        """
        Connection points:
        Assume one on each side, but they will only be active if there is a module there
        See additional picture on Github for locations

        The array specifies neighbours in the form of [0, 2.3, 1.2, 0] for platform 1.3 (example) and dtype specifies
        that we are only taking in objects of the "Module"-class
        """
        self.neighbourModules = np.array([0, 0, 0, 0],
                                         dtype=Module)  # 1 2 3 4 - Can be compared to connection point 1 2 3 4


        # Local position of Connection points on each side
        self.locPosConnections = np.array([(0, 6, self.airgap),
                                             (6, 0, self.airgap),
                                             (0, -6, self.airgap),
                                             (-6, 0, self.airgap)])



        #CREATE LISTS FOR STORAGE OF VALUES AND LATER USE

        """
        Solution of equation of motion gives eta. 

        self.eta is a temporary storage of a value that is later used in the calculations.
        self.etaTime gives 6x1 matrix with timeseries of module motions in the DOFS
        """
        #HUSK: FJERNE DOF SOM IKKE SKAL BRUKES
        self.eta = 0
        self.etaTime = [[], [], [], [], [], []]  # Surge, sway and yaw motion as timeseries

        self.nuTime = [[], [], [], [], [], []]   # Surge, sway and yaw velocity as timeseries

        """
        self.globPosPontoonTimeXYZ (tidligere absTime) contains information about every pontoon position depending on time
        """
        self.globPosPontoonTimeXYZ = [[], [], []]  # Absolute pontoon position for each time step of simulation (x,y,z)

        self.Connections = np.zeros([4], dtype=list) # Timeseries???

        # Excitation force timeseries
        self.FexcTime = [[], [], [], [], [], []]  # One for each DOF

        self.Forces = [0, 0, 0, 0, 0, 0]  # List with forces in surge, sway, heave and moment in roll, pitch and yaw

        """

        self.Fc - Contains all the temporary spring forces acting on the module from each connection point
        self.FcT - Storage of self.Fc for the whole simulation
        self.Fc6DOF - Total contribution from the spring forces acting on the module, [0:3] = Forces, [3:6] = Moments
        NB! Spring stiffness is not individual (at this moment) for each connection point.  

        """

        self.Fc = np.zeros([4], dtype=list)  # np.array(np.zeros([3]),np.zeros([3]),np.zeros([3]),np.zeros([3]))
                                             # Three spring forces in each of the connection points
        self.Fct = [[], [], [], []]
        self.Fc6DOF = np.zeros([6], dtype=float)
