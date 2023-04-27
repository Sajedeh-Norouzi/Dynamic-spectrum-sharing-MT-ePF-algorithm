# -*- coding: utf-8 -*-
"""
Created on Mon Apr 24 18:04:58 2023

@author: Sajedeh Norouzi
"""

import numpy as np
import math
from math import sqrt, log2
import scipy.io
import os

# np.random.seed(1376)
ratio_matrix = np.array ([[0,0,0,0,0,0,0,0,0,0],  #LTE 0 
                          [0,1,1,0,0,0,0,0,0,0],  #0.2 1
                          [0,1,1,0,0,0,1,0,0,0],  #0.3 2
                          [0,1,1,0,0,0,1,0,1,0],  #0.4 3
                          [0,1,1,1,0,0,1,0,1,0],  #0.5 4
                          [0,1,1,1,0,0,1,1,1,0]]) #0.6 5


# ratio_matrix = np.array ([[0,0,0,0,0,0,0,0,0,0],
#                           [0,1,1,0,0,0,0,0,0,0],
#                           [0,1,1,0,0,0,1,0,0,0],
#                           [0,1,1,0,0,0,1,0,1,0],
#                           [0,1,1,1,0,0,1,0,1,0],
#                           [0,1,1,1,0,0,1,1,1,0],
#                           [1,1,1,1,1,1,1,1,1,1]])

#_____________________UN KNOWN_______________________________
# BX= [1000,2000,1500,500,1000,2000,1500] #X for BS in each cell 
# BY= [500,500,1500,1500,2500,2500,3500] #Y for BS in each cell  

#_____________________URBAN MACRO_______________________________
BX= [2000,4000,3000,1000,2000,4000,3000] #X for BS in each cell 
BY= [1000,1000,3000,3000,5000,5000,7000] #Y for BS in each cell 

class DSS(object):

    def __init__ (self, num_of_LTE, num_of_NR, num_of_LTE_GBR, num_of_LTE_NGBR, num_of_NR_GBR, num_of_NR_NGBR, ratio, time_slot):
        self.N0 = 1.5e-16
        self.covered_area = 1000
        self.covered_inner_area = 600
        # self.covered_area = 500
        # self.covered_inner_area = 300
        self.BSPower = 15 # watt
        self.ratio = ratio
        self.n_RB = 1200
        self.ratio_number = 6 
        self.time_slot = time_slot
        self.totaln_timeslot =10
        self.GBRL_value_pframe = 15 #2000 kbps or 20kbp(10ms)
        self.GBRN_value_pframe = 250 #2000 kbps or 20kbp(10ms)
        self.GBR_rate_LTE = np.round(self.GBRL_value_pframe / (self.totaln_timeslot-np.sum(ratio_matrix[self.ratio,:])))+1
        self.GBR_rate_NR = np.round(self.GBRN_value_pframe / np.sum(ratio_matrix[self.ratio,:]))+1
        self.BW = 15
        # print ('GBR_rate_LTE',self.GBR_rate_LTE)
        # print ('GBR_rate_NR',self.GBR_rate_NR)
        
        
        
        self.num_of_LTE = num_of_LTE
        self.num_of_NR = num_of_NR
        
        self.num_of_LTE_GBR = num_of_LTE_GBR
        self.num_of_LTE_NGBR = self.num_of_LTE - self.num_of_LTE_GBR
        
        self.num_of_NR_GBR = num_of_NR_GBR
        self.num_of_NR_NGBR = self.num_of_NR - self.num_of_NR_GBR
        
        self.weight_GBR_LTE = num_of_LTE
        self.weight_GBR_NR = num_of_LTE
        

        
        
        self.weight_LTE = (self.n_RB - (self.num_of_LTE_GBR*self.weight_GBR_LTE)) / ((self.num_of_LTE * (self.num_of_LTE+1))/2)
        self.weight_NR =  (self.n_RB - (self.num_of_NR_GBR* self.weight_GBR_NR)) / ((self.num_of_NR *   (self.num_of_NR+1))/2)
        
        self.LTE_UE_X_1 = np.zeros (self.num_of_LTE)
        self.LTE_UE_X_2 = np.zeros (self.num_of_LTE)
        self.LTE_UE_X_3 = np.zeros (self.num_of_LTE)
        self.LTE_UE_X_4 = np.zeros (self.num_of_LTE)
        self.LTE_UE_X_5 = np.zeros (self.num_of_LTE)
        self.LTE_UE_X_6 = np.zeros (self.num_of_LTE)
        self.LTE_UE_X_7 = np.zeros (self.num_of_LTE)
        self.LTE_UE_Y_1 = np.zeros (self.num_of_LTE)
        self.LTE_UE_Y_2 = np.zeros (self.num_of_LTE)
        self.LTE_UE_Y_3 = np.zeros (self.num_of_LTE)
        self.LTE_UE_Y_4 = np.zeros (self.num_of_LTE)
        self.LTE_UE_Y_5 = np.zeros (self.num_of_LTE)
        self.LTE_UE_Y_6 = np.zeros (self.num_of_LTE)
        self.LTE_UE_Y_7 = np.zeros (self.num_of_LTE)
        
        self.NR_UE_X_1 = np.zeros (self.num_of_NR)
        self.NR_UE_X_2 = np.zeros (self.num_of_NR)
        self.NR_UE_X_3 = np.zeros (self.num_of_NR)
        self.NR_UE_X_4 = np.zeros (self.num_of_NR)
        self.NR_UE_X_5 = np.zeros (self.num_of_NR)
        self.NR_UE_X_6 = np.zeros (self.num_of_NR)
        self.NR_UE_X_7 = np.zeros (self.num_of_NR)
        self.NR_UE_Y_1 = np.zeros (self.num_of_NR)
        self.NR_UE_Y_2 = np.zeros (self.num_of_NR)
        self.NR_UE_Y_3 = np.zeros (self.num_of_NR)
        self.NR_UE_Y_4 = np.zeros (self.num_of_NR)
        self.NR_UE_Y_5 = np.zeros (self.num_of_NR)
        self.NR_UE_Y_6 = np.zeros (self.num_of_NR)
        self.NR_UE_Y_7 = np.zeros (self.num_of_NR)

###### 10 user        
       # self.LTE_GBR_index = np.array([4,6,8]) 
       # self.NR_GBR_index =  np.array([4,6,8])
       
        # self.LTE_GBR_index =  np.array([4]) 
        # self.NR_GBR_index  =  np.array([4]) 
        
        
###### 20 user 
        # self.LTE_GBR_index = np.array([4,6,8,14,16,18]) 
        # self.NR_GBR_index =  np.array([4,6,8,14,16,18])
                
        # self.LTE_GBR_index =  np.array([4,14]) 
        # self.NR_GBR_index  =  np.array([4,14]) 
        
###### 30 user         
        # self.LTE_GBR_index = np.array([4,6,8,14,16,18,24,26,28]) 
        # self.NR_GBR_index =  np.array([4,6,8,14,16,18,24,26,28])
               
        # self.LTE_GBR_index =  np.array([4,14,24])
        # self.NR_GBR_index  =  np.array([4,14,24]) 
        
        # self.LTE_GBR_index =  np.array([])
        # self.NR_GBR_index  =  np.array([]) 
        
###### 40 user         
        # self.LTE_GBR_index = np.array([4,6,8,14,16,18,24,26,28,34,36,38]) 
        # self.NR_GBR_index =  np.array([4,6,8,14,16,18,24,26,28,34,36,38])     
               
        # self.LTE_GBR_index =  np.array([4,14,24,34]) 
        # self.NR_GBR_index  =  np.array([4,14,24,34]) 
 
###### 50 user          
        # self.LTE_GBR_index = np.array([4,6,8,14,16,18,24,26,28,34,36,38,44,46,48]) 
        # self.NR_GBR_index =  np.array([4,6,8,14,16,18,24,26,28,34,36,38,44,46,48])      
  
        self.LTE_GBR_index = np.array([4,14,24,34,44]) 
        self.NR_GBR_index  =  np.array([4,14,24,34,44])      
        

        
    def RandomWalk(self):
        
        i1=0
        while i1< self.num_of_LTE:
            x = np.random.uniform(1000, 3000)
            y = np.random.uniform(0, 2000)
            if sqrt(((x-BX[0])**2)+((y-BY[0])**2)) <= 1000:
                self.LTE_UE_X_1 [i1] = x
                self.LTE_UE_Y_1 [i1] = y
                i1 +=1
        
        i2=0
        while i2< self.num_of_LTE:
            x = np.random.uniform(3000, 5000)
            y = np.random.uniform(0, 2000)
            if sqrt(((x-BX[1])**2)+((y-BY[1])**2)) <= 1000:
                self.LTE_UE_X_2 [i2] = x
                self.LTE_UE_Y_2 [i2] = y
                i2 +=1
                
        i3=0
        while i3< self.num_of_LTE:
            x = np.random.uniform(2000, 4000)
            y = np.random.uniform(2000, 4000)
            if sqrt(((x-BX[2])**2)+((y-BY[2])**2)) <= 1000:
                self.LTE_UE_X_3 [i3] = x
                self.LTE_UE_Y_3 [i3] = y
                i3 +=1

        i4=0
        while i4< self.num_of_LTE:
            x = np.random.uniform(0, 2000)
            y = np.random.uniform(2000, 4000)
            if sqrt(((x-BX[3])**2)+((y-BY[3])**2)) <= 1000:
                self.LTE_UE_X_4 [i4] = x
                self.LTE_UE_Y_4 [i4] = y
                i4 +=1                
         
        i5=0
        while i5< self.num_of_LTE:
            x = np.random.uniform(1000, 3000)
            y = np.random.uniform(4000, 6000)
            if sqrt(((x-BX[4])**2)+((y-BY[4])**2)) <= 1000:
                self.LTE_UE_X_5 [i5] = x
                self.LTE_UE_Y_5 [i5] = y
                i5 +=1        

        i6=0
        while i6< self.num_of_LTE:
            x = np.random.uniform(3000, 5000)
            y = np.random.uniform(4000, 6000)
            if sqrt(((x-BX[5])**2)+((y-BY[5])**2)) <= 1000:
                self.LTE_UE_X_6 [i6] = x
                self.LTE_UE_Y_6 [i6] = y
                i6 +=1               

        i7=0
        while i7< self.num_of_LTE:
            x = np.random.uniform(2000, 4000)
            y = np.random.uniform(6000, 8000)
            if sqrt(((x-BX[6])**2)+((y-BY[6])**2)) <= 1000:
                self.LTE_UE_X_7 [i7] = x
                self.LTE_UE_Y_7 [i7] = y
                i7 +=1                 
            
        
        j1=0
        while j1< self.num_of_NR:
            x = np.random.uniform(1000, 3000)
            y = np.random.uniform(0, 2000)
            if sqrt(((x-BX[0])**2)+((y-BY[0])**2)) <= 1000:
                self.NR_UE_X_1 [j1] = x
                self.NR_UE_Y_1 [j1] = y
                j1 +=1
        
        
        j2=0
        while j2< self.num_of_NR:
            x = np.random.uniform(3000, 5000)
            y = np.random.uniform(0, 2000)
            if sqrt(((x-BX[1])**2)+((y-BY[1])**2)) <= 1000:
                self.NR_UE_X_2 [j2] = x
                self.NR_UE_Y_2 [j2] = y
                j2 +=1
        
        
        j3=0
        while j3< self.num_of_NR:
            x = np.random.uniform(2000, 4000)
            y = np.random.uniform(2000, 4000)
            if sqrt(((x-BX[2])**2)+((y-BY[2])**2)) <= 1000:
                self.NR_UE_X_3 [j3] = x
                self.NR_UE_Y_3 [j3] = y
                j3 +=1

        
        j4=0
        while j4< self.num_of_NR:
            x = np.random.uniform(0, 2000)
            y = np.random.uniform(2000, 4000)
            if sqrt(((x-BX[3])**2)+((y-BY[3])**2)) <= 1000:
                self.NR_UE_X_4 [j4] = x
                self.NR_UE_Y_4 [j4] = y
                j4 +=1

        
        j5=0
        while j5< self.num_of_NR:
            x = np.random.uniform(1000, 3000)
            y = np.random.uniform(4000, 6000)
            if sqrt(((x-BX[4])**2)+((y-BY[4])**2)) <= 1000:
                self.NR_UE_X_5 [j5] = x
                self.NR_UE_Y_5 [j5] = y
                j5 +=1

        
        j6=0
        while j6< self.num_of_NR:
            x = np.random.uniform(3000, 5000)
            y = np.random.uniform(4000, 6000)
            if sqrt(((x-BX[5])**2)+((y-BY[5])**2)) <= 1000:
                self.NR_UE_X_6 [j6] = x
                self.NR_UE_Y_6 [j6] = y
                j6 +=1
                
        
        j7=0
        while j7< self.num_of_NR:
            x = np.random.uniform(2000, 4000)
            y = np.random.uniform(6000, 8000)
            if sqrt(((x-BX[6])**2)+((y-BY[6])**2)) <= 1000:
                self.NR_UE_X_7 [j7] = x
                self.NR_UE_Y_7 [j7] = y
                j7 +=1
                




    def DefineUECellWithGains(self):   #define distance user of each cells from BS lte        
        DSS.RandomWalk(self)
        self.DistanceLTEUE1 = np.zeros(self.num_of_LTE)
        self.DistanceLTEUE2 = np.zeros(self.num_of_LTE)
        self.DistanceLTEUE3 = np.zeros(self.num_of_LTE)
        self.DistanceLTEUE4 = np.zeros(self.num_of_LTE)
        self.DistanceLTEUE5 = np.zeros(self.num_of_LTE)
        self.DistanceLTEUE6 = np.zeros(self.num_of_LTE)
        self.DistanceLTEUE7 = np.zeros(self.num_of_LTE)
        #define distance user of each cells from BS NR
        self.DistanceNRUE1 = np.zeros(self.num_of_NR)
        self.DistanceNRUE2 = np.zeros(self.num_of_NR)
        self.DistanceNRUE3 = np.zeros(self.num_of_NR)
        self.DistanceNRUE4 = np.zeros(self.num_of_NR)
        self.DistanceNRUE5 = np.zeros(self.num_of_NR)
        self.DistanceNRUE6 = np.zeros(self.num_of_NR)
        self.DistanceNRUE7 = np.zeros(self.num_of_NR)
        
        
        for u in range(self.num_of_LTE):
            # ue is in cell number 1
            self.DistanceLTEUE1[u] = sqrt((abs(self.LTE_UE_X_1[u] - BX[0])) ** 2 + (abs(self.LTE_UE_Y_1[u] - BY[0])) ** 2)

            # ue is in cell number 2
            self.DistanceLTEUE2[u] = sqrt((abs(self.LTE_UE_X_2[u] - BX[1])) ** 2 + (abs(self.LTE_UE_Y_2[u] - BY[1])) ** 2)

            # ue is in cell number 3
            self.DistanceLTEUE3[u] = sqrt((abs(self.LTE_UE_X_3[u] - BX[2])) ** 2 + (abs(self.LTE_UE_Y_3[u] - BY[2])) ** 2)
            
            # ue is in cell number 4
            self.DistanceLTEUE4[u] = sqrt((abs(self.LTE_UE_X_4[u] - BX[3])) ** 2 + (abs(self.LTE_UE_Y_4[u] - BY[3])) ** 2)

            # ue is in cell number 5
            self.DistanceLTEUE5[u] = sqrt((abs(self.LTE_UE_X_5[u] - BX[4])) ** 2 + (abs(self.LTE_UE_Y_5[u] - BY[4])) ** 2)

            # ue is in cell number 6
            self.DistanceLTEUE6[u] = sqrt((abs(self.LTE_UE_X_6[u] - BX[5])) ** 2 + (abs(self.LTE_UE_Y_6[u] - BY[5])) ** 2)
            
            # ue is in cell number 7
            self.DistanceLTEUE7[u] = sqrt((abs(self.LTE_UE_X_7[u] - BX[6])) ** 2 + (abs(self.LTE_UE_Y_7[u] - BY[6])) ** 2)
            


        for u in range(self.num_of_NR):
            # ue is in cell number 1
            self.DistanceNRUE1[u] = sqrt((abs(self.NR_UE_X_1[u] - BX[0])) ** 2 + (abs(self.NR_UE_Y_1[u] - BY[0])) ** 2)

            # ue is in cell number 2
            self.DistanceNRUE2[u] = sqrt((abs(self.NR_UE_X_2[u] -  BX[1])) ** 2 + (abs(self.NR_UE_Y_2[u] - BY[1])) ** 2)

            # ue is in cell number 3
            self.DistanceNRUE3[u] = sqrt((abs(self.NR_UE_X_3[u] - BX[2])) ** 2 + (abs(self.NR_UE_Y_3[u] - BY[2])) ** 2)
            
            # ue is in cell number 4
            self.DistanceNRUE4[u] = sqrt((abs(self.NR_UE_X_4[u] - BX[3])) ** 2 + (abs(self.NR_UE_Y_4[u] - BY[3])) ** 2)

            # ue is in cell number 5
            self.DistanceNRUE5[u] = sqrt((abs(self.NR_UE_X_5[u] - BX[4])) ** 2 + (abs(self.NR_UE_Y_5[u] - BY[4])) ** 2)

            # ue is in cell number 6
            self.DistanceNRUE6[u] = sqrt((abs(self.NR_UE_X_6[u] - BX[5])) ** 2 + (abs(self.NR_UE_Y_6[u] -  BY[5])) ** 2)
            
            # ue is in cell number 7
            self.DistanceNRUE7[u] = sqrt((abs(self.NR_UE_X_7[u] - BX[6])) ** 2 + (abs(self.NR_UE_Y_7[u] - BY[6])) ** 2) 
           
        
        
        # LTE cell 1 channel gains
        L_LTE1 = 12.81 + 3.76 * np.log10(self.DistanceLTEUE1)

        c_LTE = -1 * L_LTE1 / 20

        antenna_gain_LTE = 0.9

        s_LTE = 0.8

        self.LTEChannelGains1 = pow(10, c_LTE) * math.sqrt((antenna_gain_LTE * s_LTE)) * np.random.rayleigh( scale = 1.0,size=(self.num_of_LTE))
                                                                                                                


        # LTE cell 2 channel gains
        L_LTE2 = 12.81 + 3.76 * np.log10(self.DistanceLTEUE2)

        c_LTE = -1 * L_LTE2 / 20

        antenna_gain_LTE = 0.9

        s_LTE = 0.8

        self.LTEChannelGains2 = pow(10, c_LTE) * math.sqrt((antenna_gain_LTE * s_LTE)) * np.random.rayleigh (scale = 1.0, size=(self.num_of_LTE))
              
                                                                                                          


        # LTE cell 3 channel gains
        L_LTE3 = 12.81 + 3.76 * np.log10(self.DistanceLTEUE3)

        c_LTE = -1 * L_LTE3 / 20

        antenna_gain_LTE = 0.9

        s_LTE = 0.8

        self.LTEChannelGains3 = pow(10, c_LTE) * math.sqrt((antenna_gain_LTE * s_LTE)) * np.random.rayleigh (scale = 1.0, size=(self.num_of_LTE))

        # LTE cell 4 channel gains
        L_LTE4 = 12.81 + 3.76 * np.log10(self.DistanceLTEUE4)

        c_LTE = -1 * L_LTE4 / 20

        antenna_gain_LTE = 0.9

        s_LTE = 0.8

        self.LTEChannelGains4 = pow(10, c_LTE) * math.sqrt((antenna_gain_LTE * s_LTE)) * np.random.rayleigh (scale = 1.0, size=(self.num_of_LTE))


        # LTE cell 5 channel gains
        L_LTE5 = 12.81 + 3.76 * np.log10(self.DistanceLTEUE5)

        c_LTE = -1 * L_LTE5 / 20

        antenna_gain_LTE = 0.9

        s_LTE = 0.8

        self.LTEChannelGains5 = pow(10, c_LTE) * math.sqrt((antenna_gain_LTE * s_LTE)) * np.random.rayleigh (scale = 1.0, size=(self.num_of_LTE))


        # LTE cell 6 channel gains
        L_LTE6 = 12.81 + 3.76 * np.log10(self.DistanceLTEUE6)

        c_LTE = -1 * L_LTE6 / 20

        antenna_gain_LTE = 0.9

        s_LTE = 0.8

        self.LTEChannelGains6 = pow(10, c_LTE) * math.sqrt((antenna_gain_LTE * s_LTE)) * np.random.rayleigh (scale = 1.0, size=(self.num_of_LTE))

        # LTE cell 7 channel gains
        L_LTE7 = 12.81 + 3.76 * np.log10(self.DistanceLTEUE7)

        c_LTE = -1 * L_LTE7 / 20

        antenna_gain_LTE = 0.9

        s_LTE = 0.8

        self.LTEChannelGains7 = pow(10, c_LTE) * math.sqrt((antenna_gain_LTE * s_LTE)) * np.random.rayleigh (scale = 1.0, size=(self.num_of_LTE))



        # NR cell 1 channel gains
        L_NR1 = 12.81 + 3.76 * np.log10(self.DistanceNRUE1)

        c_NR = -1 * L_NR1 / 20

        antenna_gain_NR = 0.9

        s_NR = 0.8

        self.NRChannelGains1 = pow(10, c_NR) * math.sqrt((antenna_gain_NR * s_NR)) * np.random.rayleigh (scale = 1.0, size=(self.num_of_NR))


        # NR cell 2 channel gains
        L_NR2 = 12.81 + 3.76 * np.log10(self.DistanceNRUE2)

        c_NR = -1 * L_NR2 / 20

        antenna_gain_NR = 0.9

        s_NR = 0.8

        self.NRChannelGains2 = pow(10, c_NR) * math.sqrt((antenna_gain_NR * s_NR)) * np.random.rayleigh (scale = 1.0, size=(self.num_of_NR))


        # NR cell 3 channel gains
        L_NR3 = 12.81 + 3.76 * np.log10(self.DistanceNRUE3)

        c_NR = -1 * L_NR3 / 20

        antenna_gain_NR = 0.9

        s_NR = 0.8

        self.NRChannelGains3 = pow(10, c_NR) * math.sqrt((antenna_gain_NR * s_NR)) * np.random.rayleigh (scale = 1.0, size=(self.num_of_NR))

        # NR cell 4 channel gains
        L_NR4 = 12.81 + 3.76 * np.log10(self.DistanceNRUE4)

        c_NR = -1 * L_NR4 / 20

        antenna_gain_NR = 0.9

        s_NR = 0.8

        self.NRChannelGains4 = pow(10, c_NR) * math.sqrt((antenna_gain_NR * s_NR)) * np.random.rayleigh (scale = 1.0, size=(self.num_of_NR))


        # NR cell 5 channel gains
        L_NR5 = 12.81 + 3.76 * np.log10(self.DistanceNRUE5)

        c_NR = -1 * L_NR5 / 20

        antenna_gain_NR = 0.9

        s_NR = 0.8

        self.NRChannelGains5 = pow(10, c_NR) * math.sqrt((antenna_gain_NR * s_NR)) * np.random.rayleigh (scale = 1.0, size=(self.num_of_NR))


        # NR cell 6 channel gains
        L_NR6 = 12.81 + 3.76 * np.log10(self.DistanceNRUE6)

        c_NR = -1 * L_NR6 / 20

        antenna_gain_NR = 0.9

        s_NR = 0.8

        self.NRChannelGains6 = pow(10, c_NR) * math.sqrt((antenna_gain_NR * s_NR)) * np.random.rayleigh (scale = 1.0, size=(self.num_of_NR))

        # NR cell 7 channel gains
        L_NR7 = 12.81 + 3.76 * np.log10(self.DistanceNRUE7)

        c_NR = -1 * L_NR7 / 20

        antenna_gain_NR = 0.9

        s_NR = 0.8

        self.NRChannelGains7 = pow(10, c_NR) * math.sqrt((antenna_gain_NR * s_NR)) * np.random.rayleigh (scale = 1.0, size=(self.num_of_NR))

        return self.LTEChannelGains1, self.LTEChannelGains2, self.LTEChannelGains3, self.LTEChannelGains4, self.LTEChannelGains5,\
    self.LTEChannelGains6, self.LTEChannelGains7, self.NRChannelGains1, self.NRChannelGains2, self.NRChannelGains3, self.NRChannelGains4,\
    self.NRChannelGains5, self.NRChannelGains6, self.NRChannelGains7, self.DistanceLTEUE1, self.DistanceLTEUE2, self.DistanceLTEUE3, \
    self.DistanceLTEUE4, self.DistanceLTEUE5, self.DistanceLTEUE6, self.DistanceLTEUE7, self.DistanceNRUE1,  self.DistanceNRUE2,  self.DistanceNRUE3, \
    self.DistanceNRUE4,  self.DistanceNRUE5,  self.DistanceNRUE6,  self.DistanceNRUE7
        
    
    def Interference(self):
        DSS.DefineUECellWithGains(self)
        
        self.Interference_LTE_1 = np.zeros (self.num_of_LTE)
        self.Interference_LTE_2 = np.zeros (self.num_of_LTE)
        self.Interference_LTE_3 = np.zeros (self.num_of_LTE)
        self.Interference_LTE_4 = np.zeros (self.num_of_LTE)
        self.Interference_LTE_5 = np.zeros (self.num_of_LTE)
        self.Interference_LTE_6 = np.zeros (self.num_of_LTE)
        self.Interference_LTE_7 = np.zeros (self.num_of_LTE)
        
        
        self.Interference_NR_1 = np.zeros (self.num_of_NR)
        self.Interference_NR_2 = np.zeros (self.num_of_NR)
        self.Interference_NR_3 = np.zeros (self.num_of_NR)
        self.Interference_NR_4 = np.zeros (self.num_of_NR)
        self.Interference_NR_5 = np.zeros (self.num_of_NR)
        self.Interference_NR_6 = np.zeros (self.num_of_NR)
        self.Interference_NR_7 = np.zeros (self.num_of_NR)
        
        
        self.rb_interference = int(2.5*self.n_RB/self.num_of_LTE)+1
        
        if not ratio_matrix [self.ratio, self.time_slot]:    #LTE      
            for i in range (self.num_of_LTE):
                if self.DistanceLTEUE1 [i] <= self.covered_inner_area: #Inner
                    self.Interference_LTE_1 [i]= self.BSPower * self.rb_interference *\
                                                                (np.mean(self.LTEChannelGains2)+\
                                                                 np.mean(self.LTEChannelGains3)+\
                                                                 np.mean(self.LTEChannelGains4)+\
                                                                 np.mean(self.LTEChannelGains5)+\
                                                                 np.mean(self.LTEChannelGains6)+\
                                                                 np.mean(self.LTEChannelGains7))
                    
                else: # Outer
                    self.Interference_LTE_1 [i]= self.BSPower * self.rb_interference *\
                                                                (np.mean(self.LTEChannelGains2)+\
                                                                 np.mean(self.LTEChannelGains5))

                    
            for i in range (self.num_of_LTE):
                if self.DistanceLTEUE2 [i] <= self.covered_inner_area:
                    self.Interference_LTE_2 [i]= self.BSPower * self.rb_interference *\
                                                                (np.mean(self.LTEChannelGains1)+\
                                                                 np.mean(self.LTEChannelGains3)+\
                                                                 np.mean(self.LTEChannelGains4)+\
                                                                 np.mean(self.LTEChannelGains5)+\
                                                                 np.mean(self.LTEChannelGains6)+\
                                                                 np.mean(self.LTEChannelGains7))

                else:
                    self.Interference_LTE_2 [i]= self.BSPower * self.rb_interference *\
                                                                (np.mean(self.LTEChannelGains4)+\
                                                                 np.mean(self.LTEChannelGains6))
                    
            for i in range (self.num_of_LTE):
                if self.DistanceLTEUE3 [i] <= self.covered_inner_area:
                    self.Interference_LTE_3 [i]= self.BSPower * self.rb_interference *\
                                                                (np.mean(self.LTEChannelGains1)+\
                                                                 np.mean(self.LTEChannelGains2)+\
                                                                 np.mean(self.LTEChannelGains4)+\
                                                                 np.mean(self.LTEChannelGains5)+\
                                                                 np.mean(self.LTEChannelGains6)+\
                                                                 np.mean(self.LTEChannelGains7))
                        
                else:
                    self.Interference_LTE_3 [i]= self.BSPower * self.rb_interference *\
                                                                (np.mean(self.LTEChannelGains5)+\
                                                                 np.mean(self.LTEChannelGains7))
                        
                        
                    
            for i in range (self.num_of_LTE):
                if self.DistanceLTEUE4 [i] <= self.covered_inner_area:
                    self.Interference_LTE_4 [i]= self.BSPower * self.rb_interference *\
                                                                (np.mean(self.LTEChannelGains1)+\
                                                                 np.mean(self.LTEChannelGains2)+\
                                                                 np.mean(self.LTEChannelGains3)+\
                                                                 np.mean(self.LTEChannelGains5)+\
                                                                 np.mean(self.LTEChannelGains6)+\
                                                                 np.mean(self.LTEChannelGains7))
                        
                else:
                    self.Interference_LTE_4 [i]= self.BSPower * self.rb_interference *\
                                                                (np.mean(self.LTEChannelGains2)+\
                                                                 np.mean(self.LTEChannelGains6))
                        
                        
                    
            for i in range (self.num_of_LTE):
                if self.DistanceLTEUE5 [i] <= self.covered_inner_area:
                    self.Interference_LTE_5 [i]= self.BSPower * self.rb_interference *\
                                                                (np.mean(self.LTEChannelGains1)+\
                                                                 np.mean(self.LTEChannelGains2)+\
                                                                 np.mean(self.LTEChannelGains3)+\
                                                                 np.mean(self.LTEChannelGains4)+\
                                                                 np.mean(self.LTEChannelGains6)+\
                                                                 np.mean(self.LTEChannelGains7))
                        
                else:
                    self.Interference_LTE_5 [i]= self.BSPower * self.rb_interference *\
                                                                (np.mean(self.LTEChannelGains3)+\
                                                                 np.mean(self.LTEChannelGains7))
                        
                        
            for i in range (self.num_of_LTE):
                if self.DistanceLTEUE6 [i] <= self.covered_inner_area:
                    self.Interference_LTE_6 [i]= self.BSPower * self.rb_interference *\
                                                                (np.mean(self.LTEChannelGains1)+\
                                                                 np.mean(self.LTEChannelGains2)+\
                                                                 np.mean(self.LTEChannelGains3)+\
                                                                 np.mean(self.LTEChannelGains4)+\
                                                                 np.mean(self.LTEChannelGains5)+\
                                                                 np.mean(self.LTEChannelGains7))
                        
                else:
                    self.Interference_LTE_6 [i]= self.BSPower * self.rb_interference *\
                                                                (np.mean(self.LTEChannelGains2)+\
                                                                 np.mean(self.LTEChannelGains4))
                        
                        
                    
            for i in range (self.num_of_LTE):
                if self.DistanceLTEUE7 [i] <= self.covered_inner_area:
                    self.Interference_LTE_7 [i]= self.BSPower * self.rb_interference *\
                                                                (np.mean(self.LTEChannelGains1)+\
                                                                 np.mean(self.LTEChannelGains2)+\
                                                                 np.mean(self.LTEChannelGains3)+\
                                                                 np.mean(self.LTEChannelGains4)+\
                                                                 np.mean(self.LTEChannelGains5)+\
                                                                 np.mean(self.LTEChannelGains6))
                        
                else:
                    self.Interference_LTE_7 [i]= self.BSPower * self.rb_interference *\
                                                                (np.mean(self.LTEChannelGains3)+\
                                                                 np.mean(self.LTEChannelGains5))
                   
        if ratio_matrix [self.ratio, self.time_slot]: #NR
            for i in range (self.num_of_NR):
                if self.DistanceNRUE1 [i] <= self.covered_inner_area: #Inner
                    self.Interference_NR_1 [i]= self.BSPower * self.rb_interference *\
                                                            (np.mean(self.NRChannelGains2)+\
                                                             np.mean(self.NRChannelGains3)+\
                                                             np.mean(self.NRChannelGains4)+\
                                                             np.mean(self.NRChannelGains5)+\
                                                             np.mean(self.NRChannelGains6)+\
                                                             np.mean(self.NRChannelGains7))
                                                                
                else: # Outer
                    self.Interference_NR_1 [i]= self.BSPower *  self.rb_interference *\
                                                            (np.mean(self.NRChannelGains2)+\
                                                             np.mean(self.NRChannelGains5))
                

                
            for i in range (self.num_of_NR):
                if self.DistanceNRUE2 [i] <= self.covered_inner_area:
                    self.Interference_NR_2 [i]= self.BSPower *  self.rb_interference *\
                                                                (np.mean(self.NRChannelGains1)+\
                                                                 np.mean(self.NRChannelGains3)+\
                                                                 np.mean(self.NRChannelGains4)+\
                                                                 np.mean(self.NRChannelGains5)+\
                                                                 np.mean(self.NRChannelGains6)+\
                                                                 np.mean(self.NRChannelGains7))
                else:                        
                    self.Interference_NR_2 [i]= self.BSPower * self.rb_interference *\
                                                                (np.mean(self.NRChannelGains4)+\
                                                                 np.mean(self.NRChannelGains6))
                    
            for i in range (self.num_of_NR):
                if self.DistanceNRUE3 [i] <= self.covered_inner_area:
                    self.Interference_NR_3 [i]= self.BSPower *  self.rb_interference *\
                                                                (np.mean(self.NRChannelGains1)+\
                                                                 np.mean(self.NRChannelGains2)+\
                                                                 np.mean(self.NRChannelGains4)+\
                                                                 np.mean(self.NRChannelGains5)+\
                                                                 np.mean(self.NRChannelGains6)+\
                                                                 np.mean(self.NRChannelGains7))
                else:
                    self.Interference_NR_3 [i]= self.BSPower * self.rb_interference *\
                                                                (np.mean(self.NRChannelGains5)+\
                                                                 np.mean(self.NRChannelGains7))
                        
                    
            for i in range (self.num_of_NR):
                if self.DistanceNRUE4 [i] <= self.covered_inner_area:                 
                    self.Interference_NR_4 [i]= self.BSPower *  self.rb_interference *\
                                                                (np.mean(self.NRChannelGains1)+\
                                                                 np.mean(self.NRChannelGains2)+\
                                                                 np.mean(self.NRChannelGains3)+\
                                                                 np.mean(self.NRChannelGains5)+\
                                                                 np.mean(self.NRChannelGains6)+\
                                                                 np.mean(self.NRChannelGains7))
                else:
                    self.Interference_NR_4 [i]= self.BSPower * self.rb_interference *\
                                                                (np.mean(self.NRChannelGains2)+\
                                                                 np.mean(self.NRChannelGains6))
                        
                    
            for i in range (self.num_of_NR):
                if self.DistanceNRUE5 [i] <= self.covered_inner_area:
                    self.Interference_NR_5 [i]= self.BSPower *  self.rb_interference *\
                                                                (np.mean(self.NRChannelGains1)+\
                                                                 np.mean(self.NRChannelGains2)+\
                                                                 np.mean(self.NRChannelGains3)+\
                                                                 np.mean(self.NRChannelGains4)+\
                                                                 np.mean(self.NRChannelGains6)+\
                                                                 np.mean(self.NRChannelGains7))
                else:
                    self.Interference_NR_5 [i]= self.BSPower * self.rb_interference *\
                                                                (np.mean(self.NRChannelGains3)+\
                                                                 np.mean(self.NRChannelGains7))
                        
            for i in range (self.num_of_NR):
                if self.DistanceNRUE6 [i] <= self.covered_inner_area:
                    self.Interference_NR_6 [i]= self.BSPower *  self.rb_interference *\
                                                                (np.mean(self.NRChannelGains1)+\
                                                                 np.mean(self.NRChannelGains2)+\
                                                                 np.mean(self.NRChannelGains3)+\
                                                                 np.mean(self.NRChannelGains4)+\
                                                                 np.mean(self.NRChannelGains5)+\
                                                                 np.mean(self.NRChannelGains7))
                else:
                    self.Interference_NR_6 [i]= self.BSPower * self.rb_interference *\
                                                                (np.mean(self.NRChannelGains2)+\
                                                                 np.mean(self.NRChannelGains4))
                        
                    
            for i in range (self.num_of_NR):
                if self.DistanceNRUE7 [i] <= self.covered_inner_area:
                    self.Interference_NR_7 [i]= self.BSPower *  self.rb_interference *\
                                                                (np.mean(self.NRChannelGains1)+\
                                                                 np.mean(self.NRChannelGains2)+\
                                                                 np.mean(self.NRChannelGains3)+\
                                                                 np.mean(self.NRChannelGains4)+\
                                                                 np.mean(self.NRChannelGains5)+\
                                                                 np.mean(self.NRChannelGains6))
                else:
                    self.Interference_NR_7 [i]= self.BSPower * self.rb_interference *\
                                                                (np.mean(self.NRChannelGains3)+\
                                                                 np.mean(self.NRChannelGains5))


    def BR_assignment (self):
        DSS.DefineUECellWithGains(self)
        DSS.Interference(self)            

        self.LTE_RB_assignment_1 = np.zeros (self.num_of_LTE)
        self.LTE_RB_assignment_2 = np.zeros (self.num_of_LTE)
        self.LTE_RB_assignment_3 = np.zeros (self.num_of_LTE)
        self.LTE_RB_assignment_4 = np.zeros (self.num_of_LTE)
        self.LTE_RB_assignment_5 = np.zeros (self.num_of_LTE)
        self.LTE_RB_assignment_6 = np.zeros (self.num_of_LTE)
        self.LTE_RB_assignment_7 = np.zeros (self.num_of_LTE) 
        
        self.NR_RB_assignment_1 = np.zeros (self.num_of_NR)
        self.NR_RB_assignment_2 = np.zeros (self.num_of_NR)
        self.NR_RB_assignment_3 = np.zeros (self.num_of_NR)
        self.NR_RB_assignment_4 = np.zeros (self.num_of_NR)
        self.NR_RB_assignment_5 = np.zeros (self.num_of_NR)
        self.NR_RB_assignment_6 = np.zeros (self.num_of_NR)
        self.NR_RB_assignment_7 = np.zeros (self.num_of_NR)

        self.temp_rate_LTE_cell1 = np.zeros(self.num_of_LTE)
        self.temp_rate_LTE_cell2 = np.zeros(self.num_of_LTE)
        self.temp_rate_LTE_cell3 = np.zeros(self.num_of_LTE)
        self.temp_rate_LTE_cell4 = np.zeros(self.num_of_LTE)
        self.temp_rate_LTE_cell5 = np.zeros(self.num_of_LTE)
        self.temp_rate_LTE_cell6 = np.zeros(self.num_of_LTE)
        self.temp_rate_LTE_cell7 = np.zeros(self.num_of_LTE)
        
        self.temp_rate_NR_cell1 = np.zeros (self.num_of_NR)
        self.temp_rate_NR_cell2 = np.zeros (self.num_of_NR)
        self.temp_rate_NR_cell3 = np.zeros (self.num_of_NR)
        self.temp_rate_NR_cell4 = np.zeros (self.num_of_NR)
        self.temp_rate_NR_cell5 = np.zeros (self.num_of_NR)
        self.temp_rate_NR_cell6 = np.zeros (self.num_of_NR)
        self.temp_rate_NR_cell7 = np.zeros (self.num_of_NR)
        
    
        
################## CELL 1 ##########################################
        self.temp_rate_LTE_cell1 = np.zeros(self.num_of_LTE)
        self.temp_rate_NR_cell1 = np.zeros(self.num_of_NR)
      
        if not ratio_matrix [self.ratio, self.time_slot]:    #LTE             
            for i in range (self.num_of_LTE):
                if i in self.LTE_GBR_index and np.sum(self.LTE_RB_assignment_1)<self.n_RB :
                    self.LTE_RB_assignment_1[i]  +=1
                    self.temp_rate_LTE_cell1[i]  = self.LTE_RB_assignment_1[i]*self.BW*log2 (1 +\
                                                 ((self.BSPower * self.LTEChannelGains1[i])/(self.Interference_LTE_1[i]+self.N0)))   
    
                    while (self.temp_rate_LTE_cell1[i] < self.GBR_rate_LTE) and np.sum(self.LTE_RB_assignment_1)<self.n_RB:
                        self.LTE_RB_assignment_1[i]  +=1
                        self.temp_rate_LTE_cell1[i]  = self.LTE_RB_assignment_1[i]*self.BW*log2 (1 +\
                                                     ((self.BSPower * self.LTEChannelGains1[i])/(self.Interference_LTE_1[i]+self.N0)))   
    
            
            for i in range (self.num_of_LTE) :
                if i not in self.LTE_GBR_index and np.sum(self.LTE_RB_assignment_1) < self.n_RB :
                    self.LTE_RB_assignment_1 [i] +=1
                    self.temp_rate_LTE_cell1[i] = self.LTE_RB_assignment_1[i]*self.BW*log2 (1 +\
                                                ((self.BSPower * self.LTEChannelGains1[i])/(self.Interference_LTE_1[i] +self.N0)))  
                        
            while (np.sum(self.LTE_RB_assignment_1) <self.n_RB):
                self.LTE_RB_assignment_1 [np.argmin (self.temp_rate_LTE_cell1)] +=1
                for i in range (self.num_of_LTE) :
                    self.temp_rate_LTE_cell1[i] = self.LTE_RB_assignment_1[i]*self.BW*log2 (1 +\
                                                ((self.BSPower * self.LTEChannelGains1[i])/(self.Interference_LTE_1[i] +self.N0)))   

        if  ratio_matrix [self.ratio, self.time_slot]:    #NR           
            for i in range (self.num_of_NR):
                if i in self.NR_GBR_index and np.sum(self.NR_RB_assignment_1)<self.n_RB :
                    self.NR_RB_assignment_1[i]  +=1
                    self.temp_rate_NR_cell1[i]  = self.NR_RB_assignment_1[i]*self.BW*log2 (1 +\
                                                ((self.BSPower * self.NRChannelGains1[i])/(self.Interference_NR_1[i]+self.N0)))   
    
                    while (self.temp_rate_NR_cell1[i] < self.GBR_rate_NR) and np.sum(self.NR_RB_assignment_1)<self.n_RB:
                        self.NR_RB_assignment_1[i]  +=1
                        self.temp_rate_NR_cell1[i]  = self.NR_RB_assignment_1[i]*self.BW*log2 (1 +\
                                                    ((self.BSPower * self.NRChannelGains1[i])/(self.Interference_NR_1[i]+self.N0)))   
            for i in range (self.num_of_NR) :
                if i not in self.NR_GBR_index and np.sum(self.NR_RB_assignment_1) < self.n_RB :
                    self.NR_RB_assignment_1 [i] +=1
                    self.temp_rate_NR_cell1[i] = self.NR_RB_assignment_1[i]*self.BW*log2 (1 +\
                                               ((self.BSPower * self.NRChannelGains1[i])/(self.Interference_NR_1[i] +self.N0)))   
                        
            while (np.sum(self.NR_RB_assignment_1) <self.n_RB):
                self.NR_RB_assignment_1 [np.argmin (self.temp_rate_NR_cell1)] +=1
                for i in range (self.num_of_NR) :
                    self.temp_rate_NR_cell1[i] = self.NR_RB_assignment_1[i]*self.BW*log2 (1 +\
                                               ((self.BSPower * self.NRChannelGains1[i])/(self.Interference_NR_1[i] +self.N0)))   
                        
################## CELL 2 ##########################################
        self.temp_rate_LTE_cell2 = np.zeros(self.num_of_LTE)
        self.temp_rate_NR_cell2 = np.zeros(self.num_of_NR)
      
        if not ratio_matrix [self.ratio, self.time_slot] :    #LTE             
            for i in range (self.num_of_LTE):
                if i in self.LTE_GBR_index and np.sum(self.LTE_RB_assignment_2)<self.n_RB:
                    self.LTE_RB_assignment_2[i]  +=1
                    self.temp_rate_LTE_cell2[i]  = self.LTE_RB_assignment_2[i]*self.BW*log2 (1 +\
                                                  ((self.BSPower * self.LTEChannelGains2[i])/(self.Interference_LTE_2[i]+self.N0)))   
    
                    while (self.temp_rate_LTE_cell2[i] < self.GBR_rate_LTE) and np.sum(self.LTE_RB_assignment_2)<self.n_RB:
                        self.LTE_RB_assignment_2[i] +=1
                        self.temp_rate_LTE_cell2[i]  = self.LTE_RB_assignment_2[i]*self.BW*log2 (1 +\
                                                     ((self.BSPower * self.LTEChannelGains2[i])/(self.Interference_LTE_2[i]+self.N0)))   
    
            
            for i in range (self.num_of_LTE) :
                if i not in self.LTE_GBR_index and np.sum(self.LTE_RB_assignment_2) < self.n_RB :
                    self.LTE_RB_assignment_2 [i] +=1
                    self.temp_rate_LTE_cell2[i] = self.LTE_RB_assignment_2[i]*self.BW*log2 (1 +\
                                                ((self.BSPower * self.LTEChannelGains2[i])/(self.Interference_LTE_1[i] +self.N0)))   
                        
            while (np.sum(self.LTE_RB_assignment_2) <self.n_RB):
                self.LTE_RB_assignment_2 [np.argmin (self.temp_rate_LTE_cell2)] +=1
                for i in range (self.num_of_LTE) :
                    self.temp_rate_LTE_cell2[i] = self.LTE_RB_assignment_2[i]*self.BW*log2 (1 +\
                                                ((self.BSPower * self.LTEChannelGains2[i])/(self.Interference_LTE_1[i] +self.N0)))   

        if  ratio_matrix [self.ratio, self.time_slot]:    #NR           
            for i in range (self.num_of_NR):
                if i in self.NR_GBR_index and np.sum(self.NR_RB_assignment_2) < self.n_RB:
                    self.NR_RB_assignment_2[i]  +=1
                    self.temp_rate_NR_cell2[i]  = self.NR_RB_assignment_2[i]*self.BW*log2 (1 +\
                                                ((self.BSPower * self.NRChannelGains2[i])/(self.Interference_NR_2[i]+self.N0)))   
    
                    while (self.temp_rate_NR_cell2[i] < self.GBR_rate_NR) and np.sum(self.NR_RB_assignment_2)<self.n_RB:
                        self.NR_RB_assignment_2[i]  +=1
                        self.temp_rate_NR_cell2[i]  = self.NR_RB_assignment_2[i]*self.BW*log2 (1 +\
                                                    ((self.BSPower * self.NRChannelGains2[i])/(self.Interference_NR_2[i]+self.N0)))   
            
            for i in range (self.num_of_NR) :
                if i not in self.NR_GBR_index and np.sum(self.NR_RB_assignment_2) < self.n_RB :
                    self.NR_RB_assignment_2 [i] +=1
                    self.temp_rate_NR_cell2[i] = self.NR_RB_assignment_2[i]*self.BW*log2 (1 +\
                                               ((self.BSPower * self.NRChannelGains2[i])/(self.Interference_NR_2[i] +self.N0)))  
                        
            while (np.sum(self.NR_RB_assignment_2) <self.n_RB):
                self.NR_RB_assignment_2 [np.argmin (self.temp_rate_NR_cell2)] +=1
                for i in range (self.num_of_NR) :
                    self.temp_rate_NR_cell2[i] = self.NR_RB_assignment_2[i]*self.BW*log2 (1 +\
                                               ((self.BSPower * self.NRChannelGains2[i])/(self.Interference_NR_2[i] +self.N0)))  
                        
################## CELL 3 ##########################################
        self.temp_rate_LTE_cell3 = np.zeros(self.num_of_LTE)
        self.temp_rate_NR_cell3 = np.zeros(self.num_of_NR)
      
        if not ratio_matrix [self.ratio, self.time_slot] :    #LTE             
            for i in range (self.num_of_LTE):
                if i in self.LTE_GBR_index and np.sum(self.LTE_RB_assignment_3)<self.n_RB :
                    self.LTE_RB_assignment_3[i]  +=1
                    self.temp_rate_LTE_cell3[i]  = self.LTE_RB_assignment_3[i]*self.BW*log2 (1 +\
                                                 ((self.BSPower * self.LTEChannelGains3[i])/(self.Interference_LTE_3[i]+self.N0)))   
    
                    while (self.temp_rate_LTE_cell3[i] < self.GBR_rate_LTE) and np.sum(self.LTE_RB_assignment_3)<self.n_RB:
                        self.LTE_RB_assignment_3[i]  +=1
                        self.temp_rate_LTE_cell3[i]  = self.LTE_RB_assignment_3[i]*self.BW*log2 (1 +\
                                                     ((self.BSPower * self.LTEChannelGains3[i])/(self.Interference_LTE_3[i]+self.N0)))   
    
            
            for i in range (self.num_of_LTE) :
                if i not in self.LTE_GBR_index and np.sum(self.LTE_RB_assignment_3) < self.n_RB :
                    self.LTE_RB_assignment_3 [i] +=1
                    self.temp_rate_LTE_cell3[i] = self.LTE_RB_assignment_3[i]*self.BW*log2 (1 +\
                                                ((self.BSPower * self.LTEChannelGains3[i])/(self.Interference_LTE_3[i] +self.N0)))  
                        
            while (np.sum(self.LTE_RB_assignment_3) <self.n_RB):
                self.LTE_RB_assignment_3 [np.argmin (self.temp_rate_LTE_cell3)] +=1
                for i in range (self.num_of_LTE) :
                    self.temp_rate_LTE_cell3[i] = self.LTE_RB_assignment_3[i]*self.BW*log2 (1 +\
                                                ((self.BSPower * self.LTEChannelGains3[i])/(self.Interference_LTE_3[i] +self.N0)))   

        if  ratio_matrix [self.ratio, self.time_slot]:    #NR           
            for i in range (self.num_of_NR):
                if i in self.NR_GBR_index and np.sum(self.NR_RB_assignment_3)<self.n_RB:
                    self.NR_RB_assignment_3[i] +=1
                    self.temp_rate_NR_cell3[i]  = self.NR_RB_assignment_3[i]*self.BW*log2 (1 +\
                                                                           ((self.BSPower * self.NRChannelGains3[i])/(self.Interference_NR_3[i]+self.N0)))   
    
                    while (self.temp_rate_NR_cell3[i] < self.GBR_rate_NR) and np.sum(self.NR_RB_assignment_3)<self.n_RB:
                        self.NR_RB_assignment_3[i] +=1
                        self.temp_rate_NR_cell3[i]  = self.NR_RB_assignment_3[i]*self.BW*log2 (1 +\
                                                    ((self.BSPower * self.NRChannelGains3[i])/(self.Interference_NR_3[i]+self.N0)))   
            for i in range (self.num_of_NR) :
                if i not in self.NR_GBR_index and np.sum(self.NR_RB_assignment_3) < self.n_RB :
                    self.NR_RB_assignment_3 [i] +=1
                    self.temp_rate_NR_cell3[i] = self.NR_RB_assignment_3[i]*self.BW*log2 (1 +\
                                                                               ((self.BSPower * self.NRChannelGains3[i])/(self.Interference_NR_3[i] +self.N0)))   
            while (np.sum(self.NR_RB_assignment_3) <self.n_RB):
                self.NR_RB_assignment_3 [np.argmin (self.temp_rate_NR_cell3)] +=1
                for i in range (self.num_of_NR) :
                    self.temp_rate_NR_cell3[i] = self.NR_RB_assignment_3[i]*self.BW*log2 (1 +\
                                                                               ((self.BSPower * self.NRChannelGains3[i])/(self.Interference_NR_3[i] +self.N0)))   
                        
################## CELL 4 ##########################################
        self.temp_rate_LTE_cell4 = np.zeros(self.num_of_LTE)
        self.temp_rate_NR_cell4 = np.zeros(self.num_of_NR)
      
        if not ratio_matrix [self.ratio, self.time_slot]:    #LTE             
            for i in range (self.num_of_LTE):
                if i in self.LTE_GBR_index and np.sum(self.LTE_RB_assignment_4)<self.n_RB:
                    self.LTE_RB_assignment_4[i]  +=1
                    self.temp_rate_LTE_cell4[i]  = self.LTE_RB_assignment_4[i]*self.BW*log2 (1 +\
                                                 ((self.BSPower * self.LTEChannelGains4[i])/(self.Interference_LTE_4[i]+self.N0)))   
    
                    while (self.temp_rate_LTE_cell4[i] < self.GBR_rate_LTE) and np.sum(self.LTE_RB_assignment_4)<self.n_RB:
                        self.LTE_RB_assignment_4[i]  +=1
                        self.temp_rate_LTE_cell4[i]  = self.LTE_RB_assignment_4[i]*self.BW*log2 (1 +\
                                                     ((self.BSPower * self.LTEChannelGains4[i])/(self.Interference_LTE_4[i]+self.N0)))   
    
            
            for i in range (self.num_of_LTE) :
                if i not in self.LTE_GBR_index and np.sum(self.LTE_RB_assignment_4) < self.n_RB :
                    self.LTE_RB_assignment_4 [i] +=1
                    self.temp_rate_LTE_cell4[i]  = self.LTE_RB_assignment_4[i]*self.BW*log2 (1 +\
                                                 ((self.BSPower * self.LTEChannelGains4[i])/(self.Interference_LTE_4[i] +self.N0)))   
                        
            while (np.sum(self.LTE_RB_assignment_4) <self.n_RB):
                self.LTE_RB_assignment_4 [np.argmin (self.temp_rate_LTE_cell4)] +=1
                for i in range (self.num_of_LTE) :
                    self.temp_rate_LTE_cell4[i] = self.LTE_RB_assignment_4[i]*self.BW*log2 (1 +\
                                                ((self.BSPower * self.LTEChannelGains4[i])/(self.Interference_LTE_4[i] +self.N0)))   

        if  ratio_matrix [self.ratio, self.time_slot]:    #NR           
            for i in range (self.num_of_NR):
                if i in self.NR_GBR_index and np.sum(self.NR_RB_assignment_4)<self.n_RB:
                    self.NR_RB_assignment_4[i]  +=1
                    self.temp_rate_NR_cell4[i]  = self.NR_RB_assignment_4[i]*self.BW*log2 (1 +\
                                                ((self.BSPower * self.NRChannelGains4[i])/(self.Interference_NR_4[i]+self.N0)))   
    
                    while (self.temp_rate_NR_cell4[i] < self.GBR_rate_NR) and np.sum(self.NR_RB_assignment_4)<self.n_RB:
                        self.NR_RB_assignment_4[i]  +=1
                        self.temp_rate_NR_cell4[i]  = self.NR_RB_assignment_4[i]*self.BW*log2 (1 +\
                                                    ((self.BSPower * self.NRChannelGains4[i])/(self.Interference_NR_4[i]+self.N0)))   
            for i in range (self.num_of_NR) :
                if i not in self.NR_GBR_index and np.sum(self.NR_RB_assignment_4) < self.n_RB :
                    self.NR_RB_assignment_4 [i] +=1
                    self.temp_rate_NR_cell4[i] = self.NR_RB_assignment_4[i]*self.BW*log2 (1 +\
                                               ((self.BSPower * self.NRChannelGains4[i])/(self.Interference_NR_4[i] +self.N0)))   
                        
            while (np.sum(self.NR_RB_assignment_4) <self.n_RB):
                self.NR_RB_assignment_4 [np.argmin (self.temp_rate_NR_cell4)] +=1
                for i in range (self.num_of_NR) :
                    self.temp_rate_NR_cell4[i] = self.NR_RB_assignment_4[i]*self.BW*log2 (1 +\
                                               ((self.BSPower * self.NRChannelGains4[i])/(self.Interference_NR_4[i] +self.N0)))   
                        
################## CELL 5 ##########################################
        self.temp_rate_LTE_cell5 = np.zeros(self.num_of_LTE)
        self.temp_rate_NR_cell5 = np.zeros(self.num_of_NR)
      
        if not ratio_matrix [self.ratio, self.time_slot]:    #LTE             
            for i in range (self.num_of_LTE):
                if i in self.LTE_GBR_index and np.sum(self.LTE_RB_assignment_5)<self.n_RB:
                    self.LTE_RB_assignment_5[i]  +=1
                    self.temp_rate_LTE_cell5[i]  = self.LTE_RB_assignment_5[i]*self.BW*log2 (1 +\
                                                 ((self.BSPower * self.LTEChannelGains5[i])/(self.Interference_LTE_5[i]+self.N0)))   
    
                    while (self.temp_rate_LTE_cell5[i] < self.GBR_rate_LTE) and np.sum(self.LTE_RB_assignment_5)<self.n_RB:
                        self.LTE_RB_assignment_5[i]  +=1
                        self.temp_rate_LTE_cell5[i]  = self.LTE_RB_assignment_5[i]*self.BW*log2 (1 +\
                                                     ((self.BSPower * self.LTEChannelGains5[i])/(self.Interference_LTE_5[i]+self.N0)))   
    
            
            for i in range (self.num_of_LTE) :
                if i not in self.LTE_GBR_index and np.sum(self.LTE_RB_assignment_5) < self.n_RB :
                    self.LTE_RB_assignment_5 [i] +=1
                    self.temp_rate_LTE_cell5[i] = self.LTE_RB_assignment_5[i]*self.BW*log2 (1 +\
                                                ((self.BSPower * self.LTEChannelGains5[i])/(self.Interference_LTE_5[i] +self.N0)))   
                        
            while (np.sum(self.LTE_RB_assignment_5) <self.n_RB):
                self.LTE_RB_assignment_5 [np.argmin (self.temp_rate_LTE_cell5)] +=1
                for i in range (self.num_of_LTE) :
                    self.temp_rate_LTE_cell5[i] = self.LTE_RB_assignment_5[i]*self.BW*log2 (1 +\
                                                ((self.BSPower * self.LTEChannelGains5[i])/(self.Interference_LTE_5[i] +self.N0)))   

        if  ratio_matrix [self.ratio, self.time_slot]:    #NR           
            for i in range (self.num_of_NR):
                if i in self.NR_GBR_index and np.sum(self.NR_RB_assignment_5)<self.n_RB :
                    self.NR_RB_assignment_5[i]  +=1
                    self.temp_rate_NR_cell5[i]  = self.NR_RB_assignment_5[i]*self.BW*log2 (1 +\
                                                ((self.BSPower * self.NRChannelGains5[i])/(self.Interference_NR_5[i]+self.N0)))   
    
                    while (self.temp_rate_NR_cell5[i] < self.GBR_rate_NR) and np.sum(self.NR_RB_assignment_5)<self.n_RB:
                        self.NR_RB_assignment_5[i]  +=1
                        self.temp_rate_NR_cell5[i]  = self.NR_RB_assignment_5[i]*self.BW*log2 (1 +\
                                                    ((self.BSPower * self.NRChannelGains5[i])/(self.Interference_NR_5[i]+self.N0)))  
                            
            for i in range (self.num_of_NR) :
                if i not in self.NR_GBR_index and np.sum(self.NR_RB_assignment_5) < self.n_RB :
                    self.NR_RB_assignment_5 [i] +=1
                    self.temp_rate_NR_cell5[i] = self.NR_RB_assignment_2[i]*self.BW*log2 (1 +\
                                               ((self.BSPower * self.NRChannelGains5[i])/(self.Interference_NR_5[i] +self.N0)))   
                        
            while (np.sum(self.NR_RB_assignment_5) <self.n_RB):
                self.NR_RB_assignment_5 [np.argmin (self.temp_rate_NR_cell5)] +=1
                for i in range (self.num_of_NR) :
                    self.temp_rate_NR_cell5[i] = self.NR_RB_assignment_5[i]*self.BW*log2 (1 +\
                                               ((self.BSPower * self.NRChannelGains5[i])/(self.Interference_NR_5[i] +self.N0)))   
                        
                        
################## CELL 6 ##########################################
        self.temp_rate_LTE_cell6 = np.zeros(self.num_of_LTE)
        self.temp_rate_NR_cell6 = np.zeros(self.num_of_NR)
      
        if not ratio_matrix [self.ratio, self.time_slot]:    #LTE             
            for i in range (self.num_of_LTE):
                if i in self.LTE_GBR_index and np.sum(self.LTE_RB_assignment_6)<self.n_RB:
                    self.LTE_RB_assignment_6[i]  +=1
                    self.temp_rate_LTE_cell6[i]  = self.LTE_RB_assignment_6[i]*self.BW*log2 (1 +\
                                                 ((self.BSPower * self.LTEChannelGains6[i])/(self.Interference_LTE_6[i]+self.N0)))   
    
                    while (self.temp_rate_LTE_cell6[i] < self.GBR_rate_LTE) and np.sum(self.LTE_RB_assignment_6)<self.n_RB:
                        self.LTE_RB_assignment_6[i]  +=1
                        self.temp_rate_LTE_cell6[i]  = self.LTE_RB_assignment_6[i]*self.BW*log2 (1 +\
                                                     ((self.BSPower * self.LTEChannelGains6[i])/(self.Interference_LTE_6[i]+self.N0)))   
    
            
            for i in range (self.num_of_LTE) :
                if i not in self.LTE_GBR_index and np.sum(self.LTE_RB_assignment_6) < self.n_RB :
                    self.LTE_RB_assignment_6 [i] +=1
                    self.temp_rate_LTE_cell6[i]  = self.LTE_RB_assignment_6[i]*self.BW*log2 (1 +\
                                                 ((self.BSPower * self.LTEChannelGains6[i])/(self.Interference_LTE_6[i] +self.N0)))  
                        
            while (np.sum(self.LTE_RB_assignment_6) <self.n_RB):
                self.LTE_RB_assignment_6 [np.argmin (self.temp_rate_LTE_cell6)] +=1
                for i in range (self.num_of_LTE) :
                    self.temp_rate_LTE_cell6[i] = self.LTE_RB_assignment_6[i]*self.BW*log2 (1 +\
                                                ((self.BSPower * self.LTEChannelGains6[i])/(self.Interference_LTE_6[i] +self.N0)))   

        if  ratio_matrix [self.ratio, self.time_slot]:    #NR           
            for i in range (self.num_of_NR):
                if i in self.NR_GBR_index and np.sum(self.NR_RB_assignment_6)<self.n_RB:
                    self.NR_RB_assignment_6[i]  +=1
                    self.temp_rate_NR_cell6[i]  = self.NR_RB_assignment_6[i]*self.BW*log2 (1 +\
                                                ((self.BSPower * self.NRChannelGains6[i])/(self.Interference_NR_6[i]+self.N0)))   
    
                    while (self.temp_rate_NR_cell6[i] < self.GBR_rate_NR) and np.sum(self.NR_RB_assignment_6)<self.n_RB:
                        self.NR_RB_assignment_6[i]  +=1
                        self.temp_rate_NR_cell6[i]  = self.NR_RB_assignment_6[i]*self.BW*log2 (1 +\
                                                    ((self.BSPower * self.NRChannelGains6[i])/(self.Interference_NR_6[i]+self.N0)))   
                            
            for i in range (self.num_of_NR) :
                if i not in self.NR_GBR_index and np.sum(self.NR_RB_assignment_6) < self.n_RB :
                    self.NR_RB_assignment_6 [i] +=1
                    self.temp_rate_NR_cell6[i]  = self.NR_RB_assignment_6[i]*self.BW*log2 (1 +\
                                                                               ((self.BSPower * self.NRChannelGains6[i])/(self.Interference_NR_6[i] +self.N0)))   
            while (np.sum(self.NR_RB_assignment_6) <self.n_RB):
                self.NR_RB_assignment_6 [np.argmin (self.temp_rate_NR_cell6)] +=1
                for i in range (self.num_of_NR) :
                    self.temp_rate_NR_cell6[i] = self.NR_RB_assignment_6[i]*self.BW*log2 (1 +\
                                               ((self.BSPower * self.NRChannelGains6[i])/(self.Interference_NR_6[i] +self.N0)))   
                        
################## CELL 7 ##########################################
        self.temp_rate_LTE_cell7 = np.zeros(self.num_of_LTE)
        self.temp_rate_NR_cell7 = np.zeros(self.num_of_NR)
      
        if not ratio_matrix [self.ratio, self.time_slot]:    #LTE             
            for i in range (self.num_of_LTE):
                if i in self.LTE_GBR_index and np.sum(self.LTE_RB_assignment_7)<self.n_RB:
                    self.LTE_RB_assignment_7[i]  +=1
                    self.temp_rate_LTE_cell7[i]  = self.LTE_RB_assignment_7[i]*self.BW*log2 (1 +\
                                                 ((self.BSPower * self.LTEChannelGains7[i])/(self.Interference_LTE_7[i]+self.N0)))   
    
                    while (self.temp_rate_LTE_cell7[i] < self.GBR_rate_LTE) and np.sum(self.LTE_RB_assignment_7)<self.n_RB:
                        self.LTE_RB_assignment_7[i]  +=1
                        self.temp_rate_LTE_cell7[i]  = self.LTE_RB_assignment_7[i]*self.BW*log2 (1 +\
                                                     ((self.BSPower * self.LTEChannelGains7[i])/(self.Interference_LTE_7[i]+self.N0)))   
    
            
            for i in range (self.num_of_LTE) :
                if i not in self.LTE_GBR_index  and np.sum(self.LTE_RB_assignment_7)<self.n_RB:
                    self.LTE_RB_assignment_7 [i] +=1
                    self.temp_rate_LTE_cell7[i]  = self.LTE_RB_assignment_7[i]*self.BW*log2 (1 +\
                                                 ((self.BSPower * self.LTEChannelGains7[i])/(self.Interference_LTE_7[i] +self.N0))) 
                        
            while (np.sum(self.LTE_RB_assignment_7) <self.n_RB):
                self.LTE_RB_assignment_7 [np.argmin (self.temp_rate_LTE_cell7)] +=1
                for i in range (self.num_of_LTE) :
                    self.temp_rate_LTE_cell7[i] = self.LTE_RB_assignment_7[i]*self.BW*log2 (1 +\
                                                ((self.BSPower * self.LTEChannelGains7[i])/(self.Interference_LTE_7[i] +self.N0)))   

        if  ratio_matrix [self.ratio, self.time_slot]:    #NR           
            for i in range (self.num_of_NR):
                if i in self.NR_GBR_index and np.sum(self.NR_RB_assignment_7)<self.n_RB:
                    self.NR_RB_assignment_7[i] +=1
                    self.temp_rate_NR_cell7[i]  = self.NR_RB_assignment_7[i]*self.BW*log2 (1 +\
                                                ((self.BSPower * self.NRChannelGains7[i])/(self.Interference_NR_7[i]+self.N0)))   
    
                    while (self.temp_rate_NR_cell7[i] < self.GBR_rate_NR) and np.sum(self.NR_RB_assignment_7)<self.n_RB:
                        self.NR_RB_assignment_7[i]  +=1
                        self.temp_rate_NR_cell7[i]  = self.NR_RB_assignment_7[i]*self.BW*log2 (1 +\
                                                    ((self.BSPower * self.NRChannelGains7[i])/(self.Interference_NR_7[i]+self.N0)))   
                            
            for i in range (self.num_of_NR) :
                if i not in self.NR_GBR_index and np.sum(self.NR_RB_assignment_7) < self.n_RB :
                    self.NR_RB_assignment_7 [i] +=1
                    self.temp_rate_NR_cell7[i]  = self.NR_RB_assignment_7[i]*self.BW*log2 (1 +\
                                                ((self.BSPower * self.NRChannelGains7[i])/(self.Interference_NR_7[i] +self.N0))) 
                        
            while (np.sum(self.NR_RB_assignment_7) <self.n_RB):
                self.NR_RB_assignment_7 [np.argmin (self.temp_rate_NR_cell7)] +=1
                for i in range (self.num_of_LTE) :
                    self.temp_rate_NR_cell7[i] = self.NR_RB_assignment_7[i]*self.BW*log2 (1 +\
                                               ((self.BSPower * self.NRChannelGains7[i])/(self.Interference_NR_7[i] +self.N0)))   
                        
                        
                        
        return self.LTE_RB_assignment_1, self.temp_rate_LTE_cell1, self.NR_RB_assignment_1, self.temp_rate_NR_cell1,\
            self.LTE_RB_assignment_2, self.temp_rate_LTE_cell2, self.NR_RB_assignment_2, self.temp_rate_NR_cell2,\
                self.LTE_RB_assignment_3, self.temp_rate_LTE_cell3, self.NR_RB_assignment_3, self.temp_rate_NR_cell3,\
                    self.LTE_RB_assignment_4, self.temp_rate_LTE_cell4, self.NR_RB_assignment_4, self.temp_rate_NR_cell4,\
                        self.LTE_RB_assignment_5, self.temp_rate_LTE_cell5, self.NR_RB_assignment_5, self.temp_rate_NR_cell5,\
                        self.LTE_RB_assignment_6, self.temp_rate_LTE_cell6, self.NR_RB_assignment_6, self.temp_rate_NR_cell6,\
                        self.LTE_RB_assignment_7, self.temp_rate_LTE_cell7, self.NR_RB_assignment_7, self.temp_rate_NR_cell7
                        
                
                
 

 #### ratio 0.2=1, 0.3=2, 0.4=3, 0.5=4, 0.6=5, 0=0
# label1 = "ratio4/"
# label2 = "t0/"
                    
# num_lte = 10
# num_nr = 10
# timeslot = 8
# test = DSS (10,10,1,9,1,9,4, timeslot)


# num_lte = 20
# num_nr = 20
# timeslot = 8
# test = DSS (20,20,6,14,6,14, 4,timeslot)
  

# num_lte = 30
# num_nr = 30
# timeslot = 8
# test = DSS (30,30,9,21,9,21, 4, timeslot)


# num_lte = 40
# num_nr = 40
# timeslot = 8
# test = DSS (40,40,12,28,12,28, 4, timeslot)


num_lte = 50
num_nr = 50
timeslot =8
test = DSS (50,50,5,45,5,45, 4, timeslot)

  

lch1, lch2, lch3, lch4, lch5, lch6, lch7, nch1, nch2, nch3, nch4, nch5, nch6, nch7,\
    dl1, dl2, dl3, dl4, dl5, dl6, dl7, dn1, dn2, dn3, dn4, dn5, dn6, dn7 = test.DefineUECellWithGains()
    
LTE_RB1, RL1, NR_RB1, RN1, LTE_RB2, RL2, NR_RB2, RN2, LTE_RB3, RL3, NR_RB3, RN3, LTE_RB4, RL4, NR_RB4, RN4, \
    LTE_RB5, RL5, NR_RB5, RN5, LTE_RB6, RL6, NR_RB6, RN6, LTE_RB7, RL7, NR_RB7, RN7 = test.BR_assignment()
    
  
RL_total =sum(RL1 + RL2 + RL3 + RL4 + RL5 + RL6 + RL7)
RN_total = sum(RN1 + RN2 + RN3 + RN4 + RN5 + RN6 + RN7)

BW_LTE_total = np.zeros ([7,num_lte])
BW_NR_total = np.zeros ([7,num_nr])
Rate_LTE_total = np.zeros ([7,num_lte])
Rate_NR_total = np.zeros ([7,num_nr])


for i in range (num_lte):
    BW_LTE_total [0,i] = LTE_RB1 [i]
    BW_LTE_total [1,i] = LTE_RB2 [i]
    BW_LTE_total [2,i] = LTE_RB3 [i]
    BW_LTE_total [3,i] = LTE_RB4 [i]
    BW_LTE_total [4,i] = LTE_RB5 [i]
    BW_LTE_total [5,i] = LTE_RB6 [i]
    BW_LTE_total [6,i] = LTE_RB7 [i]
    
    Rate_LTE_total [0,i] = RL1 [i]
    Rate_LTE_total [1,i] = RL2 [i]
    Rate_LTE_total [2,i] = RL3 [i]
    Rate_LTE_total [3,i] = RL4 [i]
    Rate_LTE_total [4,i] = RL5 [i]
    Rate_LTE_total [5,i] = RL6 [i]
    Rate_LTE_total [6,i] = RL7 [i]
    
for i in range (num_nr):    
    BW_NR_total [0,i] = NR_RB1 [i]
    BW_NR_total [1,i] = NR_RB2 [i]
    BW_NR_total [2,i] = NR_RB3 [i]
    BW_NR_total [3,i] = NR_RB4 [i]
    BW_NR_total [4,i] = NR_RB5 [i]
    BW_NR_total [5,i] = NR_RB6 [i]
    BW_NR_total [6,i] = NR_RB7 [i]
    
    Rate_NR_total [0,i] = RN1 [i]
    Rate_NR_total [1,i] = RN2 [i]
    Rate_NR_total [2,i] = RN3 [i]
    Rate_NR_total [3,i] = RN4 [i]
    Rate_NR_total [4,i] = RN5 [i]
    Rate_NR_total [5,i] = RN6 [i]
    Rate_NR_total [6,i] = RN7 [i]
    


fairness_LTE = 0
fairness_NR = 0


fairness_LTE = (pow(np.sum(Rate_LTE_total),2)) / (7*num_lte*np.sum(pow(Rate_LTE_total,2)))
fairness_NR =  (pow(np.sum(Rate_NR_total ),2)) / (7*num_nr*np.sum(pow(Rate_NR_total, 2 )))

# print ('BW_LTE_total', BW_LTE_total)

# print ('BW_NR_total', BW_NR_total)

print ('Rate_LTE_total', Rate_LTE_total)

print ('Rate_NR_total', RN_total)
     
print ('RL_total', RL_total)

print ('RN_total', RN_total)

print ('fairness_LTE',timeslot,'=', fairness_LTE)

print ('fairness_NR', timeslot,'=',fairness_NR)

# current_dir = os.path.dirname(os.path.realpath(__file__))
# rate_lte_cell1 = os.path.join(current_dir, "model/" +label1 + label2 +'/RL_total.mat')
# scipy.io.savemat(rate_lte_cell1, {'RL_total': RL_total})           

# current_dir = os.path.dirname(os.path.realpath(__file__))
# rate_lte_cell1 = os.path.join(current_dir, "model/" +label1 + label2 +'/RN_total.mat')
# scipy.io.savemat(rate_lte_cell1, {'RN_total': RN_total})           

# current_dir = os.path.dirname(os.path.realpath(__file__))
# rate_lte_cell1 = os.path.join(current_dir, "model/" +label1 + label2 +'/BW_LTE_total.mat')
# scipy.io.savemat(rate_lte_cell1, {'BW_LTE_total': BW_LTE_total})           

# current_dir = os.path.dirname(os.path.realpath(__file__))
# rate_lte_cell1 = os.path.join(current_dir, "model/" +label1 + label2 +'/BW_NR_total.mat')
# scipy.io.savemat(rate_lte_cell1, {'BW_NR_total': BW_NR_total})           

# current_dir = os.path.dirname(os.path.realpath(__file__))
# rate_lte_cell1 = os.path.join(current_dir, "model/" +label1 + label2 +'/Rate_LTE_total.mat')
# scipy.io.savemat(rate_lte_cell1, {'Rate_LTE_total': Rate_LTE_total})           

# current_dir = os.path.dirname(os.path.realpath(__file__))
# rate_lte_cell1 = os.path.join(current_dir, "model/" +label1 + label2 +'/Rate_NR_total.mat')
# scipy.io.savemat(rate_lte_cell1, {'Rate_NR_total': Rate_NR_total})  


# current_dir = os.path.dirname(os.path.realpath(__file__))
# rate_lte_cell1 = os.path.join(current_dir, "model/" +label1 + label2 +'/fairness_LTE.mat')
# scipy.io.savemat(rate_lte_cell1, {'fairness_LTE': fairness_LTE}) 


# current_dir = os.path.dirname(os.path.realpath(__file__))
# rate_lte_cell1 = os.path.join(current_dir, "model/" +label1 + label2 +'/fairness_NR.mat')
# scipy.io.savemat(rate_lte_cell1, {'fairness_NR': fairness_NR})