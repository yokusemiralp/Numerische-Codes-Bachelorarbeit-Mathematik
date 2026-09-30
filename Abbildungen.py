import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches



"""


Dieser Code wurde verwendet, um die Abbildungen 4.1, 4,2, 5.1, und 5.2 zu plotten.



"""

#-----------------------------Verwendete Beispielparameter--------------------------------------

A = -5
B =  5
C = -5
D =  5
rho = 0.5
N1 = 16
N2 = 16
ecken1 = [(A,C),(B,C),(B,D),(A,D)] # Ecken des rechteckigen Gebiets Omega_y'
ecken2 = [(A,C-rho*A),(B,C-rho*B),(B,D-rho*B),(A,D-rho*A)] #Alle 4 Ecken des Paralelogramms


#------------------------------------Abbildung 4.1a-------------------------------------------

fig1,ax1 = plt.subplots(figsize=(4,4))
ax1.add_patch(patches.Polygon(ecken1,closed=True,facecolor="peachpuff",edgecolor="darkorange",linewidth=1.8,alpha=0.5)) 


for x, y in ecken1:
    ax1.plot(x,y,"ko", markersize=4)

ax1.text(A-0.8,D+0.3,r"$(A,D)$")   
ax1.text(B+0.2,D+0.2,r"$(B,D)$")  
ax1.text(A-1.2,C-0.7,r"$(A,C)$")  
ax1.text(B+0.2,C-0.5,r"$(B,C)$")  
ax1.set_xlim(-7,7)
ax1.set_ylim(-7,7)
ax1.set_xlabel(r"$z_1$")
ax1.set_ylabel(r"$z_2$")
ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)
plt.tight_layout()
plt.show()


#----------------------------------- Abbildung 4.1b --------------------------------------

fig2,ax2 = plt.subplots(figsize=(4,4))
ax2.add_patch(patches.Polygon(ecken2,closed=True,facecolor="lightblue",edgecolor="navy",linewidth=1.8,alpha=0.5)) 

for x, y in ecken2:
    ax2.plot(x,y,"ko", markersize=4)

ax2.text(A-2.0,D-rho*A+0.4,r"$(A,D-\rho A)$") 
ax2.text(B+0.2,D-rho*B+0.2,r"$(B,D-\rho B)$") 
ax2.text(A-3.0,C-rho*A-1.0,r"$(A,C-\rho A)$") 
ax2.text(B+0.2,C-rho*B-0.5,r"$(B,C-\rho B)$")  
ax2.set_xlim(-8,8)
ax2.set_ylim(-9,9)
ax2.set_xlabel(r"$z_1$")
ax2.set_ylabel(r"$z_2$")
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)
plt.tight_layout()
plt.show()



#--------------------------------  Abbildung 4.2,5.1 -------------------------------------

z2_min = min(C-rho*A,C-rho*B)
z2_max = max(D-rho*A,D-rho*B)
z1 = np.linspace(A,B,N1+1)
z2 = np.linspace(z2_min,z2_max,N2+1)
Z1,Z2 = np.meshgrid(z1,z2,indexing="ij")

#Parallelogram wird geplottet
fig3,ax3 = plt.subplots(figsize=(4,4))
ax3.add_patch(patches.Polygon(ecken2,closed=True,facecolor="lightblue",edgecolor="navy",linewidth=1.8,alpha=0.5)) 

#Gitterpunkte werden geplottet
for i in range(N1+1):
    for j in range(N2+1):
        if i%2==0 and j%2==0:
            ax3.plot(Z1[i,j],Z2[i,j],"o",color="black",markersize=2)     
        else:
            ax3.plot(Z1[i,j],Z2[i,j],"o",markerfacecolor="black",markeredgecolor="black",markersize=2) #Für Abbildung 5.1b) markerfacecolor="none",markeredgecolor="lightgrey" 
                   
#Gitterlinien zwischen den Gitterpunkten 
for x in z1:
    ax3.plot([x,x],[z2_min,z2_max],color="lightgrey",linewidth=0.5,zorder=0)
for y in z2:
    ax3.plot([A,B],[y,y],color="lightgrey",linewidth=0.5,zorder=0)

ax3.text(A-2.0,D-rho*A+0.4,r"$(A,D-\rho A)$") 
ax3.text(B+0.2,D-rho*B+0.2,r"$(B,D-\rho B)$") 
ax3.text(A-3.0,C-rho*A-1.0,r"$(A,C-\rho A)$") 
ax3.text(B+0.2,C-rho*B-0.5,r"$(B,C-\rho B)$") 
ax3.set_xlim(-8,8)
ax3.set_ylim(-9,9)
ax3.set_xticks([-5,-2.5,0,2.5,5])
ax3.set_yticks([-7.5,-5,-2.5,0,2.5,5,7.5])
ax3.set_xlabel(r"$z_1$")
ax3.set_ylabel(r"$z_2$")
ax3.spines["top"].set_visible(False)
ax3.spines["right"].set_visible(False)
plt.tight_layout()
plt.show()



#---------------------------- Abbildung 5.2 V-Zyklus plot -------------------------------------

fig4, ax4 = plt.subplots(figsize=(4,4))
x2 = [-3,-2,-1,0,1,2,3]
y2 = [3,2,1,0,1,2,3]
ax4.plot(x2,y2,"-ko",linewidth=0.8)
ax4.plot(0,0,"ks")
ax4.set_aspect("equal")
ax4.axis("off")
ax4.text(-3.3,3,r"$\Omega_z^{h_{L}}$",va="center",ha="right",fontsize=13)
ax4.text(-2.3,2,r"$\Omega_z^{h_{L-1}}$",va="center",ha="right",fontsize=13)
ax4.text(-1.3,1,r"$\Omega_z^{h_{L-2}}$",va="center",ha="right",fontsize=13)
ax4.text(0.0,-0.4,r"$\Omega_z^{h_0}$",va="center", ha="center",fontsize=13)
ax4.text(1.3,1,r"$\Omega_z^{h_{L-2}}$",va="center",ha="left",fontsize=13)
ax4.text(2.3,2,r"$\Omega_z^{h_{L-1}}$",va="center",ha="left",fontsize=13)
ax4.text(3.3,3,r"$\Omega_z^{h_{L}}$",va="center",ha="left",fontsize=13)
plt.show()