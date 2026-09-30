import numpy as np 
import matplotlib.pyplot as plt 



"""


Dieser Code beinhaltet die loglog-Fehlerplots in Abbildung 6.1 und 6.2


"""



N = np.array([32,64,128,256,512])

rel_fehler1 = [6.58e-04,1.20e-04,2.83e-05,5.90e-06,9.06e-07]  # (100,90)
rel_fehler2 = [7.51e-03,1.83e-03,4.56e-04,1.14e-04,2.85e-05]  # (100,100)
rel_fehler3 = [1.35e-02,2.99e-03,6.20e-04,1.34e-04,5.18e-05]  # (100,110)
rel_fehler4 = [8.50e-03,3.73e-03,9.16e-04,2.22e-04,5.20e-05]  # (90,100)
rel_fehler5 = [5.68e-04,8.85e-05,2.32e-06,4.78e-06,2.48e-06]  # (110,100)

fig,ax=plt.subplots()
ax.loglog(N,rel_fehler1,marker="o",label=r"$(S_1,S_2)=(100,90)$")
ax.loglog(N,rel_fehler2,marker="x",label=r"$(S_1,S_2)=(100,100)$")
ax.loglog(N,rel_fehler3,marker="v",label=r"$(S_1,S_2)=(100,110)$")
ax.loglog(N,rel_fehler4,marker="*",label=r"$(S_1,S_2)=(90,100)$")
ax.loglog(N,rel_fehler5,marker="s",label=r"$(S_1,S_2)=(110,100)$")
ax.set_xlabel(r"$N_1=N_2$")
ax.set_ylabel("relativer Fehler")
ax.set_xscale("log",base=2)
ax.set_xticks(N)
ax.set_xticklabels([32,64,128,256,512])
ax.plot(N,2*N**-2.0,"k--",label=r"$\mathcal{O}(N_1^{-2})$") #Referenzgerade
ax.legend()
plt.tight_layout()
plt.show()

#beginnt mit N_tau = 8

# rho = 0.9, Start N_tau = 8, Reihenfolge: N = 32, 64, 128, 256, 512 


T2rel_fehler1 = [5.95e-03,2.42e-03,5.98e-04,1.49e-04,3.73e-05]  # (100,90)
T2rel_fehler2 = [1.48e-01,2.23e-02,4.34e-03,1.07e-03,2.66e-04]  # (100,100)
T2rel_fehler3 = [1.15e-01,4.95e-02,1.29e-02,3.11e-03,7.70e-04]  # (100,110)
T2rel_fehler4 = [1.65e-01,8.46e-02,2.26e-02,5.56e-03,1.39e-03]  # (90,100)
T2rel_fehler5 = [5.34e-03,2.31e-03,6.18e-04,1.52e-04,3.79e-05] # (110,100)



T2fig,T2ax=plt.subplots()
T2ax.loglog(N,T2rel_fehler1,marker="o",label=r"$(S_1,S_2) = (100,90)$")
T2ax.loglog(N,T2rel_fehler2,marker="x",label=r"$(S_1,S_2) = (100,100)$")
T2ax.loglog(N,T2rel_fehler3,marker="v",label=r"$(S_1,S_2) = (100,110)$")
T2ax.loglog(N,T2rel_fehler4,marker="*",label=r"$(S_1,S_2) = (90,100)$")
T2ax.loglog(N,T2rel_fehler5,marker="s",label=r"$(S_1,S_2) = (110,100)$")
T2ax.set_xlabel(r"$N_1=N_2$")
T2ax.set_ylabel("relativer Fehler")
T2ax.set_xscale("log",base=2)
T2ax.set_xticks(N)
T2ax.plot(N,20*N**-2.0,"k--",label=r"$\mathcal{O}(N_1^{-2})$")  #Referenzgerade
T2ax.legend(loc="upper right")
T2ax.set_xticklabels([32,64,128,256,512])
plt.tight_layout()
plt.show()