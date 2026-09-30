import numpy as np
from scipy.stats import norm
import time
from scipy.sparse import lil_matrix
from scipy.sparse import eye  
from scipy.sparse.linalg import spsolve 

def Gitter(A,B,C,D,N_1,N_2,rho):
    """
    Aufbau des Rechengitters nach Abschnitt 4.2
    Erzeugt das kartesische Hintergrundgitter G^h auf dem Rechteck [A, B]x[z2_min, z2_max], das das Gebiet Omega_z^h umschließt.
    Alle Gitterpunkte in Omega_z^h, sowie alle Unbekannten Punkte (ohne Randpunkte) werden durch eine Bool-Abfrage mit TRUE markiert. 
    """
    z2_min = min(C-rho*A,C-rho*B)
    z2_max = max(D-rho*A,D-rho*B)
    z1 = np.linspace(A,B,N_1+1)        
    z2 = np.linspace(z2_min,z2_max,N_2+1)
    dz1 = (B-A)/N_1            #Schrittweiten
    dz2 = (z2_max-z2_min)/N_2
    Z1,Z2 = np.meshgrid(z1,z2,indexing="ij")  #Gitter

    Punkte_innerhalb = np.zeros((N_1+1,N_2+1),dtype=bool)  #alle Punkte im Rechengitter (TRUE), inklusive auch Punkte auf dem Rand 
    for i in range(N_1+1):
        for j in range(N_2+1):
            if (Z1[i,j]>=A and Z1[i,j]<=B and Z2[i,j]>=C-rho*Z1[i,j] and Z2[i,j]<=D-rho*Z1[i,j]):
             Punkte_innerhalb[i,j] = True 

    tol=1e-12
    Punkte_unbekannt = np.zeros((N_1+1,N_2+1),dtype=bool)   #Punkte auf dem Rand sind hier FALSE 
    for i in range(N_1+1):
        for j in range(N_2+1):
            if (Punkte_innerhalb[i,j] and (Z1[i,j]>A+tol and Z1[i,j]<B-tol and Z2[i,j]>C-rho*Z1[i,j]+tol and Z2[i,j]<D-rho*Z1[i,j]-tol)):
             Punkte_unbekannt[i,j] = True       
    return(z1,z2,dz1,dz2,Z1,Z2,Punkte_innerhalb,Punkte_unbekannt)


def k_idx(Punkte_unbekannt):
    """
    Nummeriert die Unbekannten Gitterpunkte lexikografisch und zählt diese.
    """
    idx = -np.ones(Punkte_unbekannt.shape,dtype=int)
    gesamt = 0 
    for j in range(Punkte_unbekannt.shape[1]):
        for i in range(Punkte_unbekannt.shape[0]):
            if Punkte_unbekannt[i,j] == True: 
                idx[i,j]=gesamt
                gesamt+=1
    return idx,gesamt


def trafo_a1_a2_b(r,q1,q2,sigma1,sigma2,rho):  
    """
    Bestimmung der Transformationsparameter a_1,a_2 und b aus Abschnitt 3.2 
    """
    M = np.array([[sigma1**2,rho*sigma1*sigma2],[rho*sigma1*sigma2,sigma2**2]],dtype=float)
    rhs = np.array([0.5*sigma1**2+q1-r,0.5*sigma2**2+q2-r],dtype=float)
    a1,a2 = np.linalg.solve(M,rhs)
    b = (r-q1-0.5*sigma1**2)*a1+(r-q2-0.5*sigma2**2)*a2+0.5*sigma1**2*a1**2+0.5*sigma2**2*a2**2+rho*sigma1*sigma2*a1*a2-r
    return a1,a2,b


def payoff_call(S1,S2,tau,K,r,q_1,q_2):
    return np.maximum(S1*np.exp(-q_1*tau) - S2*np.exp(-q_2*tau) - K*np.exp(-r*tau), 0.0)


def bilde_L_SW(z1,z2,dz1,dz2,Punkte_unbekannt,idx,A,B,C,D,rho):
    """
    Aufbau der diskreten Shortley-Weller-Operatormatrix L_h^{SW} mit scipy.sparse lil_matrix. Speichert alle nichtnull Einträge zur Optimierung.
    L_h^{SW} wird mit einer rho Fallunterscheidung aufgebaut, da je nach Gebietsform beispielsweise die unteren/oberen Randnahen Punkte nicht nur 
    fehlende Nachbarn nach unten/oben sondern auch nach links/rechts haben können.
    Die Funktion geht jeweils durch jeden Gitterpunkt (i,j), berechnet die zugehörigen Shortley-Weller Abstände zu den Nachbarn,
    berechnet die Gewichte und bildet anschließend die ensprechende Zeile von L_H^{SW}.
    Ränder speichert für den Randvektor eine Liste an Tupeln (k,alpha,z1_rand,z2_rand) mit Gitterindex k, Gewicht alpha und den exakten Koordinaten des Randpunktes.
    Die Gewichte alpha_L, alpha_R, alpha_U, alpha_O werden für den Linien-GS-Glätter pro Gitterpunkt zurückgegeben.
    """
    n = np.max(idx)+1 #Anzahl an innere Unbekannte 
    L_h = lil_matrix((n,n))
    Ränder = []

    a_L = np.zeros(Punkte_unbekannt.shape) 
    a_R = np.zeros(Punkte_unbekannt.shape)
    a_U = np.zeros(Punkte_unbekannt.shape)
    a_O = np.zeros(Punkte_unbekannt.shape)

    for i in range(1,len(z1)-1):    
        for j in range(1,len(z2)-1):  
            if Punkte_unbekannt[i,j]==False: 
                continue              
            k = idx[i,j]              

            if rho > 0:                                   
                if Punkte_unbekannt[i-1,j]==True:         #Falls Linker Nachbar (i-1,j) von (i,j) eine Unbekannte ist: Abstand 1 
                    s_L = 1.0                               
                else:
                    D_L=(z1[i]-A)/dz1                     #Falls (i-1,j) außerhalb \Omega_z, rechne Abstand zum Rand 
                    D_U=(z2[j]-C+rho*z1[i])/(rho*dz1)     #D_U ist der Shortley-Weller Abstand zur unteren schrägen Kante zum linken Nachbarn außerhalb \Omega_z
                    if D_L <= D_U:                        
                        s_L = D_L
                    else:
                        s_L = D_U
                if Punkte_unbekannt[i+1,j]==True:
                    s_R=1.0
                else:
                    D_R=(B-z1[i])/dz1
                    D_O=(D-rho*z1[i]-z2[j])/(rho*dz1) 
                    if D_R<=D_O:
                        s_R = D_R
                    else:
                        s_R = D_O
            elif rho<0:
                if Punkte_unbekannt[i-1,j]==True:
                    s_L = 1.0
                else:
                    D_L=(z1[i]-A)/dz1
                    D_O=(D-rho*z1[i]-z2[j])/(-rho*dz1)
                    if D_L <= D_O:
                        s_L = D_L
                    else:
                        s_L = D_O
                if Punkte_unbekannt[i+1,j]==True:
                    s_R=1.0
                else:
                    D_R=(B-z1[i])/dz1
                    D_U=(z2[j]-C+rho*z1[i])/(-rho*dz1) 
                    if D_R<=D_U:
                        s_R = D_R
                    else:
                        s_R = D_U
            elif rho == 0:
                if Punkte_unbekannt[i-1,j]==True:
                    s_L = 1.0
                else:
                    s_L=(z1[i]-A)/dz1
                if Punkte_unbekannt[i+1,j] == True: 
                        s_R = 1.0 
                else:
                        s_R = (B-z1[i])/dz1
            if Punkte_unbekannt[i,j-1]==True:
                    s_U=1.0
            else:
                    s_U = (z2[j]-(C-rho*z1[i]))/dz2
            if Punkte_unbekannt[i,j+1]==True:
                s_O = 1.0
            else:
                s_O = ((D-rho*z1[i])-z2[j])/dz2


            alpha_L = 1/(dz1**2*s_L*(s_L+s_R))
            alpha_R = 1/(dz1**2*s_R*(s_L+s_R))
            alpha_O = (1-rho**2)/((dz2**2)*s_O*(s_U+s_O))
            alpha_U = ((1-rho**2))/((dz2**2)*s_U*(s_U+s_O))

            a_L[i,j] = alpha_L
            a_R[i,j] = alpha_R
            a_U[i,j] = alpha_U
            a_O[i,j] = alpha_O

            L_h[k,k] += -(alpha_L+alpha_R+alpha_O+alpha_U) #Diagonale von L_h
           
            #Linker Rand
            if Punkte_unbekannt[i-1,j]==True: 
                L_h[k,idx[i-1,j]] = alpha_L
            else:
                # Falls außerhalb, exakte Koordinate berechnen: Wir gehen von z1[i] um s_L nach links
                z1_rand = z1[i]-s_L*dz1
                z2_rand = z2[j]
                Ränder.append((k,alpha_L,z1_rand,z2_rand))
                
            #Rechter Rand
            if Punkte_unbekannt[i+1,j]==True:
                L_h[k,idx[i+1,j]]=alpha_R
            else:
                z1_rand = z1[i]+s_R*dz1
                z2_rand = z2[j]
                Ränder.append((k,alpha_R,z1_rand,z2_rand))
                
            #Unterer Rand
            if Punkte_unbekannt[i,j-1]==True:
                L_h[k,idx[i,j-1]]=alpha_U
            else:
                z1_rand = z1[i]
                z2_rand = z2[j]-s_U*dz2
                Ränder.append((k,alpha_U,z1_rand,z2_rand))
                
            #Oberer Rand
            if Punkte_unbekannt[i,j+1]==True:
                L_h[k,idx[i,j+1]] = alpha_O
            else:
                z1_rand = z1[i]
                z2_rand = z2[j]+s_O*dz2
                Ränder.append((k,alpha_O,z1_rand,z2_rand))

    return L_h.tocsr(),Ränder,a_L,a_R,a_U,a_O  


def CN_Matrix(L_h,dtau):
    n = L_h.shape[0]
    I = eye(n,format="csr")
    A_h = I-0.5*dtau*L_h
    B_h = I+0.5*dtau*L_h
    return A_h.tocsr(),B_h.tocsr()



def Vzuw(V,z1,z2,tau,sigma_1,sigma_2,rho,a_1,a_2,b): #Transformationsfunktion 
    return V*np.exp(-a_1*sigma_1*z1-a_2*sigma_2*(z2+rho*z1)-b*tau)

def Randvektor_g(Ränder,n,tau,S1_0,S2_0,sigma_1,sigma_2,rho,a_1,a_2,b_param,K,r,q_1,q_2):
    vektor_g = np.zeros(n)

    for k,alpha,z1_rand,z2_rand in Ränder:
        S1_rand = S1_0*np.exp(sigma_1*z1_rand)                 #Kurse am Rand 
        S2_rand = S2_0*np.exp(sigma_2*(z2_rand+rho*z1_rand))
        V = payoff_call(S1_rand,S2_rand,tau,K,r,q_1,q_2)      #Approximative Dirichlet-Randbedingungen                
        vektor_g[k] += alpha*Vzuw(V,z1_rand,z2_rand,tau,sigma_1,sigma_2,rho,a_1,a_2,b_param)
    return vektor_g

def start_w0(z1,z2,Punkte_unbekannt,idx,S1_0,S2_0,sigma_1,sigma_2,rho,a_1,a_2,K): 
    """
    Aufbau des Startvektors w^0 für die CN Zeitschleife.
    Für jede Unbekannte wird der zugehörige Eintrag von w^0 punktweise aus 
    der Anfangsbedingung in den transformierten Koordinaten zur Fälligkeit tau = 0 bestimmt. 
    """
    n = np.max(idx)+1
    w0 = np.zeros(n)  
    for i in range(len(z1)):     
        for j in range(len(z2)):
            if Punkte_unbekannt[i,j]==False:   
                continue
            if Punkte_unbekannt[i,j]==True:
                X1 = sigma_1*z1[i]
                X2 = sigma_2*(z2[j]+rho*z1[i])
                S1 = S1_0*np.exp(X1)
                S2 = S2_0*np.exp(X2)
                payoff = max(S1-S2-K,0.0)
                k = idx[i,j]
                w0[k] = payoff*np.exp(-a_1*X1-a_2*X2) 
    return w0


def Thomas_Algo(l,d,r,g):
    """
    Löst ein Tridiagonalsystem aus dem Linienglätter mit dem Thomas-Algorithmus. Siehe Algorithmus 5.1
    l: untere Nebendiagonale 
    d: diagonale 
    r: obere Nebendiagonale 
    g: rechte Seite
    """
    n = len(d)
    r_stern = np.zeros(n-1,dtype=float)
    g_stern = np.zeros(n,dtype = float)
    x_lösung = np.zeros(n,dtype=float)

    r_stern[0] = r[0]/d[0] 
    g_stern[0]= g[0]/d[0]

    for i in range(1,len(r)):
        nenner = (d[i] - r_stern[i-1]*l[i-1])
        r_stern[i] = r[i]/nenner
        g_stern[i] = (g[i]-g_stern[i-1]*l[i-1])/nenner

    letztes_n = len(g_stern) - 1 

    g_stern[letztes_n] = (g[letztes_n]-g_stern[letztes_n-1]*l[letztes_n-1])/(d[letztes_n]-r_stern[letztes_n-1]*l[letztes_n-1])
    x_lösung[n-1] = g_stern[n-1]

    for i in range(n-2,-1,-1):
        x_lösung[i] = g_stern[i] - r_stern[i]*x_lösung[i+1]
    return x_lösung

def hflinien(idx,Punkte_unbekannt):
    """
    Bereitet die z1-Linien für den Linien-Gauß-Seidel vor.
    Für jede feste Gitterlinie j werden für alle Unbekannten die Indizes (i,j,k) gesammelt.
    """
    Nx,Ny = Punkte_unbekannt.shape 
    z1_linien=[]
    for j in range(Ny): 
        xlinie=[(i,j,idx[i,j]) for i in range(Nx) if Punkte_unbekannt[i,j]==True]
        if len(xlinie)>0:                                                         
            z1_linien.append(xlinie)                                               
    return z1_linien                    

def z1LinienGS(Level,w,rhs):
    """
    Die z1-Linien werden von unten nach oben durchlaufen. 
    Für jede Linie wird das Tridiagonalsystem (l,d,r,g) nach Abschnitt 5.3.2 Aufgebaut 
    und mit dem Thomas-Algorithmus gelöst.
    """
    Punkte_unbekannt = Level["Punkte_unbekannt"]
    idx = Level["idx"]
    a_L = Level["a_L"]
    a_R = Level["a_R"]
    a_U = Level["a_U"]
    a_O = Level["a_O"]
    dtau = Level["dtau"]
    z1_linien = Level["z1_linien"]

    for linie in z1_linien:                   
        n = len(linie)
        idx_k = [k for (i,j,k) in linie]   
        l = np.zeros(n-1)
        d = np.zeros(n)
        r = np.zeros(n-1)
        g = np.zeros(n)

        for p, (i,j,k) in enumerate(linie):
            d[p] = 1+0.5*dtau*(a_L[i,j]+a_R[i,j]+a_O[i,j]+a_U[i,j])
            if p>0:
                l[p-1] = -0.5*dtau*a_L[i,j]    
            if p<n-1:
                r[p] = -0.5*dtau*a_R[i,j] 
            g[p] = rhs[k]       
            if Punkte_unbekannt[i,j-1]==True:
                g[p] += 0.5*dtau*a_U[i,j]*w[idx[i,j-1]]    #bereits aktualisiert 
            if Punkte_unbekannt[i,j+1]==True:
                g[p] += 0.5*dtau*a_O[i,j]*w[idx[i,j+1]]    #aus dem vorherigen Schritt
            
        if n == 1: #für n=1 löse direkt
            w[idx_k] = g/d
        else:
            w[idx_k] =Thomas_Algo(l,d,r,g)
    return w

def Prolong_matrix(idx_H,idx_h,Punkte_unbekannt_h,Punkte_unbekannt_H):
    """
    Die bilineare Prolongation wird als lil_matrix aus Effizienzgründen gebaut. 
    Die Restriktion ergibt sich aus R = 1/4*P. 
    """
    n_h = np.max(idx_h)+1 
    n_H = np.max(idx_H)+1
    N1_h,N2_h = Punkte_unbekannt_h.shape
    N1_H,N2_H = Punkte_unbekannt_H.shape
    P = lil_matrix((n_h,n_H))

    for i in range(N1_h):
        for j in range(N2_h):
            if Punkte_unbekannt_h[i,j]==False:
                continue
            k_h = idx_h[i,j]
            I=i//2
            J=j//2
            if i%2==0 and j%2==0:
                beitrag = [(I,J,1)]
            elif i%2==1 and j%2==0:
                beitrag = [(I,J,0.5),(I+1,J,0.5)]
            elif i%2==0 and j%2==1:
                beitrag = [(I,J,0.5),(I,J+1,0.5)]
            elif i%2==1 and j%2==1:
                beitrag = [(I,J,0.25),(I+1,J,0.25),(I,J+1,0.25),(I+1,J+1,0.25)]
            
            for Ix,Jx,wert in beitrag:
                if 0<=Ix<N1_H and 0<=Jx<N2_H and Punkte_unbekannt_H[Ix,Jx]==True:
                    P[k_h,idx_H[Ix,Jx]] = wert
    return P.tocsr()


def GS_sparse(A_h,w,b):
    """
    Aufbau des punktweisen Gauß-Seidel-Glätters.
    Arbeitet mit der CSR-Struktur der Systemmatrix A_h.
    -data speichert alle nichtnull Werte der Matrix 
    -indices speichert zu jedem Wert die Spalte j
    -indptr speichert wo jede Zeile in data beginnt 
    """
    eintrag = A_h.data     
    indize_j = A_h.indices
    grenze = A_h.indptr 
    n = len(b)
    for i in range(n):
        summe = 0.0
        diag = 0.0
        for k in range(grenze[i],grenze[i+1]): 
            j = indize_j[k]
            if j==i:
                diag = eintrag[k]
            else:
                summe += eintrag[k]*w[j]
        w[i] = (b[i]-summe)/diag
    return w


def Gitterlevel(A,B,C,D,N_1,N_2,rho,dtau): 
    """
    Aufbau der Gitterhierarchie. 
    Für jedes Level werden alle benötigten Größen in ein Dictionary gespeichert.
    Level[0]: feinstes Gitter 
    Level[-1]: gröbste Gitter
    """
    Level = []
    while N_1>=4 and N_2>=4:
        z1,z2,dz1,dz2,Z1,Z2,Punkte_innerhalb,Punkte_unbekannt = Gitter(A,B,C,D,N_1,N_2,rho)
        idx,counter = k_idx(Punkte_unbekannt)
        L_h,Ränder,a_L,a_R,a_U,a_O = bilde_L_SW(z1,z2,dz1,dz2,Punkte_unbekannt,idx,A,B,C,D,rho)
        A_h,B_h = CN_Matrix(L_h,dtau)
        z1_linien = hflinien(idx,Punkte_unbekannt)
        Level.append({"N_1":N_1,"N_2":N_2,"z1":z1,"z2":z2,"dz1":dz1,"dz2":dz2,"Punkte_unbekannt":Punkte_unbekannt,"idx":idx,"L_h":L_h,"A_h":A_h,"B_h":B_h,"Ränder":Ränder,"a_L":a_L,"a_R":a_R,"a_U":a_U,"a_O":a_O,"dtau":dtau,"z1_linien":z1_linien}) 
        N_1//=2
        N_2//=2
    
    for l in range(len(Level)-1):
        P=Prolong_matrix(Level[l+1]["idx"],Level[l]["idx"],Level[l]["Punkte_unbekannt"],Level[l+1]["Punkte_unbekannt"])
        Level[l]["P"] = P
        Level[l]["R"] = (0.25*P.T).tocsr()
    return Level


def V_Zyklus(Level,l,w,b,nu1,nu2): 
    """
    V-Zyklus mit: 
    nu_1 Vorglättungsschritten 
    Grobgitterkorrektur mit rekursivem V-Zyklus aufruf
    nu_2 Nachglättungsschritten 
    Punktweise GS wurde auskommentiert als Glätter und kann bei Bedarf getestet werden. 
    """
    A_h = Level[l]["A_h"]

    if l==len(Level)-1:       #exakt Lösen auf dem gröbsten Gitter
        return spsolve(A_h,b)
        
    idx_h = Level[l]["idx"]
    idx_H = Level[l+1]["idx"]
    Punkte_unbekannt_h = Level[l]["Punkte_unbekannt"]
    Punkte_unbekannt_H = Level[l+1]["Punkte_unbekannt"]

    # nu_1 Vorglättungsschritte
    for _ in range(nu1):     
       w = z1LinienGS(Level[l],w,b) 
       #w = GS_sparse(A_h,w,b)

    #Grobgitterkorrektur  
    r_h = b-A_h@w 
    r_H = Level[l]["R"]@r_h
    e_H_start = np.zeros_like(r_H) #Startvektor für den Fehler

    e_H = V_Zyklus(Level,l+1,e_H_start,r_H,nu1=nu1,nu2=nu2)
    e_h = Level[l]["P"]@e_H
    w = w+e_h 

    # nu_2 Nachglättungsschritte
    for _ in range(nu2):
        w = z1LinienGS(Level[l],w,b)
        #w = GS_sparse(A_h,w,b)
    return w



def CN_Zeitschleife_MG(w0,Level,N_tau,dtau,A,B,C,D,rho,S1_0,S2_0,sigma_1,sigma_2,a_1,a_2,b_param,K,r,q_1,q_2,k_max,nu1,nu2,eps):
    """
    V-Zyklus eingebettet in der CN Zeitschleife.
    Löst in jedem Zeitschritt das LGS A_h*w^n+1 = B_h + 0.5*dtau(g^n + g^n+1) iterativ mit dem V-Zyklus,
    bis die Norm des Residums die Toleranz eps relativ zum Startresiduum unterschreitet und zählt die benötigten V-Zyklen pro Zeitschritt.
    """
    A_h = Level[0]["A_h"]
    B_h = Level[0]["B_h"]
    Ränder = Level[0]["Ränder"]
    idx = Level[0]["idx"]
    n_unbekannte = np.max(idx)+1
    
    Zykl_ges = 0
    w = w0.copy()
    for n in range(N_tau):  
        tau_alt = n*dtau     #Zeitpunkte
        tau_neu = (n+1)*dtau
        
        # Randvektoren direkt mit exakten Koordinaten berechnen
        g_alt = Randvektor_g(Ränder,n_unbekannte,tau_alt,S1_0,S2_0,sigma_1,sigma_2,rho,a_1,a_2,b_param,K,r,q_1,q_2)
        g_neu = Randvektor_g(Ränder,n_unbekannte,tau_neu,S1_0,S2_0,sigma_1,sigma_2,rho,a_1,a_2,b_param,K,r,q_1,q_2)
        
        rhs = B_h@w + 0.5*dtau*(g_alt+g_neu) 
        w_neu = w.copy()

        #Abbruchkriterium aufbauen 
        r0 = np.linalg.norm(rhs-A_h@w_neu)  #Startresiduum
        for _ in range(k_max): 
            w_neu=V_Zyklus(Level,0,w_neu,rhs,nu1,nu2)
            Zykl_ges += 1
            if np.linalg.norm(rhs-A_h@w_neu)/r0 <= eps:
                break
        w = w_neu
    Zykl_schnitt = Zykl_ges/N_tau
    return w, Zykl_schnitt


def Optionspreis(w_T,Level,T,b):
    """
    Ließt den Optionspreis zum heutigen Zeitpunkt t=0 aus der Lösung w_T am Gitterpunkt z = (0,0) ab. 
    """
    fein = Level[0]
    z1 = fein["z1"]
    z2 = fein["z2"]
    idx = fein["idx"]
    i0 = np.argmin(np.abs(z1))
    j0 = np.argmin(np.abs(z2))
    V0 = np.exp(b*T)*w_T[idx[i0,j0]]
    return V0

def margrabe_preis(S1_0,S2_0,sigma_1,sigma_2,rho,T):
    sigma = np.sqrt(sigma_1**2 + sigma_2**2 - 2*rho*sigma_1*sigma_2)
    d1 = (np.log(S1_0/S2_0) + (0.5*sigma**2)*T)/(sigma*np.sqrt(T))
    d2 = d1-sigma*np.sqrt(T)
    V0_exakt = S1_0*norm.cdf(d1) - S2_0*norm.cdf(d2)
    return V0_exakt



#------------------------------------------Testcodes-------------------------------------------------



def Benchmark_test(A,B,C,D,sigma_1,sigma_2,rho,r,q_1,q_2,K,T,N_liste,N_tau,k_max,nu1,nu2,eps):
    """
    Testet die im BENCHOP gegebenen Basiswerte und rechnet die relativen Fehler aus.
    Gibt den numerisch berechneten Optionspreis, den exakten Margrabe-Preis, den relativen Fehler,
    die Anzahl der V-Zyklen pro Zeitschritt und die benötigte Rechenzeit aus. 
    N_tau wird bei jeder Gitterverfeinerung mitverdoppelt.
    """
    Basiswerte = [(100,90),(100,100),(100,110),(90,100),(110,100)]
    #Basiswerte = [(100,100)] #isolierter Test 
    a_1,a_2,b_param = trafo_a1_a2_b(r,q_1,q_2,sigma_1,sigma_2,rho)
    
    for N in N_liste:
        dtau = T/N_tau
        Level = Gitterlevel(A,B,C,D,N,N,rho,dtau)
        fein = Level[0]
        print(f"N = {N},N_tau={N_tau}")
        
        for S1_0,S2_0 in Basiswerte:
            t0 = time.time()
            w0 = start_w0(fein["z1"],fein["z2"],fein["Punkte_unbekannt"],fein["idx"],S1_0,S2_0,sigma_1,sigma_2,rho,a_1,a_2,K)
            w_T,Zykl_schnitt = CN_Zeitschleife_MG(w0,Level,N_tau,dtau,A,B,C,D,rho,S1_0,S2_0,sigma_1,sigma_2,a_1,a_2,b_param,K,r,q_1,q_2,k_max,nu1,nu2,eps)
            V0 = Optionspreis(w_T,Level,T,b_param)
            t1 = time.time()

            V0_exakt = margrabe_preis(S1_0,S2_0,sigma_1,sigma_2,rho,T)
            rel = abs(V0-V0_exakt)/abs(V0_exakt)
            
            print(f"({S1_0:5.0f},{S2_0:5.0f}) | Multigrid Optionspreis = {V0:9.6f} | Margrabe Preis = {V0_exakt:9.6f} | relativer Fehler = {rel:.2e} | Zyklen/Schnitt = {Zykl_schnitt} |Zeit ={t1-t0:.1f}s")
        N_tau *= 2
    return


#Test der Kovnvergenzrate 
def teste_konvergenzrate(Level,n_zyklus,nu1,nu2): 
    """
    Rechnet die durchschnittlichen Residuenreduktionsfaktoren in jedem V-Zyklus aus  
    """
    A_h=Level[0]["A_h"]
    n = A_h.shape[0]
    b = np.zeros(n)
    w = np.random.default_rng(0).uniform(-1,1,n)

    d=[]
    d_vektor = b-A_h@w 
    d_norm = np.linalg.norm(d_vektor)
    d.append(d_norm)
    for m in range(1,n_zyklus+1):
        w = V_Zyklus(Level,0,w,b,nu1,nu2)
        d.append(np.linalg.norm(b-A_h@w))

    k0 = 3
    q_dach = (d[-1]/d[k0])**(1/(n_zyklus-k0))
    return q_dach

def h_unabh(A,B,C,D,rho,N_tau,T,N_liste,n_zyklus,nu1,nu2):
    dtau = T/N_tau
    for N in N_liste:
        Level = Gitterlevel(A,B,C,D,N,N,rho,dtau)
        q_dach = teste_konvergenzrate(Level,n_zyklus,nu1,nu2)
        print(f"N_1=N_2={N}: q_dach = {q_dach:.3f}")





#BENCHOP Test mit rho = 0.5 
Test_1 = Benchmark_test(A=-5,B=5,C=-5,D=5,sigma_1=0.15,sigma_2=0.15,rho=0.5,r=0.03,q_1=0.0,q_2=0.0,K=0.0,T=1.0,N_liste=(32,64,128,256,512),N_tau=8,k_max=50,nu1=1,nu2=1,eps=1e-5)
print(Test_1)



#Erweiterter Test mit rho = 0.9
Test_2 = Benchmark_test(A=-5,B=5,C=-5,D=5,sigma_1=0.15,sigma_2=0.15,rho=0.9,r=0.03,q_1=0.0,q_2=0.0,K=0.0,T=1.0,N_liste=(32,64,128,256,512),N_tau=8,k_max=50,nu1=1,nu2=1,eps=1e-5)
print(Test_2)



for rho in (0.0,0.5,0.9,0.99):      #durchschnittliche Residuenreduktionsfaktoren im elliptischen Grenzfall, siehe Tabelle 6.1.
    print(f"rho = {rho}")
    h_unabh(A=-5,B=5,C=-5,D=5,rho=rho,N_tau=1e-8,T=1,N_liste=(32,64,128,256,512,1024,2048),n_zyklus=12,nu1=1,nu2=1)


for rho in (0.0,0.5,0.9,0.99):     #durchschnittliche Residuenreduktionsfaktoren für ein festes dtau = 0.02, siehe Tabelle 6.3.
    print(f"rho = {rho}")
    h_unabh(A=-5,B=5,C=-5,D=5,rho=rho,N_tau=50,T=1,N_liste=(32,64,128,256,512,1024,2048),n_zyklus=12,nu1=1,nu2=1)

