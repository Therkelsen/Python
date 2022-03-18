import roboticstoolbox as rtb

puma = rtb.models.DH.Puma560()

# print(puma)


## forward kinematics
# T = puma.fkine(qz) # home pose 
# print(T)


## inverse  kinematics (numerical)
# T = puma.fkine([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]) # user defined
# sol = puma.ikine_LM(T)
# print(sol)


## inverse  kinematics (analytical)
# T = puma.fkine([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]) # user defined
# sol = puma.ikine_a(T, config="lun")
# print(sol)


## 3D plot
#q = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
#puma.plot(q, block=True)


## joint path
# traj = rtb.jtraj(puma.qz, puma.qr, 100)
# #traj.q.shape
# rtb.qplot(traj.q, block=True)


## Cartesian path
# import numpy as np
# from spatialmath import SE3
# t = np.arange(0, 2, 0.010)
# T0 = SE3(0.6, -0.5, 0.0)
# T1 = SE3(0.4, 0.5, 0.2)
# Ts = rtb.tools.trajectory.ctraj(T0, T1, len(t))
# #len(Ts)
# sol = puma.ikine_LM(Ts)
# #sol.q.shape
# rtb.qplot(sol.q, block=True)


## 3D animation
import numpy as np
from spatialmath import SE3
t = np.arange(0, 2, 0.010)
T0 = SE3(0.6, -0.5, 0.0)
T1 = SE3(0.4, 0.5, 0.2)
Ts = rtb.tools.trajectory.ctraj(T0, T1, len(t))
sol = puma.ikine_LM(Ts)
puma.plot(sol.q, block=True)


## free plot
# pyp = rtb.backends.PyPlot.PyPlot()
# pyp.launch()
# pyp.add(puma)
# q = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
# puma.q = q
# pyp.step()
# pyp.hold()