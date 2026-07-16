from rtde_control import RTDEControlInterface as RTDEControl
from rtde_receive import RTDEReceiveInterface as RTDEReceive

robot_ip = "127.0.0.1"

rtde_c = RTDEControl(robot_ip)
rtde_r = RTDEReceive(robot_ip)

pose = rtde_r.getActualTCPPose()
pose[2] += 0.05

rtde_c.moveL(pose, 3, 130)
rtde_c.stopScript()