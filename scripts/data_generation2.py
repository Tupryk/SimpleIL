import os
import numpy as np
import robotic as ry
from tqdm import tqdm
import matplotlib.pyplot as plt
from Robotic_Manipulation.shelf import generate_target_box
import Robotic_Manipulation.manipulation as manip

if not os.path.exists(f"./sim_recs"):
    os.makedirs(f"./sim_recs")

# Define the new config
C = ry.Config()
C.addFile(ry.raiPath("scenarios/pandaSingle.g"))

generate_target_box(C, [-.3, .3, .75])

box_frame = C.addFrame("box") \
    .setPosition([.3, .3, .8]) \
    .setShape(ry.ST.ssBox, size=[.05, .05, .12, .001]) \
    .setColor([1., 1., 0.]) \
    .setContact(1) \
    .setMass(1.)

# For convenience, a few definitions:
gripper = "l_gripper"
box = "box"
table = "table"

C.view()

bot = ry.BotOp(C, useRealRobot=False)
bot.home(C)
bot.gripperMove(ry._left)
bot.sync(C, 2.)
qHome = C.getJointState()

def pick_place(obj: str, grasp_direction: str, place_direction: str, info: str="", vis: bool=False) -> bool:
    M = manip.ManipulationModelling(C, info, helpers=[gripper])
    M.setup_pick_and_place_waypoints(gripper, obj)
    M.grasp_box(1., gripper, box, "l_palm", grasp_direction)
    M.target_relative_xy_position(2., "box", "target_box_inside", [0, 0])
    M.set_relative_distance(2., "box", "target_box_inside", .1)
    M.solve()
    if not M.feasible:
        return False, [], []

    M1 = M.sub_motion(0)
    M1.keep_distance([.3,.7], "l_palm", obj, margin=.05)
    M1.retract([.0, .2], gripper)
    M1.approach([.8, 1.], gripper)
    path1 = M1.solve()
    if not M1.feasible:
        return False, [], []

    M2 = M.sub_motion(1)
    M2.keep_distance([], table, "panda_collCameraWrist")
    M2.keep_distance([.2, .8], table, obj, .04)
    M2.keep_distance([], "l_palm", obj)
    path2 = M2.solve()
    if not M2.feasible:
        return False, [], []

    if vis:
        M1.play(C, 1.)
        C.attach(gripper, obj)
        M2.play(C, 1.)
        C.attach(table, obj)

    return True, path1, path2

attempt_count = 10_000
success_count = 0

for l in tqdm(range(attempt_count)):

    grasp_direction = np.random.choice(["x", "y"])  # "z" not possible: box too large
    place_direction = np.random.choice(["z", "zNeg"])

    success, path1, path2 = pick_place(box, grasp_direction, place_direction, str(l), vis=False)

    if success:
        folder = f"./sim_recs/rec{success_count:04}"
        if not os.path.exists(folder):
            os.makedirs(folder)
        if not os.path.exists(f"{folder}/images"):
            os.makedirs(f"{folder}/images")

        gen_file = open(f"{folder}/proprioceptives.txt", 'w')
        img_idx = 0
        bot.move(path1, [3])
        while bot.getTimeToEnd() > 0.:
            bot.sync(C, 0.)
            q = bot.get_q().tolist()
            q.append(0.)
            gen_file.write(f"{q}\n".replace("[", "").replace("]", "").replace(",", ""))
            rgb, depth = bot.getImageAndDepth("cameraTop")
            plt.imsave(f"{folder}/images/{img_idx:04}.jpg", rgb)
            img_idx += 1

        bot.gripperClose(ry._left)
        while not bot.gripperDone(ry._left):
            bot.sync(C)
            
        bot.move(path2, [3])
        while bot.getTimeToEnd() > 0.:
            bot.sync(C, 0.)
            q = bot.get_q().tolist()
            q.append(1.)
            gen_file.write(f"{q}\n".replace("[", "").replace("]", "").replace(",", ""))
            rgb, depth = bot.getImageAndDepth("cameraTop")
            plt.imsave(f"{folder}/images/{img_idx:04}.jpg", rgb)
            img_idx += 1

        bot.gripperMove(ry._left)
        while not bot.gripperDone(ry._left):
            bot.sync(C)

        q = bot.get_q().tolist()
        q.append(1.)
        gen_file.write(f"{q}\n".replace("[", "").replace("]", "").replace(",", ""))
        rgb, depth = bot.getImageAndDepth("cameraTop")
        plt.imsave(f"{folder}/images/{img_idx:04}.jpg", rgb)
        
        del bot
        x_dev = np.random.random() * .4 -.2
        y_dev = np.random.random() * .3 -.15
        box_frame.setQuaternion([1., 0., 0., 0.])
        box_frame.setPosition([.3 + x_dev, .3 + y_dev, .8])
        C.setJointState(qHome)
        C.view()
        bot = ry.BotOp(C, useRealRobot=False)
        bot.home(C)
        bot.gripperMove(ry._left)
        bot.sync(C, 2.)

    success_count += 1 if success else 0
	
print(f"Successful motions: {success_count}/{attempt_count}")
