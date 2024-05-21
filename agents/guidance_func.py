import numpy as np


#gloabal perception functions
def get_position(obj_name):
    print("get_position: ",obj_name)
    return np.array([0,0,0])

def get_size(obj_name):
    print("get_size: ",obj_name)
    return np.array([0.2,0.2])

def guidance(state):
    # Unpack the robot's current state
    robot_x, robot_y, robot_z, robot_rx, robot_ry, robot_rz, gripper = state

    # Get the position and size of the can
    can_x, can_y, can_z = get_position('handle')
    can_height, can_width = get_size('handle')

    return robot_z
    # Calculate the distance to the can in the xy-plane
    distance_xy = np.sqrt((can_x - robot_x)**2 + (can_y - robot_y)**2)

    # Calculate the vertical distance to the top of the can
    distance_z = abs(robot_z - (can_z + can_height / 2))

    # Calculate how well the gripper is adjusted to the size of the can
    gripper_adjustment = abs(gripper - can_width)

    # Define a score that combines these distances and the gripper adjustment
    # Lower score is better. A score of 0 means the task is completed.
    score = distance_xy + distance_z + gripper_adjustment

    return score