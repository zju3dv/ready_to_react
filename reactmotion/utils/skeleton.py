MOTIVE_NUM_JOINTS = 21

MOTIVE_JOINTS_NAMES = [
    'Hips', # 0
    'Spine', # 1
    'Spine1', # 2
    'Neck', # 3
    'Head', # 4
    'LeftShoulder', # 5
    'LeftArm', # 6
    'LeftForeArm', # 7 
    'LeftHand', # 8
    'RightShoulder', # 9
    'RightArm', # 10
    'RightForeArm', # 11
    'RightHand', # 12
    'LeftUpLeg', # 13
    'LeftLeg', # 14
    'LeftFoot', # 15
    'LeftToeBase', # 16
    'RightUpLeg', # 17
    'RightLeg', # 18
    'RightFoot', # 19
    'RightToeBase', # 20
]

MOTIVE_VR3JOINTS = [
    "Head",
    "LeftHand",
    "RightHand",
]

MOTIVE_JOINTS_PARENTS = [
    -1, 0, 1, 2, 3,
              2, 5, 6, 7,
              2, 9, 10, 11,
        0, 13, 14, 15,
        0, 17, 18, 19,
]

MOTIVE_ALIGNMENT = [
    0, 0, 1, 0, 1,
             0, 1, 1, 1,
             0, 1, 1, 1,
       0, 1, 1, 1,
       0, 1, 1, 1,
]


# In this list, the rotation can be directly mapped between the two skeletons
SMPL_MOTIVE_DIRECT_MAPPING = [
    ("pelvis", "Hips"),
    # ("left_hip", "LeftUpLeg"),
    # ("right_hip", "RightUpLeg"),
    # ("spine1", "Spine"),
    ("left_knee", "LeftLeg"),
    ("right_knee", "RightLeg"),
    # ("spine2", "Spine"),
    ("left_ankle", "LeftFoot"),
    ("right_ankle", "RightFoot"),
    # ("spine3", "Spine1"),
    ("left_foot", "LeftToeBase"),
    ("right_foot", "RightToeBase"),
    ("neck", "Neck"),
    # ("left_collar", "LeftShoulder"),
    # ("right_collar", "RightShoulder"),
    ("head", "Head"),
    # ("left_shoulder", "LeftArm"),
    # ("right_shoulder", "RightArm"),
    ("left_elbow", "LeftForeArm"),
    ("right_elbow", "RightForeArm"),
    ("left_wrist", "LeftHand"),
    ("right_wrist", "RightHand"),
]

# It is an approximate mapping between the two skeletons
SMPL_MOTIVE_MAPPING = [
    ("pelvis", "Hips"),
    ("right_hip", "RightUpLeg"),
    ("right_knee", "RightLeg"),
    ("right_ankle", "RightFoot"),
    ("right_foot", "RightToeBase"),
    ("left_hip", "LeftUpLeg"),
    ("left_knee", "LeftLeg"),
    ("left_ankle", "LeftFoot"),
    ("left_foot", "LeftToeBase"),
    ("spine1", "Spine"),
    # ("spine2", "Spine"),
    ("spine3", "Spine1"),
    ("neck", "Neck"),
    ("head", "Head"),
    # ("left_collar", "LeftShoulder"),
    ("left_shoulder", "LeftArm"),
    ("left_elbow", "LeftForeArm"),
    ("left_wrist", "LeftHand"),
    # ("right_collar", "RightShoulder"),
    ("right_shoulder", "RightArm"),
    ("right_elbow", "RightForeArm"),
    ("right_wrist", "RightHand"),
]

# for fast indexing, skeleton group
SG = {
    'motive': {
        'parents': MOTIVE_JOINTS_PARENTS,
        'njoints': MOTIVE_NUM_JOINTS,
        'vr3joints': MOTIVE_VR3JOINTS,
        'alignment': MOTIVE_ALIGNMENT,
        'names': MOTIVE_JOINTS_NAMES,
        'toe': [
            MOTIVE_JOINTS_NAMES.index("LeftToeBase"),
            MOTIVE_JOINTS_NAMES.index("RightToeBase"),
        ],
        'foot': [
            MOTIVE_JOINTS_NAMES.index("LeftFoot"),
            MOTIVE_JOINTS_NAMES.index("LeftToeBase"),
            MOTIVE_JOINTS_NAMES.index("RightFoot"),
            MOTIVE_JOINTS_NAMES.index("RightToeBase"),
        ],
    }
}