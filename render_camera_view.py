'''
    This script is used to render output of model in camera view.
    Useful to create a demo video.
    6 cameras are used.
'''


import json
import math
import cv2
import os
import os.path as osp 
import sys
import time
from datetime import datetime
from typing import Tuple, List, Iterable
import matplotlib.pyplot as plt
import numpy as np
import sklearn.metrics
from PIL import Image
from matplotlib import rcParams
from matplotlib.axes import Axes
from pyquaternion import Quaternion
from tqdm import tqdm


from nuscenes.utils.data_classes import LidarPointCloud, RadarPointCloud, Box
from nuscenes.utils.geometry_utils import view_points, box_in_image, BoxVisibility, transform_matrix
from nuscenes.nuscenes import NuScenes


assert len(sys.argv) == 2, "\nProvide a single json file containing bounding boxes in Lidar View. \
                            \nExpected order of the file is as the same as mmdetection3d's\n"

results_file = sys.argv[1]

with open(results_file, 'r') as res_file:
    results = res_file.read()

results = json.loads(results)
# results => ['meta', 'results']
results = results['results']

import pdb; pdb.set_trace()

def get_boxes(nusc, sample_data_token: str) -> List[Box]:
    """
    Instantiates Boxes for all annotation for a particular sample_data record. If the sample_data is a
    keyframe, this returns the annotations for that sample. But if the sample_data is an intermediate
    sample_data, a linear interpolation is applied to estimate the location of the boxes at the time the
    sample_data was captured.
    :param sample_data_token: Unique sample_data identifier.
    """
    global results
    # Retrieve sensor & pose records
    sd_record = nusc.get('sample_data', sample_data_token)
    curr_sample_record = nusc.get('sample', sd_record['sample_token'])

    # if curr_sample_record['prev'] == "" or sd_record['is_key_frame']:
    # if sd_record['is_key_frame']:
        # If no previous annotations available, or if sample_data is keyframe just return the current ones.
        # boxes = list(map(get_box, nusc, curr_sample_record['anns']))

    boxes = []
    predictions = results[curr_sample_record['token']]

    for pred in predictions:
        box = Box(pred['translation'], pred['size'], Quaternion(pred['rotation']),
           name=pred['detection_name'], token=None)
        if pred['detection_score']>=0.7:
            boxes.append(box)

    # else:
    #     boxes = []

    return boxes


# def get_sample_data_path(nusc, sample_data_token: str) -> str:
#         """ Returns the path to a sample_data. """

#     sd_record = nusc.get('sample_data', sample_data_token)
#     return osp.join(nusc.dataroot, sd_record['filename'])


def get_sample_data(nusc, 
                    sample_data_token: str,
                    box_vis_level: BoxVisibility = BoxVisibility.ANY,
                    use_flat_vehicle_coordinates: bool = False) -> Tuple[str, List[Box], np.array]:
    """
    Returns the data path as well as all annotations related to that sample_data.
    Note that the boxes are transformed into the current sensor's coordinate frame.
    :param sample_data_token: Sample_data token.
    :param box_vis_level: If sample_data is an image, this sets required visibility for boxes.
    :param selected_anntokens: If provided only return the selected annotation.
    :param use_flat_vehicle_coordinates: Instead of the current sensor's coordinate frame, use ego frame which is
                                         aligned to z-plane in the world.
    :return: (data_path, boxes, camera_intrinsic <np.array: 3, 3>)
    """

    # Retrieve sensor & pose records
    sd_record = nusc.get('sample_data', sample_data_token)
    cs_record = nusc.get('calibrated_sensor', sd_record['calibrated_sensor_token'])
    sensor_record = nusc.get('sensor', cs_record['sensor_token'])
    pose_record = nusc.get('ego_pose', sd_record['ego_pose_token'])

    data_path = nusc.get_sample_data_path(sample_data_token)

    if sensor_record['modality'] == 'camera':
        cam_intrinsic = np.array(cs_record['camera_intrinsic'])
        imsize = (sd_record['width'], sd_record['height'])
    else:
        assert False, "Only camera modalities are supported"

    # Retrieve all sample annotations and map to sensor coordinate system.
    boxes = get_boxes(nusc, sample_data_token)

    # Make list of Box objects including coord system transforms.
    box_list = []
    for box in boxes:
        if use_flat_vehicle_coordinates:
            # Move box to ego vehicle coord system parallel to world z plane.
            yaw = Quaternion(pose_record['rotation']).yaw_pitch_roll[0]
            box.translate(-np.array(pose_record['translation']))
            box.rotate(Quaternion(scalar=np.cos(yaw / 2), vector=[0, 0, np.sin(yaw / 2)]).inverse)
        else:
            # Move box to ego vehicle coord system.
            box.translate(-np.array(pose_record['translation']))
            box.rotate(Quaternion(pose_record['rotation']).inverse)

            #  Move box to sensor coord system.
            box.translate(-np.array(cs_record['translation']))
            box.rotate(Quaternion(cs_record['rotation']).inverse)

        if sensor_record['modality'] == 'camera' and not box_in_image(box, cam_intrinsic, imsize, vis_level=box_vis_level):
            continue

        box_list.append(box)

    return data_path, box_list, cam_intrinsic



def render_scene(nusc,
                     scene_token: str,
                     freq: float = 10,
                     imsize: Tuple[float, float] = (640, 360),
                     out_path: str = None) -> None:
    """
    Renders a full scene with all camera channels.
    :param scene_token: Unique identifier of scene to render.
    :param freq: Display frequency (Hz).
    :param imsize: Size of image to render. The larger the slower this will run.
    :param out_path: Optional path to write a video file of the rendered frames.
    """

    assert imsize[0] / imsize[1] == 16 / 9, "Aspect ratio should be 16/9."

    if out_path is not None:
        assert osp.splitext(out_path)[-1] == '.avi'

    # Get records from DB.
    scene_rec = nusc.get('scene', scene_token)
    first_sample_rec = nusc.get('sample', scene_rec['first_sample_token'])
    last_sample_rec = nusc.get('sample', scene_rec['last_sample_token'])

    # Set some display parameters.
    layout = {
        'CAM_FRONT_LEFT': (0, 0),
        'CAM_FRONT': (imsize[0], 0),
        'CAM_FRONT_RIGHT': (2 * imsize[0], 0),
        'CAM_BACK_LEFT': (0, imsize[1]),
        'CAM_BACK': (imsize[0], imsize[1]),
        'CAM_BACK_RIGHT': (2 * imsize[0], imsize[1]),
    }

    colors = {
        "pedestrian": (0, 0, 230),  # Blue
        "barrier": (112, 128, 144),  # Slategrey
        "traffic_cone": (47, 79, 79),  # Darkslategrey
        "bicycle": (220, 20, 60),  # Crimson
        "bus": (255, 127, 80),  # Coral
        "car": (255, 158, 0),  # Orange
        "construction_vehicle": (233, 150, 70),  # Darksalmon
        "motorcycle": (255, 61, 99),  # Red
        "trailer": (255, 140, 0),  # Darkorange
        "truck": (255, 99, 71),  # Tomato
    }

    horizontal_flip = ['CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT']  # Flip these for aesthetic reasons.

    time_step = 1 / freq * 1e6  # Time-stamps are measured in micro-seconds.

    window_name = '{}'.format(scene_rec['name'])
    cv2.namedWindow(window_name)
    cv2.moveWindow(window_name, 0, 0)

    canvas = np.ones((2 * imsize[1], 3 * imsize[0], 3), np.uint8)
    if out_path is not None:
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        out = cv2.VideoWriter(out_path, fourcc, freq, canvas.shape[1::-1])
    else:
        out = None

    # Load first sample_data record for each channel.
    current_recs = {}  # Holds the current record to be displayed by channel.
    prev_recs = {}  # Hold the previous displayed record by channel.
    for channel in layout:
        current_recs[channel] = nusc.get('sample_data', first_sample_rec['data'][channel])
        prev_recs[channel] = None

    current_time = first_sample_rec['timestamp']

    while current_time < last_sample_rec['timestamp']:

        current_time += time_step

        # For each channel, find first sample that has time > current_time.
        for channel, sd_rec in current_recs.items():
            while sd_rec['timestamp'] < current_time and sd_rec['next'] != '':
                sd_rec = nusc.get('sample_data', sd_rec['next'])
                current_recs[channel] = sd_rec

        # if not sd_rec['is_key_frame']:
        #     continue

        # Now add to canvas
        for channel, sd_rec in current_recs.items():

            # Only update canvas if we have not already rendered this one.
            if not sd_rec == prev_recs[channel]:

                # Get annotations and params from DB.
                impath, boxes, camera_intrinsic = get_sample_data(nusc, sd_rec['token'], box_vis_level=BoxVisibility.ANY)
                if len(boxes) == 0:
                    print("Empty boxes!!!")
                    print(impath)
                    print()

                # Load and render.
                if not osp.exists(impath):
                    raise Exception('Error: Missing image %s' % impath)
                im = cv2.imread(impath)
                for box in boxes:
                    c = colors[box.name]
                    # c = (255, 158, 0)
                    box.render_cv2(im, view=camera_intrinsic, normalize=True, colors=(c, c, c))

                im = cv2.resize(im, imsize)
                if channel in horizontal_flip:
                    im = im[:, ::-1, :]

                canvas[
                    layout[channel][1]: layout[channel][1] + imsize[1],
                    layout[channel][0]:layout[channel][0] + imsize[0], :
                ] = im

                prev_recs[channel] = sd_rec  # Store here so we don't render the same image twice.

        # Show updated canvas.
        cv2.imshow(window_name, canvas)
        if out_path is not None:
            out.write(canvas)

        key = cv2.waitKey(1)  # Wait a very short time (1 ms).

        if key == 32:  # if space is pressed, pause.
            key = cv2.waitKey()

        if key == 27:  # if ESC is pressed, exit.
            cv2.destroyAllWindows()
            break

    cv2.destroyAllWindows()
    if out_path is not None:
        out.release()


# nusc_test_set = NuScenes(version='v1.0-test', dataroot='./test_set/')
# obtain some scene token from nusc_test_set obj., e.g: scene  
# render_scene(nusc_test_set, scene, out_path=None)      # Provide out_path to save rendering as a video 