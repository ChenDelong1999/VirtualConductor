from scipy import signal
from scipy.signal import savgol_filter, medfilt
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import torch
import math
from queue import Queue, LifoQueue, PriorityQueue
import time
import tqdm

RED = (0, 0, 255)
GREEN = (0, 255, 0)
BLUE = (255, 0, 0)
CYAN = (255, 255, 0)
YELLOW = (0, 255, 255)
ORANGE = (0, 165, 255)
PURPLE = (255, 0, 255)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)


def smooth_motion(kp_pred, kernel=11, order=5):
    for i in range(kp_pred.shape[1]):
        for j in range(2):
            data = kp_pred[:, i, j]
            data_smooth = savgol_filter(data, kernel, order)
            kp_pred[:, i, j] = data_smooth
    return kp_pred


def norm_motion(kp_pred, width, height):
    kp_pred /= width

    # shoulder width
    shoulder = np.average(kp_pred[:, 5, 0] - kp_pred[:, 6, 0])
    kp_pred *= 1 / 5 / shoulder

    # hip centering
    hip_x = np.average(kp_pred[:, 11, 0] + kp_pred[:, 12, 0]) / 2
    hip_y = np.average(kp_pred[:, 11, 1] + kp_pred[:, 12, 1]) / 2

    for i in range(17):
        kp_pred[:, i, 0] -= hip_x - 0.5
        kp_pred[:, i, 1] -= hip_y - 0.75

    return kp_pred


def vis_img(img, kp_preds, kp_scores, hand_trace):
    """
    frame: frame image
    im_res: im_res of predictions
    format: coco or mpii
    return rendered image
    """

    ''' --- MOCO keypoints ---
    0 Nose, 1 LEye, 2 REye, 3 LEar, 4 REar
    5 LShoulder, 6 RShoulder, 7 LElbow, 8 RElbow, 9 LWrist, 10 RWrist
    11 LHip, 12 RHip, 
    (discarded: 13 LKnee, 14 Rknee, 15 LAnkle, 16 RAnkle, 17 Neck)
    '''

    kp_num = 17
    l_pair = [
        (0, 1), (0, 2), (1, 3), (2, 4),  # Head
        (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
        (17, 11), (17, 12),  # Body
        (11, 13), (12, 14), (13, 15), (14, 16),  # Legs
        # add body outline:
        (11, 12), (5, 11), (6, 12)
    ]
    black = (120, 120, 120)
    blue = (216, 164, 78)
    red = (54, 41, 159)
    white = (255, 255, 255)
    line_color = [blue, blue, blue, blue,
                  blue, blue, blue, blue, blue,
                  black, black, black, black, black, black,
                  blue, blue, blue  # cdl add body outline
                  ]

    trace_head = np.array(red)
    trace_end = np.array(white)
    for i in range(len(hand_trace)):
        alpha = i / len(hand_trace)

        color = alpha * trace_head + (1 - alpha) * trace_end
        for j in range(len(hand_trace[i])):
            color_factor = i / len(hand_trace)
            cv2.circle(img, (int(hand_trace[i, j, 0]), int(hand_trace[i, j, 1])), 2, color, 2)

    part_line = {}
    # --- Draw points --- #
    vis_thres = 0.4
    for n in range(kp_scores.shape[0]):
        if kp_scores[n] <= vis_thres:
            continue
        cor_x, cor_y = int(kp_preds[n, 0]), int(kp_preds[n, 1])
        part_line[n] = (int(cor_x), int(cor_y))

    # --- Draw limbs --- #
    for i, (start_p, end_p) in enumerate(l_pair):
        if start_p in part_line and end_p in part_line:
            start_xy = part_line[start_p]
            end_xy = part_line[end_p]
            X = (start_xy[0], end_xy[0])
            Y = (start_xy[1], end_xy[1])
            mX = np.mean(X)
            mY = np.mean(Y)

            length = ((Y[0] - Y[1]) ** 2 + (X[0] - X[1]) ** 2) ** 0.5
            angle = math.degrees(math.atan2(Y[0] - Y[1], X[0] - X[1]))
            stickwidth = (kp_scores[start_p] + kp_scores[end_p]) + 1
            polygon = cv2.ellipse2Poly((int(mX), int(mY)), (int(length / 2), int(stickwidth)), int(angle), 0, 360, 10)
            if i < len(line_color):
                cv2.fillConvexPoly(img, polygon, line_color[i])
            else:
                cv2.line(img, start_xy, end_xy, (128, 128, 128), 1)

    '''for n in [1,2]:
        cor_x, cor_y = int(kp_preds[n, 0]), int(kp_preds[n, 1])
        cv2.circle(img, (int(cor_x), int(cor_y)), 9, white, 9)
        cv2.circle(img, (int(cor_x), int(cor_y)), 2, black, 2)
        cv2.circle(img, (int(cor_x), int(cor_y)), 10, black, 2)'''

    for n in [9, 10]:
        cor_x, cor_y = int(kp_preds[n, 0]), int(kp_preds[n, 1])
        cv2.circle(img, (int(cor_x), int(cor_y)), 9, white, 9)
        cv2.circle(img, (int(cor_x), int(cor_y)), 2, red, 2)
        cv2.circle(img, (int(cor_x), int(cor_y)), 10, red, 2)

    return img


def vis_motion(motions, kp_score=None, save_path='../test/result', name='_[name]_', post_processing=True):
    # motions [num_conductor, num_frame, 13, 2]
    if kp_score is None:  # confidence
        kp_score = np.zeros((motions[0].shape[0], 17))
        kp_score[:, :13] = 1

    window = 600
    trace_len = 30
    hand_traces = []
    video_file = save_path + name + '.avi'
    wirter = cv2.VideoWriter(video_file, 0, cv2.VideoWriter_fourcc(*'XVID'), 30,
                             (1 + len(motions) * window, window))

    for i in range(len(motions)):
        motions[i] *= window
        motion = motions[i]
        motion = smooth_motion(motion, kernel=19)
        hand_trace = np.ones((motion.shape[0] + trace_len, 2, 2)) * -1
        hand_trace[trace_len:, :, :] = motion[:, 9:11, :]
        hand_traces.append(hand_trace)

    for f in tqdm.tqdm(range(motions[0].shape[0])):
        frame = np.ones((window, 1, 3), np.uint8) * 255
        for i in range(len(motions)):
            motion = motions[i]
            background = np.ones((window, window, 3), np.uint8) * 255
            img = vis_img(background, motion[f], kp_score[f], hand_traces[i][f:f + trace_len, :, :])
            frame = np.concatenate((frame, img), axis=1)
        # cv2.imshow("frame", frame)
        # save img
        # if f % 30 == 0:
        #    cv2.imwrite(save_path + '{}.png'.format(str(np.random.rand())), frame)
        # key = cv2.waitKey(1)
        # if key == 27:  # press Esc
        #    break
        wirter.write(frame)
    wirter.release()
    cv2.destroyAllWindows()
    return video_file


def filter(keypoints, freq_low=0.4, freq_high=5, sample_rate=25, mode='high pass'):
    
    highpass_pose = np.zeros_like(keypoints)
    wnl = 2 * freq_low / sample_rate
    wnh = 2 * freq_high / sample_rate

    high_b, high_a = signal.butter(8, [wnl, wnh], 'bandpass', output='ba')

    for i in range(13):
        for j in range(2):
            highpass_pose[:,i,j] = signal.filtfilt(high_b, high_a, keypoints[:,i,j])

    if mode == 'high pass':
        return highpass_pose


def COCO_to_CM100(kp_pred):
    return kp_pred[:, :13, :]


def CM100_to_COCO(kp_pred):
    kp_pred_moco = np.zeros((kp_pred.shape[0], 17, 2))
    kp_pred_moco[:, :13, :] = kp_pred
    return kp_pred_moco
