from scipy.ndimage.filters import gaussian_filter
from draw import draw_points_lines, draw_box
import math
import cv2
import time
import torch
import numpy as np
stride = 8
boxsize = 800
thre = 0.02
mapIdx = [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]
limbSeq = [[1, 2], [2, 3], [3, 4], [4, 5], [5, 1]]

def padRightDownCorner(img, stride, padValue):
    h = img.shape[0]
    w = img.shape[1]

    pad = 4 * [None]
    pad[0] = 0  # up
    pad[1] = 0  # left
    pad[2] = 0 if (h % stride == 0) else stride - (h % stride)  # down
    pad[3] = 0 if (w % stride == 0) else stride - (w % stride)  # right

    img_padded = img
    pad_up = np.tile(img_padded[0:1, :, :] * 0 + padValue, (pad[0], 1, 1))
    img_padded = np.concatenate((pad_up, img_padded), axis=0)
    pad_left = np.tile(img_padded[:, 0:1, :] * 0 + padValue, (1, pad[1], 1))
    img_padded = np.concatenate((pad_left, img_padded), axis=1)
    pad_down = np.tile(img_padded[-2:-1, :, :] * 0 + padValue, (pad[2], 1, 1))
    img_padded = np.concatenate((img_padded, pad_down), axis=0)
    pad_right = np.tile(img_padded[:, -2:-1, :] * 0 + padValue, (1, pad[3], 1))
    img_padded = np.concatenate((img_padded, pad_right), axis=1)

    return img_padded, pad

def normalize(origin_img):
    image = origin_img.astype(np.float32) / 255.
    means = [0.485, 0.456, 0.406]
    stds = [0.229, 0.224, 0.225]

    preprocessed_img = image.copy()[:, :, ::-1]
    for i in range(3):
        preprocessed_img[:, :, i] = preprocessed_img[:, :, i] - means[i]
        preprocessed_img[:, :, i] = preprocessed_img[:, :, i] / stds[i]

    image = preprocessed_img.astype(np.float32)
    return image

def post_process(model, input_path, args):

    origin_img = cv2.imread(input_path)
    normed_img = normalize(origin_img)

    height, width, _ = normed_img.shape

    multiplier = [boxsize / height]

    last_heatmap_avg = np.zeros((height, width, 6))  # num_point
    paf_avg = np.zeros((height, width, 5 * 2))  # num_vector

    def heatmap_avg(heat, heatmap_avg):
        heatmap = heat.data.cpu().numpy()
        heatmap = np.transpose(np.squeeze(heatmap), (1, 2, 0))  # (h, w, c)
        heatmap = cv2.resize(heatmap, (0, 0), fx=stride, fy=stride,
                             interpolation=cv2.INTER_CUBIC)
        heatmap = heatmap[:imgToTest_padded.shape[0] - pad[2],
                  :imgToTest_padded.shape[1] - pad[3], :]
        heatmap = cv2.resize(heatmap, (width, height),
                             interpolation=cv2.INTER_CUBIC)
        heatmap_avg = heatmap_avg + heatmap / len(multiplier)
        return heatmap_avg


    for m in range(len(multiplier)):
        scale = multiplier[m]

        # preprocess
        imgToTest = cv2.resize(normed_img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        imgToTest_padded, pad = padRightDownCorner(imgToTest, stride, 0)
        input_img = np.transpose(imgToTest_padded[:, :, :, np.newaxis], (3, 2, 0, 1))  # required shape (1, c, h, w)
        input_var = torch.autograd.Variable(torch.from_numpy(input_img).cuda().float())

        # get the features
        out_scales = model(input_var)


        heat6 = out_scales[-1]
        vec6 = out_scales[-2]

        # get the first stage heatmap_avg
        last_heatmap_avg = heatmap_avg(heat6, last_heatmap_avg)

        paf_avg = heatmap_avg(vec6, paf_avg)


    last_stage_paf_map = cv2.resize(paf_avg, (origin_img.shape[1], origin_img.shape[0]),interpolation=cv2.INTER_CUBIC)

    point_tic = time.time()
    all_peaks = []  # all of the possible points by classes.
    peak_counter = 0

    for part in range(5):
        map_ori = last_heatmap_avg[:, :, part]
        map = gaussian_filter(map_ori, sigma=3)

        map_left = np.zeros(map.shape)
        map_left[:, 1:] = map[:, :-1]
        map_right = np.zeros(map.shape)
        map_right[:, :-1] = map[:, 1:]
        map_up = np.zeros(map.shape)
        map_up[1:, :] = map[:-1, :]
        map_down = np.zeros(map.shape)
        map_down[:-1, :] = map[1:, :]

        # get the salient point and its score > thre_point
        peaks_binary = np.logical_and.reduce(
            (map >= map_left, map >= map_right, map >= map_up, map >= map_down, map > thre))
        peaks = list(zip(np.nonzero(peaks_binary)[1], np.nonzero(peaks_binary)[0]))  # (w, h)

        # a point format: (w, h, score, number)
        peaks_with_score = [x + (map_ori[x[1], x[0]],) for x in peaks]
        id = range(peak_counter, peak_counter + len(peaks))
        peaks_with_score_and_id = [peaks_with_score[i] + (id[i],) for i in range(len(id))]

        all_peaks.append(peaks_with_score_and_id)
        peak_counter += len(peaks)
    point_toc = time.time()

    line_tic = time.time()
    connection_all = []  # save all of the possible lines by classes.
    special_k = []  # save the lines, which haven't legal points.
    mid_num = 10  # could adjust to accelerate (small) or improve accuracy(large).

    for k in range(len(mapIdx)):

        score_mid = paf_avg[:, :, [x for x in mapIdx[k]]]
        candA = all_peaks[limbSeq[k][0] - 1]
        candB = all_peaks[limbSeq[k][1] - 1]
        lenA = len(candA)
        lenB = len(candB)

        if lenA != 0 and lenB != 0:
            connection_candidate = []
            for i in range(lenA):
                for j in range(lenB):
                    vec = np.subtract(candB[j][:2], candA[i][:2])  # the vector of BA
                    norm = math.sqrt(vec[0] * vec[0] + vec[1] * vec[1])
                    if norm == 0:
                        continue
                    vec = np.divide(vec, norm)

                    startend = list(zip(np.linspace(candA[i][0], candB[j][0], num=mid_num),
                                        np.linspace(candA[i][1], candB[j][1], num=mid_num)))

                    # get the vector between A and B.
                    vec_x = np.array([score_mid[int(round(startend[I][1])), int(round(startend[I][0])), 0] for I in
                                      range(len(startend))])
                    vec_y = np.array([score_mid[int(round(startend[I][1])), int(round(startend[I][0])), 1] for I in
                                      range(len(startend))])

                    score_midpts = np.multiply(vec_x, vec[0]) + np.multiply(vec_y, vec[1])
                    score_with_dist_prior = sum(score_midpts) / len(score_midpts) + min(0.5 * height / norm - 1,0)  # ???
                    criterion1 = len(np.nonzero(score_midpts > thre)[0]) > 0.6 * len(score_midpts)
                    criterion2 = score_with_dist_prior > 0
                    if criterion1 and criterion2:
                        connection_candidate.append(
                            [i, j, score_with_dist_prior, score_with_dist_prior + candA[i][2] + candB[j][2]])

            # sort the possible line from large to small order.
            connection_candidate = sorted(connection_candidate, key=lambda x: x[3],
                                          reverse=True)  # different from openpose, I think there should be sorted by x[3]
            connection = np.zeros((0, 5))

            for c in range(len(connection_candidate)):
                i, j, s = connection_candidate[c][0: 3]
                if (i not in connection[:, 3] and j not in connection[:, 4]):
                    # the number of A point, the number of B point, score, A point, B point
                    connection = np.vstack([connection, [candA[i][3], candB[j][3], s, i, j]])
                    if len(connection) >= min(lenA, lenB):
                        break
            connection_all.append(connection)
        else:
            special_k.append(k)
            connection_all.append([])

    subset = -1 * np.ones((0, 6 + 1))
    candidate = np.array([item for sublist in all_peaks for item in sublist])

    for k in range(len(mapIdx)):
        if k not in special_k:
            partAs = connection_all[k][:, 0]
            partBs = connection_all[k][:, 1]
            indexA, indexB = np.array(limbSeq[k]) - 1

            for i in range(len(connection_all[k])):
                found = 0
                flag = [False, False]
                subset_idx = [-1, -1]
                for j in range(len(subset)):
                    # fix the bug, found == 2 and not joint will lead someone occur more than once.
                    # if more than one, we choose the subset, which has a higher score.
                    if subset[j][indexA] == partAs[i]:
                        if flag[0] == False:
                            flag[0] = found
                            subset_idx[found] = j
                            flag[0] = True
                            found += 1
                        else:
                            ids = subset_idx[flag[0]]
                            if subset[ids][-1] < subset[j][-1]:
                                subset_idx[flag[0]] = j
                    if subset[j][indexB] == partBs[i]:
                        if flag[1] == False:
                            flag[1] = found
                            subset_idx[found] = j
                            flag[1] = True
                            found += 1
                        else:
                            ids = subset_idx[flag[1]]
                            if subset[ids][-1] < subset[j][-1]:
                                subset_idx[flag[1]] = j

                if found == 1:
                    j = subset_idx[0]
                    if (subset[j][indexB] != partBs[i]):
                        subset[j][indexB] = partBs[i]
                        subset[j][-1] += 1
                        subset[j][-2] += candidate[partBs[i].astype(int), 2] + connection_all[k][i][2]
                elif found == 2:  # if found equals to 2 and disjoint, merge them
                    j1, j2 = subset_idx
                    membership = ((subset[j1] >= 0).astype(int) + (subset[j2] >= 0).astype(int))[:-2]
                    if len(np.nonzero(membership == 2)[0]) == 0:  # merge
                        subset[j1][:-2] += (subset[j2][:-2] + 1)
                        subset[j1][-2:] += subset[j2][-2:]
                        subset[j1][-2] += connection_all[k][i][2]
                        subset = np.delete(subset, j2, 0)
                    else:  # as like found == 1
                        subset[j1][indexB] = partBs[i]
                        subset[j1][-1] += 1
                        subset[j1][-2] += candidate[partBs[i].astype(int), 2] + connection_all[k][i][2]
                elif not found and k < len(mapIdx):
                    row = -1 * np.ones(6+1)
                    row[indexA] = partAs[i]
                    row[indexB] = partBs[i]
                    row[-1] = 2
                    row[-2] = sum(candidate[connection_all[k][i, :2].astype(int), 2]) + connection_all[k][i][2]
                    subset = np.vstack([subset, row])

    for i in range(len(subset)):
        subset[i][-1] = np.sum(np.array(subset[i][:5] > -1))

    # delete som rows of subset which has few parts occur
    deleteIdx = []
    for i in range(len(subset)):
        if subset[i][-1] < 4 or subset[i][-2] / subset[i][-1] < 0.1:
            deleteIdx.append(i)
    subset = np.delete(subset, deleteIdx, axis=0)
    line_toc = time.time()

    tradition_box_img, bbox_img, _, _ = draw_box(origin_img, last_stage_paf_map, subset, candidate, uv_thre=0.005)

    points_lines_img, points_dics = draw_points_lines(origin_img, subset, candidate)

    return points_lines_img, bbox_img
