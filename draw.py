import numpy as np
import cv2
import time
import copy
import math

thre_point = 0.02
num_points = 6
mapIdx = [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]
limbSeq = [[0, 1], [1, 2], [2, 3], [3, 4], [4, 0]]
colors = [[0, 0, 255], [255, 0, 255], [255, 85, 0], [0, 255, 255], [0, 255, 0]]

def draw_points_lines(canvas, subset, candidate):
    draw_result_tic = time.time()
    # draw points
    points_dics = []

    ob_num = len(subset)
    for human_id in range(ob_num):
        point_dics = {}
        point_weizhi = []
        for point_index in range(num_points - 1):
            index_ = subset[human_id][point_index]

            if index_ != -1:
                point_weizhi.append([int(candidate[int(index_)][0]), int(candidate[int(index_)][1])])
                center = (int(candidate[int(index_)][0]), int(candidate[int(index_)][1]))
                cv2.circle(canvas, center, 50, colors[point_index], thickness=-1)
            else:
                point_weizhi.append([-100, -100])

        point_dics["points"] = point_weizhi
        point_dics["score"] = subset[human_id][-2] / subset[human_id][-1]
        points_dics.append(point_dics)

    for i in range(len(mapIdx)):
        for n in range(len(subset)):
            index = subset[n][np.array(limbSeq[i])]
            if -1 in index:
                continue
            cur_canvas = canvas.copy()

            Y = candidate[index.astype(int), 0]
            X = candidate[index.astype(int), 1]
            mX = np.mean(X)
            mY = np.mean(Y)
            length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
            angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))  # (-180 , 180)
            if i in range(4):
                polygon = cv2.ellipse2Poly((int(mY), int(mX)), (int(length / 2), 35), int(angle), 0, 360, 5)
                cv2.fillConvexPoly(cur_canvas, polygon, colors[i])

            canvas = cv2.addWeighted(canvas, 0.4, cur_canvas, 0.6, 0)
    draw_result_toc = time.time()
    return canvas, points_dics

def draw_box(ori_img, last_stage_paf_map, subset, candidate, uv_thre):
    box_img = copy.deepcopy(ori_img)
    tradition_box_img = copy.deepcopy(ori_img)
    backbone_img = copy.deepcopy(ori_img)
    dics = []  # 存储每个bbox的信息
    for k in range(len(subset)):
        points_index = subset[k]   # 1 X 7
        points = []
        for index in points_index[:5]:
            if index > -1:
                points.append(candidate[int(index)])
            elif index == -1:
                points.append(np.array([-1, -1, -1, -1]))
        # print(points)
        points = np.array(points)

        dis_matrix_all = np.zeros((len(points), len(points)))  # 5X5
        for i, points_i in enumerate(points):
            for j, points_j in enumerate(points):
                distance = ((points_i[0] - points_j[0]) ** 2 + (points_i[1] - points_j[1]) ** 2) ** 0.5
                dis_matrix_all[i][j] = distance

        for i, point_0 in enumerate(points[:, 0]):
            if point_0 == -1:
                dis_matrix_all[i, :] = 0
                dis_matrix_all[:, i] = 0
        max_dis = np.max(dis_matrix_all)

        box_points_index = list(np.argwhere(np.mat(dis_matrix_all) == max_dis)[0])

        X=[]
        Y=[]
        for i in range(5):
            if points[i][-1] != -1:
                X.append(points[i][0])
                Y.append(points[i][1])
                break
        for i in range(4,-1,-1):
            if points[i][-1] != -1:
                X.append(points[i][0])
                Y.append(points[i][1])
                break
        backbone_img = cv2.line(backbone_img, (int(X[0]), int(Y[0])), (int(X[1]), int(Y[1])), (0, 0, 255), thickness=10)

        mask = points[:, 0] >= 0
        z = np.polyfit(points[mask,0], points[mask,1], 1)
        # print(z)


        U_start = last_stage_paf_map[:, :, mapIdx[0][0]] * -1  # x
        V_start = last_stage_paf_map[:, :, mapIdx[0][1]] # y
        M_start = np.zeros(U_start.shape, dtype='bool')
        M_start[abs(U_start) + abs(V_start) < uv_thre] = True

        U_end = last_stage_paf_map[:, :, mapIdx[3][0]] * -1  # x
        V_end = last_stage_paf_map[:, :, mapIdx[3][1]]  # y
        M_end = np.zeros(U_end.shape, dtype='bool')
        M_end[abs(U_end) + abs(V_end) < uv_thre] = True

        start_point = points[box_points_index[0]]
        end_point = points[box_points_index[1]]

        def find_min_range(M_start, M_end, point1, point2):

            length_start = 0
            length_end = 0
            y1 = int(point1[1])
            x1 = int(point1[0])
            y2 = int(point2[1])
            x2 = int(point2[0])
            # 判断绝缘子是横的还是竖的
            is_axis_y = True
            if abs(z[0]) >= 0 and abs(z[0]) <= 1:
                is_axis_y = False
            x1_start = copy.deepcopy(x1)
            y1_start = copy.deepcopy(y1)
            x2_end = copy.deepcopy(x2)
            y2_end = copy.deepcopy(y2)
            while M_start[y1_start][x1_start] == False:
                length_start += 1
                if is_axis_y:
                    y1_start -= 1
                else:
                    x1_start -= 1
                if x1_start <= 0 or y1_start <= 0:
                    break
            while M_end[y2_end][x2_end] == False:
                length_end += 1
                if is_axis_y:
                    y2_end += 1
                else:
                    x2_end += 1
                if x2_end >= ori_img.shape[1] or y2_end >= ori_img.shape[0]:
                    break
            return (int(length_start), int(length_end))

        length = find_min_range(M_start, M_end, start_point, end_point)
        sin_theat = np.abs(z[0]) / (z[0] * z[0] + 1) ** 0.5
        if z[0] >= 0:
            cos_theat = 1 / (z[0] * z[0] + 1) ** 0.5
        else:
            cos_theat = - 1 / (z[0] * z[0] + 1) ** 0.5

        point_box_length = (length[0] + length[1] - 1) * sin_theat
        box_length = round(max_dis + point_box_length)

        flag_x_y = []
        limbs_width = []
        for j in range(5):
            U = last_stage_paf_map[:, :, mapIdx[j][0]] * -1  # x
            V = last_stage_paf_map[:, :, mapIdx[j][1]]  # y
            M = np.zeros(U.shape, dtype='bool')
            M[abs(U) + abs(V)< uv_thre] = True

            limb_start = points[limbSeq[j][0]]
            limb_end = points[limbSeq[j][1]]
            if limb_start[0] == -1 or limb_end[0] == -1:
                continue

            def find_min_width(M, point):
                width = 0
                y = int(point[1])
                x = int(point[0])

                is_axis_y = True
                if abs(z[0]) >= 0 and abs(z[0]) <= 1:
                    is_axis_y = False
                x_right = copy.deepcopy(x)
                y_bottom = copy.deepcopy(y)
                x_left = copy.deepcopy(x)
                y_top = copy.deepcopy(y)
                while M[y_bottom][x_right] == False:
                    width += 1
                    if is_axis_y:
                        x_right += 1
                    else:
                        y_bottom += 1
                    if x_right >= ori_img.shape[1] or y_bottom >= ori_img.shape[0]:
                        break
                while M[y_top][x_left] == False:
                    width += 1
                    if is_axis_y:
                        x_left -= 1
                    else:
                        y_top -= 1
                    if x_left <= 0 or y_top <= 0:
                        break
                return width - 1, is_axis_y

            start_width, is_axis_y = find_min_width(M, limb_start[:2])
            flag_x_y.append(is_axis_y)
            end_width, is_axis_y = find_min_width(M, limb_end[:2])
            flag_x_y.append(is_axis_y)
            limb_width = (start_width + end_width) / 2
            limbs_width.append(limb_width)

        max_width = max(limbs_width)

        def compute_box_width(init_width, flag_x_y):
            count_true = 0

            for flag in flag_x_y:
                if flag:
                    count_true += 1
            if count_true > int(len(flag_x_y) / 2):

                res = round(init_width * sin_theat)
            else:
                res = round(abs(init_width * cos_theat))
            return res

        box_width = compute_box_width(max_width, flag_x_y)
        # print("box width:" + str(box_width))
        # print("box length:" + str(box_length))

        def is_rect(points):
            """

            :param points:
            :return:
            """
            def oushi2(point1, point2):
                return (point1[1] - point2[1]) ** 2 + (point1[0] - point2[0]) ** 2
            s1 = oushi2(points[1], points[0])
            s2 = oushi2(points[2], points[0])
            s3 = oushi2(points[3], points[0])
            max_s = max(s1, s2, s3)
            if (s1 + s2 + s3 - max_s)/max_s > 0.97:
                return True
            else:
                return False

        def find_fourth_points(points, z, box_width, length, max_dis):
            """
            根据区域边界点坐标，中轴线斜率，box 长宽确定四点坐标
            :param points: 所有点坐标
            :param z:  中轴线斜率
            :param box_width: box宽度
            :param length:  （length_right, length_left）
            :param max_dis:  (最远关键点的距离)
            :return:
            """
            box_points_index = list(np.argwhere(np.mat(dis_matrix_all) == max_dis)[0])
            start_point_0 = points[box_points_index[0]]
            end_point_0 = points[box_points_index[1]]

            y_start_range = (length[0]-1) * sin_theat * sin_theat # len * sin(theat)
            x_start_range = (length[0]-1) * cos_theat * sin_theat # len * cos(theat)
            y_end_range = (length[1]-1) * sin_theat * sin_theat  # len * sin(theat)
            x_end_range = (length[1]-1) * cos_theat * sin_theat  # len * cos(theat)
            start_box_mid_point = copy.deepcopy(start_point_0)
            end_box_mid_point = copy.deepcopy(end_point_0)

            start_box_mid_point[0] =start_point_0[1] - y_start_range
            start_box_mid_point[1] =start_point_0[0] - x_start_range
            end_box_mid_point[0] = end_point_0[1] + y_end_range
            end_box_mid_point[1] = end_point_0[0] + x_end_range

            def find_box(width, z, start_mid, end_mid):
                length_k = z[0]
                width_k = - 1.0 / length_k

                b_start = start_mid[0] - start_mid[1] * width_k
                b_end = end_mid[0] - end_mid[1] * width_k

                def find_point(k,b,mid_point):
                    A = k ** 2 + 1
                    B = 2 * k * b - 2 * k * mid_point[0] - 2 * mid_point[1]
                    C = (b - mid_point[0]) ** 2 + mid_point[1] ** 2 - width ** 2 / 4
                    x1 = (-B+math.sqrt(B**2-4*A*C))/(2*A)
                    x2 = (-B-math.sqrt(B**2-4*A*C))/(2*A)
                    y1 = x1 * k + b
                    y2 = x2 * k + b
                    return [[round(x1), round(y1)], [round(x2), round(y2)]]

                two_point = find_point(width_k, b_start, start_mid)
                fouth_point = find_point(width_k, b_end, end_mid)
                two_point.append(fouth_point[1])
                two_point.append(fouth_point[0])

                return two_point


            return find_box(box_width, z, start_box_mid_point, end_box_mid_point)

        box_4 = find_fourth_points(points, z, box_width, length, max_dis)
        box_4 = np.array(box_4).astype('int32')
        if not is_rect(box_4):
            continue
        else:
            print(box_4)
            print("box width:" + str(box_width))
            print("box length:" + str(box_length))

        dic = {}
        dic["bbox"] = box_4
        dic["score"] = subset[k][-2] / subset[k][-1]
        dics.append(dic)

        xs = box_4[:, 0]
        ys = box_4[:, 1]
        minx = int(round((sorted(xs)[0] + sorted(xs)[1]) / 2.0))
        maxx = int(round((sorted(xs)[2] + sorted(xs)[3]) / 2.0))
        miny = int(round((sorted(ys)[0] + sorted(ys)[1]) / 2.0))
        maxy = int(round((sorted(ys)[2] + sorted(ys)[3]) / 2.0))

        tradition_box_img = cv2.rectangle(tradition_box_img, (minx, miny), (maxx, maxy), (0, 0, 255), thickness=15,)

        box_img = cv2.polylines(box_img, [box_4], 1, (0, 0, 255), thickness=15)

    return tradition_box_img, box_img, dics, backbone_img
