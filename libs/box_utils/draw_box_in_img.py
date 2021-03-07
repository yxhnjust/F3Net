# -*- coding: utf-8 -*-

from __future__ import absolute_import, print_function, division

import numpy as np

from PIL import Image, ImageDraw, ImageFont
import cv2

from libs.configs import cfgs
from libs.label_name_dict.label_dict import LABEL_NAME_MAP
NOT_DRAW_BOXES = 0
ONLY_DRAW_BOXES = -1
ONLY_DRAW_BOXES_WITH_SCORES = -2


FONT = ImageFont.load_default()


def draw_a_rectangel_in_img(draw_obj, box, color, width, method):
    '''
    use draw lines to draw rectangle. since the draw_rectangle func can not modify the width of rectangle
    :param draw_obj:
    :param box: [x1, y1, x2, y2]
    :return:
    '''
    # color = (0, 255, 0)
    if method == 0:
        x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
        top_left, top_right = (x1, y1), (x2, y1)
        bottom_left, bottom_right = (x1, y2), (x2, y2)

        draw_obj.line(xy=[top_left, top_right],
                      fill=color,
                      width=width)
        draw_obj.line(xy=[top_left, bottom_left],
                      fill=color,
                      width=width)
        draw_obj.line(xy=[bottom_left, bottom_right],
                      fill=color,
                      width=width)
        draw_obj.line(xy=[top_right, bottom_right],
                      fill=color,
                      width=width)
    else:
        x_c, y_c, w, h, theta = box[0], box[1], box[2], box[3], box[4]
        rect = ((x_c, y_c), (w, h), theta)
        rect = cv2.boxPoints(rect)
        rect = np.int0(rect)
        draw_obj.line(xy=[(rect[0][0], rect[0][1]), (rect[1][0], rect[1][1])],
                      fill=color,
                      width=width)
        draw_obj.line(xy=[(rect[1][0], rect[1][1]), (rect[2][0], rect[2][1])],
                      fill=color,
                      width=width)
        draw_obj.line(xy=[(rect[2][0], rect[2][1]), (rect[3][0], rect[3][1])],
                      fill=color,
                      width=width)
        draw_obj.line(xy=[(rect[3][0], rect[3][1]), (rect[0][0], rect[0][1])],
                      fill=color,
                      width=width)


def draw_a_quad_in_img(draw_obj, box, width):
    '''
    use draw lines to draw rectangle. since the draw_rectangle func can not modify the width of rectangle
    :param draw_obj:
    :param box: [x1, y1, x2, y2]
    :return:
    '''
    point1, point2 = (box[0], box[1]), (box[2], box[3])
    point3, point4 = (box[4], box[5]), (box[6], box[7])

    draw_obj.line(xy=[point1, point2],
                  fill=(255, 0, 0),
                  width=width)
    draw_obj.line(xy=[point2, point3],
                  fill=(0, 255, 0),
                  width=width)
    draw_obj.line(xy=[point3, point4],
                  fill=(0, 0, 255),
                  width=width)
    draw_obj.line(xy=[point4, point1],
                  fill=(255, 255, 0),
                  width=width)


def only_draw_scores(draw_obj, box, score, color):

    x, y = box[0], box[1]
    draw_obj.rectangle(xy=[x, y, x+60, y+10],
                       fill=color)
    draw_obj.text(xy=(x, y),
                  text="obj:" + str(round(score, 2)),
                  fill='black',
                  font=FONT)


def draw_label_with_scores(draw_obj, box, label, score, color):
    x, y = box[0], box[1]
    draw_obj.rectangle(xy=[x, y, x + 60, y + 10],
                       fill=color)

    txt = LABEL_NAME_MAP[label] + ':' + str(round(score, 2))
    draw_obj.text(xy=(x, y),
                  text=txt,
                  fill='black',
                  font=FONT)


def draw_boxes_with_label_and_scores(img_array, boxes, labels, scores, method, in_graph=True, is_quad=False):
    if in_graph:
        if cfgs.NET_NAME in ['resnet152_v1d', 'resnet101_v1d', 'resnet50_v1d']:
            img_array = (img_array * np.array(cfgs.PIXEL_STD) + np.array(cfgs.PIXEL_MEAN_)) * 255
        else:
            img_array = img_array + np.array(cfgs.PIXEL_MEAN)
    img_array.astype(np.float32)
    boxes = boxes.astype(np.int64)
    labels = labels.astype(np.int32)
    img_array = np.array(img_array * 255 / np.max(img_array), dtype=np.uint8)

    img_obj = Image.fromarray(img_array)
    raw_img_obj = img_obj.copy()

    draw_obj = ImageDraw.Draw(img_obj)
    num_of_objs = 0
    for box, a_label, a_score in zip(boxes, labels, scores):

        if a_label != NOT_DRAW_BOXES:
            num_of_objs += 1
            if is_quad:
                draw_a_quad_in_img(draw_obj, box, width=3)
            else:
                draw_a_rectangel_in_img(draw_obj, box, color='red', width=3, method=method)
            if a_label == ONLY_DRAW_BOXES:  # -1
                continue
            elif a_label == ONLY_DRAW_BOXES_WITH_SCORES:  # -2
                 only_draw_scores(draw_obj, box, a_score, color='White')
                 continue
            else:
                draw_label_with_scores(draw_obj, box, a_label, a_score, color='White')

    out_img_obj = Image.blend(raw_img_obj, img_obj, alpha=0.7)

    return np.array(out_img_obj)


def draw_boxes(img_array, boxes, labels, scores, color, method, in_graph=True, is_quad=False):
    if in_graph:
        if cfgs.NET_NAME in ['resnet152_v1d', 'resnet101_v1d', 'resnet50_v1d']:
            img_array = (img_array * np.array(cfgs.PIXEL_STD) + np.array(cfgs.PIXEL_MEAN_)) * 255
        else:
            img_array = img_array + np.array(cfgs.PIXEL_MEAN)
    img_array.astype(np.float32)
    boxes = boxes.astype(np.int64)
    labels = labels.astype(np.int32)
    img_array = np.array(img_array * 255 / np.max(img_array), dtype=np.uint8)

    img_obj = Image.fromarray(img_array)
    raw_img_obj = img_obj.copy()

    draw_obj = ImageDraw.Draw(img_obj)
    num_of_objs = 0
    for box, a_label, a_score in zip(boxes, labels, scores):

        if a_label != NOT_DRAW_BOXES:
            num_of_objs += 1
            if is_quad:
                draw_a_quad_in_img(draw_obj, box, width=3)
            else:
                draw_a_rectangel_in_img(draw_obj, box, color=color, width=3, method=method)
            # draw_a_rectangel_in_img(draw_obj, box, color=STANDARD_COLORS[1], width=3, method=method)
            if a_label == ONLY_DRAW_BOXES:  # -1
                continue
            elif a_label == ONLY_DRAW_BOXES_WITH_SCORES:  # -2
                 only_draw_scores(draw_obj, box, a_score, color='White')
                 continue
            else:
                draw_label_with_scores(draw_obj, box, a_label, a_score, color='White')

    out_img_obj = Image.blend(raw_img_obj, img_obj, alpha=0.7)

    return np.array(out_img_obj)







