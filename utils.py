import sys
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
import xml.etree.ElementTree as ET
from PIL import Image, ImageDraw
from keras.utils import to_categorical


def scale_proposals(proposals,scale):
  new_proposals = np.zeros(proposals.shape)
  # invert x:
  new_proposals[:,:] = scale*proposals[:,:]
  return new_proposals

  
def mirror_proposals(proposals):
  new_proposals = np.zeros(proposals.shape)
  # invert x:
  new_proposals[:,0] = 1024 - proposals[:,2] -1
  new_proposals[:,2] = 1024 - proposals[:,0] -1
  # y unchanges:
  new_proposals[:,3] = proposals[:,3]
  new_proposals[:,1] = proposals[:,1]
  return new_proposals

def flip_proposals(proposals):
  new_proposals = np.zeros(proposals.shape)
  # x unchanged:
  new_proposals[:,0] = proposals[:,0]
  new_proposals[:,2] = proposals[:,2]
  # invert y:
  new_proposals[:,1] = 1024 - proposals[:,3] -1
  new_proposals[:,3] = 1024 - proposals[:,1] -1
  return new_proposals


def xywh_xyXY(true_bboxes):
  true_bboxes_aux=np.zeros(true_bboxes.shape)
  true_bboxes_aux[:,2] = true_bboxes[:,2]+true_bboxes[:,0] - 1
  true_bboxes_aux[:,3] = true_bboxes[:,3]+true_bboxes[:,1] - 1
  true_bboxes_aux[:,0] = true_bboxes[:,0] 
  true_bboxes_aux[:,1] = true_bboxes[:,1] 
  return true_bboxes_aux

def yx_exchange( box ):
  box_new = np.zeros(box.shape) # Exchanging x and y for image.crop_and_resize
  box_new[0] = box[1]
  box_new[1] = box[0]
  box_new[2] = box[3]
  box_new[3] = box[2]
  return box_new

def extract_crop(roi):
  crop = np.zeros(roi.shape) # Exchanging x and y for image.crop_and_resize
  crop[0] = roi[1]/image_size  
  crop[1] = roi[0]/image_size
  crop[2] = roi[3]/image_size
  crop[3] = roi[2]/image_size
  return crop

def extract_crop(roi):
  crop = np.zeros(roi.shape) # Exchanging x and y for image.crop_and_resize
  crop[0] = roi[1]/image_size  
  crop[1] = roi[0]/image_size
  crop[2] = roi[3]/image_size
  crop[3] = roi[2]/image_size
  return crop


def draw_anchors_from_image(im, anchors, pad_size=50):
    w,h=im.size
    a4im = Image.new('RGB',
                    (w+2*pad_size, h+2*pad_size),   # A4 at 72dpi
                    (255, 255, 255))  # White
    a4im.paste(im, (pad_size,pad_size))  # Not centered, top-left corner
    for a in anchors:
        a=(a+pad_size).astype(int).tolist()
        draw = ImageDraw.Draw(a4im)
        draw.rectangle(a,outline=(255,0,0), fill=None,width=1)
    return a4im

def draw_anchors_from_image_scores(im, anchors,scores, pad_size=50, FG_TH = 0.7, BG_TH = 0.3):
    w,h=im.size
    a4im = Image.new('RGB',
                    (w+2*pad_size, h+2*pad_size),   # A4 at 72dpi
                    (255, 255, 255))  # White
    a4im.paste(im, (pad_size,pad_size))  # Not centered, top-left corner
    for i in range(len(anchors)):
        a = anchors[i]
        s = scores[i]
        a=(a+pad_size).astype(int).tolist()
        draw = ImageDraw.Draw(a4im)
        #score_aux = round(10000*(s[0]-0.99),2)
        score_aux = round(s,2)
        if score_aux>=FG_TH:
          draw.rectangle(a,outline=(255,0,0), fill=None,width=3)
          draw.text((a[0], a[1]-3), str(score_aux))
        elif score_aux<BG_TH:
          draw.rectangle(a,outline=(0,255,0), fill=None,width=3)
          draw.text((a[0], a[1]-3), str(score_aux))
        else:
          draw.rectangle(a,outline=(0,0,255), fill=None,width=1)
          draw.text((a[0], a[1]-3), str(score_aux)) 
    return a4im


def enlarge(true_bboxes, bbox_padding = 15):
  FOV_size = 280
  
  gt_bboxes_w = true_bboxes[:,2] - true_bboxes[:,0] + 1
  gt_bboxes_h = true_bboxes[:,3] - true_bboxes[:,1] + 1

  #only enlarge bboxes smaller than FOV
  enlarged_bboxes = np.zeros(true_bboxes.shape)
  condition = (gt_bboxes_w < FOV_size) & (gt_bboxes_h < FOV_size)
  enlarged_bboxes[:,0] = np.where(condition  , true_bboxes[:,0]-bbox_padding , true_bboxes[:,0])
  enlarged_bboxes[:,2] = np.where(condition , true_bboxes[:,2]+bbox_padding , true_bboxes[:,2])

  enlarged_bboxes[:,1] = np.where(condition  , true_bboxes[:,1]-bbox_padding , true_bboxes[:,1])
  enlarged_bboxes[:,3] = np.where(condition  , true_bboxes[:,3]+bbox_padding , true_bboxes[:,3])

  # Make sure enlarged objects fit FOV
  bboxes_w = enlarged_bboxes[:,2] - enlarged_bboxes[:,0] + 1
  bboxes_h = enlarged_bboxes[:,3] - enlarged_bboxes[:,1] + 1

  condition_w = (bboxes_w>FOV_size)
  condition_h = (bboxes_h>FOV_size)

  delta_w = enlarged_bboxes[condition_w,2] - enlarged_bboxes[condition_w,0] + 1 - FOV_size
  delta_h = enlarged_bboxes[condition_h,3] - enlarged_bboxes[condition_h,1] + 1 - FOV_size

  enlarged_bboxes_v2 = np.zeros(true_bboxes.shape)
  enlarged_bboxes_v2[:,:] = enlarged_bboxes[:,:]
  enlarged_bboxes_v2[condition_w,0] = enlarged_bboxes_v2[condition_w,0] + delta_w/2
  enlarged_bboxes_v2[condition_h,1] = enlarged_bboxes_v2[condition_h,1] + delta_h/2
  enlarged_bboxes_v2[condition_w,2] = enlarged_bboxes_v2[condition_w,2] - delta_w/2
  enlarged_bboxes_v2[condition_h,3] = enlarged_bboxes_v2[condition_h,3] - delta_h/2

  del enlarged_bboxes
  del gt_bboxes_w
  del gt_bboxes_h
  del bboxes_w
  del bboxes_h
  del delta_w
  del delta_h

  return enlarged_bboxes_v2

# Huber loss
def huberGTZ(y_true, y_pred):
    nd=tf.where(tf.not_equal(y_true,0))
    y_true=tf.gather_nd(y_true,nd)
    y_pred=tf.gather_nd(y_pred,nd)
    h = tf.keras.losses.Huber()
    return h(y_true, y_pred)

def loss_cls(y_true, y_pred):
    condition = K.not_equal(y_true, -1)
    indices = tf.where(condition)

    target = tf.gather_nd(y_true, indices)
    output = tf.gather_nd(y_pred, indices)

    loss = K.binary_crossentropy(target, output)
    return K.mean(loss)


def smoothL1(y_true, y_pred):
    nd=tf.where(tf.not_equal(y_true,0))
    y_true=tf.gather_nd(y_true,nd)
    y_pred=tf.gather_nd(y_pred,nd)
    x = tf.keras.losses.Huber(y_true,y_pred)
#     x   = K.switch(x < HUBER_DELTA, 0.5 * x ** 2, HUBER_DELTA * (x - 0.5 * HUBER_DELTA))
    return x


def draw_anchors(img_path, anchors, pad_size=50):
    im = Image.open(img_path)
    w,h=im.size
    a4im = Image.new('RGB',
                    (w+2*pad_size, h+2*pad_size),   # A4 at 72dpi
                    (255, 255, 255))  # White
    a4im.paste(im, (pad_size,pad_size))  # Not centered, top-left corner
    for a in anchors:
        a=(a+pad_size).astype(int).tolist()
        draw = ImageDraw.Draw(a4im)
        draw.rectangle(a,outline=(255,0,0), fill=None,width=3)
    return a4im

def generate_anchors(base_width=16, base_height=16, ratios=[0.6,1,2], scales=np.asarray([1,1.5,2])):
    """
    Generate anchor (reference) windows by enumerating aspect ratios X
    scales wrt a reference (0, 0, w_stride-1, h_stride-1) window.
    """

    base_anchor = np.array([1, 1, base_width, base_height]) - 1
    ratio_anchors = _ratio_enum(base_anchor, ratios)
    anchors = np.vstack([_scale_enum(ratio_anchors[i, :], scales)
                         for i in range(ratio_anchors.shape[0])])
    return anchors

def _whctrs(anchor):
    """
    Return width, height, x center, and y center for an anchor (window).
    """

    w = anchor[2] - anchor[0] + 1
    h = anchor[3] - anchor[1] + 1
    x_ctr = anchor[0] + 0.5 * (w - 1)
    y_ctr = anchor[1] + 0.5 * (h - 1)
    return w, h, x_ctr, y_ctr

def _mkanchors(ws, hs, x_ctr, y_ctr):
    """
    Given a vector of widths (ws) and heights (hs) around a center
    (x_ctr, y_ctr), output a set of anchors (windows).
    """

    ws = ws[:, np.newaxis]
    hs = hs[:, np.newaxis]
    anchors = np.hstack((x_ctr - 0.5 * (ws - 1),
                         y_ctr - 0.5 * (hs - 1),
                         x_ctr + 0.5 * (ws - 1),
                         y_ctr + 0.5 * (hs - 1)))
    return anchors

def _ratio_enum(anchor, ratios):
    """
    Enumerate a set of anchors for each aspect ratio wrt an anchor.
    """

    w, h, x_ctr, y_ctr = _whctrs(anchor)
    size = w * h
    size_ratios = size / ratios
    ws = np.round(np.sqrt(size_ratios))
    hs = np.round(ws * ratios)
    anchors = _mkanchors(ws, hs, x_ctr, y_ctr)
    return anchors

def _scale_enum(anchor, scales):
    """
    Enumerate a set of anchors for each scale wrt an anchor.
    """

    w, h, x_ctr, y_ctr = _whctrs(anchor)
    ws = w * scales
    hs = h * scales
    anchors = _mkanchors(ws, hs, x_ctr, y_ctr)
    return anchors

def clip_boxes(boxes, im_shape):
    """
    Clip boxes to image boundaries.
    """

    # x1 >= 0
    boxes[:, 0::4] = np.maximum(np.minimum(boxes[:, 0::4], im_shape[1] - 1), 0)
    # y1 >= 0
    boxes[:, 1::4] = np.maximum(np.minimum(boxes[:, 1::4], im_shape[0] - 1), 0)
    # x2 < im_shape[1]
    boxes[:, 2::4] = np.maximum(np.minimum(boxes[:, 2::4], im_shape[1] - 1), 0)
    # y2 < im_shape[0]
    boxes[:, 3::4] = np.maximum(np.minimum(boxes[:, 3::4], im_shape[0] - 1), 0)
    return boxes

def filter_boxes(boxes, min_size):
    """Remove all boxes with any side smaller than min_size."""
    ws = boxes[:, 2] - boxes[:, 0] + 1
    hs = boxes[:, 3] - boxes[:, 1] + 1
    keep = np.where((ws >= min_size) & (hs >= min_size))[0]
    return keep

def py_cpu_nms(dets, thresh):
    """Pure Python NMS baseline."""
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0] # index for first(highest score) entry
        keep.append(i) # keep it (high score)
        xx1 = np.maximum(x1[i], x1[order[1:]]) #find the maximun coord of all the boxes
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]]) #
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1) #maximum sizes
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h 
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]

        order = order[inds + 1]

    return keep

 
def bbox_transform(ex_rois, gt_rois):
    ex_widths = ex_rois[:, 2] - ex_rois[:, 0] + 1.0
    ex_heights = ex_rois[:, 3] - ex_rois[:, 1] + 1.0
    ex_ctr_x = ex_rois[:, 0] + 0.5 * ex_widths
    ex_ctr_y = ex_rois[:, 1] + 0.5 * ex_heights

    gt_widths = gt_rois[:, 2] - gt_rois[:, 0] + 1.0
    gt_heights = gt_rois[:, 3] - gt_rois[:, 1] + 1.0
    gt_ctr_x = gt_rois[:, 0] + 0.5 * gt_widths
    gt_ctr_y = gt_rois[:, 1] + 0.5 * gt_heights

    targets_dx = (gt_ctr_x - ex_ctr_x) / ex_widths
    targets_dy = (gt_ctr_y - ex_ctr_y) / ex_heights
    targets_dw = np.log(gt_widths / ex_widths)
    targets_dh = np.log(gt_heights / ex_heights)

    targets = np.stack((targets_dx, targets_dy, targets_dw, targets_dh))

    targets = np.transpose(targets)

    return targets


def bbox_transform_inv(boxes, deltas):
    if boxes.shape[0] == 0:
        return np.zeros((0, deltas.shape[1]), dtype=deltas.dtype)

    boxes = boxes.astype(deltas.dtype, copy=False)

    widths = boxes[:, 2] - boxes[:, 0] + 1.0
    heights = boxes[:, 3] - boxes[:, 1] + 1.0
    ctr_x = boxes[:, 0] + 0.5 * widths
    ctr_y = boxes[:, 1] + 0.5 * heights

    dx = deltas[:, 0::4]
    dy = deltas[:, 1::4]
    dw = deltas[:, 2::4]
    dh = deltas[:, 3::4]

    pred_ctr_x = dx * widths[:, np.newaxis] + ctr_x[:, np.newaxis]
    pred_ctr_y = dy * heights[:, np.newaxis] + ctr_y[:, np.newaxis]
    pred_w = np.exp(dw) * widths[:, np.newaxis]
    pred_h = np.exp(dh) * heights[:, np.newaxis]

    pred_boxes = np.zeros(deltas.shape, dtype=deltas.dtype)
    # x1
    pred_boxes[:, 0::4] = pred_ctr_x - 0.5 * pred_w
    # y1
    pred_boxes[:, 1::4] = pred_ctr_y - 0.5 * pred_h
    # x2
    pred_boxes[:, 2::4] = pred_ctr_x + 0.5 * pred_w
    # y2
    pred_boxes[:, 3::4] = pred_ctr_y + 0.5 * pred_h

    return pred_boxes

def bbox_overlaps(boxes, query_boxes):
    """
    Parameters
    ----------
    boxes: (N, 4) ndarray of float
    query_boxes: (K, 4) ndarray of float
    Returns
    -------
    overlaps: (N, K) ndarray of overlap between boxes and query_boxes
    """
    boxes=boxes.astype(int)
    N = boxes.shape[0]
    K = query_boxes.shape[0]

    overlaps = np.zeros((N, K), dtype=np.float)

    for k in range(K):
        box_area = ((query_boxes[k, 2] - query_boxes[k, 0] + 1) * (query_boxes[k, 3] - query_boxes[k, 1] + 1))
        for n in range(N):
            iw = (min(boxes[n, 2], query_boxes[k, 2]) - max(boxes[n, 0], query_boxes[k, 0]) + 1)
            if iw > 0:
                ih = (min(boxes[n, 3], query_boxes[k, 3]) - max(boxes[n, 1], query_boxes[k, 1]) + 1)

                if ih > 0:
                    ua = float((boxes[n, 2] - boxes[n, 0] + 1) * (boxes[n, 3] - boxes[n, 1] + 1) + box_area - iw * ih)
                    overlaps[n, k] = iw * ih / ua

    return overlaps

def unmap(data, count, inds, fill=0):
    """ Unmap a subset of item (data) back to the original set of items (of
    size count) """
    if len(data.shape) == 1:
        ret = np.empty((count, ), dtype=np.float32)
        ret.fill(fill)
        ret[inds] = data
    else:
        ret = np.empty((count, ) + data.shape[1:], dtype=np.float32)
        ret.fill(fill)
        ret[inds, :] = data
    return ret
