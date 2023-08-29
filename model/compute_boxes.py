""" box_ultis
 compute  area, iou
 compute encode,decode
 compute Non maximum suppresion
 compute predict

 """

from libs import *

def compute_area(top_left,bot_right):

    '''
    args:
        top_left : tensor(num_boxes,2)
        bot_right: tensor(num_boxes,2)
    returns:
        area: tensor(num_boxes,2)
    '''
    hw = tf.clip_by_value(bot_right-top_left,0.0,512.0) # if value < 0.0 -> value = 0.0; if value > 512.0  -> value = 512
    area = hw[...,0]*hw[...,1]
    
def compute_iou(boxes_a, boxes_b):
    '''
    args:
        boxes_a: tensor(num_boxes,4)
        boxes_b: tensor(num_boxes,4)
    return -> tensor(num_boxes_a,num_boxes_b)
    '''
    boxes_a = tf.expand_dims(boxes_a,1) #-> num_boxes_a, 1, 4
    boxes_b = tf.expand_dims(boxes_b,0) # -> 1, num_boxes_b, 4
    top_left = tf.math.maximum(boxes_a[...,:2],boxes_b[...,:2])
    bot_right = tf.math.minimum(boxes_a[...,2:],boxes_b[...,2:])
    
    overlap_area = compute_area(top_left,bot_right)
    area_a = compute_area(boxes_a[...,:2],boxes_a[...,2:])
    area_b = compute_area(boxes_b[...,:2],boxes_b[...,2:])

    overlap = overlap_area / (area_a + area_b - overlap_area)

    return overlap


def transform_center_to_corner(boxes,data_type = True):
    """
        if box type default_box(cx,cy,w,h)
            Transform boxes of format (cx, cy, w, h)  to format (xmin, ymin, xmax, ymax)
        else:
            return boxes
    """
    if data_type == True:
        corner_box = tf.concat([
            boxes[...,:2] - boxes[...,2:]/2,
            boxes[...,:2] + boxes[...,2:]/2],
            axis=-1
        )
        return  corner_box
    else :
        return boxes


def encode(default_boxes,boxes,variance=[0.1,0.2]):

    """
        default_boxes: cx,cy,w,h
        boxes: cx,cy,w,h
        variance: phuong sai cho center va size
        -> locs: regisssion values, tensor
    """
    transformed_boxes = transform_center_to_corner(boxes)

    locs =  tf.concat([
        (transformed_boxes[..., :2] - default_boxes[:, :2]
         ) / (default_boxes[:, 2:] * variance[0]),
        tf.math.log(transformed_boxes[..., 2:] / default_boxes[:, 2:]) / variance[1]],
        axis = -1)
    return locs

def decode(default_boxes,locs,variance = [0.1,0.2]):
    """
    default_boxes: tensor(num_default,4) of format (cx,cy,w,h)
    locs: tensor(batch_size, num_default,4)   of format (cx,cy,w,h)
    variance: phuong sai cho center va size

    return -> boxes: tensor (num_default,4) of format(xmin,ymin,xmax,ymax) 
    """

    locs = tf.concat([
        locs[...,2]*variance[0]*default_boxes[:,2:]+default_boxes[:,:2],
        tf.math.exp(locs[...,2:]*variance[1])*default_boxes[:,2:]],
        axis=-1
    )
    boxes = transform_center_to_corner(locs)

    return boxes

def compute_target(default_boxes, gt_boxes, gt_labels, iou_threshold=0.5):
    """
    Args:
        default_boxes : tensor(num_default_box,4) <=> x_center,y_center,w,h
        gt_boxes: tensor(num_gt,4) <=> xmin,ymin,xmax,ymax
        gt_labels: tensor(num_gt,)
    
    return:->
        gt_confs: classification targets, tensor (num_default,)
        gt_locs: regression targets, tensor (num_default, 4)
    
    """

    tranformed_default_box = transform_center_to_corner(default_boxes)
    iou = compute_iou(tranformed_default_box,gt_boxes)
    best_default_iou = tf.math.reduce_max(iou,0)
    best_defaut_idx = tf.math.reduce_max(iou,0)

    best_gt_idx = tf.tensor_scatter_nd_update(
        best_gt_idx,
        tf.expand_dims(best_defaut_idx,1),
        tf.range(best_defaut_idx.shape[0],dtype=tf.int64)
    )

    best_gt_iou = tf.tensor_scatter_nd_update(
        best_gt_iou,
        tf.expand_dims(best_defaut_idx,1),
        tf.ones_like(best_defaut_idx,dtype=tf.float32)
    )

    gt_confs = tf.gather(gt_labels,best_gt_idx)
    gt_locs = encode(default_boxes,gt_boxes)

    return gt_confs,gt_locs


def compute_nms(boxes,scores,nms_threshold, limit = 200):
    """
    Non Maximum Suppression
        
        boxes: tensor(num_boxes,4) of format (xmin,ymin,xmax,ymax)
        scores: tensor(numboxes,)
        nms_threshold: NMS threshold
        limit: maximum number of boxes to help
        
    returns:
        idx: indices of kelp boxes
    """
    if boxes.shape[0] == 0:
        return tf.constant([],dtype = tf.int32)
    selected = [0]
    idx = tf.argsort(scores,direction= "DESCENDING")
    idx = idx[:limit]
    boxes = tf.gather(boxes,idx)
    iou = compute_iou(boxes,boxes)

    while True:
        row = iou[selected[-1]]
        next_indices = row <= nms_threshold

        iou = tf.where(
            tf.expand_dims(tf.math.logical_not(next_indices),0),
            tf.ones_like(iou,dtype = tf.float32),
            iou
        )

        if not tf.math.reduce_any(next_indices):
            break
        selected.append(tf.argsort(tf.dtypes.cast(next_indices,tf.int32),direction="DESENDING")[0].numpy())

    return tf.gather(idx,selected)
    
    

        








