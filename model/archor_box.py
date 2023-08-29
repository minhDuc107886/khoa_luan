''' create default_boxes'''

from libs import *

def generate_default_boxes(config):
    """
        config:information of feature maps
        scales: boxes'size ralative to image's size
        fm_sizes: size of feature maps
        ratios: box ratios used in each feature maps
        returns:
        default_boxes: tensor of shape(num_default,4)
                with format (cx,xy,w,h)
    """

    default_boxes = []
    scales = config['scales']
    fm_sizes = config['fm_sizes']
    ratios = config['ratios']
    

    for m,fm_sizes in enumerate(fm_sizes):
        for i,j in itertools.product(range(fm_sizes),repeat=2):
            cx = (j+0.5)/fm_sizes
            cy = (i+0.5)/fm_sizes

            default_boxes.append(
                [cx,cy,scales[m],scales[m]]
            )
            
            default_boxes.append([
                    cx,cy,
                    math.sqrt(scales[m]*scales[m+1]),
                    math.sqrt(scales[m]*scales[m+1])
                 ])
            
            for ratio in ratios[m]:
                r = math.sqrt(ratio)

                default_boxes.append([
                    cx,cy,
                    scales[m]*r,
                    scales[m]/r
                ])

                default_boxes.append([
                    cx,cy,
                    scales[m]/r,
                    scales[m]*r
                ])

    default_boxes = tf.constant(default_boxes)
    default_boxes = tf.clip_by_value(default_boxes,0.0,1.0)

    return default_boxes

if __name__=="__main__":
    config = {
        'ratios': [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
        'scales': [0.1, 0.2, 0.375, 0.55, 0.725, 0.9, 1.075],
        'fm_sizes': [38, 19, 10, 5, 3, 1],
        'image_size': 300
        }
    
    default_boxes_ssd300 = generate_default_boxes(config)
    print( f"GPU available: {tf.test.is_gpu_available()}")
    print(len(default_boxes_ssd300)) #8732 boxes
