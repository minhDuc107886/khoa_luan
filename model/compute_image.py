from libs import *
from compute_boxes import compute_iou

class ImageVisualizer(object):
    """
    
    """
    def __init__(self,idx_to_name,class_colors =None, save_dir = None):
        self.idx_to_name = idx_to_name
        if class_colors is None or len(class_colors) != len(self.idx_to_name):
            self.class_colors = [[0,255,0]]*len(self.idx_to_name)
        else:
            self.class_colors = class_colors
        
        if save_dir is None:
            self.save_dir = "./"
        else:
            self.save_dir = save_dir

        os.makedirs(self.save_dir,exist_ok= True)

    
    def save_image(self, img, boxes, labels, name):

        """
        draw boxes and labels then save to dir
        args: 
            img: numpy array (w,h,3)
            boxes: numpy array(num_boxes,4) xmin,ymin,xmax,ymax
            labels: numpyarray( len == numboxes)
            name: name of image
        """
        plt.figure()
        fig,ax = plt.subplot(1)
        ax.imshow(img)
        save_path = os.path.join(self.save_dir,name)

        for i,box in enumerate(boxes):
            idx = labels[i]-1
            cls_name = self.idx_to_name[idx]
            top_left = (boxes[0],boxes[1])
            bot_right = (boxes[2],boxes[3])
            ax.add_patch(patches.Rectangle(
                (box[0],box[1]),
                box[2]-box[0],box[3]-box[1],
                linewidth = 1,edgecolor = (0.,1.,0.),facecolor = "none")
                )
            plt.text(box[0],box[1],s=cls_name,color = "white",verticalalignment = "top",bbox = {"color":(0.,1.,0.),"pad":0})

        plt.axis("off")
        plt.savefig(save_path,bbox_inches="tight",pad_inches=0.0)
        plt.close("all")

    



