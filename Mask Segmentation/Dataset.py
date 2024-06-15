import os
import torch
from torchvision import tv_tensors
from torch.utils.data import Dataset
from torchvision.io import read_image, ImageReadMode
from torchvision.ops.boxes import masks_to_boxes
from torchvision.transforms.v2 import functional as F

class CustomeDataSet(Dataset):
    def __init__(self,root,transforms):
        self.root = root
        self.transforms = transforms
         # load all image files, sorting them
        self.images = list(sorted(os.listdir(os.path.join(root,"Path_to_images"))))
        self.masks = list(sorted(os.listdir(os.path.join(root,"Path_to_masks"))))

    def __getitem__(self, index):
        # Load images and masks
        image_path = os.path.join(self.root, "Path_to_images", self.images[index])
        mask_path = os.path.join(self.root, "Path_to_masks", self.masks[index])     
        image = read_image(image_path)
        # Load mask images in grayscal.
        mask = read_image(mask_path, mode=ImageReadMode.GRAY)
        # Instances are encoded as different colors
        object_ids = torch.unique(mask)
        # First id is the background, so remove it
        object_ids = object_ids[1:]
        num_objects = len(object_ids)
        # split the color-encoded mask into a set of binary masks

        masks = (mask == object_ids[:, None, None]).to(dtype=torch.uint8)
        # Get bounding box coordinates for each mask.
        boxes = masks_to_boxes(masks)
         # There is only one class.
        
        labels = torch.ones((num_objects,), dtype=torch.int64)
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # Suppose all instances are not crowd.
       
        iscrowd = torch.zeros((num_objects,), dtype=torch.int64)
        # Wrap sample and targets into torchvision tv_tensors:
        image = tv_tensors.Image(image)
        target = {}
        target["boxes"] = tv_tensors.BoundingBoxes(boxes,format="XYXY",canvas_size = F.get_size(image))
        target["masks"] = tv_tensors.Mask(masks)
        target["labels"] = labels 
        target["image_id"] = index
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            image, target = self.transforms(image,target)
        return image, target

    def __len__(self):
        return len(self.images)