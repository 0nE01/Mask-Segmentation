# Modules
import torch
import torchvision
import utils
from Dataset import CustomeDataSet
from torch.utils.data import DataLoader
from torchvision.io import read_image
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.transforms import v2 as T
from engine import train_one_epoch, evaluate
from torchvision.utils import  draw_segmentation_masks, draw_bounding_boxes
import matplotlib.pyplot as plt
# Device
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# You Can Download required torch packages with links below

# wget https://raw.githubusercontent.com/pytorch/vision/main/references/detection/engine.py
# wget https://raw.githubusercontent.com/pytorch/vision/main/references/detection/utils.py
# wget https://raw.githubusercontent.com/pytorch/vision/main/references/detection/coco_utils.py
# wget https://raw.githubusercontent.com/pytorch/vision/main/references/detection/coco_eval.py
# wget https://raw.githubusercontent.com/pytorch/vision/main/references/detection/transforms.py



def get_model(num_classes: int):
    # Load an instance segmentation model pre-trained .
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights="DEFAULT")
    # Get number of input features for the classifier.
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # Replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    # Now get the number of input features for the mask classifier.
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_leayers = 256
    # Replace the mask predictor with a new one.
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,hidden_leayers,num_classes)
    return model


def get_transforms(train: bool):
    # Create a transforms compose.
    transforms_list = []
    if train :
        transforms_list.append(T.RandomHorizontalFlip(0.5))
    transforms_list.append(T.ToDtype(torch.float,scale=True))
    transforms_list.append(T.ToPureTensor())
    return T.Compose(transforms_list)


# Create two dataset and split it for train and test.
train_dataset = CustomeDataSet("path_to_dataset",get_transforms(train=True))
test_dataset = CustomeDataSet("path_to_dataset",get_transforms(train=False))
indices = torch.randperm(len(train_dataset)).tolist()
train_dataset = torch.utils.data.Subset(train_dataset, indices[:-50])
test_dataset = torch.utils.data.Subset(test_dataset, indices[-50:])

# Dataloaders.
train_dataloader = DataLoader(
    train_dataset,
    batch_size=2,
    shuffle=True,
    collate_fn=utils.collate_fn
)

test_dataloader = DataLoader(
    test_dataset,
    batch_size=1,
    shuffle=False,
    collate_fn=utils.collate_fn
)


model = get_model(2)
model.to(device)

# SGD optimizer.
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(
    params,
    lr=0.005,
    momentum=0.9,
    weight_decay=0.0005
)
# lr_scheduler
lr_scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer,
    step_size=3,
    gamma=0.1
)

# Training loop.
for epoch in range(10):
    # Train for one epoch, printing every 10 iterations.
    train_one_epoch(model, optimizer, train_dataloader, device, epoch, print_freq=10)
    # Update the learning rate.
    lr_scheduler.step()
   # Evaluate on the test dataset.
    # If your using colab you can delete this line of code
    # It's not necessary and can effect your ram usage and maybe crash colab notebook.
    evaluate(model, test_dataloader, device=device)

# Get transform for evaluation.
eval_transform = get_transforms(train=False)

image = read_image("path_to_image")
model.eval()
with torch.inference_mode():
    x = eval_transform(image)
    x = x[:3, ...].to(device)
    predictions = model([x, ])
   
    con = predictions[0]["scores"] > 0.9
    predictions[0]["boxes"] = predictions[0]["boxes"][con]
    predictions[0]["labels"] = predictions[0]["labels"][con]
    predictions[0]["scores"] = predictions[0]["scores"][con]
    predictions[0]["masks"] = predictions[0]["masks"][con]
    pred = predictions[0]

pred_labels = [f"confidence: {score:.3f}" for label, score in zip(pred["labels"], pred["scores"])]
pred_boxes = pred["boxes"].long()
output_image = draw_bounding_boxes(image, pred_boxes, pred_labels, colors="red")
masks = (pred["masks"] > 0.7).squeeze(1)
output_image = draw_segmentation_masks(output_image, masks, alpha=0.5, colors="blue")

plt.figure(figsize=(12, 12))
plt.figure(figsize=(12, 12))
plt.xlabel("")
plt.ylabel("")
plt.xticks([])  
plt.yticks([])  

plt.imshow(output_image.permute(1, 2, 0))
