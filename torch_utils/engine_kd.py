import math
import sys
import time

import torch
import torch.nn as nn
import torchvision.models.detection.mask_rcnn
import torch.nn.functional as F
from torch_utils import utils
from torch_utils.coco_eval import CocoEvaluator
from torch_utils.coco_utils import get_coco_api_from_dataset
from utils.general import save_validation_results
import numpy as np
from torchvision.ops import box_iou
from scipy.optimize import linear_sum_assignment


class KnowledgeDistillationLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=0.5, gamma=0.1, delta=0.1, temperature=1.0, iou_threshold=0.5):
        super(KnowledgeDistillationLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.temperature = temperature
        self.iou_threshold = iou_threshold

    def forward(self, student_outputs, teacher_outputs): #, student_features, teacher_features):

        # print ("student_outputs ", student_outputs.keys())
        # print ("teacher_outputs ", teacher_outputs.keys())

        batch_size = len(student_outputs)
        for i in range(batch_size):
            student_output = student_outputs[i]
            teacher_output = teacher_outputs[i]
            # Extract detections
            student_boxes, student_scores, student_labels = self._extract_detections(student_output)
            teacher_boxes, teacher_scores, teacher_labels = self._extract_detections(teacher_output)

            # Match detections using IoU
            matched_indices = self._match_detections(student_boxes, teacher_boxes)

            # Classification distillation loss
            class_loss = self._calculate_classification_loss(
                student_scores, student_labels, teacher_scores, teacher_labels, matched_indices
            )

            # Bounding box distillation loss
            bbox_loss = self._calculate_bbox_loss(student_boxes, teacher_boxes, matched_indices)

            # # Feature map distillation loss (Mean Squared Error)
            # feature_loss = 0
            # for sf, tf in zip(student_features, teacher_features):
            #     feature_loss += F.mse_loss(sf, tf, reduction='mean')

            # # Attention map distillation loss (Mean Squared Error)
            # student_attention = student_outputs.get('attention_maps', [])
            # teacher_attention = teacher_outputs.get('attention_maps', [])
            
            # attention_loss = 0
            # for sa, ta in zip(student_attention, teacher_attention):
            #     attention_loss += F.mse_loss(sa, ta, reduction='mean')

        total_loss = (self.alpha * class_loss + 
                      self.beta * bbox_loss) #+ 
                    #   self.gamma * feature_loss) # +
                    #   self.delta * attention_loss)

        return total_loss

    def _extract_detections(self, outputs):
        # Extract boxes, scores, and labels
        boxes = outputs['boxes']
        scores = outputs['scores']
        labels = outputs['labels']
        return boxes, scores, labels

    def _match_detections(self, student_boxes, teacher_boxes):
        # Compute IoU matrix
        iou_matrix = box_iou(student_boxes, teacher_boxes)

        # Use Hungarian algorithm to find the best matching pairs
        matched_indices = linear_sum_assignment(-iou_matrix.detach().cpu().numpy())

        # Filter matches based on IoU threshold
        matches = [
            (s_idx, t_idx) for s_idx, t_idx in zip(*matched_indices)
            if iou_matrix[s_idx, t_idx] > self.iou_threshold
        ]
        return matches

    def _calculate_classification_loss(self, student_scores, student_labels, teacher_scores, teacher_labels, matches):
        class_loss = 0
        for s_idx, t_idx in matches:
            student_logit = student_scores[s_idx]
            teacher_logit = teacher_scores[t_idx]

            # Use KL Divergence for classification distillation
            class_loss += F.kl_div(
                F.log_softmax(student_logit / self.temperature, dim=0),
                F.softmax(teacher_logit / self.temperature, dim=0),
                reduction='batchmean'
            ) * (self.temperature ** 2)

        return class_loss / len(matches) if matches else torch.tensor(0.0, device=student_scores.device)

    def _calculate_bbox_loss(self, student_boxes, teacher_boxes, matches):
        bbox_loss = 0
        for s_idx, t_idx in matches:
            student_box = student_boxes[s_idx]
            teacher_box = teacher_boxes[t_idx]

            # Use Smooth L1 Loss for bounding box distillation
            bbox_loss += F.smooth_l1_loss(student_box, teacher_box, reduction='mean')

        return bbox_loss / len(matches) if matches else torch.tensor(0.0, device=student_boxes.device)

def compute_iou(box1, box2):
    """Compute IoU between two sets of boxes."""
    x1 = torch.max(box1[:, None, 0], box2[:, 0])
    y1 = torch.max(box1[:, None, 1], box2[:, 1])
    x2 = torch.min(box1[:, None, 2], box2[:, 2])
    y2 = torch.min(box1[:, None, 3], box2[:, 3])

    inter_area = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)
    box1_area = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
    box2_area = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])
    union_area = box1_area[:, None] + box2_area - inter_area

    return inter_area / union_area

def match_boxes(teacher_boxes, student_boxes, iou_threshold=0.5):
    """Match teacher and student boxes based on IoU."""
    iou_matrix = compute_iou(teacher_boxes, student_boxes)
    matched_indices = (iou_matrix >= iou_threshold).nonzero(as_tuple=False)

    matched_teacher_boxes = teacher_boxes[matched_indices[:, 0]]
    matched_student_boxes = student_boxes[matched_indices[:, 1]]

    return matched_teacher_boxes, matched_student_boxes, matched_indices[:, 1]

class DistillationLoss(nn.Module):
    def __init__(self, alpha=0.5, temperature=2.0, iou_threshold=0.5):
        super(DistillationLoss, self).__init__()
        self.alpha = alpha
        self.temperature = temperature
        self.iou_threshold = iou_threshold
        self.mse_loss = nn.MSELoss()
        self.kl_loss = nn.KLDivLoss(reduction='batchmean')

    def forward(self, teacher_outputs, student_outputs): #, student_outputs2):
        teacher_boxes = teacher_outputs[0]['boxes']
        student_boxes = student_outputs[0]['boxes']

        teacher_scores = teacher_outputs[0]['scores']
        student_scores = student_outputs[0]['scores']

        # Match boxes based on IoU
        matched_teacher_boxes, matched_student_boxes, matched_indices = match_boxes(
            teacher_boxes, student_boxes, self.iou_threshold)

        # print ("here")

        if len(matched_indices) == 0:
            return torch.tensor(0.0, device=teacher_boxes.device)
            #sum(student_outputs.values())  # No matches, return standard loss

        # Calculate the bounding box regression loss (L2 loss)
        bbox_loss = self.mse_loss(matched_student_boxes, matched_teacher_boxes)

        # # Calculate the classification loss (KL divergence)
        # matched_teacher_scores = teacher_scores[matched_indices]
        # matched_student_scores = student_scores[matched_indices]

        # print ("matched_teacher_scores ", matched_teacher_scores.shape)
        # print ("matched_student_scores ", matched_student_scores.shape)

        # student_scores_soft = torch.log_softmax(matched_student_scores / self.temperature, dim=0)
        # teacher_scores_soft = torch.softmax(matched_teacher_scores / self.temperature, dim=0)
        # class_loss = self.kl_loss(student_scores_soft, teacher_scores_soft) * (self.temperature ** 2)

        # # Combine the standard student losses
        # # standard_loss = sum(student_outputs.values())

        # # Calculate the total distillation loss
        # total_loss = self.alpha * bbox_loss + (1 - self.alpha) * (class_loss)

        return bbox_loss #+ class_loss


def train_one_epoch(
    model_teacher,
    model_student, 
    optimizer, 
    data_loader, 
    device, 
    epoch, 
    train_loss_hist,
    print_freq, 
    scaler=None,
    scheduler=None
):
    
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = f"Epoch: [{epoch}]"

    # List to store batch losses.
    batch_loss_list = []
    batch_loss_cls_list = []
    batch_loss_box_reg_list = []
    batch_loss_objectness_list = []
    batch_loss_rpn_list = []
    batch_loss_kd_list = []

    lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1.0 / 1000
        warmup_iters = min(1000, len(data_loader) - 1)

        lr_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=warmup_factor, total_iters=warmup_iters
        )

    # Loss function
    kd_loss_fn = DistillationLoss() #KnowledgeDistillationLoss()

    temperature = 1.0
    step_counter = 0
    for images, targets in metric_logger.log_every(data_loader, print_freq, header):
        step_counter += 1
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device).to(torch.int64) for k, v in t.items()} for t in targets]

        with torch.no_grad():
           teacher_predictions = model_teacher(images, targets)

        model_student.eval()  # Switch to evaluation mode for distillation
        student_predictions = model_student(images, targets)

        # print ("teacher_predictions ", teacher_predictions)
        # # print ("student_predictions ", student_predictions.keys())

        # # # Extract feature maps if needed
        # student_features = model_student.backbone(images)
        # teacher_features = model_teacher.backbone(images)

        # Calculate distillation loss
        # kd_loss = kd_loss_fn(student_predictions, teacher_predictions) #, student_features, teacher_features)

        # Compute distillation loss using the predictions
        kd_loss = distillation_loss(student_predictions, teacher_predictions, temperature)
        # kd_loss = kd_loss_fn(teacher_predictions, student_predictions)

        model_student.train()

        # with torch.cuda.amp.autocast(enabled=scaler is not None):
        loss_dict = model_student(images, targets)

        # kd_loss = torch.tensor(0.0, device=device)
        loss_dict['kd_loss'] = kd_loss
        # loss_dict['kd_loss'] = 0.
        losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        loss_value = losses_reduced.item()

        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(losses).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            losses.backward()
            optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        batch_loss_list.append(loss_value)
        batch_loss_cls_list.append(loss_dict_reduced['loss_classifier'].detach().cpu())
        batch_loss_box_reg_list.append(loss_dict_reduced['loss_box_reg'].detach().cpu())
        batch_loss_objectness_list.append(loss_dict_reduced['loss_objectness'].detach().cpu())
        batch_loss_rpn_list.append(loss_dict_reduced['loss_rpn_box_reg'].detach().cpu())
        batch_loss_kd_list.append(loss_dict_reduced['kd_loss'].detach().cpu())
        train_loss_hist.send(loss_value)

        if scheduler is not None:
            scheduler.step(epoch + (step_counter/len(data_loader)))

    return (
        metric_logger, 
        batch_loss_list, 
        batch_loss_cls_list, 
        batch_loss_box_reg_list, 
        batch_loss_objectness_list, 
        batch_loss_rpn_list, 
        batch_loss_kd_list
    )


def distillation_loss(student_predictions, teacher_predictions, temperature=1.0):
    # Ensure that the predictions from both models are aligned
    student_scores = []
    teacher_scores = []
    for student_output, teacher_output in zip(student_predictions, teacher_predictions):
        # Ensure both have the same number of detections
        # print ("student_output ", student_output)
        # print ("teacher_output ", teacher_output)
        min_detections = min(student_output['scores'].shape[0], teacher_output['scores'].shape[0])
        # print ("min_detections ", min_detections)
        student_scores.append(student_output['scores'][:min_detections])
        teacher_scores.append(teacher_output['scores'][:min_detections])

    student_scores = torch.cat(student_scores)
    teacher_scores = torch.cat(teacher_scores)

    # Scale the scores by the temperature
    student_scores /= temperature
    teacher_scores /= temperature

    # Compute the KL divergence loss
    loss = torch.nn.functional.kl_div(
        torch.nn.functional.log_softmax(student_scores, dim=-1),
        torch.nn.functional.softmax(teacher_scores, dim=-1),
        reduction='batchmean'
    )
    # print ("loss kd ", loss)
    return loss


def _get_iou_types(model):
    model_without_ddp = model
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model_without_ddp = model.module
    iou_types = ["bbox"]
    if isinstance(model_without_ddp, torchvision.models.detection.MaskRCNN):
        iou_types.append("segm")
    if isinstance(model_without_ddp, torchvision.models.detection.KeypointRCNN):
        iou_types.append("keypoints")
    return iou_types


@torch.inference_mode()
def evaluate(
    model, 
    data_loader, 
    device, 
    save_valid_preds=False,
    out_dir=None,
    classes=None,
    colors=None
):
    n_threads = torch.get_num_threads()
    # FIXME remove this and make paste_masks_in_image run on the GPU
    torch.set_num_threads(1)
    cpu_device = torch.device("cpu")
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Test:"

    coco = get_coco_api_from_dataset(data_loader.dataset)
    iou_types = _get_iou_types(model)
    coco_evaluator = CocoEvaluator(coco, iou_types)

    counter = 0
    for images, targets in metric_logger.log_every(data_loader, 100, header):
        counter += 1
        images = list(img.to(device) for img in images)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        model_time = time.time()
        outputs = model(images)

        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
        model_time = time.time() - model_time

        res = {target["image_id"].item(): output for target, output in zip(targets, outputs)}
        evaluator_time = time.time()
        coco_evaluator.update(res)
        evaluator_time = time.time() - evaluator_time
        metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)

        if save_valid_preds and counter == 1:
            # The validation prediction image which is saved to disk
            # is returned here which is again returned at the end of the
            # function for WandB logging.
            val_saved_image = save_validation_results(
                images, outputs, counter, out_dir, classes, colors
            )
        elif save_valid_preds == False and counter == 1:
            val_saved_image = np.ones((1, 64, 64, 3))
            

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    coco_evaluator.accumulate()
    stats = coco_evaluator.summarize()
    torch.set_num_threads(n_threads)
    return stats, val_saved_image
