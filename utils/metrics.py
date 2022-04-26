import os
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.metrics import f1_score, jaccard_score, precision_score, recall_score

def IOU(true, pred):
    """
    Calls chosen optimizer from the pytorch library.

    Args:
        true (numpy array): true label (binary image)
        pred (numpy array): predicted label (binary image)

    Returns:
        iou_score (float): iou score
    """
    intersection = np.logical_and(true, pred)
    union = np.logical_or(true, pred)
    iou = intersection.sum() / union.sum()

    iou_score = round(iou, 4)

    return iou_score

def JaccardIndex(pred, true):
    """
    Calculates intersection and union for a batch of images.

    Args:
        pred (torch.Tensor): a tensor of predictions
        true (torch.Tensor): a tensor of labels

    Returns:
        iou (float): total intersection of pixels in percent
    """
    valid_pixel_mask = true.ne(255)  # valid pixel mask
    true = true.masked_select(valid_pixel_mask).to("cpu")
    pred = pred.masked_select(valid_pixel_mask).to("cpu")

    # Intersection and union totals
    intersection = np.logical_and(true, pred)
    union = np.logical_or(true, pred)
    iou = intersection.sum() / union.sum()
    return iou

def IoULoss(pred, true):
    """
    Calculates IoU loss for a batch of images.

    Args:
        pred (torch.Tensor): a tensor of predictions
        true (torch.Tensor): a tensor of labels

    Returns:
        iou loss (float): 1 - iou
    """
    IoU = intersection_over_union(pred, true)

    return 1 - IoU

def calculate_scores(y_true, y_pred):
    score_df = pd.DataFrame(data=[[0,0,0,0,0]], columns=['Date', 'Jaccard', 'F1_score', 'Precision', 'Recall'])

    # Add date & time
    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S") # dd/mm/YY H:M:S
    score_df['Date'] = dt_string

    # Calculate scores
    f1_sc = f1_score(y_true=y_true, y_pred=y_pred)
    score_df['F1_score'] = f1_sc
    jaccard_sc = jaccard_score(y_true=y_true, y_pred=y_pred)
    score_df['Jaccard'] = jaccard_sc
    recall_sc = recall_score(y_true=y_true, y_pred=y_pred)
    score_df['Recall'] = recall_sc
    precision_sc = precision_score(y_true=y_true, y_pred=y_pred)
    score_df['Precision'] = precision_sc

    score_df_path = 'temp/score_df.csv'
    if not os.path.exists(score_df_path):
        score_df.to_csv(score_df_path, index=False)
    else:
        temp_score_df = pd.read_csv(score_df_path)
        frames = [score_df, temp_score_df]
        score_df = pd.concat(frames)
        score_df.to_csv(score_df_path, index=False)

    score_df = score_df.reset_index(drop=True)

    return score_df