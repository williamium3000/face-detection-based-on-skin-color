def calculate_tp(pred_boxes, pred_scores, gt_boxes, gt_difficult, iou_thresh = 0.5):
    """
        calculate tp/fp for all predicted bboxes for one class of one image.
        对于匹配到同一gt的不同bboxes，让score最高tp = 1，其它的tp = 0
    Args:
        pred_boxes: Tensor[N, 4], 某张图片中某类别的全部预测框的坐标 (x0, y0, x1, y1)
        pred_scores: Tensor[N, 1], 某张图片中某类别的全部预测框的score
        gt_boxes: Tensor[M, 4], 某张图片中某类别的全部gt的坐标 (x0, y0, x1, y1)
        gt_difficult: Tensor[M, 1], 某张图片中某类别的gt中是否为difficult目标的值
        iou_thresh: iou 阈值

    Returns:
        gt_num: 某张图片中某类别的gt数量
        tp_list: 记录某张图片中某类别的预测框是否为tp的情况
        confidence_score: 记录某张图片中某类别的预测框的score值 (与tp_list相对应)
    """
    if gt_boxes.numel() == 0:
        return 0, [], []

    # 若无对应的boxes，则 tp 为空
    if pred_boxes.numel() == 0:
        return len(gt_boxes), [], []

    # 否则计算所有预测框与gt之间的iou
    ious = pred_boxes.new_zeros((len(gt_boxes), len(pred_boxes)))
    for i in range(len(gt_boxes)):
        gb = gt_boxes[i]
        area_pb = (pred_boxes[:, 2] - pred_boxes[:, 0]) * (pred_boxes[:, 3] - pred_boxes[:, 1])
        area_gb = (gb[2] - gb[0]) * (gb[3] - gb[1])
        xx1 = pred_boxes[:, 0].clamp(min = gb[0].item())  # [N-1,]
        yy1 = pred_boxes[:, 1].clamp(min = gb[1].item())
        xx2 = pred_boxes[:, 2].clamp(max = gb[2].item())
        yy2 = pred_boxes[:, 3].clamp(max = gb[3].item())
        inter = (xx2 - xx1).clamp(min = 0) * (yy2 - yy1).clamp(min = 0)  # [N-1,]
        ious[i] = inter / (area_pb + area_gb - inter)
    # 每个预测框的最大iou所对应的gt记为其匹配的gt
    max_ious, max_ious_idx = ious.max(dim = 0)

    not_difficult_gt_mask = gt_difficult == 0
    gt_num = not_difficult_gt_mask.sum().item()
    if gt_num == 0:
        return 0, [], []

    # 保留 max_iou 中属于 非difficult 目标的预测框，即应该去掉与 difficult gt 相匹配的预测框，不参与p-r计算
    # 如果去掉与 difficult gt 对应的iou分数后，候选框的最大iou依然没有发生改变，则可认为此候选框不与difficult gt相匹配，应该保留
    not_difficult_pb_mask = (ious[not_difficult_gt_mask].max(dim = 0)[0] == max_ious)
    max_ious, max_ious_idx = max_ious[not_difficult_pb_mask], max_ious_idx[not_difficult_pb_mask]
    if max_ious_idx.numel() == 0:
        return gt_num, [], []

    confidence_score = pred_scores.view(-1)[not_difficult_pb_mask]
    tp_list = torch.zeros_like(max_ious)
    for i in max_ious_idx[max_ious > iou_thresh].unique():
        gt_mask = (max_ious > iou_thresh) * (max_ious_idx == i)
        idx = (confidence_score * gt_mask.float()).argmax()
        tp_list[idx] = 1

    return gt_num, tp_list.tolist(), confidence_score.tolist()


def calculate_pr(gt_num, tp_list, confidence_score):
    """
    calculate all p-r pairs among different score_thresh for one class, using `tp_list` and `confidence_score`.

    Args:
        gt_num (Integer): 某张图片中某类别的gt数量
        tp_list (List): 记录某张图片中某类别的预测框是否为tp的情况
        confidence_score (List): 记录某张图片中某类别的预测框的score值 (与tp_list相对应)

    Returns:
        recall
        precision

    """
    if gt_num == 0:
        return [0], [0]
    if isinstance(tp_list, (tuple, list)):
        tp_list = np.array(tp_list)
    if isinstance(confidence_score, (tuple, list)):
        confidence_score = np.array(confidence_score)

    assert len(tp_list) == len(confidence_score), "len(tp_list) and len(confidence_score) should be same"

    if len(tp_list) == 0:
        return [0], [0]

    sort_mask = np.argsort(-confidence_score)
    tp_list = tp_list[sort_mask]
    recall = np.cumsum(tp_list) / gt_num
    precision = np.cumsum(tp_list) / (np.arange(len(tp_list)) + 1)

    return recall.tolist(), precision.tolist()


def voc_ap(rec, prec, use_07_metric = False):
    """Compute VOC AP given precision and recall. If use_07_metric is true, uses
    the VOC 07 11-point method (default:False).
    """
    if isinstance(rec, (tuple, list)):
        rec = np.array(rec)
    if isinstance(prec, (tuple, list)):
        prec = np.array(prec)
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap

def get_ap(pred_boxes, pred_scores, gt_boxes, iou_thresh = 0.5):
    gt_difficult = np.zeros((gt_boxes.shape[0], 1))
    gt_num, tp_list, confidence_score = calculate_tp(pred_boxes, pred_scores, gt_boxes, gt_difficult, iou_thresh = 0.5)
    rec, prec = calculate_pr(gt_num, tp_list, confidence_score)
    ap = voc_ap(rec, prec, use_07_metric = False)
    return ap