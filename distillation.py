import torch

def distillation_loss(student_predictions, teacher_predictions, temperature=1.0):
    # Ensure that the predictions from both models are aligned
    student_scores = []
    teacher_scores = []
    for student_output, teacher_output in zip(student_predictions, teacher_predictions):
        # Ensure both have the same number of detections
        min_detections = min(student_output['scores'].shape[0], teacher_output['scores'].shape[0])
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
    return loss

def resize_and_match_channels(student_features, teacher_features):
    resized_student_features = []
    conv_layers = []
    for sf, tf in zip(student_features, teacher_features):
        resized_sf = torch.nn.functional.interpolate(sf, size=tf.shape[2:], mode='bilinear', align_corners=False)
        if sf.shape[1] != tf.shape[1]:
            conv = torch.nn.Conv2d(sf.shape[1], tf.shape[1], kernel_size=1).to(sf.device)
            resized_sf = conv(resized_sf)
            conv_layers.append(conv)
        resized_student_features.append(resized_sf)
    return resized_student_features, conv_layers

def compute_pearson_correlation(x, y):
    x = x.view(x.size(0), -1)
    y = y.view(y.size(0), -1)
    mean_x = x.mean(dim=1, keepdim=True)
    mean_y = y.mean(dim=1, keepdim=True)
    xm = x - mean_x
    ym = y - mean_y
    r_num = torch.sum(xm * ym, dim=1)
    r_den = torch.sqrt(torch.sum(xm ** 2, dim=1)) * torch.sqrt(torch.sum(ym ** 2, dim=1))
    r = r_num / r_den
    return torch.mean(r)

def pkd_loss(student_features, teacher_features, alpha=0.5):
    student_features, conv_layers = resize_and_match_channels(student_features, teacher_features)
    loss = 0
    for sf, tf in zip(student_features, teacher_features):
        pcc_loss = 1 - compute_pearson_correlation(sf, tf)
        loss += pcc_loss
    return alpha * loss, conv_layers