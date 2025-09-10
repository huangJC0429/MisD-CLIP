import torch
from torch import nn, optim
from torch.nn import functional as F
import numpy as np
from sklearn.isotonic import IsotonicRegression
import seaborn as sns
# import networkx as nx
from matplotlib.pyplot import MultipleLocator
import matplotlib.pyplot as plt
import matplotlib as mpl
from utils import *
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve, precision_score, recall_score

class _ECELoss(nn.Module):
    """
    Calculates the Expected Calibration Error of a model.
    (This isn't necessary for temperature scaling, just a cool metric).

    The input to this loss is the logits of a model, NOT the softmax scores.

    This divides the confidence outputs into equally-sized interval bins.
    In each bin, we compute the confidence gap:

    bin_gap = | avg_confidence_in_bin - accuracy_in_bin |

    We then return a weighted average of the gaps, based on the number
    of samples in each bin

    See: Naeini, Mahdi Pakdaman, Gregory F. Cooper, and Milos Hauskrecht.
    "Obtaining Well Calibrated Probabilities Using Bayesian Binning." AAAI.
    2015.
    """
    def __init__(self, n_bins=20):
        """
        n_bins (int): number of confidence interval bins
        """
        super(_ECELoss, self).__init__()
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]

    def forward(self, logits, labels): # label 1,3,.,3
        logits = logits.cuda().softmax(dim=-1).to('cpu')
        labels = torch.tensor(labels)
        softmaxes = logits# .softmax(dim=-1)# F.softmax(logits, dim=1)
        confidences, predictions = torch.max(softmaxes, 1)
        accuracies = predictions.eq(labels)
        ece = torch.zeros(1, device=logits.device)
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            # Calculated |confidence - accuracy| in each bin
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece +=  torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

        return ece

def brier_score_criterion(logits, labels, nclass):
    logits = torch.softmax(logits, dim=1)
    labels = F.one_hot(labels, nclass)
    loss_criterion = nn.MSELoss()
    brier_score = loss_criterion(logits, labels) * nclass
    return brier_score

def irova_calibrate(logit, label, logit_eval):
    p = np.exp(logit) / np.sum(np.exp(logit), 1)[:, None]
    p_eval = np.exp(logit_eval) / np.sum(np.exp(logit_eval), 1)[:, None]

    for ii in range(p_eval.shape[1]):
        ir = IsotonicRegression(out_of_bounds='clip')
        y_ = ir.fit_transform(p[:, ii], label[:, ii])
        p_eval[:, ii] = ir.predict(p_eval[:, ii]) + 1e-9 * p_eval[:, ii]

    return p_eval

def plot_acc_calibration(output, labels, n_bins, title):
    output = output.cuda().softmax(dim=-1).to('cpu')
    pred_label = torch.max(output, 1)[1]
    p_value = torch.max(output, 1)[0]
    ground_truth = labels
    confidence_all, confidence_acc = np.zeros(n_bins), np.zeros(n_bins)
    for index, value in enumerate(p_value):
        #value -= suboptimal_prob[index]
        interval = int(value / (1 / n_bins) -0.0001)
        confidence_all[interval] += 1
        if pred_label[index] == ground_truth[index]:
            confidence_acc[interval] += 1
    for index, value in enumerate(confidence_acc):
        if confidence_all[index] == 0:
            confidence_acc[index] = 0
        else:
            confidence_acc[index] /= confidence_all[index]

    start = np.around(1/n_bins/2, 3)
    step = np.around(1/n_bins, 3)
    plt.figure(figsize=(7, 6))
    plt.rcParams['axes.labelweight'] = 'bold'
    plt.rcParams["font.weight"] = "bold"
    plt.bar(np.around(np.arange(start, 1.0, step), 3), confidence_acc,
            alpha=0.7, width=0.03, color='#010FCC', label='Outputs')
    plt.bar(np.around(np.arange(start, 1.0, step), 3),
            np.around(np.arange(start, 1.0, step), 3), alpha=0.7, width=0.03, color='#b5b5b5', label='Expected')
    plt.plot([0,1], [0,1], ls='--',c='k')
    plt.xlabel('Confidence', fontsize=14)
    plt.ylabel('Accuracy', fontsize=14)
    plt.tick_params(labelsize=13)
    plt.xlim(0, 1.0)
    plt.ylim(0, 1.0)
    #title = 'Uncal. - Cora - 20 - GCN'
    plt.title(title, fontsize=16, fontweight="bold")
    plt.legend(fontsize=14)
    # plt.savefig("./methods/ablation_fig/"+title+".pdf", format='png', dpi=300,
    #             pad_inches=0, bbox_inches = 'tight')
    plt.savefig('./methods/vis/' + title +'.pdf', format='png', dpi=300,
                pad_inches=0, bbox_inches = 'tight')
    plt.show()

def plot_histograms(content_a, content_b, title, labeltitle, n_bins=50, norm_hist=True):
    # Plot histogram of correctly classified and misclassified examples
    global conf_histogram

    plt.figure(figsize=(6, 4))
    plt.rcParams['axes.labelweight'] = 'bold'
    plt.rcParams["font.weight"] = "bold"
    sns.distplot(content_a, kde=False, bins=n_bins, norm_hist=False, fit=None, label=labeltitle[0])
    sns.distplot(content_b, kde=False, bins=n_bins, norm_hist=False,  fit=None, label=labeltitle[1])
    plt.xlabel('Confidence', fontsize=14)
    plt.ylabel('Density', fontsize=14)
    plt.tick_params(labelsize=13)
    plt.title(title, fontsize=16, fontweight="bold")
    plt.legend(fontsize=14)
    # plt.savefig('output/' + title +'.png', format='png', transparent=True, dpi=300,
    #             pad_inches=0, bbox_inches = 'tight')
    plt.show()



def get_confidence(output, with_softmax=False):

    if not with_softmax:
        output = torch.softmax(output, dim=1)

    confidence, pred_label = torch.max(output, dim=1)

    return confidence, pred_label



def intra_distance_loss(output, labels):
    #loss = torch.ones(1, requires_grad = True)
    output = torch.softmax(output, dim=1)
    pred_max_index = torch.max(output, 1)[1]
    correct_i = torch.where(pred_max_index==labels)
    incorrect_i = torch.where(pred_max_index!=labels)
    output = torch.sort(output, dim=1, descending=True)
    pred,sub_pred = output[0][:,0], output[0][:,1]
    loss = (torch.sum(1 - pred[correct_i] + sub_pred[correct_i]) + torch.sum(pred[incorrect_i]-sub_pred[incorrect_i])) / labels.size()[0]
    return loss


def get_FPR90(confidence_t, correct_index_t, flag="correct"):
    # 计算FPR90%, flag== “correct”represent the positive sample is correct prediction，"incorrect" for incorrect prediction
    if flag == "correct":
        idxs = np.argsort(-confidence_t)  # large -> small
    elif flag == "incorrect":
        idxs = np.argsort(confidence_t)  # small -> large
    else:
        print("pleas input correct form for get FPR90!")
        exit()
    scores_sorted = confidence_t[idxs]
    labels_sorted = correct_index_t[idxs]
    # print(confidence_t)
    # print(idxs)
    if flag == "incorrect":
        labels_sorted = 1-labels_sorted
    target_recall = 0.90  # N%

    labels_sorted = np.asarray(labels_sorted)


    total_positives = np.sum(labels_sorted == 1)
    total_negatives = np.sum(labels_sorted == 0)


    if total_positives == 0:
        raise ValueError("No positive samples, can not compute FPRN.")

    # the number of positive samples that need to be detected.
    target_tp = int(np.ceil(target_recall * total_positives))


    tp = 0
    idx = 0
    while tp < target_tp and idx < len(labels_sorted):
        if labels_sorted[idx] == 1:
            tp += 1
        idx += 1


    fp = np.sum(labels_sorted[:idx] == 0)


    fpr_n = fp / total_negatives if total_negatives > 0 else 0.0
    return fpr_n

def get_FPR95(confidence_t, correct_index_t, flag="correct"):

    if flag == "correct":
        idxs = np.argsort(-confidence_t)
    elif flag == "incorrect":
        idxs = np.argsort(confidence_t)
    else:
        print("pleas input correct form for get FPR90!")
        exit()
    scores_sorted = confidence_t[idxs]
    labels_sorted = correct_index_t[idxs]
    if flag == "incorrect":
        labels_sorted = 1-labels_sorted
    target_recall = 0.95  # N%

    labels_sorted = np.asarray(labels_sorted)


    total_positives = np.sum(labels_sorted == 1)
    total_negatives = np.sum(labels_sorted == 0)


    if total_positives == 0:
        raise ValueError("No positive samples, can not compute FPRN.")


    target_tp = int(np.ceil(target_recall * total_positives))


    tp = 0
    idx = 0
    while tp < target_tp and idx < len(labels_sorted):
        if labels_sorted[idx] == 1:
            tp += 1
        idx += 1

    fp = np.sum(labels_sorted[:idx] == 0)


    fpr_n = fp / total_negatives if total_negatives > 0 else 0.0
    return fpr_n

def write_floats_to_txt(filename, float_list):
    with open(filename, "a") as f:
        for num in float_list:
            f.write(f"{num}")
            f.write(f",")
        f.write(f"\n")

def get_AUROC_AUPR_FPR(confidence_t, correct_index_t, isprint=True):
    # print(confidence_t)
    # print(correct_index_t.numpy())
    # print(correct_index_t.int())
    confidence_t = confidence_t.numpy()
    correct_index_t = np.array(correct_index_t, dtype=int)
    # print(confidence_t)
    # print(correct_index_t)
    y_pred = confidence_t
    # print(y_pred)
    # exit()
    y_true = correct_index_t
    auroc = roc_auc_score(y_true,y_pred)
    aupr_cor = average_precision_score(y_true,y_pred)
    aupr_incor = average_precision_score(1-y_true,1-y_pred)


    threshold = 0.5
    y_pred_class = np.where(y_pred > threshold, 1, 0)

    precision = precision_score(y_true, y_pred_class)

    fpr_n_cor_90 = get_FPR90(confidence_t,correct_index_t,"correct") # This is for correct predicted sample
    fpr_n_incor_90 = get_FPR90(confidence_t, correct_index_t, "incorrect")  # This is for incorrect predicted sample

    fpr_n_cor_95 = get_FPR95(confidence_t,correct_index_t,"correct") # This is for correct predicted sample
    fpr_n_incor_95 = get_FPR95(confidence_t, correct_index_t, "incorrect")  # This is for incorrect predicted sample

    # write_floats_to_txt("./para_exp_tau.txt", [auroc, aupr_cor, aupr_incor, fpr_n_cor_90, fpr_n_incor_90])

    if isprint:
        print(f"AUROC: {auroc:.3f}")
        print(f"AUPR_cor: {aupr_cor:.3f}")
        print(f"AUPR_incor: {aupr_incor:.3f}")
        print(f"threshold{threshold}时，Precision = {precision:.3f}")
        print(f"FPR95%_cor: {fpr_n_cor_95:.3f}")
        print(f"FPR95%_incor: {fpr_n_incor_95:.3f}")
        print(f"FPR90%_cor: {fpr_n_cor_90:.3f}")
        print(f"FPR90%_incor: {fpr_n_incor_90:.3f}")

    return auroc, aupr_cor, aupr_incor, fpr_n_cor_95, fpr_n_incor_95, fpr_n_cor_90, fpr_n_incor_90

def get_metric(prob_preds, numeric_testlabels, isprint=True):
    confidence_t = torch.softmax(prob_preds, dim=1).cpu().detach()
    confidence_t = torch.max(confidence_t, 1)[0]
    pred_label = torch.max(prob_preds, 1)[1]
    # correct_index_t = numeric_testlabels == pred_label.tolist()
    correct_index_t = [a == b for a, b in zip(pred_label.tolist(), numeric_testlabels)]

    # measure discrimination ability
    auroc, aupr_cor, aupr_incor, fpr_n_cor_95, fpr_n_incor_95, fpr_n_cor_90, fpr_n_incor_90 = get_AUROC_AUPR_FPR(confidence_t, correct_index_t, isprint=isprint)
    return auroc, aupr_cor, aupr_incor, fpr_n_cor_95, fpr_n_incor_95, fpr_n_cor_90, fpr_n_incor_90


def visual_tau(tau, mask):

    tau = tau.numpy()
    mask = np.array(mask, dtype=bool)


    true_values = tau[mask]
    false_values = tau[~mask]

    true_mean = true_values.mean()
    false_mean = false_values.mean()

    true_std = true_values.std()
    false_std = false_values.std()

    print(true_mean, false_mean, true_std, false_std)

    # 全局字体设置
    plt.rcParams.update({
        "font.size": 18,
        "axes.titlesize": 20,
        "axes.labelsize": 20,
        "xtick.labelsize": 18,
        "ytick.labelsize": 18,
        "legend.fontsize": 18
    })

    # 柱状图 + 误差棒
    plt.figure(figsize=(4, 6))
    plt.bar(
        ["Cor", "Incor"],
        [true_mean, false_mean],
        yerr=[true_std, false_std],
        capsize=8,
        color=["skyblue", "salmon"],
        alpha=0.8
    )

    plt.ylabel("Aver." + r"$\tau$")
    plt.title("Datasets")
    plt.grid(True, linestyle="--", alpha=0.6, axis="y")
    plt.tight_layout()
    plt.show()