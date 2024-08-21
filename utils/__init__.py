import os
import math
import torch
import numpy
import random
import logging
import skimage.transform
import matplotlib.pyplot as plt

from .metric_utils import get_metrics
from PIL import Image


LOGGER_NAME = "Show-Attend-Tell"

LOGGER = logging.getLogger(LOGGER_NAME)
LOGGER.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(name)s : %(message)s")
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
LOGGER.addHandler(stream_handler)


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    numpy.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    # torch.use_deterministic_algorithms(True) # CUDA >= 10.2


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    numpy.random.seed(worker_seed)
    random.seed(worker_seed)


def setup_env():
    """
    .env
        MASTER_ADDR : master ip
        MASTER_PORT : master port
    """
    os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ['TORCH_USE_CUDA_DSA'] = '1'


def save_loss_history(base_path, phase, model_name, loss_history, step):
    losses = []
    length = len(loss_history)
    chunk_step = length // step
    for i in range(0, length, chunk_step):
        cur = [loss for _, loss in loss_history[i: i+chunk_step]]
        losses.append(sum(cur) / len(cur))

    plt.title(f"{phase} Loss History")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    plt.plot(range(1, step + 1), losses, marker='o', label=model_name)
    plt.xticks(range(1, step + 1))
    plt.legend()
    plt.savefig(f"{base_path}/{phase}_loss.png", bbox_inches='tight', dpi=300)


def save_history(base_path, history, metric_type='bleu'):
    """
    Args:
        metric_type (str) : metric name
    """
    if metric_type == 'bleu':
        bleu2 = [bleu for bleu, _ in history]
        bleu4 = [bleu for _, bleu in history]
        epoch = len(history)

    plt.title(f"{metric_type.upper()} Score History")
    plt.xlabel("Epoch")
    plt.ylabel(metric_type.upper())

    if metric_type == 'bleu':
        plt.plot(range(1, epoch + 1), bleu2, marker='o', label='BLEU-2')
        plt.plot(range(1, epoch + 1), bleu4, marker='o', label='BLEU-4')
    else:
        plt.plot(range(1, epoch + 1), bleu2, marker='o', label=metric_type.upper())

    plt.xticks(range(1, epoch + 1))
    plt.legend()
    plt.savefig(f"{base_path}/{metric_type}.png", bbox_inches='tight', dpi=300)


def save_pred_figure(base_path, sampled_images, sampled_labels, sampled_preds):
    for img, label, pred in zip(sampled_images, sampled_labels, sampled_preds):
        plt.figure()
        image_name = img[img.rfind('/')+1:img.rfind('.')]
        ax = plt.gca()
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)
        img = plt.imread(img)
        h, w, _ = img.shape

        plt.imshow(img)
        plt.text(w/2, h+20, 'label: ' + label, horizontalalignment='center')
        plt.text(w/2, h+40, 'pred: ' + pred, horizontalalignment='center')

        plt.savefig(f"{base_path}/{image_name}_pred.jpg")


def save_attention_figure(base_path, encoded_hidden_size, transform, sampled_images, sampled_scores, sampled_preds):
    for image, score, pred in zip(sampled_images, sampled_scores, sampled_preds):
        image_name = image[image.rfind('/')+1:image.rfind('.')]
        image = transform(Image.open(image)).permute(1, 2, 0)
        score = score.view(encoded_hidden_size, encoded_hidden_size, -1)
        pred = pred.split()

        plt.figure(figsize=(4*3, math.ceil(len(pred)/4)*3))
        for i in range(len(pred)):
            cur_score = skimage.transform.pyramid_expand(score[:, :, i].numpy(), upscale=18, sigma=8)
            plt.subplot(math.ceil(len(pred)/4), 4, i+1)
            ax = plt.gca()
            ax.axes.xaxis.set_visible(False)
            ax.axes.yaxis.set_visible(False)
            plt.imshow(image)
            plt.imshow(cur_score, cmap='gray', alpha=0.7)
            plt.text(5, 15, pred[i], bbox={'facecolor':'w','boxstyle':'square','alpha':1})
        plt.savefig(f"{base_path}/{image_name}_attention.jpg")