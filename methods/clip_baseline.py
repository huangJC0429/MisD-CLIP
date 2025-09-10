import logging

import clip
import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt


g = torch.Generator()
g.manual_seed(0)

log = logging.getLogger(__name__)


class ClipBaseline(object):
    def __init__(
        self, config, label_to_idx, classes, seen_classes, unseen_classes, device
    ):
        """This class is CLIP model.

        :param config: class object with model configurations in the
                       file models_config/clip_baseline_config.yml
        :param label_to_idx: dictionary (key, value):(class name, id)
        :param classes: list of class names
        :param seen_classes: list on seen classes' names
        :param unseen_classes: list on unseen classes' names
        :param device: device in use

        """
        self.config = config
        self.classes = classes
        self.seen_classes = seen_classes
        self.unseen_classes = unseen_classes
        self.label_to_idx = label_to_idx

        self.device = device
        self.model, self.transform = clip.load(
            self.config.VIS_ENCODER, device=self.device
        )
        self.template = self.config.PROMPT_TEMPLATE

    def test_predictions(self, data):
        """
        :param data: test dataset
        """

        # Declare data pre-processing
        data.transform = self.transform
        # Define the data loader
        test_loader = torch.utils.data.DataLoader(
            data, batch_size=self.config.BATCH_SIZE
        )

        # Build textual prompts
        prompts = [
            self.template.format(" ".join(i.split("_"))) for i in self.classes
        ]
        log.info(f"Number of prompts: {len(prompts)}")

        # Encode text
        text = clip.tokenize(prompts).to(self.device)
        text_features = self.model.encode_text(text)

        log.info(f"Start inference for test data")
        predictions = []
        images = []
        prob_preds = []
        # Iterate through data loader
        for img, _, _, img_path in tqdm(test_loader):
            with torch.no_grad():
                logits_per_image, logits_per_text = self.model(
                    img.to(self.device), text.to(self.device)
                )
                probs = logits_per_image.softmax(dim=-1)
                idx_preds = torch.argmax(probs, dim=1)

                predictions += [self.classes[i] for i in idx_preds]
                images += [i for i in img_path]
                prob_preds += [logits_per_image]

        prob_preds = torch.cat(prob_preds, axis=0).detach().to('cpu')
        df_predictions = pd.DataFrame({"id": images, "class": predictions})

        return df_predictions, images, predictions, prob_preds

    def plot_acc_predicted_class(self, logits, labels):
        preds = logits.argmax(dim=1)
        num_classes = logits.shape[1]
        labels = torch.tensor(labels)

        # print(preds)
        # print(labels)
        #
        # print((preds.eq(torch.tensor(labels))).sum()/logits.shape[0])
        # exit()

        pred_count = {}
        correct_count = {}

        for cls in range(num_classes):
            pred_mask = preds == cls
            pred_count[cls] = pred_mask.int().sum().item()
            correct_count[cls] = (pred_mask & (labels == cls)).int().sum().item()
        # print(pred_count)
        # print(labels)
        # print(preds)
        # print(correct_count)
        # exit()


        sorted_classes = sorted(pred_count.keys(), key=lambda k: pred_count[k], reverse=True)
        x_labels = [str(cls) for cls in sorted_classes]
        x = np.arange(len(sorted_classes))
        pred_vals = [pred_count[cls] for cls in sorted_classes]
        correct_vals = [correct_count.get(cls, 0) for cls in sorted_classes]
        print(pred_vals)
        print(correct_vals)


        width = 0.4
        plt.figure(figsize=(10, 6))
        plt.bar(x , pred_vals, width=width, color='yellow', alpha=0.7, label='Predicted Count')
        plt.bar(x , correct_vals, width=width, color='blue', alpha=0.7, label='Correct Count')

        plt.xticks(x, x_labels)
        plt.xlabel('Predicted Class (sorted by frequency)')
        plt.ylabel('Sample Count')
        plt.title('Prediction and Correct Count per Predicted Class')
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.show()