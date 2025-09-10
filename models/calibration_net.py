import copy
import logging
import os.path as osp
import numpy as np
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from clip import clip
import random
import pandas as pd

from accelerate import Accelerator
accelerator = Accelerator()


log = logging.getLogger(__name__)
g = torch.Generator()


class CN(nn.Module):
    def __init__(self, config, label_to_idx, classes, device):
        super(CN, self).__init__()
        self.config = config
        self.device = device
        self.clip_model, self.transform = clip.load(
            self.config.VIS_ENCODER, device=self.device
        )
        self.label_to_idx  = label_to_idx
        self.tau = config.tau
        self.classes = classes
        self.dropout = config.dropout

        self.template = self.config.PROMPT_TEMPLATE

        self.confidence_MLP = nn.Linear(len(classes), self.config.n_hid)
        self.img_MLP = nn.Linear(512, self.config.n_hid)
        self.text_MLP = nn.Linear(512, self.config.n_hid)
        self.out = nn.Linear(3*self.config.n_hid, 1)
        self.t = None


    def test_scale(self, logits, image_emb, text_emb):
        c = F.dropout(logits, self.dropout, training=self.training)
        c = self.confidence_MLP(c)

        image_emb = F.dropout(image_emb, self.dropout, training=self.training)
        image_emb = self.img_MLP(image_emb)

        text_emb = self.text_MLP(text_emb)

        x = torch.cat((c, image_emb, text_emb), dim=-1)

        x = F.dropout(x, self.dropout, training=self.training)
        x = self.out(x)
        # Softplus
        self.t = torch.log(torch.exp(x) + torch.tensor(1.1))
        return logits * (2-self.t)



    def forward(self, logits, image_emb, text_emb):
        # c = F.dropout(logits, self.dropout, training=self.training)
        # print(c)
        c = self.confidence_MLP(logits)

        image_emb = F.dropout(image_emb, self.dropout, training=self.training)
        image_emb = self.img_MLP(image_emb)
        text_emb = self.text_MLP(text_emb)

        x = torch.cat((c, image_emb, text_emb), dim=-1)
        x = F.relu(x)

        # x = F.dropout(x, self.dropout, training=self.training)
        x = self.out(x)
        # Softplus
        self.t = torch.log(torch.exp(x) + torch.tensor(1.1))
        # t = torch.ones((75,1)).to(self.device)
        # print(t.shape)
        # print(confidence.shape)
        # exit()
        # print(logits.max(1)[0])
        # print(self.t)
        return logits * self.t
        # print(torch.softmax(logits * t, dim=1).max(1)[0])
        # exit()
        #
        # return confidence*t


class CalibrationNet():
    """Calibration network"""

    def __init__(self, config, label_to_idx, classes, alpha, device):
        super(CalibrationNet, self).__init__()
        self.config = config
        self.device = device
        self.clip_model, self.transform = clip.load(
            self.config.VIS_ENCODER, device=self.device
        )
        self.label_to_idx  = label_to_idx
        self.tau = config.tau
        self.classes = classes
        self.meta_net = None
        self.best_model = None

        self.template = self.config.PROMPT_TEMPLATE
        self.alpha = alpha


    def sigmoid(self, z):
        return 1 / (1+torch.exp(-z))

    def NSF(self, x):
        numerator = self.sigmoid((x - 0.5) / self.tau) - self.sigmoid(torch.tensor([-0.5]).to(self.device) / self.tau)
        denominator = self.sigmoid(torch.tensor([0.5]).to(self.device) / self.tau) - self.sigmoid(torch.tensor([-0.5]).to(self.device) / self.tau)
        return numerator / denominator

    def NSF_loss(self, prob, y):
        '''
        NRM - PRM
        '''
        # print(conf.max(1)[0])
        # exit()
        conf, predictions  = torch.max(prob, 1)
        # print(predictions)
        # print(y)
        # print(self.sigmoid(conf))
        # exit()
        correct_i = torch.where(predictions == y)
        incorrect_i = torch.where(predictions != y)
        loss = (torch.sum(1-self.NSF(conf[correct_i])) + torch.sum(self.NSF(conf[incorrect_i]))) / conf.size()[0]
        # loss = torch.mean(1 - self.NSF(conf[correct_i])) + torch.mean(self.NSF(conf[incorrect_i]))
        return  0.5*loss

    def label_smoothing(self, targets, num_classes, epsilon=0.1):
        """
        Convert class indices to smoothed one-hot vectors.
        """
        # one-hot
        one_hot = F.one_hot(targets, num_classes=num_classes).float()
        # label smoothing
        smooth = one_hot * (1 - epsilon) + (epsilon / num_classes)
        return smooth

    def instance_weight_smooth_CE(self, preds, targets, num_classes):
        """
        For misclassified samples, the higher the confidence, the higher weight should to apply to constrain.
        Compute cross entropy loss with unifrom label smoothing.
        preds: logits (before softmax), shape (batch_size, num_classes)
        targets: class indices, shape (batch_size,)
        """
        smoothed_labels = self.label_smoothing(targets, num_classes, (num_classes-1)/num_classes)
        log_probs = F.log_softmax(preds, dim=1)
        probs = F.softmax(preds, dim=1)
        # log_probs = torch.log(probs)
        # print(log_probs)
        # Calculate instance-level weights
        conf = torch.max(probs, 1)[0]
        instance_weights = self.NSF(conf)

        if self.config.fusion == 'sum':
            loss = -(torch.sum(smoothed_labels * log_probs, dim=1) ).mean() + torch.mean(instance_weights)# /mean_weight # nll loss
        elif self.config.fusion == 'mul':
            loss = -(torch.sum(smoothed_labels * log_probs, dim=1) * instance_weights).mean()


        # print(loss)
        # exit()
        return loss

    def instance_weight_CE(self, preds, y, num_class):
        """
        For right predicted samples, the higher the confidence, the lower weight should to apply to constrain.
        """
        log_probs = F.log_softmax(preds, dim=1)
        probs = F.softmax(preds, dim=1)
        # log_probs = torch.log(probs)
        # print(log_probs)
        # Calculate instance-level weights
        conf = torch.max(probs, 1)[0]
        instance_weights = 1 - self.NSF(conf)
        # instance_weights = instance_weights / instance_weights.mean()
        # mean_weight = instance_weights.mean()

        # print("correct_mean weight", str(mean_weight))


        y = F.one_hot(y, num_class)

        if self.config.fusion == 'sum':
            loss = -(torch.sum(y * log_probs, dim=1)).mean() + torch.mean(instance_weights) # Entropy loss
        elif self.config.fusion == 'mul':
            loss = -(torch.sum(y * log_probs, dim=1) * instance_weights).mean()
        # loss = -(torch.sum(y * log_probs, dim=1)).mean()
        return loss


    def DCE_loss(self, preds, y):
        predictions = torch.max(preds, 1)[1]
        correct_i = torch.where(predictions == y)
        incorrect_i = torch.where(predictions != y)
        if self.alpha == 1.0:
            loss = self.alpha*self.instance_weight_CE(preds[correct_i], y[correct_i], preds.shape[1]) + \
                   (1.01-self.alpha)*self.instance_weight_smooth_CE(preds[incorrect_i], y[incorrect_i], preds.shape[1])
        else:
            loss = self.alpha * self.instance_weight_CE(preds[correct_i], y[correct_i], preds.shape[1]) + \
                   (1.0 - self.alpha) * self.instance_weight_smooth_CE(preds[incorrect_i], y[incorrect_i],
                                                                        preds.shape[1])


        return loss

    # def forward(self, confidence, image_emb):
    #     encoded_text = self.clip_model.encode_text(text)
    #     return encoded_text


    def OOD_loss(self, preds, y):
        # This is a baseline method of OOD detection
        predictions = torch.max(preds, 1)[1]
        correct_i = torch.where(predictions == y)
        incorrect_i = torch.where(predictions != y)

        cor_preds = preds[correct_i]
        cor_y = y[correct_i]
        incor_preds = preds[incorrect_i]
        incor_y = y[incorrect_i]

        cor_log_probs = F.log_softmax(cor_preds, dim=1)
        cor_probs = F.softmax(cor_preds, dim=1)
        cor_conf = torch.max(cor_probs, 1)[0]

        incor_log_probs = F.log_softmax(incor_preds, dim=1)
        incor_probs = F.softmax(incor_preds, dim=1)
        incor_conf = torch.max(incor_probs, 1)[0]

        num_class = preds.shape[1]
        cor_y = F.one_hot(cor_y, num_class)

        # print(cor_y.shape)
        # print(cor_log_probs.shape)
        # print(cor_conf.shape)
        # exit()
        loss_cor = -(torch.sum(cor_y * cor_log_probs, dim=1) * (1-cor_conf)).mean()
        loss_incor = (torch.sum(incor_probs * incor_log_probs, dim=1) * incor_conf).mean()

        return loss_cor + 0.5*loss_incor



    def train_cn(self, train_data, val_data):

        train_loader = torch.utils.data.DataLoader(
            train_data,
            batch_size=len(train_data),
            shuffle=True,
            worker_init_fn=seed_worker,
            generator=g,
        )
        val_loader = torch.utils.data.DataLoader(
            val_data, batch_size=len(val_data)# self.config.BATCH_SIZE
        )

        # Declare data pre-processing
        train_data.transform = self.transform
        val_data.transform = self.transform

        best_val_accuracy = 0
        best_prompt = None
        loss = None
        if val_loader is not None:
            log.info(f"Size of validation dataset: {len(val_data.filepaths)}")

        # Build textual prompts
        prompts = [
            self.template.format(" ".join(i.split("_"))) for i in self.classes
        ]
        log.info(f"Number of prompts: {len(prompts)}")  # a photo of XXX

        # Encode text
        text = clip.tokenize(prompts).to(self.device)

        for i, (img, _, _, label, img_path) in enumerate(train_loader):  # batch_size = len(train_data)
            # Calculate the output probability of CLIP
            with torch.no_grad():
                logits_per_image_train, logits_per_text = self.clip_model(  # [16, 47],   [47, 16]
                    img.to(self.device), text.to(self.device)
                ) # already multiply logit_score
                # logit_scale = self.clip_model.logit_scale.exp()
                # print(logit_scale)
                # exit()
                logits_per_image_train = logits_per_image_train.float()
                probs = logits_per_image_train.softmax(dim=-1).float()
                idx_preds = torch.argmax(probs, dim=1)
                # print(probs.max(dim=1)[0])
                # exit()
            with torch.no_grad():
                img_embs = self.clip_model.encode_image(img.to(self.device)).float()
                text_embs = self.clip_model.encode_text(text).float()

                # img_embs = img_embs / img_embs.norm(dim=-1, keepdim=True)
                # text_embs = text_embs / text_embs.norm(dim=-1, keepdim=True)
                p_text_embs = text_embs[torch.argmax(img_embs @ text_embs.t(), dim=-1)] # Predicted Test Embs


        for i, (val_img, _, _, val_label, img_path) in enumerate(val_loader):  # batch_size = len(train_data)
            # Calculate the output probability of CLIP
            with torch.no_grad():
                logits_per_image_val, logits_per_text = self.clip_model(  # [16, 47],   [47, 16]
                    val_img.to(self.device), text.to(self.device)
                )
                # logit_scale = self.clip_model.logit_scale.exp()
                # print(logit_scale)
                # exit()

                logits_per_image_val = logits_per_image_val.float()
                val_probs = logits_per_image_val.softmax(dim=-1).float()
                idx_val_preds = torch.argmax(val_probs, dim=1)
            with torch.no_grad():
                val_img_embs = self.clip_model.encode_image(val_img.to(self.device)).float()
                val_text_embs = self.clip_model.encode_text(text).float()

                # val_img_embs = val_img_embs / val_img_embs.norm(dim=-1, keepdim=True)
                # val_text_embs = val_text_embs / val_text_embs.norm(dim=-1, keepdim=True)

                val_p_text_embs = val_text_embs[torch.argmax(val_img_embs @ val_text_embs.t(), dim=-1)]

        label = label.to(self.device)
        val_label = val_label.to(self.device)
        self.meta_net = CN(self.config, self.label_to_idx, self.classes, self.device).to(self.device)

        self.optimizer = torch.optim.Adam(
            self.meta_net.parameters(),
            lr=self.config.META_LR,
            weight_decay=self.config.DECAY
        )
        # print(self.config.META_LR)
        # exit()
        nll = nn.CrossEntropyLoss().to(self.device)

        # print(logits_per_image_train.sort(dim=-1)[0])
        # print(logits_per_image_val.sort(dim=-1)[0])


        best = 999999999
        losses = []
        # Training the meta net.
        for epoch in range(self.config.EPOCHS):
            self.meta_net.train()
            self.optimizer.zero_grad()

            # Train
            cali_logits = self.meta_net(logits_per_image_train, img_embs, p_text_embs)
            # print(self.meta_net.t)
            # # 差一个softmax
            # prob = torch.softmax(cali_logits, dim=1)
            # nsf_loss = self.NSF_loss(prob, label)
            # # nll loss
            # nll_loss = nll(cali_logits, label)
            # loss = 0.0*nll_loss + 1.0*nsf_loss
            loss = self.DCE_loss(cali_logits, label) # self.OOD_loss(cali_logits, label) #
            losses.append(loss.detach().cpu().numpy().item())

            loss.backward()
            self.optimizer.step()

            # for name, param in self.meta_net.named_parameters():
            #     if param.requires_grad:
            #         print(name, param.grad)  #

            # validation
            self.meta_net.eval()
            with torch.no_grad():
                val_logits = self.meta_net(logits_per_image_val, val_img_embs, val_p_text_embs)
                # val_prob = F.softmax(val_logits, dim=1)
                # nsf_loss = self.NSF_loss(val_prob, val_label)
                # # nll loss
                # nll_loss = nll(val_logits, val_label)
                # val_loss = 0.0*nll_loss + 1.0*nsf_loss

                val_loss = self.DCE_loss(val_logits, val_label)

            log.info(f"Training loss after Epoch {epoch}: {loss}, Val loss: {val_loss}")
            # print(f"Training loss after Epoch {epoch}: {loss}, Val loss: {val_loss}")

            if val_loss < best:
                best = val_loss
                self.best_model = copy.deepcopy(self.meta_net)

            # self.best_model = copy.deepcopy(self.meta_net)
        torch.save(self.best_model,
                   "./saved_model/" + self.config.DATASET_NAME + "_" + str(self.config.N_LABEL) + ".pth")
        # print(self.best_model.t)
        # exit()
        # losses = np.array(losses).reshape(1,-1).squeeze().tolist()
        # print(range(len(losses)))
        # print(losses)
        # import matplotlib.pyplot as plt
        # plt.plot(range(len(losses)), losses)
        # plt.show()
        # exit()


    def test_cn(self, test_data):

        test_data.transform = self.transform
        # Define the data loader
        test_loader = torch.utils.data.DataLoader(
            test_data, batch_size=self.config.BATCH_SIZE
        )

        # Build textual prompts
        prompts = [
            self.template.format(" ".join(i.split("_"))) for i in self.classes
        ]
        log.info(f"Number of prompts: {len(prompts)}")  # a photo of XXX

        # Encode text
        text = clip.tokenize(prompts).to(self.device)
        text_features = self.clip_model.encode_text(text)

        log.info(f"Start inference for test data")
        predictions = []
        images = []
        prob_preds = []
        clip_logits = []
        taus = []


        for img, _, _, img_path in test_loader:
            with torch.no_grad():
                logits_per_image, logits_per_text = self.clip_model(  # [16, 47],   [47, 16]
                    img.to(self.device), text.to(self.device)
                )

                logits_per_image = logits_per_image.float()
                probs = logits_per_image.softmax(dim=-1)
                idx_preds = torch.argmax(probs, dim=1)


                # Calibration
                img_embs = self.clip_model.encode_image(img.to(self.device)).float()
                test_text_embs = self.clip_model.encode_text(text).float()

                # img_embs = img_embs / img_embs.norm(dim=-1, keepdim=True)
                # test_text_embs = test_text_embs / test_text_embs.norm(dim=-1, keepdim=True)

                test_p_text_embs = test_text_embs[torch.argmax(img_embs @ test_text_embs.t(), dim=-1)]

                self.best_model.eval()
                cali_test_logits = self.best_model(logits_per_image, img_embs, test_p_text_embs)
                # print(self.best_model.t)

                # preparing tau for visualiztion.
                taus += [self.best_model.t]

                predictions += [self.classes[i] for i in idx_preds]
                images += [i for i in img_path]
                clip_logits += [logits_per_image]
                prob_preds += [cali_test_logits]  # probs, logits_per_image

        clip_logits = torch.cat(clip_logits, axis=0).detach().to('cpu') # the original logits outputed by CLIP
        prob_preds = torch.cat(prob_preds, axis=0).detach().to('cpu')
        df_predictions = pd.DataFrame({"id": images, "class": predictions})
        taus = torch.cat(taus).detach().to('cpu')

        return df_predictions, images, predictions, prob_preds, clip_logits #, taus




def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)