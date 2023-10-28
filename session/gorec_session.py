import os
import time
import torch
from tqdm import tqdm
import sys
import criterion
import tool
from metric import evaluation
from metric import pgd_evaluate
import numpy as np
from collections import defaultdict
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import torch.nn.functional as F


class GoRec_session(object):
    def __init__(self, env, model, loader):
        self.env = env
        self.model = model
        self.dataset = loader
        self.data_loadr = torch.utils.data.DataLoader(self.dataset, batch_size=env.args.batch_size, shuffle=True)
        self.optimizer = torch.optim.Adam(
            [{'params': filter(lambda p: p.requires_grad, self.model.parameters()), 'lr': self.env.args.lr},])
        self.criterion_mse = torch.nn.MSELoss()
        # self.criterion_infonce = criterion.InfoNCE(env.args.ssl_temp)
        self.criterion_l2 = criterion.l2_regularization
        self.criterion_uni = criterion.uniformity
        self.criterion_kl = torch.nn.KLDivLoss()
        self.early_stop = 0
        self.best_epoch = 0
        self.total_epoch = 0
        
        self.best_ndcg = defaultdict(float)
        self.best_hr = defaultdict(float)
        self.best_recall = defaultdict(float)
        self.test_ndcg = defaultdict(float)
        self.test_hr = defaultdict(float)
        self.test_recall = defaultdict(float)

    def train_epoch(self):
        t = time.time()
        self.model.train()
        self.total_epoch += 1

        all_loss, all_rec_loss, all_uni_loss, all_kl_loss = 0., 0., 0., 0.
        for indexs in self.data_loadr:
            warm = self.dataset.item_emb[indexs].to(self.env.device)
            side_information = torch.tensor(self.dataset.feature[indexs], dtype=torch.float32).to(self.env.device)
            side_information = torch.nn.functional.normalize(side_information)
            rec_warm, mu, log_variances, z, zgc = self.model(warm, side_information)

            rec_loss = self.criterion_mse(rec_warm, warm.to(self.env.device))
            uni_loss = self.criterion_uni(mu)
            uni_loss= self.env.args.uni_coeff * uni_loss
            kl_loss = self.criterion_kl(z, zgc)
            kl_loss = self.env.args.kl_coeff * kl_loss

            loss = rec_loss + kl_loss + uni_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            all_loss += loss
            all_rec_loss += rec_loss
            all_kl_loss += kl_loss
            all_uni_loss += uni_loss

        return all_loss / len(self.data_loadr), all_rec_loss / len(self.data_loadr), all_uni_loss / len(self.data_loadr), all_kl_loss / len(self.data_loadr), time.time() - t
 
    def train(self, epochs):
        for epoch in range(self.env.args.ckpt_start_epoch, epochs):
            loss, rec_loss, uni_loss, kl_loss, train_time = self.train_epoch()
            print(
                f'TRAIN:epoch = {epoch}/{epochs} loss = {loss:.5f}, rec_loss = {rec_loss:.5f}, kl_loss = {kl_loss:.5f}, train_time = {train_time:.2f}')

            if epoch % self.env.args.eva_interval == 0:
                self.early_stop += self.env.args.eva_interval
                range4eval = self.env.args.range4eval
                hr, recall, ndcg, val_time = self.evaluate_model(eval(self.env.args.topk), mode='val', range4eval=range4eval)
                thr, trecall, tndcg, test_time = self.evaluate_model(eval(self.env.args.topk), mode='test', range4eval=range4eval)

                if self.env.args.tensorboard:
                    for key in hr.keys():
                        self.env.w.add_scalar(
                            f'Val/hr@{key}', hr[key], self.total_epoch)
                        self.env.w.add_scalar(
                            f'Val/recall@{key}', hr[key], self.total_epoch)
                        self.env.w.add_scalar(
                            f'Val/ndcg@{key}', ndcg[key], self.total_epoch)
                key = list(hr.keys())[0]
                print(
                    f'epoch = {epoch} hr@{key} = {hr[key]:.5f}, recall@{key} = {recall[key]:.5f}, ndcg@{key} = {ndcg[key]:.5f}, val_time = {val_time+test_time:.2f}')

                if ndcg[list(hr.keys())[0]] > self.best_ndcg[list(hr.keys())[0]]:
                    # self.update_memory()
                    
                    self.early_stop = 0
                    for key in hr.keys():
                        tool.cprint(
                            f'epoch = {epoch} hr@{key} = {hr[key]:.5f}, recall@{key} = {recall[key]:.5f}, ndcg@{key} = {ndcg[key]:.5f}, val_time = {val_time:.2f}')

                    for key in hr.keys():
                        self.best_hr[key] = hr[key]
                        self.best_recall[key] = recall[key]
                        self.best_ndcg[key] = ndcg[key]
                        self.test_hr[key] = thr[key]
                        self.test_recall[key] = trecall[key]
                        self.test_ndcg[key] = tndcg[key]

                    if self.env.args.save:
                        self.save_model(epoch)
                        print('save ckpt')
                    self.best_epoch = epoch
                    if self.env.args.log:
                        self.env.val_logger.info(f'EPOCH[{epoch}/{epochs}]')
                        for key in hr.keys():
                            self.env.val_logger.info(
                                f'hr@{key} = {hr[key]:.5f}, recall@{key} = {recall[key]:.5f}, ndcg@{key} = {ndcg[key]:.5f}, val_time = {val_time:.2f}')

                        self.env.test_logger.info(f'EPOCH[{epoch}/{epochs}]')
                        for key in thr.keys():
                            self.env.test_logger.info(
                                f'hr@{key} = {thr[key]:.5f}, recall@{key} = {trecall[key]:.5f}, ndcg@{key} = {tndcg[key]:.5f}, test_time = {test_time:.2f}')
            if self.env.args.log:
                self.env.train_logger.info(
                    f'EPOCH[{epoch}/{epochs}] loss = {loss:.5f}, rec_loss = {rec_loss:.5f}, uni_loss = {uni_loss:.5f}, kl_loss = {kl_loss:.5f}')

            if self.env.args.tensorboard:
                self.env.w.add_scalar(f'Train/loss', loss, self.total_epoch)
                self.env.w.add_scalar(
                    f'Train/rec_loss', rec_loss, self.total_epoch)
                self.env.w.add_scalar(
                    f'Train/uni_loss', uni_loss, self.total_epoch)
                self.env.w.add_scalar(
                    f'Train/kl_loss', kl_loss, self.total_epoch)

            if self.early_stop > self.env.args.early_stop:
                break



    def evaluate_model(self, top_list, mode='test', range4eval='all'):
        self.model.eval()
        t = time.time()

        user_emb = self.dataset.user_emb.detach().cpu().numpy()
        item_emb = self.dataset.item_emb

        side_information = torch.tensor(self.dataset.feature, dtype=torch.float32).to(self.env.device)
        side_information = torch.nn.functional.normalize(side_information)

        item_emb = self.dataset.cluster_cfmean.to(self.env.device)
        rec_warm = self.model(item_emb, side_information, gen_size=self.dataset.m_item)
        item_emb = rec_warm.detach().cpu().numpy()

        if mode == 'test':
            if range4eval == 'all':
                hr, recall, ndcg = evaluation.num_faiss_evaluate(self.dataset.test_data,
                                                                    list(
                                                                        self.dataset.test_data.keys()),
                                                                    list(
                                                                        range(self.dataset.m_item)),
                                                                    self.dataset.train_data,
                                                                    top_list, user_emb, item_emb)
            elif range4eval == 'cold':
                hr, ndcg, _, recall = pgd_evaluate.evaluate(self.dataset.test_data,
                                                self.dataset.test_user_list,
                                                self.dataset.train_data,
                                                set(self.dataset.cold_item_index),
                                                top_list, user_emb, item_emb, 4)
        elif mode == 'val':
            if range4eval == 'all':
                hr, recall, ndcg = evaluation.num_faiss_evaluate(self.dataset.val_data,
                                                                    list(
                                                                        self.dataset.val_data.keys()),
                                                                    list(
                                                                        range(self.dataset.m_item)),
                                                                    self.dataset.train_data,
                                                                    top_list, user_emb, item_emb)
            elif range4eval == 'cold':
                hr, ndcg, _, recall = pgd_evaluate.evaluate(self.dataset.val_data,
                                                self.dataset.val_user_list,
                                                self.dataset.train_data,
                                                set(self.dataset.cold_item_index),
                                                top_list, user_emb, item_emb, 4)
        return hr, recall, ndcg, time.time() - t

    def save_ckpt(self, path):
        torch.save(self.model.state_dict(), path)

    def save_model(self, current_epoch):
        model_state_file = os.path.join(
            self.env.CKPT_PATH, f'{self.env.args.suffix}_epoch{current_epoch}.pth')
        self.save_ckpt(model_state_file)
        if self.best_epoch is not None and current_epoch != self.best_epoch:
            old_model_state_file = os.path.join(
                self.env.CKPT_PATH, f'{self.env.args.suffix}_epoch{self.best_epoch}.pth')
            if os.path.exists(old_model_state_file):
                os.system('rm {}'.format(old_model_state_file))
