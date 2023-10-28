import torch
import os
import numpy as np
import scipy.sparse as sp
from collections import defaultdict
from tqdm import tqdm
import faiss

class Loader4AE(torch.utils.data.Dataset):
    def __init__(self, env):

        self.env = env

        if env.args.dataset == 'baby':
            self.n_user, self.m_item = 19445, 7050
        elif env.args.dataset == 'clothing':
            self.n_user, self.m_item = 39387, 23033
        elif env.args.dataset == 'sports':
            self.n_user, self.m_item = 35598, 18357
        
        train_file = os.path.join(self.env.DATA_PATH, 'train.txt')
        val_file = os.path.join(self.env.DATA_PATH, 'val.txt')
        test_file = os.path.join(self.env.DATA_PATH, 'test.txt')

        self.train_data = defaultdict(list)
        count = 0
        with open(train_file) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip('\n').split(' ')
                    items = [int(i) for i in l[1:]]
                    count += len(items)
                    uid = int(l[0])
                    self.train_data[uid].extend(items)

        self.val_user_list = []
        self.val_data = defaultdict(list)
        with open(val_file) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip('\n').split(' ')
                    uid = int(l[0])
                    if l[1] == '':
                        continue
                    else:
                        items = [int(i) for i in l[1:]]
                    self.val_user_list.append(uid)
                    self.val_data[uid].extend(items)
                    
        self.test_user_list = []
        self.test_data = defaultdict(list)
        with open(test_file) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip('\n').split(' ')
                    uid = int(l[0])
                    if l[1] == '':
                        continue
                    else:
                        items = [int(i) for i in l[1:]]
                    self.test_data[uid].extend(items)
                    self.test_user_list.append(uid)
        all_item_index = np.array(list(range(self.m_item)))
        self.cold_item_index = np.load(os.path.join(self.env.DATA_PATH, 'cold_item_index.npy'))
        self.warm_item_index = np.setdiff1d(all_item_index, self.cold_item_index, assume_unique=False)

        self.image_feat, self.text_feat = self.load_mutimedia_feature()
        self.feature = np.concatenate([self.image_feat, self.text_feat], axis=1)
        self.user_emb, self.item_emb = self.load_cf_embedding(self.env.args.pre_model)
        self.cluster_label = np.load(os.path.join(self.env.DATA_PATH, f'cluster_label.npy'))

        cluster_cfmean = []
        for i in range(self.env.args.pre_cluster_num):
            cluster_cfmean.append(self.item_emb[np.where(self.cluster_label==i)].mean(0))
        cluster_cfmean = torch.stack(cluster_cfmean)

        self.cluster_cfmean = cluster_cfmean[self.cluster_label]

    def load_mutimedia_feature(self):
        image_file = os.path.join(self.env.DATA_PATH, 'image_feat.npy')
        text_file = os.path.join(self.env.DATA_PATH, 'text_feat.npy')
        image_feat = np.load(image_file)
        text_feat = np.load(text_file)
        return image_feat, text_feat

    def load_cf_embedding(self, pre_model):
        if pre_model == 'lgcn':
            user_file = os.path.join(self.env.DATA_PATH, f'lgcn_uemb_{self.env.args.dataset}.pt')
            item_file = os.path.join(self.env.DATA_PATH, f'lgcn_iemb_{self.env.args.dataset}.pt')
        elif pre_model == 'mf':
            user_file = os.path.join(self.env.DATA_PATH, f'mf_uemb_{self.env.args.dataset}.pt')
            item_file = os.path.join(self.env.DATA_PATH, f'mf_iemb_{self.env.args.dataset}.pt')
        elif pre_model == 'vbpr':
            user_file = os.path.join(self.env.DATA_PATH, f'vbpr_uemb_{self.env.args.dataset}.pt')
            item_file = os.path.join(self.env.DATA_PATH, f'vbpr_iemb_{self.env.args.dataset}.pt')
        elif pre_model == 'vbprm':
            user_file = os.path.join(self.env.DATA_PATH, f'vbprm_uemb_{self.env.args.dataset}.pt')
            item_file = os.path.join(self.env.DATA_PATH, f'vbprm_iemb_{self.env.args.dataset}.pt')
        elif pre_model == 'simgcl':
            user_file = os.path.join(self.env.DATA_PATH, f'simgcl_uemb_{self.env.args.dataset}.pt')
            item_file = os.path.join(self.env.DATA_PATH, f'simgcl_iemb_{self.env.args.dataset}.pt')
        elif pre_model == 'lgcnf':
            user_file = os.path.join(self.env.DATA_PATH, f'lgcnf_uemb_{self.env.args.dataset}.pt')
            item_file = os.path.join(self.env.DATA_PATH, f'lgcnf_iemb_{self.env.args.dataset}.pt')
        elif pre_model == 'grcn':
            user_file = os.path.join(self.env.DATA_PATH, f'grcn_uemb_{self.env.args.dataset}.pt')
            item_file = os.path.join(self.env.DATA_PATH, f'grcn_iemb_{self.env.args.dataset}.pt')
        elif pre_model == 'vsgcl':
            user_file = os.path.join(self.env.DATA_PATH, f'vsgcl_uemb_{self.env.args.dataset}.pt')
            item_file = os.path.join(self.env.DATA_PATH, f'vsgcl_iemb_{self.env.args.dataset}.pt')
        elif pre_model == 'vlgcn':
            user_file = os.path.join(self.env.DATA_PATH, f'vlgcn_uemb_{self.env.args.dataset}.pt')
            item_file = os.path.join(self.env.DATA_PATH, f'vlgcn_iemb_{self.env.args.dataset}.pt')
        user_emb = torch.load(user_file, map_location=torch.device('cpu'))
        item_emb = torch.load(item_file, map_location=torch.device('cpu'))
        return user_emb, item_emb
      
    def __getitem__(self, index):
        warm_item = self.warm_item_index[index]
        return warm_item

    def __len__(self):
        return len(self.warm_item_index)