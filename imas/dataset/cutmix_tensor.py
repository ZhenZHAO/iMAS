import numpy as np
import random
import torch
import scipy.stats as stats


def rand_bbox(size, lam=None):
    # past implementation
    if len(size) == 4:
        W = size[2]
        H = size[3]
    elif len(size) == 3:
        W = size[1]
        H = size[2]
    else:
        raise Exception
    B = size[0]
    
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    cx = np.random.randint(size=[B, ], low=int(W/8), high=W)
    cy = np.random.randint(size=[B, ], low=int(H/8), high=H)
    
    
#     print(cx,"\n", cy, "\n", cx - cut_w // 2)
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)

    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    
#     print("="*20)
#     print(cut_w, cut_h)
#     print([(x,y) for x,y in zip(bbx2-bbx1, bby2-bby1)])

    return bbx1, bby1, bbx2, bby2


def rand_bbox_by_v(size, v=0.2):
    # past implementation
    if len(size) == 4:
        W = size[2]
        H = size[3]
    elif len(size) == 3:
        W = size[1]
        H = size[2]
    else:
        raise Exception
    B = size[0]
    
    cut_rat = v
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    cx = np.random.randint(size=[B, ], low=int(W/16), high=W)
    cy = np.random.randint(size=[B, ], low=int(H/16), high=H)
    
#     print(cx,"\n", cy, "\n", cx - cut_w // 2)
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)

    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    
#     print("="*20)
#     print(cut_w, cut_h)
#     print([(x,y) for x,y in zip(bbx2-bbx1, bby2-bby1)])

    return bbx1, bby1, bbx2, bby2


# # # # # # # # # # # # # # # # # # # # # 
# # 1.1 cutmix using beta distr
# # # # # # # # # # # # # # # # # # # # # 
def cut_mix(unlabeled_image, unlabeled_mask, unlabeled_logits):
    mix_unlabeled_image = unlabeled_image.clone()
    mix_unlabeled_target = unlabeled_mask.clone()
    mix_unlabeled_logits = unlabeled_logits.clone()
    
    # get the random mixing objects
    u_rand_index = torch.randperm(unlabeled_image.size()[0])[:unlabeled_image.size()[0]]
    # print(u_rand_index)
    
    # get box
    u_bbx1, u_bby1, u_bbx2, u_bby2 = rand_bbox(unlabeled_image.size(), lam=np.random.beta(4, 4))
    
    # cut & paste
    for i in range(0, mix_unlabeled_image.shape[0]):
        mix_unlabeled_image[i, :, u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]] = \
            unlabeled_image[u_rand_index[i], :, u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]]
        # label is of 3 dimensions
#         mix_unlabeled_target[i, :, u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]] = \
#             unlabeled_mask[u_rand_index[i], :, u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]]
        mix_unlabeled_target[i, u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]] = \
            unlabeled_mask[u_rand_index[i], u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]]
        
        mix_unlabeled_logits[i, u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]] = \
            unlabeled_logits[u_rand_index[i], u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]]

    del unlabeled_image, unlabeled_mask, unlabeled_logits

    return mix_unlabeled_image, mix_unlabeled_target, mix_unlabeled_logits


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # 1.2 cutmix using hardness (cut easy for hard, cut hard for easy)
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
def cut_mix_by_hardness_beta(unlabeled_image, unlabeled_mask, unlabeled_logits,
                           hardness=0.99):
    # note
    # in this implementation, the larger hardness means the instance is easier to learn
    # # # # # # # # # # # # # # # # # # 
    mix_unlabeled_image = unlabeled_image.clone()
    mix_unlabeled_target = unlabeled_mask.clone()
    mix_unlabeled_logits = unlabeled_logits.clone()
    
    # 1. get box
    u_bbx1, u_bby1, u_bbx2, u_bby2 = rand_bbox(unlabeled_image.size(), lam=np.random.beta(4, 4))
    
    # 2. get hardness
    if (isinstance(hardness, list) or isinstance(hardness, tuple)) and len(hardness) == unlabeled_image.shape[0]:
        hardness_lst = np.array(hardness)
        flag_different_hardness = True
    elif isinstance(hardness, float):
        flag_different_hardness = False
    else:
        flag_different_hardness = False
        raise ValueError
    
    # 3. get index
    if flag_different_hardness:
        tmp_dict = {x:y for x,y in zip(np.argsort(hardness_lst), np.argsort(hardness_lst)[::-1])}
        u_rand_index = [tmp_dict[i] for i in range(len(hardness_lst))]
    else:
        u_rand_index = torch.randperm(unlabeled_image.size()[0])[:unlabeled_image.size()[0]]
    
    # 4. apply  cutmix
    for i in range(0, mix_unlabeled_image.shape[0]):
        mix_unlabeled_image[i, :, u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]] = \
            unlabeled_image[u_rand_index[i], :, u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]]

        mix_unlabeled_target[i, u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]] = \
            unlabeled_mask[u_rand_index[i], u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]]
        
        mix_unlabeled_logits[i, u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]] = \
            unlabeled_logits[u_rand_index[i], u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]]
    
    # return mixed results
    del unlabeled_image, unlabeled_mask, unlabeled_logits
    return mix_unlabeled_image, mix_unlabeled_target, mix_unlabeled_logits



# # # # # # # # # # # # # # # # # # # # # 
# # 2.1 cutmix using uniform range
# # # # # # # # # # # # # # # # # # # # # 
def cut_mix_using_v(unlabeled_image, unlabeled_mask, unlabeled_logits, scale=[0.3, 0.7]):
    mix_unlabeled_image = unlabeled_image.clone()
    mix_unlabeled_target = unlabeled_mask.clone()
    mix_unlabeled_logits = unlabeled_logits.clone()
    
    # 0. get the random mixing objects
    u_rand_index = torch.randperm(unlabeled_image.size()[0])[:unlabeled_image.size()[0]]
    # print(u_rand_index)
    
    # 1. get box
    min_v, max_v = min(scale), max(scale)
    v = float(max_v - min_v) * random.random()
    v += min_v
    u_bbx1, u_bby1, u_bbx2, u_bby2 = rand_bbox_by_v(unlabeled_image.size(), v=v)
    
    # 2. cut & paste
    for i in range(0, mix_unlabeled_image.shape[0]):
        mix_unlabeled_image[i, :, u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]] = \
            unlabeled_image[u_rand_index[i], :, u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]]
        
        mix_unlabeled_target[i, u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]] = \
            unlabeled_mask[u_rand_index[i], u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]]
        
        mix_unlabeled_logits[i, u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]] = \
            unlabeled_logits[u_rand_index[i], u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]]

    del unlabeled_image, unlabeled_mask, unlabeled_logits

    return mix_unlabeled_image, mix_unlabeled_target, mix_unlabeled_logits



# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # 2.2 cutmix using hardness for adjusting trigger probability
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
def cut_mix_by_hardness_for_prob(unlabeled_image, unlabeled_mask, unlabeled_logits, 
                                 hardness=0.99, max_prob=0.8, scale=[0.3, 0.7], flag_hardness_batch=True):
    mix_unlabeled_image = unlabeled_image.clone()
    mix_unlabeled_target = unlabeled_mask.clone()
    mix_unlabeled_logits = unlabeled_logits.clone()
    
    # 0. get the random mixing objects
    u_rand_index = torch.randperm(unlabeled_image.size()[0])[:unlabeled_image.size()[0]]
    
    # 1. get box
    min_v, max_v = min(scale), max(scale)
    v = float(max_v - min_v) * random.random()
    v += min_v
    u_bbx1, u_bby1, u_bbx2, u_bby2 = rand_bbox_by_v(unlabeled_image.size(), v=v)
    
    # 2. get hardness
    if (isinstance(hardness, list) or isinstance(hardness, tuple)) and len(hardness) == unlabeled_image.shape[0]:
        hardness_lst = np.array(hardness)
    elif isinstance(hardness, float):
        hardness_lst = np.array([hardness] * unlabeled_image.shape[0])
    else:
        raise ValueError
    hardness_lst *= max_prob
    
    # 3. judge batch-hardness
    if flag_hardness_batch:
        avg_hardness = np.mean(hardness_lst)
        if np.random.uniform() > avg_hardness:
            # print("Not apply cutmix - batch")
            return unlabeled_image, unlabeled_mask, unlabeled_logits
        else:
            pass
            # print("will apply cutmix - batch")
    
    # 4. apply  cutmix
    for i in range(0, mix_unlabeled_image.shape[0]):
        # judge instance-hardness
        if not flag_hardness_batch and np.random.uniform() > hardness_lst[i]:
            continue
        mix_unlabeled_image[i, :, u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]] = \
            unlabeled_image[u_rand_index[i], :, u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]]
        
        mix_unlabeled_target[i, u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]] = \
            unlabeled_mask[u_rand_index[i], u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]]
        
        mix_unlabeled_logits[i, u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]] = \
            unlabeled_logits[u_rand_index[i], u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]]

    del unlabeled_image, unlabeled_mask, unlabeled_logits

    return mix_unlabeled_image, mix_unlabeled_target, mix_unlabeled_logits


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # 2.3. cutmix using hardness (cut easy for hard, cut hard for easy)
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
def get_adpative_magnituide(v_min, v_max, confidence, flag_tough_max=True, sigma=None):
    assert 0 <= confidence <= 1
    # mean
    if flag_tough_max:
        var_mu = v_min + (v_max - v_min) * confidence 
    else:
        var_mu = v_max - (v_max - v_min) * confidence 
    # sigma
    if sigma is None:
        var_sigma = max(var_mu - v_min, v_max - var_mu)
        var_sigma /=3.0
    else:
        var_sigma = sigma
    # print("="*10,f"mu:{var_mu}, std:{var_sigma}")
    # truncated norm
    a = (v_min - var_mu) / var_sigma
    b = (v_max - var_mu) / var_sigma
    rv = stats.truncnorm(a,b, loc=var_mu, scale=var_sigma)
    return rv.rvs()


def cut_mix_by_hardness(unlabeled_image, unlabeled_mask, unlabeled_logits,
                           hardness=0.99, scale=[0.55, 0.95], 
                           flag_hardness_random=False, 
                           flag_hardness_gaussion=False):
    # note
    # in this implementation, the larger hardness means the instance is easier to learn
    # # # # # # # # # # # # # # # # # # 
    mix_unlabeled_image = unlabeled_image.clone()
    mix_unlabeled_target = unlabeled_mask.clone()
    mix_unlabeled_logits = unlabeled_logits.clone()
    
    # 1. get hardness
    avg_hardness = None
    if (isinstance(hardness, list) or isinstance(hardness, tuple)) and len(hardness) == unlabeled_image.shape[0]:
        hardness_lst = np.array(hardness)
        flag_different_hardness = True
        avg_hardness = hardness_lst.mean()
    elif isinstance(hardness, float):
        flag_different_hardness = False
        avg_hardness = hardness
    else:
        flag_different_hardness = False
        raise ValueError
    
    # 2. get box
    min_v, max_v = min(scale), max(scale)
    if flag_hardness_gaussion:
        v = get_adpative_magnituide(min_v, max_v, avg_hardness, flag_tough_max=True)
    else:
        v = float(max_v - min_v) * random.random()
        if flag_hardness_random:
            v *= avg_hardness
        v += min_v

    u_bbx1, u_bby1, u_bbx2, u_bby2 = rand_bbox_by_v(unlabeled_image.size(), v=v)
    
    
    
    # 3. get index
    if flag_different_hardness:
        tmp_dict = {x:y for x,y in zip(np.argsort(hardness_lst), np.argsort(hardness_lst)[::-1])}
        u_rand_index = [tmp_dict[i] for i in range(len(hardness_lst))]
    else:
        u_rand_index = torch.randperm(unlabeled_image.size()[0])[:unlabeled_image.size()[0]]
    
    # 4. apply  cutmix
    for i in range(0, mix_unlabeled_image.shape[0]):
        mix_unlabeled_image[i, :, u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]] = \
            unlabeled_image[u_rand_index[i], :, u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]]

        mix_unlabeled_target[i, u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]] = \
            unlabeled_mask[u_rand_index[i], u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]]
        
        mix_unlabeled_logits[i, u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]] = \
            unlabeled_logits[u_rand_index[i], u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]]
    
    # return mixed results
    del unlabeled_image, unlabeled_mask, unlabeled_logits
    return mix_unlabeled_image, mix_unlabeled_target, mix_unlabeled_logits
