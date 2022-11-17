import numpy as np
import scipy.stats as stats
from PIL import Image, ImageOps, ImageFilter, ImageEnhance
import random
import collections
import cv2
import torch
from torchvision import transforms


# # # # # # # # # # # # # # # # # # # # # # # # 
# # # 1. Augmentation for image and labels
# # # # # # # # # # # # # # # # # # # # # # # # 
class Compose(object):
    def __init__(self, segtransforms):
        self.segtransforms = segtransforms

    def __call__(self, image, label, modifier=0.999):
        for idx, t in enumerate(self.segtransforms):
            if isinstance(t, strong_img_aug):
                image = t(image, modifier)
            else:
                image, label = t(image, label)
        return image, label


class ToTensorAndNormalize(object):
    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        assert len(mean) == len(std)
        assert len(mean) == 3
        self.normalize = transforms.Normalize(mean, std)
        self.to_tensor = transforms.ToTensor()

    def __call__(self, in_image, in_label):
        in_image = Image.fromarray(np.uint8(in_image))
        image = self.normalize(self.to_tensor(in_image))
        label = torch.from_numpy(np.array(in_label, dtype=np.int32)).long()

        return image, label


class Resize(object):
    def __init__(self, base_size, ratio_range, scale=True, bigger_side_to_base_size=True):
        assert isinstance(ratio_range, collections.Iterable) and len(ratio_range) == 2
        self.base_size = base_size
        self.ratio_range = ratio_range
        self.scale = scale
        self.bigger_side_to_base_size = bigger_side_to_base_size

    def __call__(self, in_image, in_label):
        w, h = in_image.size
        
        if isinstance(self.base_size, int):
            # obtain long_side
            if self.scale:
                long_side = random.randint(int(self.base_size * self.ratio_range[0]), 
                                        int(self.base_size * self.ratio_range[1]))
            else:
                long_side = self.base_size
                
            # obtain new oh, ow
            if self.bigger_side_to_base_size:
                if h > w:
                    oh = long_side
                    ow = int(1.0 * long_side * w / h + 0.5)
                else:
                    oh = int(1.0 * long_side * h / w + 0.5)
                    ow = long_side
            else:
                oh, ow = (long_side, int(1.0 * long_side * w / h + 0.5)) if h < w else (
                        int(1.0 * long_side * h / w + 0.5), long_side)
                
            image = in_image.resize((ow, oh), Image.BILINEAR)
            label = in_label.resize((ow, oh), Image.NEAREST)
            return image, label
        elif (isinstance(self.base_size, list) or isinstance(self.base_size, tuple)) and len(self.base_size) == 2:
            if self.scale:
                # scale = random.random() * 1.5 + 0.5  # Scaling between [0.5, 2]
                scale = self.ratio_range[0] + random.random() * (self.ratio_range[1] - self.ratio_range[0])
                # print("="*100, h, self.base_size[0])
                # print("="*100, w, self.base_size[1])
                oh, ow = int(self.base_size[0] * scale), int(self.base_size[1] * scale)
            else:
                oh, ow = self.base_size
            image = in_image.resize((ow, oh), Image.BILINEAR)
            label = in_label.resize((ow, oh), Image.NEAREST)
            # print("="*100, in_image.size, image.size)
            return image, label

        else:
            raise ValueError


class Crop(object):
    def __init__(self, crop_size, crop_type="rand", mean=[0.485, 0.456, 0.406], ignore_value=255):
        if (isinstance(crop_size, list) or isinstance(crop_size, tuple)) and len(crop_size) == 2:
            self.crop_h, self.crop_w = crop_size
        elif isinstance(crop_size, int):
            self.crop_h, self.crop_w = crop_size, crop_size
        else:
            raise ValueError
        
        self.crop_type = crop_type
        self.image_padding = (np.array(mean) * 255.).tolist()
        self.ignore_value = ignore_value

    def __call__(self, in_image, in_label):
        # Padding to return the correct crop size
        w, h = in_image.size
        pad_h = max(self.crop_h - h, 0)
        pad_w = max(self.crop_w - w, 0)
        pad_kwargs = {
            "top": 0,
            "bottom": pad_h,
            "left": 0,
            "right": pad_w,
            "borderType": cv2.BORDER_CONSTANT, 
        }
        if pad_h > 0 or pad_w > 0:
            image = cv2.copyMakeBorder(np.asarray(in_image, dtype=np.float32), 
                                       value=self.image_padding, **pad_kwargs)
            label = cv2.copyMakeBorder(np.asarray(in_label, dtype=np.int32), 
                                       value=self.ignore_value, **pad_kwargs)
            image = Image.fromarray(np.uint8(image))
            label = Image.fromarray(np.uint8(label))
        else:
            image = in_image
            label = in_label
        
        # cropping
        w, h = image.size
        if self.crop_type == "rand":
            x = random.randint(0, w - self.crop_w)
            y = random.randint(0, h - self.crop_h)
        else:
            x = (w - self.crop_w) // 2
            y = (h - self.crop_h) // 2
        image = image.crop((x, y, x + self.crop_w, y + self.crop_h))
        label = label.crop((x, y, x + self.crop_w, y + self.crop_h))
        return image, label


class RandomFlip(object):
    def __init__(self, prob=0.5, flag_hflip=True,):
        self.prob = prob
        if flag_hflip:
            self.type_flip = Image.FLIP_LEFT_RIGHT
        else:
            self.type_flip = Image.FLIP_TOP_BOTTOM
            
    def __call__(self, in_image, in_label):
        if random.random() < self.prob:
            in_image = in_image.transpose(self.type_flip)
            in_label = in_label.transpose(self.type_flip)
        return in_image, in_label


# # # # # # # # # # # # # # # # # # # # # # # # 
# # # 2. Strong Augmentation for image only
# # # # # # # # # # # # # # # # # # # # # # # # 

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
        # var_sigma /=3.0
        var_sigma /=2.0
    else:
        var_sigma = sigma
    # print("="*10,f"mu:{var_mu}, std:{var_sigma}")
    # truncated norm
    a = (v_min - var_mu) / var_sigma
    b = (v_max - var_mu) / var_sigma
    rv = stats.truncnorm(a,b, loc=var_mu, scale=var_sigma)
    return rv.rvs()


def img_aug_identity(img, modifier=0.99, scale=None, flag_map_random=True, flag_map_gaussian=False):
    return img


def img_aug_autocontrast(img, modifier=0.99, scale=None, flag_map_random=True, flag_map_gaussian=False):
    return ImageOps.autocontrast(img)


def img_aug_equalize(img, modifier=.99, scale=None, flag_map_random=True, flag_map_gaussian=False):
    return ImageOps.equalize(img)


def img_aug_invert(img, modifier=.99, scale=None, flag_map_random=True, flag_map_gaussian=False):
    return ImageOps.invert(img)


def img_aug_blur(img, modifier=0.99, scale=[0.1, 2.0], flag_map_random=True, flag_map_gaussian=False):
    # sigma = np.random.uniform(scale[0], scale[1])
    assert 0<= modifier <= 1.0
    min_v, max_v = min(scale), max(scale)
    if flag_map_gaussian:
        sigma = get_adpative_magnituide(min_v, max_v, modifier, flag_tough_max=True)
    else:
        v = float(max_v - min_v)
        if flag_map_random:
            v *= random.random()
        v *= modifier
        sigma = min_v + v
    # print(f"sigma:{sigma}")
    return img.filter(ImageFilter.GaussianBlur(radius=sigma))


def img_aug_contrast(img, modifier=0.99, scale=[0.05, 0.95], flag_map_random=True, flag_map_gaussian=False):
    assert 0<= modifier <= 1.0
    min_v, max_v = min(scale), max(scale)
    if flag_map_gaussian:
        v = get_adpative_magnituide(min_v, max_v, modifier, flag_tough_max=False)
    else:
        v = float(max_v - min_v)
        if flag_map_random:
            v *= random.random()
        v *= modifier
        v = max_v - v
    # v = float(max_v - min_v)*random.random()
    # # print(min_v, max_v, v)
    # v *= modifier
    # v = max_v - v
    # # print(f"final:{v}")
    return ImageEnhance.Contrast(img).enhance(v)


def img_aug_brightness(img, modifier=0.99, scale=[0.15, 0.95], flag_map_random=True, flag_map_gaussian=False):
    assert 0<= modifier <= 1.0
    min_v, max_v = min(scale), max(scale)
    if flag_map_gaussian:
        v = get_adpative_magnituide(min_v, max_v, modifier, flag_tough_max=False)
    else:
        v = float(max_v - min_v)
        if flag_map_random:
            v *= random.random()
        v *= modifier
        v = max_v - v

    # v = float(max_v - min_v)*random.random()
    # # print(min_v, max_v, v)
    # v *= modifier
    # v = max_v - v
    # # print(f"final:{v}")
    return ImageEnhance.Brightness(img).enhance(v)


def img_aug_color(img, modifier=0.99, scale=[0.1, 0.95], flag_map_random=True, flag_map_gaussian=False):
    assert 0<= modifier <= 1.0
    min_v, max_v = min(scale), max(scale)

    if flag_map_gaussian:
        v = get_adpative_magnituide(min_v, max_v, modifier, flag_tough_max=False)
    else:
        v = float(max_v - min_v)
        if flag_map_random:
            v *= random.random()
        v *= modifier
        v = max_v - v

    # v = float(max_v - min_v)*random.random()
    # # print(min_v, max_v, v)
    # v *= modifier
    # v = max_v - v
    # # print(f"final:{v}")
    return ImageEnhance.Color(img).enhance(v)


def img_aug_sharpness(img, modifier=0.99, scale=[0.05, 0.95], flag_map_random=True, flag_map_gaussian=False):
    assert 0<= modifier <= 1.0
    min_v, max_v = min(scale), max(scale)

    if flag_map_gaussian:
        v = get_adpative_magnituide(min_v, max_v, modifier, flag_tough_max=False)
    else:
        v = float(max_v - min_v)
        if flag_map_random:
            v *= random.random()
        v *= modifier
        v = max_v - v

    # v = float(max_v - min_v)*random.random()
    # # print(min_v, max_v, v)
    # v *= modifier
    # v = max_v - v
    # # print(f"final:{v}")

    return ImageEnhance.Sharpness(img).enhance(v)


def img_aug_hue(img, modifier=0.99, scale=[0, 0.5], flag_map_random=True, flag_map_gaussian=False):
    assert 0<= modifier <= 1.0
    min_v, max_v = min(scale), max(scale)
    if flag_map_gaussian:
        v = get_adpative_magnituide(min_v, max_v, modifier, flag_tough_max=False)
    else:
        v = float(max_v - min_v)
        if flag_map_random:
            v *= random.random()
        v *= modifier
        v += min_v

    # v = float(max_v - min_v)*random.random()
    # v *= modifier
    # v += min_v
    if np.random.random() < 0.5:
        hue_factor = -v
    else:
        hue_factor = v
    # print(f"Final-V:{hue_factor}")
    input_mode = img.mode
    if input_mode in {"L", "1", "I", "F"}:
        return img
    h, s, v = img.convert("HSV").split()
    np_h = np.array(h, dtype=np.uint8)
    # uint8 addition take cares of rotation across boundaries
    with np.errstate(over="ignore"):
        np_h += np.uint8(hue_factor * 255)
    h = Image.fromarray(np_h, "L")
    img = Image.merge("HSV", (h, s, v)).convert(input_mode)
    return img


def img_aug_posterize(img, modifier=0.99, scale=[4, 8], flag_map_random=True, flag_map_gaussian=False):
    assert 0<= modifier <= 1.0
    min_v, max_v = min(scale), max(scale)
    v = float(max_v - min_v)*random.random()
    
    if flag_map_gaussian:
        v = get_adpative_magnituide(min_v, max_v, modifier, flag_tough_max=False)
        v = int(np.ceil(v))
    else:
        v = float(max_v - min_v)
        if flag_map_random:
            v *= random.random()
        v *= modifier
        v = int(np.ceil(v))
        v = max(1, v)
        v = max_v - v
        
    # # print(min_v, max_v, v)
    # v *= modifier
    # v = int(np.ceil(v))
    # v = max(1, v)
    # v = max_v - v
    # # print(f"final:{v}")
    return ImageOps.posterize(img, v)


def img_aug_solarize(img, modifier=0.99, scale=[1, 256], flag_map_random=True, flag_map_gaussian=False):
    assert 0<= modifier <= 1.0
    min_v, max_v = min(scale), max(scale)

    if flag_map_gaussian:
        v = get_adpative_magnituide(min_v, max_v, modifier, flag_tough_max=False)
        v = int(np.ceil(v))
        v = max(1, v)
        v = min(v, 256)
    else:
        v = float(max_v - min_v)
        if flag_map_random:
            v *= random.random()
        v *= modifier
        v = int(np.ceil(v))
        v = max_v - v
        v = max(1, v)
        v = min(v, 256)
        

    # v = float(max_v - min_v)*random.random()
    # # print(min_v, max_v, v)
    # v *= modifier
    # v = int(np.ceil(v))
    # v = max(1, v)
    # v = max_v - v
    # # print(f"final:{v}")
    return ImageOps.solarize(img, v)

def augment_list():  
    l = [
        (img_aug_identity, None),
        (img_aug_autocontrast, None),
        (img_aug_equalize, None),
        (img_aug_invert, None),  # hard comment for 10 augs
        (img_aug_blur, [0.1, 2.0]),
        (img_aug_contrast, [0.1, 0.95]),
        (img_aug_brightness, [0.1, 0.95]),
        (img_aug_color, [0.1, 0.95]),
        (img_aug_sharpness, [0.1, 0.95]),
        (img_aug_posterize, [4, 8]),
        (img_aug_solarize, [1, 256]), # hard comment for 11/10 augs
        (img_aug_hue, [0, 0.5])
    ]
    return l



# def augment_list_wide():  
#     l = [
#         (img_aug_identity, None),
#         (img_aug_autocontrast, None),
#         (img_aug_equalize, None),
#         (img_aug_invert, None),  # hard comment for 10 augs
#         (img_aug_blur, [0.1, 2.0]),
#         (img_aug_contrast, [0.05, 0.95]),
#         (img_aug_brightness, [0.05, 0.95]),
#         (img_aug_color, [0.05, 0.95]),
#         (img_aug_sharpness, [0.05, 0.95]),
#         (img_aug_posterize, [4, 8]),
#         (img_aug_solarize, [1, 256]), # hard comment for 11/10 augs
#         (img_aug_hue, [0, 0.5])
#     ]
#     return l


class strong_img_aug:
    def __init__(self, num_augs, flag_map_random=True, flag_map_gaussian=False):
        self.n = num_augs
        self.augment_list = augment_list()
        self.flag_map_random = flag_map_random
        self.flag_map_gaussian = flag_map_gaussian
        
    def __call__(self, img, modifier=0.999):
        ops = random.choices(self.augment_list, k=self.n)
        for op, scales in ops:
            # print("="*20, str(op))
            img = op(img, modifier, scales, self.flag_map_random, self.flag_map_gaussian)
        return img
