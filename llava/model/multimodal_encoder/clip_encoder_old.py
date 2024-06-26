import torch
import torch.nn as nn

from transformers import CLIPVisionModel, CLIPImageProcessor, CLIPVisionConfig,SiglipImageProcessor, SiglipVisionModel
import open_clip
from collections import namedtuple

class CLIPVisionTower(nn.Module):
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__()

        self.is_loaded = False

        self.vision_tower_name = vision_tower
        self.select_layer = args.mm_vision_select_layer
        self.select_feature = getattr(args, 'mm_vision_select_feature', 'patch')

        if not delay_load:
            self.load_model()
        elif getattr(args, 'unfreeze_mm_vision_tower', False):
            self.load_model()
        else:
            self.cfg_only = CLIPVisionConfig.from_pretrained(self.vision_tower_name)

    def load_model(self, device_map=None):
        if self.is_loaded:
            print('{} is already loaded, `load_model` called again, skipping.'.format(self.vision_tower_name))
            return

        self.image_processor = CLIPImageProcessor.from_pretrained(self.vision_tower_name)
        self.vision_tower = CLIPVisionModel.from_pretrained(self.vision_tower_name, device_map=device_map)
        self.vision_tower.requires_grad_(False)

        self.is_loaded = True

    def feature_select(self, image_forward_outs):
        image_features = image_forward_outs.hidden_states[self.select_layer]
        if self.select_feature == 'patch':
            image_features = image_features[:, 1:]
        elif self.select_feature == 'cls_patch':
            image_features = image_features
        else:
            raise ValueError(f'Unexpected select feature: {self.select_feature}')
        return image_features

    @torch.no_grad()
    def forward(self, images):
        if type(images) is list:
            image_features = []
            for image in images:
                image_forward_out = self.vision_tower(image.to(device=self.device, dtype=self.dtype).unsqueeze(0), output_hidden_states=True)
                image_feature = self.feature_select(image_forward_out).to(image.dtype)
                image_features.append(image_feature)
        else:
            image_forward_outs = self.vision_tower(images.to(device=self.device, dtype=self.dtype), output_hidden_states=True)
            image_features = self.feature_select(image_forward_outs).to(images.dtype)

        return image_features

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return self.vision_tower.dtype

    @property
    def device(self):
        return self.vision_tower.device

    @property
    def config(self):
        if self.is_loaded:
            return self.vision_tower.config
        else:
            return self.cfg_only

    @property
    def hidden_size(self):
        return self.config.hidden_size

    @property
    def num_patches_per_side(self):
        return self.config.image_size // self.config.patch_size

    @property
    def num_patches(self):
        return (self.config.image_size // self.config.patch_size) ** 2



class CLIPVisionTowerS2(CLIPVisionTower):
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__(vision_tower, args, delay_load)

        self.s2_scales = getattr(args, 's2_scales', '336,672,1008')
        self.s2_scales = list(map(int, self.s2_scales.split(',')))
        self.s2_scales.sort()
        self.s2_split_size = self.s2_scales[0]
        self.s2_image_size = self.s2_scales[-1]

        try:
            from s2wrapper import forward as multiscale_forward
        except ImportError:
            raise ImportError('Package s2wrapper not found! Please install by running: \npip install git+https://github.com/bfshi/scaling_on_scales.git')
        self.multiscale_forward = multiscale_forward

        # change resize/crop size in preprocessing to the largest image size in s2_scale
        if not delay_load or getattr(args, 'unfreeze_mm_vision_tower', False):
            self.image_processor.size['shortest_edge'] = self.s2_image_size
            self.image_processor.crop_size['height'] = self.image_processor.crop_size['width'] = self.s2_image_size

    def load_model(self, device_map=None):
        if self.is_loaded:
            print('{} is already loaded, `load_model` called again, skipping.'.format(self.vision_tower_name))
            return

        self.image_processor = CLIPImageProcessor.from_pretrained(self.vision_tower_name)
        self.vision_tower = CLIPVisionModel.from_pretrained(self.vision_tower_name, device_map='cuda:0')
        self.vision_tower.requires_grad_(False)

        self.image_processor.size['shortest_edge'] = self.s2_image_size
        self.image_processor.crop_size['height'] = self.image_processor.crop_size['width'] = self.s2_image_size

        self.is_loaded = True

    @torch.no_grad()
    def forward_feature(self, images):
        image_forward_outs = self.vision_tower(images.to(device=self.device, dtype=self.dtype), output_hidden_states=True)
        image_features = self.feature_select(image_forward_outs).to(images.dtype)
        return image_features

    @torch.no_grad()
    def forward(self, images):
        if type(images) is list:
            image_features = []
            for image in images:
                image_feature = self.multiscale_forward(self.forward_feature, image.unsqueeze(0), img_sizes=self.s2_scales, max_split_size=self.s2_split_size)
                image_features.append(image_feature)
        else:
            image_features = self.multiscale_forward(self.forward_feature, images, img_sizes=self.s2_scales, max_split_size=self.s2_split_size)

        return image_features

    @property
    def hidden_size(self):
        return self.config.hidden_size * len(self.s2_scales)



# bio_image_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch16")
# bio_openai_vision_tower = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch16")
# bio_vision_config = bio_openai_vision_tower.config

# del bio_openai_vision_tower
        

# bio_model, _, _ = open_clip.create_model_and_transforms('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
# biovision_tower=bio_model.visual.trunk
# biovision_tower.requires_grad_(False)
# setattr(biovision_tower, 'config', bio_vision_config)
# biovision_tower.requires_grad_(False)


class CLIPVisionTower2(nn.Module):
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__()

        self.is_loaded = False
        self.cache_dir = "./cache_dir"
        self.vision_tower_name = vision_tower
        self.select_layer = args.mm_vision_select_layer
        self.select_feature = getattr(args, 'mm_vision_select_feature', 'patch')
        self.delay_load = False
        if not delay_load:
            self.load_model()
        elif getattr(args, 'unfreeze_mm_vision_tower', False):
            self.load_model()
        else:
            self.cfg_only = CLIPVisionConfig.from_pretrained(self.vision_tower_name)

    def load_hf_vision_tower(self, vision_tower_name, device_map="cuda:0"):
        image_processor = CLIPImageProcessor.from_pretrained(vision_tower_name)
        vision_tower = CLIPVisionModel.from_pretrained(vision_tower_name)
        return vision_tower, image_processor
    
    def load_open_clip_vision_tower(self, vision_tower_name, device_map="cuda:0"):
        open_clip_model, open_clip_processor, _ = open_clip.create_model_and_transforms(vision_tower_name, precision=torch.bfloat16)
        cfg = open_clip.CLIPVisionCfg()
        def preprocess(image,**kwargs):
            pixel_values = open_clip_processor(image).unsqueeze(0)
            return dict(pixel_values=pixel_values)
    
        image_processor_placeholder = namedtuple('image_processor_placeholder', ['preprocess', 'image_mean'])
        image_processor = image_processor_placeholder(preprocess=preprocess, image_mean=open_clip_processor.transforms[3].mean)
        vision_tower = open_clip_model.visual 
        cfg.hidden_size = vision_tower.trunk.embed_dim
        vision_tower.config = cfg
        vision_tower.to(device_map)
        return vision_tower, image_processor
    
    def load_siglip_vision_tower(self, vision_tower_name, device_map="cuda:0"):
        image_processor = SiglipImageProcessor.from_pretrained(vision_tower_name, cache_dir=self.cache_dir)
        vision_tower = SiglipVisionModel.from_pretrained(vision_tower_name, cache_dir=self.cache_dir)
        return vision_tower, image_processor

    def load_model(self, device_map=None):
        if self.is_loaded:
            print('{} is already loaded, `load_model` called again, skipping.'.format(self.vision_tower_name))
            return
        device_map='cuda:0'
        self.vision_tower, self.image_processor = self.load_hf_vision_tower(vision_tower_name=self.vision_tower_name, device_map=device_map)
        self.vision_tower_derma, self.image_processor_derma = self.load_hf_vision_tower(vision_tower_name="jiviai/jivi_derma_clip_patch_14_336")
        # self.vision_tower_open_clip, self.image_processor_open_clip = self.load_open_clip_vision_tower(device_map)
        self.is_loaded = True
        self.vision_tower_derma.requires_grad_(False)
        self.vision_tower.requires_grad_(False)

    def feature_select(self, image_forward_outs):
        image_features = image_forward_outs.hidden_states[self.select_layer]
        if self.select_feature == 'patch':
            image_features = image_features[:, 1:]
        elif self.select_feature == 'cls_patch':
            image_features = image_features
        else:
            raise ValueError(f'Unexpected select feature: {self.select_feature}')
        return image_features
    
    def feature_select_open_clip(self, image_forward_outs):
        image_features = image_forward_outs[self.select_layer]
        return image_features
    
    def feature_select_siglip(self, image_forward_outs):
        image_features = image_forward_outs.hidden_states[self.select_layer]
        if self.select_feature == 'patch':
            image_features = image_features[:, 1:]
        elif self.select_feature == 'cls_patch':
            image_features = image_features
        else:
            raise ValueError(f'Unexpected select feature: {self.select_feature}')
        return image_features


    @torch.no_grad()
    def forward(self, images,images_derma):
        if type(images) is list:
            image_features = []
            image_features_derma=[]
            for image,image_derma in zip(images,images_derma):
                image_forward_out = self.vision_tower(image.to(device=self.device, dtype=self.dtype).unsqueeze(0), output_hidden_states=True)
                
                image_feature = self.feature_select(image_forward_out).to(image.dtype)
                
                ####
                image_forward_out_derma = self.vision_tower_derma(image_derma.to(device=self.vision_tower_derma.device, dtype=self.dtype), output_hidden_states=True)
                image_features_derma = self.feature_select(image_forward_out_derma).to(images.dtype)    
                # image_forward_out_open_clip = self.vision_tower_open_clip(image_open_clip.to(device=self.device, dtype=self.dtype).unsqueeze(0), output_hidden_states=True)
                # image_feature_open_clip = self.feature_select(image_forward_out_open_clip).to(image.dtype)
                # # print(image_feature_open_clip.shape)
                image_features_derma.append(image_features_derma)
                ####
                image_features.append(image_feature)
        else:
            image_forward_outs = self.vision_tower(images.to(device=self.vision_tower.device, dtype=self.dtype), output_hidden_states=True)
            image_features = self.feature_select(image_forward_outs).to(images.dtype)

            image_forward_outs_derma = self.vision_tower_derma(images_derma.to(device=self.vision_tower_derma.device, dtype=self.dtype), output_hidden_states=True)
            image_features_derma = self.feature_select(image_forward_outs_derma).to(images.dtype)
            
            ############################
            # image_forward_out_open_clip = self.vision_tower_open_clip(images_derma.to(device=self.vision_tower_open_clip.device, dtype=self.dtype), output_)
            # image_features_derma = self.feature_select_open_clip(image_forward_out_open_clip).to(images.dtype)
            # print(image_features_derma.shape)
            ##############################
        return image_features,image_features_derma

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return self.vision_tower.dtype

    @property
    def device(self):
        return self.vision_tower.device

    @property
    def config(self):
        if self.is_loaded:
            return self.vision_tower.config
        else:
            return self.cfg_only

    @property
    def hidden_size(self):
        return self.config.hidden_size

    @property
    def num_patches_per_side(self):
        return self.config.image_size // self.config.patch_size

    @property
    def num_patches(self):
        return (self.config.image_size // self.config.patch_size) ** 2
