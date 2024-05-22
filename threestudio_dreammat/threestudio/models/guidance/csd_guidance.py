import random
from contextlib import contextmanager
from dataclasses import dataclass, field
import math
import cv2
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import (
    DDPMScheduler,
    DPMSolverMultistepScheduler,
    StableDiffusionPipeline,
    StableDiffusionControlNetPipeline,
    UNet2DConditionModel,
    ControlNetModel,
    DDIMScheduler,
)
from diffusers.loaders import AttnProcsLayers
from diffusers.models.attention_processor import LoRAAttnProcessor
from diffusers.models.embeddings import TimestepEmbedding
from diffusers.utils.import_utils import is_xformers_available
from threestudio.utils.ops import perpendicular_component

import threestudio
from threestudio.models.prompt_processors.base import PromptProcessorOutput
from threestudio.utils.base import BaseObject
from threestudio.utils.misc import C, cleanup, parse_version
from threestudio.utils.typing import *
from controlnet_aux import NormalBaeDetector, CannyDetector, PidiNetDetector, HEDdetector
import pdb

class ToWeightsDType(nn.Module):
    def __init__(self, module: nn.Module, dtype: torch.dtype):
        super().__init__()
        self.module = module
        self.dtype = dtype

    def forward(self, x: Float[Tensor, "..."]) -> Float[Tensor, "..."]:
        return self.module(x).to(self.dtype)


@threestudio.register("stable-diffusion-triple-guidance")
class StableDiffusionCSDGuidance(BaseObject):
    @dataclass
    class Config(BaseObject.Config):
        width:int=512
        height:int=512
        cache_dir: str = "../../model/SD"
        pretrained_model_name_or_path: str = "runwayml/stable-diffusion-v1-5"
        controlnet_path:str = None
        enable_memory_efficient_attention: bool = False
        enable_sequential_cpu_offload: bool = False
        enable_attention_slicing: bool = False
        enable_channels_last_format: bool = False
        half_precision_weights: bool = True

        # set up for potential control net
        use_controlnet: bool = False
        condition_scale: float = 1.5
        control_anneal_start_step: Optional[int] = None
        control_anneal_end_scale: Optional[float] = None
        control_types: List = field(default_factory=lambda: ['depth', 'canny'])  
        condition_scales: List = field(default_factory=lambda: [1.0, 1.0])
        condition_scales_anneal: List = field(default_factory=lambda: [1.0, 1.0])
        p2p_condition_type: str = 'p2p'
        canny_lower_bound: int = 50
        canny_upper_bound: int = 100

        min_step_percent: float = 0.02
        max_step_percent: float = 0.98

        cond_scale: float = 1
        uncond_scale: float = 0
        null_scale: float = -1
        noise_scale: float = 0
        perpneg_scale: float = 0.0

        view_dependent_prompting: bool = True

        grad_clip_val: Optional[float] = None
        grad_normalize: Optional[
            bool
        ] = False 

    cfg: Config

    def configure(self) -> None:
        threestudio.info(f"Loading Stable Diffusion ...")
        self.use_controlnet = self.cfg.use_controlnet

        self.weights_dtype = (
            torch.float16 if self.cfg.half_precision_weights else torch.float32
        )

        if self.use_controlnet:
            self.start_condition_scale = self.cfg.condition_scale
            multi_controlnet_processor = []
            for i, control_type in enumerate(self.cfg.control_types):
                if control_type == 'normal':
                    controlnet_name_or_path: str = "lllyasviel/control_v11p_sd15_normalbae"
                    self.preprocessor_normal = NormalBaeDetector.from_pretrained(
                        "lllyasviel/Annotators",
                        cache_dir=self.cfg.cache_dir,
                        #local_files_only=True
                    )
                    self.preprocessor_normal.model.to(self.device)
                elif control_type == 'canny':
                    controlnet_name_or_path: str = "lllyasviel/control_v11p_sd15_canny"
                    self.preprocessor_canny = CannyDetector()

                elif control_type == 'self-normal':
                    controlnet_name_or_path: str = "lllyasviel/control_v11p_sd15_normalbae"
                elif control_type == 'hed':
                    controlnet_name_or_path: str = "lllyasviel/control_v11p_sd15_scribble"
                    self.preprocessor_hed = HEDdetector.from_pretrained(
                        'lllyasviel/Annotators',
                        cache_dir=self.cfg.cache_dir,
                        #local_files_only=True
                        )
                elif control_type == 'p2p':
                    controlnet_name_or_path: str = "lllyasviel/control_v11e_sd15_ip2p"
                elif control_type == 'depth':
                    controlnet_name_or_path: str = "lllyasviel/control_v11f1p_sd15_depth"
                elif control_type == 'light':
                    controlnet_name_or_path: str = self.cfg.controlnet_path
                print(control_type)
                controlnet = ControlNetModel.from_pretrained(
                    controlnet_name_or_path,
                    torch_dtype=self.weights_dtype,
                    cache_dir=self.cfg.cache_dir,
                    local_files_only=True
                )

                multi_controlnet_processor.append(controlnet)

            pipe_kwargs = {
                #"tokenizer": None,
                "safety_checker": None,
                "feature_extractor": None,
                "controlnet": multi_controlnet_processor,
                "requires_safety_checker": False,
                "torch_dtype": self.weights_dtype,
                "cache_dir": self.cfg.cache_dir,
                #"local_files_only": True,
            }

            self.pipe = StableDiffusionControlNetPipeline.from_pretrained(
                self.cfg.pretrained_model_name_or_path,
                **pipe_kwargs).to(self.device)

            #self.pipe.unet.load_attn_procs("lora-library/watercolor",cache_dir=self.cfg.cache_dir)

            #self.scheduler.set_timesteps(self.cfg.diffusion_steps)
        else:
            pipe_kwargs = {
                "tokenizer": None,
                "safety_checker": None,
                "feature_extractor": None,
                "requires_safety_checker": False,
                "torch_dtype": self.weights_dtype,
                "cache_dir": self.cfg.cache_dir,
                #"local_files_only": True,
            }
            self.pipe = StableDiffusionPipeline.from_pretrained(
                self.cfg.pretrained_model_name_or_path,
                **pipe_kwargs,
            ).to(self.device)
        
        self.scheduler = DDIMScheduler.from_pretrained(
                #self.cfg.ddim_scheduler_name_or_path,
                self.cfg.pretrained_model_name_or_path,
                subfolder="scheduler",
                torch_dtype=self.weights_dtype,
                #cache_dir=self.cfg.cache_dir,
                cache_dir=self.cfg.cache_dir,
                #local_files_only=True,
                )
        if self.cfg.enable_memory_efficient_attention:
            if parse_version(torch.__version__) >= parse_version("2"):
                threestudio.info(
                    "PyTorch2.0 uses memory efficient attention by default."
                )
            elif not is_xformers_available():
                threestudio.warn(
                    "xformers is not available, memory efficient attention is not enabled."
                )
            else:
                self.pipe.enable_xformers_memory_efficient_attention()

        if self.cfg.enable_sequential_cpu_offload:
            self.pipe.enable_sequential_cpu_offload()

        if self.cfg.enable_attention_slicing:
            self.pipe.enable_attention_slicing(1)

        if self.cfg.enable_channels_last_format:
            self.pipe.unet.to(memory_format=torch.channels_last)

        # Create model
        self.vae = self.pipe.vae.eval()
        self.unet = self.pipe.unet.eval()
        if self.use_controlnet:
            self.controlnet = self.pipe.controlnet.eval()

        for p in self.vae.parameters():
            p.requires_grad_(False)
        for p in self.unet.parameters():
            p.requires_grad_(False)


        self.num_train_timesteps = self.scheduler.config.num_train_timesteps
        self.set_min_max_steps()

        self.alphas: Float[Tensor, "..."] = self.scheduler.alphas_cumprod.to(
            self.device
        )

        self.noise_scale = 0.0
        self.cond_scale = 1.0
        self.uncond_scale = -0.0
        self.null_scale = -1.0
        self.perpneg_scale = 0.0
        

        threestudio.info(f"Loaded Stable Diffusion!")


    def multi_control_forward(
            self,
            sample: torch.FloatTensor,
            timestep: Union[torch.Tensor, float, int],
            encoder_hidden_states: torch.Tensor,
            controlnet_cond: List[torch.tensor],
            conditioning_scale: List[float],
            class_labels: Optional[torch.Tensor] = None,
            timestep_cond: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            cross_attention_kwargs: Optional[Dict[str, Any]] = None,
    ):
        for i, (image, scale, controlnet) in enumerate(zip(controlnet_cond, conditioning_scale, self.controlnet.nets)):
            down_samples, mid_sample = controlnet(
                sample.to(self.weights_dtype),
                timestep.to(self.weights_dtype),
                encoder_hidden_states.to(self.weights_dtype),
                image.to(self.weights_dtype),
                scale,
                class_labels,
                timestep_cond,
                attention_mask,
                cross_attention_kwargs,
                return_dict=False,
            )

            # merge samples
            if i == 0:
                down_block_res_samples, mid_block_res_sample = down_samples, mid_sample
            else:
                down_block_res_samples = [
                    samples_prev + samples_curr
                    for samples_prev, samples_curr in zip(down_block_res_samples, down_samples)
                ]
                mid_block_res_sample += mid_sample

        return down_block_res_samples, mid_block_res_sample

    @torch.cuda.amp.autocast(enabled=False)
    def forward_controlnet(
        self,
        latents: Float[Tensor, "..."],
        t: Float[Tensor, "..."],
        image_cond: Float[Tensor, "..."],
        condition_scale: float,
        encoder_hidden_states: Float[Tensor, "..."],
    ) -> Float[Tensor, "..."]:
        return self.controlnet(
            latents.to(self.weights_dtype),
            t.to(self.weights_dtype),
            encoder_hidden_states=encoder_hidden_states.to(self.weights_dtype),
            controlnet_cond=image_cond.to(self.weights_dtype),
            conditioning_scale=condition_scale,
            return_dict=False,
        )

    @torch.cuda.amp.autocast(enabled=False)
    def forward_unet(
        self,
        unet: UNet2DConditionModel,
        latents: Float[Tensor, "..."],
        t: Float[Tensor, "..."],
        encoder_hidden_states: Float[Tensor, "..."],
        class_labels: Optional[Float[Tensor, "B 16"]] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        down_block_additional_residuals: Optional[Float[Tensor, "..."]] = None,
        mid_block_additional_residual: Optional[Float[Tensor, "..."]] = None
    ) -> Float[Tensor, "..."]:
        input_dtype = latents.dtype
        return unet(
            latents.to(self.weights_dtype),
            t.to(self.weights_dtype),
            encoder_hidden_states=encoder_hidden_states.to(self.weights_dtype),
            class_labels=class_labels,
            cross_attention_kwargs=cross_attention_kwargs,
            down_block_additional_residuals=down_block_additional_residuals,
            mid_block_additional_residual=mid_block_additional_residual
        ).sample.to(input_dtype)

    @torch.cuda.amp.autocast(enabled=False)
    def encode_images(
        self, imgs: Float[Tensor, "B 3 512 512"]
    ) -> Float[Tensor, "B 4 64 64"]:
        input_dtype = imgs.dtype
        imgs = imgs * 2.0 - 1.0
        posterior = self.vae.encode(imgs.to(self.weights_dtype)).latent_dist
        latents = posterior.sample() * self.vae.config.scaling_factor
        return latents.to(input_dtype)

    @torch.cuda.amp.autocast(enabled=False)
    def decode_latents(
        self,
        latents: Float[Tensor, "B 4 H W"],
        latent_height: int = 64,
        latent_width: int = 64,
    ) -> Float[Tensor, "B 3 512 512"]:
        input_dtype = latents.dtype
        latents = F.interpolate(
            latents, (latent_height, latent_width), mode="bilinear", align_corners=False
        )
        latents = 1 / self.vae.config.scaling_factor * latents
        image = self.vae.decode(latents.to(self.weights_dtype)).sample
        image = (image * 0.5 + 0.5).clamp(0, 1)
        return image.to(input_dtype)

    @contextmanager
    def disable_unet_class_embedding(self, unet: UNet2DConditionModel):
        class_embedding = unet.class_embedding
        try:
            unet.class_embedding = None
            yield unet
        finally:
            unet.class_embedding = class_embedding

    def compute_with_perpneg(
        self,
        condition_scales,
        prompt_utils,
        latents_noisy,
        t, 
        elevation: Float[Tensor, "B"],
        azimuth: Float[Tensor, "B"],
        camera_distances: Float[Tensor, "B"],
        image_cond: Float[Tensor, "B 3 512 512"],
    ):
        (
            text_embeddings,
            neg_guidance_weights,
        ) = prompt_utils.get_text_embeddings_perp_neg(
            elevation,
            azimuth,
            camera_distances,
            self.cfg.view_dependent_prompting,
            return_null_text_embeddings=True,
        )
        batch_size = latents_noisy.shape[0]
        with torch.no_grad():
            latent_model_input = torch.cat([latents_noisy] * 5, dim=0)
            if not all(scale == 0 for scale in condition_scales):
                down_block_res_samples, mid_block_res_sample = self.multi_control_forward(
                    latent_model_input,
                    torch.cat([t] * 5),
                    encoder_hidden_states=text_embeddings,
                    controlnet_cond=image_cond,
                    conditioning_scale=condition_scales
                )
                with self.disable_unet_class_embedding(self.unet) as unet:
                    noise_pred = self.forward_unet(
                        unet,
                        latent_model_input,
                        torch.cat([t] * 5),
                        encoder_hidden_states=text_embeddings,
                        cross_attention_kwargs=None,
                        down_block_additional_residuals=down_block_res_samples,
                        mid_block_additional_residual=mid_block_res_sample
                    )
            else:
                with self.disable_unet_class_embedding(self.unet) as unet:
                    noise_pred = self.forward_unet(
                        unet,
                        latent_model_input,
                        torch.cat([t] * 5),
                        encoder_hidden_states=text_embeddings,
                        cross_attention_kwargs=None,
                    )

        noise_pred_text = noise_pred[:batch_size]
        noise_pred_uncond = noise_pred[batch_size : batch_size * 2]
        noise_pred_neg = noise_pred[batch_size * 2 : batch_size * 4]
        noise_pred_null = noise_pred[batch_size * 4 :]

        e_pos = noise_pred_text - noise_pred_uncond
        accum_grad = 0
        n_negative_prompts = neg_guidance_weights.shape[-1]
        for i in range(n_negative_prompts):
            e_i_neg = noise_pred_neg[i::n_negative_prompts] - noise_pred_uncond
            accum_grad += neg_guidance_weights[:, i].view(
                -1, 1, 1, 1
            ) * perpendicular_component(e_i_neg, e_pos)
        noise_pred_perpneg = accum_grad

        return noise_pred_text, noise_pred_uncond, noise_pred_null, noise_pred_perpneg


    def my_compute_without_perpneg(
        self,
        condition_scales,
        prompt_utils,
        latents_noisy,
        t, 
        elevation: Float[Tensor, "B"],
        azimuth: Float[Tensor, "B"],
        camera_distances: Float[Tensor, "B"],
        image_cond: Float[Tensor, "B 3 512 512"],
    ):
        text_embeddings = prompt_utils.get_text_embeddings(
                elevation,
                azimuth,
                camera_distances,
                self.cfg.view_dependent_prompting,
                return_null_text_embeddings=True,
            )
        

        with torch.no_grad():            
            latent_model_input = torch.cat([latents_noisy] * 2, dim=0)
            #if not all(scale == 0 for scale in condition_scales):
            down_block_res_samples, mid_block_res_sample = self.multi_control_forward(
                latents_noisy,
                t,
                encoder_hidden_states=text_embeddings[0:1],
                controlnet_cond=image_cond,
                conditioning_scale=condition_scales
            )


            # down_block_res_samples, mid_block_res_sample = self.multi_control_forward(latents_noisy,t,encoder_hidden_states=text_embeddings[0:1],controlnet_cond=image_cond,conditioning_scale=condition_scales)
            # down_block_res_samples, mid_block_res_sample = self.multi_control_forward(torch.cat([latents_noisy] * 3, dim=0),torch.cat([t] * 3),encoder_hidden_states=text_embeddings,controlnet_cond=image_cond,conditioning_scale=condition_scales)
            # import ipdb
            # ipdb.set_trace()
            with self.disable_unet_class_embedding(self.unet) as unet:
                noise_pred_text = self.forward_unet(
                    unet,
                    latents_noisy,
                    t,
                    encoder_hidden_states=text_embeddings[0:1],
                    cross_attention_kwargs=None,
                    down_block_additional_residuals=down_block_res_samples,
                    mid_block_additional_residual=mid_block_res_sample
                )
            
            with self.disable_unet_class_embedding(self.unet) as unet:
                noise_pred = self.forward_unet(
                    unet,
                    latent_model_input,
                    torch.cat([t] * 2),
                    encoder_hidden_states=text_embeddings[1:3],
                    cross_attention_kwargs=None,
                )

        noise_pred_uncond, noise_pred_null = noise_pred.chunk(2)

        return noise_pred_text, noise_pred_uncond, noise_pred_null    


    def compute_without_perpneg(
        self,
        condition_scales,
        prompt_utils,
        latents_noisy,
        t, 
        elevation: Float[Tensor, "B"],
        azimuth: Float[Tensor, "B"],
        camera_distances: Float[Tensor, "B"],
        image_cond: Float[Tensor, "B 3 512 512"],
    ):
        text_embeddings = prompt_utils.get_text_embeddings(
                elevation,
                azimuth,
                camera_distances,
                self.cfg.view_dependent_prompting,
                return_null_text_embeddings=True,
            )
        with torch.no_grad():            
            latent_model_input = torch.cat([latents_noisy] * 3, dim=0)
            if not all(scale == 0 for scale in condition_scales):
                down_block_res_samples, mid_block_res_sample = self.multi_control_forward(
                    latent_model_input,
                    torch.cat([t] * 3),
                    encoder_hidden_states=text_embeddings,
                    controlnet_cond=image_cond,
                    conditioning_scale=condition_scales
                )
                with self.disable_unet_class_embedding(self.unet) as unet:
                    noise_pred = self.forward_unet(
                        unet,
                        latent_model_input,
                        torch.cat([t] * 3),
                        encoder_hidden_states=text_embeddings,
                        cross_attention_kwargs=None,
                        down_block_additional_residuals=down_block_res_samples,
                        mid_block_additional_residual=mid_block_res_sample
                    )
            else:
                with self.disable_unet_class_embedding(self.unet) as unet:
                    noise_pred = self.forward_unet(
                        unet,
                        latent_model_input,
                        torch.cat([t] * 3),
                        encoder_hidden_states=text_embeddings,
                        cross_attention_kwargs=None,
                    )

        noise_pred_text, noise_pred_uncond, noise_pred_null = noise_pred.chunk(3)

        return noise_pred_text, noise_pred_uncond, noise_pred_null

    def compute_grad_sds(
        self,
        prompt_utils,
        condition_scales,
        latents: Float[Tensor, "B 4 64 64"],
        image_cond: Float[Tensor, "B 3 512 512"],
        elevation: Float[Tensor, "B"],
        azimuth: Float[Tensor, "B"],
        camera_distances: Float[Tensor, "B"],
    ):
        
        B = latents.shape[0]

        # random timestamp
        t = torch.randint(
            self.min_step,
            self.max_step + 1,
            [B],
            dtype=torch.long,
            device=self.device,
        )
        # add noise
        noise = torch.randn_like(latents)
        latents_noisy = self.scheduler.add_noise(latents, noise, t)

        if prompt_utils.use_perp_neg:
            noise_pred_text, noise_pred_uncond, noise_pred_null, noise_pred_perpneg = self.compute_with_perpneg(
                condition_scales, prompt_utils, latents_noisy, t, elevation, azimuth, camera_distances, image_cond
            )
        else:
            noise_pred_text, noise_pred_uncond, noise_pred_null = self.compute_without_perpneg(
                condition_scales, prompt_utils, latents_noisy, t, elevation, azimuth, camera_distances, image_cond
            )
        

        w = (1 - self.alphas[t]).view(-1, 1, 1, 1)
            
        # print("cond_scale:"+str(self.cond_scale))
        # print("uncond_scale:"+str(self.uncond_scale))
        # print("null_scale:"+str(self.null_scale))
        # print("noise_scale:"+str(self.noise_scale))
        # print("\n")
        grad = w * (self.cond_scale * noise_pred_text
            + self.uncond_scale * noise_pred_uncond
            + self.null_scale * noise_pred_null
            + self.noise_scale * noise
        )

        if prompt_utils.use_perp_neg:
            grad += w * self.perpneg_scale * noise_pred_perpneg

        guidance_eval_utils = {
                    "uncond_m_noise_norm": (noise_pred_uncond - noise).norm(),
                    "text_m_noise_norm": (noise_pred_text - noise).norm(),
                    "text_m_uncond_norm": (noise_pred_text - noise_pred_uncond).norm(),
                    "text_m_null_norm": (noise_pred_text - noise_pred_null).norm(),
                    "null_m_uncond_norm": (noise_pred_null - noise_pred_uncond).norm(),
                    "noise_norm": noise.norm(),
                    "uncond_norm": noise_pred_uncond.norm(),
                    "text_norm": noise_pred_text.norm(),
                }

        return grad, guidance_eval_utils

    def get_latents(
        self, rgb_BCHW: Float[Tensor, "B C H W"], rgb_as_latents=False
    ) -> Float[Tensor, "B 4 64 64"]:
        if rgb_as_latents:
            latents = F.interpolate(
                rgb_BCHW, (64, 64), mode="bilinear", align_corners=False
            )
        else:
            if rgb_BCHW.shape[2]!=self.cfg.height:

                rgb_BCHW_512 = F.interpolate(
                    rgb_BCHW, (self.cfg.width, self.cfg.height), mode="bilinear", align_corners=False
                )
                # encode image into latents with vae
                latents = self.encode_images(rgb_BCHW_512)
            else:
                latents = self.encode_images(rgb_BCHW)
        return latents

    def prepare_image_cond(self, control_type, cond_rgb: Float[Tensor, "B H W C"]):
        if control_type == 'normal':
            cond_rgb = (cond_rgb[0].detach().cpu().numpy() * 255).astype(np.uint8).copy()
            detected_map = self.preprocessor_normal(cond_rgb)
            control = torch.from_numpy(np.array(detected_map)).float().to(self.device) / 255.0
            control = control.unsqueeze(0)
            control = control.permute(0, 3, 1, 2)
        elif control_type == 'canny':
            control = []
            for i in range(cond_rgb.size()[0]):
                cond_rgb_ = cond_rgb[i]
                
                cond_rgb_ = (cond_rgb_.detach().cpu().numpy() * 255).astype(np.uint8).copy()
                blurred_img = cv2.blur(cond_rgb_, ksize=(5, 5))
                detected_map = self.preprocessor_canny(blurred_img, self.cfg.canny_lower_bound,
                                                       self.cfg.canny_upper_bound,cond_rgb_.shape[0],cond_rgb_.shape[0])
                
                control_ = torch.from_numpy(np.array(detected_map)).float().to(self.device) / 255.0
                control.append(control_)
            control = torch.stack(control, dim=0)
            #control = control.unsqueeze(-1).repeat(1, 1, 1, 3)
            #control = control.repeat(1, 1, 1, 3)

            control = control.permute(0, 3, 1, 2)
        elif control_type == 'self-normal':
            control = cond_rgb.permute(0, 3, 1, 2)
        elif control_type == 'hed':
            cond_rgb = (cond_rgb[0].detach().cpu().numpy() * 255).astype(np.uint8).copy()
            detected_map = self.preprocessor_hed(cond_rgb, scribble=True)
            control = torch.from_numpy(np.array(detected_map)).float().to(self.device) / 255.0
            control = control.unsqueeze(0)
            control = control.permute(0, 3, 1, 2)
        elif control_type == 'p2p':
            control = cond_rgb.permute(0, 3, 1, 2)
        elif control_type == 'depth':
            control = cond_rgb.permute(0, 3, 1, 2)
            control = control.repeat(1, 3, 1, 1)
        elif control_type == 'light':
            control = cond_rgb.permute(0, 3, 1, 2)

        if control.shape[2]!=self.cfg.height:
            return F.interpolate(
                control, (512, 512), mode="bilinear", align_corners=False
            )
        else:
            return control

    def __call__(
        self,
        rgb: Float[Tensor, "B H W C"],
        prompt_utils: PromptProcessorOutput,
        elevation: Float[Tensor, "B"],
        azimuth: Float[Tensor, "B"],
        camera_distances: Float[Tensor, "B"],
        env_id: Float[Tensor, "B"],
        rgb_as_latents=False,
        **kwargs,
    ):
        batch_size = rgb.shape[0]

        rgb_BCHW = rgb.permute(0, 3, 1, 2)
        latents = self.get_latents(rgb_BCHW, rgb_as_latents=rgb_as_latents)
        if self.use_controlnet:
            image_cond = []
            condition_scales = []
            cond_rgb = kwargs.get('cond_rgb', None)
            cond_depth = kwargs.get('cond_depth', None)
            cond_light = kwargs.get('condition_map',None)
            cond_normal = kwargs.get('cond_normal',None)
            for k in range(len(self.cfg.control_types)):
                control_type = self.cfg.control_types[k]
                if control_type == 'canny':
                    control_cond = self.prepare_image_cond(control_type, cond_rgb)
                elif control_type == 'depth':
                    control_cond = self.prepare_image_cond(control_type, cond_depth)
                elif control_type == 'light':
                    control_cond = kwargs.get('condition_map')
                    control_cond = self.prepare_image_cond(control_type, cond_light)
                elif control_type == 'self-normal':
                    control_cond = self.prepare_image_cond(control_type, cond_normal)
                else:
                    control_cond = self.prepare_image_cond(control_type, cond_rgb)
                image_cond.append(control_cond)

                condition_scales.append(self.cfg.condition_scales[k])

            grad, guidance_eval_utils = self.compute_grad_sds(
                prompt_utils, condition_scales, latents, image_cond, elevation, azimuth, camera_distances
            )
        else:
            condition_scales = [0]
            image_cond = []
            grad, guidance_eval_utils = self.compute_grad_sds(
                prompt_utils, condition_scales, latents, image_cond, elevation, azimuth, camera_distances
            )


        grad = torch.nan_to_num(grad)

        if self.cfg.grad_clip_val is not None:
            grad = grad.clamp(-self.cfg.grad_clip_val, self.cfg.grad_clip_val)
        if self.cfg.grad_normalize:
            grad = grad / (grad.norm(2) + 1e-8)

        # reparameterization trick
        # d(loss)/d(latents) = latents - target = latents - (latents - grad) = grad
        target = (latents - grad).detach()
        loss_sds = 0.5 * F.mse_loss(latents, target, reduction="sum") / batch_size

        guidance_out = {
            "loss_sds": loss_sds,
            "grad_norm": grad.norm(),
        }
        guidance_out.update(guidance_eval_utils)

        return guidance_out

    @torch.cuda.amp.autocast(enabled=False)
    def set_min_max_steps(self, min_step_percent=0.02, max_step_percent=0.98):
        self.min_step = int(self.num_train_timesteps * min_step_percent)
        self.max_step = int(self.num_train_timesteps * max_step_percent)

    def update_step(self, epoch: int, global_step: int, on_load_weights: bool = False):
        self.noise_scale = C(self.cfg.noise_scale, epoch, global_step)
        self.cond_scale=C(self.cfg.cond_scale, epoch, global_step)
        self.uncond_scale=C(self.cfg.uncond_scale, epoch, global_step)
        self.null_scale=C(self.cfg.null_scale, epoch, global_step)
        self.perpneg_scale=C(self.cfg.perpneg_scale, epoch, global_step)

        self.set_min_max_steps(
            min_step_percent=C(self.cfg.min_step_percent, epoch, global_step),
            max_step_percent=C(self.cfg.max_step_percent, epoch, global_step),
        )
        
        if (
            self.use_controlnet
            and self.cfg.control_anneal_start_step is not None
            and global_step > self.cfg.control_anneal_start_step
        ):
            self.cfg.condition_scales = self.cfg.condition_scales_anneal
    
