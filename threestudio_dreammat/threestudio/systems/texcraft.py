import os
import shutil
from dataclasses import dataclass, field
import random
import torch
import numpy as np
import matplotlib.pyplot as plt
import threestudio
from threestudio.systems.base import BaseLift3DSystem
from threestudio.utils.misc import cleanup, get_device
from threestudio.utils.ops import binary_cross_entropy, dot
from threestudio.utils.typing import *
import torch.nn.functional as F
from threestudio.utils.ops import (
            get_mvp_matrix,
            get_projection_matrix,
            get_ray_directions,
            get_rays,
        )
import math
import pdb
from threestudio.utils.ops import get_activation
@threestudio.register("texcraft-system")
class SegControlnetMagic3D(BaseLift3DSystem):
    @dataclass
    class Config(BaseLift3DSystem.Config):
        render_depth: Optional[bool] = True
        save_train_image: bool = False
        save_train_image_iter: int = 20
        init_step: int = 0
        init_width:int=512
        init_height:int=512
        test_background_white: Optional[bool] = False

    cfg: Config

    def configure(self):
        # create geometry, material, background, renderer
        super().configure()
        # only used in training
        self.prompt_processor = threestudio.find(self.cfg.prompt_processor_type)(
            self.cfg.prompt_processor
        )
        self.guidance = threestudio.find(self.cfg.guidance_type)(self.cfg.guidance)
  
        self.init_batch = self.make_init_batch(
            width=self.cfg.init_width,
            height=self.cfg.init_height,
            eval_camera_distance=2.8,
            eval_fovy_deg=45.,
            eval_elevation_deg=15.0,
            n_views=4,
            eval_batch_size=4,
        )



    def forward(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        render_out = self.renderer(render_depth=self.cfg.render_depth, **batch)
        return {
            **render_out,
        }

    def on_fit_start(self) -> None:
        super().on_fit_start()

    def training_step(self, batch, batch_idx):
        # prompt_utils = self.prompt_processor()
        
        # if self.true_global_step < self.cfg.init_step:
        #     init_batch = self.get_random_init_batch()
        #     out = self(init_batch)
        #     batch['cond_rgb']=out.get('comp_normal', None)
        #     batch['cond_depth']=out.get('depth', None)
        #     guidance_out = self.guidance(
        #         out["comp_rgb"], prompt_utils, **batch, batch_idx=batch_idx, rgb_as_latents=False, 
        #     )

        # else:
        #     out = self(batch)
        #     batch['cond_rgb']=out.get('comp_normal', None)
        #     batch['cond_depth']=out.get('depth', None)
        #     guidance_out = self.guidance(
        #         out["comp_rgb"], prompt_utils, **batch, batch_idx=batch_idx, rgb_as_latents=False, 
        #     )
        prompt_utils = self.prompt_processor()
        if self.true_global_step < self.cfg.init_step:
            init_batch = self.get_random_init_batch()
            out = self(init_batch)
            # guidance_inp = get_activation("lin2srgb")(out["comp_rgb"])
            # batch['cond_rgb']=out.get('comp_normal', None)
            # batch['cond_depth']=out.get('comp_depth', None)
            # guidance_out = self.guidance(
            #     guidance_inp, prompt_utils, **batch, rgb_as_latents=False
            # )

        else:
            out = self(batch)
            
        #guidance_inp = get_activation("lin2srgb")(out["comp_rgb"])
        guidance_inp=out["comp_rgb"].clamp(0.0, 1.0)
        batch['cond_normal']=out.get('comp_normal', None)
        batch['cond_depth']=out.get('comp_depth', None)
        #batch['condition_map'][...,1:4]=out.get('comp_normal', None)
        
        guidance_out = self.guidance(
            guidance_inp, prompt_utils, **batch, rgb_as_latents=False, 
        )

        loss = 0.0
        for name, value in guidance_out.items():
            # self.log(f"train/{name}", value)
            if name.startswith("loss_"):
                loss += value * self.C(self.cfg.loss[name.replace("loss_", "lambda_")])

        for name, value in self.cfg.loss.items():
            self.log(f"train_params/{name}", self.C(value))

        if self.cfg.save_train_image:
            if self.true_global_step%self.cfg.save_train_image_iter == 0:
                srgb=get_activation("lin2srgb")(out["comp_rgb"][0].detach())
                self.save_image_grid(
                f"train/it{self.true_global_step}.png",
                [
                    {
                        "type": "rgb",
                        "img": out["comp_rgb"][0].detach(),#srgb,
                        "kwargs": {"data_format": "HWC"},
                    },
                    {
                        "type": "rgb",
                        "img": out["comp_normal"][0].detach(),
                        "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                    },
                    {
                        "type": "rgb",
                        "img": out["albedo"][0],
                        "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                    },
                    {
                        "type": "grayscale",
                        "img": out["metalness"][0, :, :, 0],
                        "kwargs": {"cmap": None, "data_range": (0, 1)},
                    },
                    {
                        "type": "grayscale",
                        "img": out["roughness"][0, :, :, 0],
                        "kwargs": {"cmap": None, "data_range": (0, 1)},
                    },
                ],
                name="train_step",
                step=self.true_global_step,
            )

        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        out = self(batch)
        srgb=get_activation("lin2srgb")(out["comp_rgb"][0].detach())
        self.save_image_grid(
            f"validate/it{self.true_global_step}-{batch['index'][0]}.png",
            [
                {
                    "type": "rgb",
                    "img": out["comp_rgb"][0].detach(),#srgb,
                    "kwargs": {"data_format": "HWC"},
                },
                {
                    "type": "rgb",
                    "img": out["comp_normal"][0].detach(),
                    "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                },
                {
                    "type": "rgb",
                    "img": out["albedo"][0],
                    "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                },
                {
                    "type": "rgb",
                    "img": out["metalness"][0],
                    "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                },
                {
                    "type": "rgb",
                    "img": out["roughness"][0],
                    "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                },
            ],
            
            # claforte: TODO: don't hardcode the frame numbers to record... read them from cfg instead.
            name=f"validation_step_batchidx_{batch_idx}"
            if batch_idx in [0, 7, 15, 23, 29]
            else None,
            step=self.true_global_step,
        )

    def on_validation_epoch_end(self):
        pass
        # filestem = f"it{self.true_global_step}-val"
        # self.save_img_sequence(
        #     filestem,
        #     filestem,
        #     "(\d+)\.png",
        #     save_format="mp4",
        #     fps=30,
        #     name="validation_epoch_end",
        #     step=self.true_global_step,
        # )
        # shutil.rmtree(
        #     os.path.join(self.get_save_dir(), f"it{self.true_global_step}-val")
        # )

    def test_step(self, batch, batch_idx):
        out = self(batch)
        srgb=get_activation("lin2srgb")(out["comp_rgb"][0].detach())
        self.save_image_grid(
            f"it{self.true_global_step}-test/view/{batch['index'][0]}.png",
            [
                {
                    "type": "rgb",
                    "img": out["comp_rgb"][0].detach(),#srgb,
                    "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                },
                {
                    "type": "rgb",
                    "img": out["comp_normal"][0],
                    "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                },
                {
                    "type": "rgb",
                    "img": out["albedo"][0],
                    "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                },
                {
                    "type": "rgb",
                    "img": out["metalness"][0],
                    "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                },
                {
                    "type": "rgb",
                    "img": out["roughness"][0],
                    "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                },
            ],
            name="test_step",
            step=self.true_global_step,
        )
        self.save_img(out["albedo"][0],f"it{self.true_global_step}-test/albedo/{batch['index'][0]}.png")
        self.save_img(out["roughness"][0],f"it{self.true_global_step}-test/roughness/{batch['index'][0]}.png")
        self.save_img(out["metalness"][0],f"it{self.true_global_step}-test/metallic/{batch['index'][0]}.png")
        self.save_img(out["comp_rgb"][0],f"it{self.true_global_step}-test/render/{batch['index'][0]}.png")

    def on_test_epoch_end(self):
        viewpath="it"+str(self.true_global_step)+"-test/view"
        self.save_gif(viewpath,fps=30)
        # self.save_img_sequence(
        #     f"it{self.true_global_step}-test",
        #     f"it{self.true_global_step}-test",
        #     "(\d+)\.png",
        #     save_format="mp4",
        #     fps=30,
        #     name="test",
        #     step=self.true_global_step,
        # )

#------------------------below is getting the init batch, 4 views, HARD CODING NOW (TODO)-------------------
    def make_init_batch(self,
        width=512,
        height=512,
        eval_camera_distance=1.5,
        eval_fovy_deg=70.,
        eval_elevation_deg=15.0,
        n_views=4,
        eval_batch_size=4,
        ):

        azimuth_deg = torch.linspace(0, 360.0, n_views + 1)[: n_views]
        elevation_deg: Float[Tensor, "B"] = torch.full_like(
            azimuth_deg, eval_elevation_deg
        )
        camera_distances: Float[Tensor, "B"] = torch.full_like(
            elevation_deg, eval_camera_distance
        )

        elevation = elevation_deg * math.pi / 180
        azimuth = azimuth_deg * math.pi / 180

        # convert spherical coordinates to cartesian coordinates
        # right hand coordinate system, x back, y right, z up
        # elevation in (-90, 90), azimuth from +x to +y in (-180, 180)
        camera_positions: Float[Tensor, "B 3"] = torch.stack(
            [
                camera_distances * torch.cos(elevation) * torch.cos(azimuth),
                camera_distances * torch.cos(elevation) * torch.sin(azimuth),
                camera_distances * torch.sin(elevation),
            ],
            dim=-1,
        )

        # default scene center at origin
        center: Float[Tensor, "B 3"] = torch.zeros_like(camera_positions)
        # default camera up direction as +z
        up: Float[Tensor, "B 3"] = torch.as_tensor([0, 0, 1], dtype=torch.float32)[
                                    None, :
                                    ].repeat(eval_batch_size, 1)

        fovy_deg: Float[Tensor, "B"] = torch.full_like(
            elevation_deg, eval_fovy_deg
        )
        fovy = fovy_deg * math.pi / 180

        # sample light distance from a uniform distribution bounded by light_distance_range
        light_distances: Float[Tensor, "B"] = (
                torch.rand(eval_batch_size)
                * (1.5 - 0.8)
                + 0.8
        )

        # sample light direction within restricted angle range (pi/3)
        local_z = F.normalize(camera_positions, dim=-1)
        local_x = F.normalize(
            torch.stack(
                [local_z[:, 1], -local_z[:, 0], torch.zeros_like(local_z[:, 0])],
                dim=-1,
            ),
            dim=-1,
        )
        local_y = F.normalize(torch.cross(local_z, local_x, dim=-1), dim=-1)
        rot = torch.stack([local_x, local_y, local_z], dim=-1)
        light_azimuth = (
                torch.rand(eval_batch_size) * math.pi - 2 * math.pi
        )  # [-pi, pi]
        light_elevation = (
                torch.rand(eval_batch_size) * math.pi / 3 + math.pi / 6
        )  # [pi/6, pi/2]
        light_positions_local = torch.stack(
            [
                light_distances
                * torch.cos(light_elevation)
                * torch.cos(light_azimuth),
                light_distances
                * torch.cos(light_elevation)
                * torch.sin(light_azimuth),
                light_distances * torch.sin(light_elevation),
            ],
            dim=-1,
        )
        light_positions = (rot @ light_positions_local[:, :, None])[:, :, 0]

        lookat: Float[Tensor, "B 3"] = F.normalize(center - camera_positions, dim=-1)
        right: Float[Tensor, "B 3"] = F.normalize(torch.cross(lookat, up), dim=-1)
        up = F.normalize(torch.cross(right, lookat), dim=-1)
        c2w3x4: Float[Tensor, "B 3 4"] = torch.cat(
            [torch.stack([right, up, -lookat], dim=-1), camera_positions[:, :, None]],
            dim=-1,
        )
        c2w: Float[Tensor, "B 4 4"] = torch.cat(
            [c2w3x4, torch.zeros_like(c2w3x4[:, :1])], dim=1
        )
        c2w[:, 3, 3] = 1.0

        proj_mtx: Float[Tensor, "B 4 4"] = get_projection_matrix(
            fovy, width / height, 0.1, 1000.0
        )  # FIXME: hard-coded near and far
        mvp_mtx, w2c= get_mvp_matrix(c2w, proj_mtx)
        
        return {
            'mvp_mtx': mvp_mtx.cuda(),
            'camera_positions': camera_positions.cuda(),
            'light_positions': light_positions.cuda(),
            'height': width,
            'width': height,
            "elevation": elevation_deg.cuda(),
            "azimuth": azimuth_deg.cuda(),
            "camera_distances": camera_distances.cuda(),
            "c2w": c2w.cuda(),
        }
    
    def get_random_init_batch(self):
        views = len(self.init_batch['mvp_mtx'])
        i = random.randint(0, views-1)

        return {
                'mvp_mtx': self.init_batch['mvp_mtx'][i:i + 1],
                'camera_positions': self.init_batch['camera_positions'][i:i + 1],
                'light_positions': self.init_batch['light_positions'][i:i + 1],
                'height': self.init_batch['height'],
                'width': self.init_batch['width'],
                "elevation": self.init_batch['elevation'][i:i + 1],
                "azimuth": self.init_batch['azimuth'][i:i + 1],
                "camera_distances": self.init_batch['camera_distances'][i:i + 1],
                "c2w": self.init_batch['c2w'][i:i + 1],
                }