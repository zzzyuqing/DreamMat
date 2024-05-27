from dataclasses import dataclass

import nerfacc
import torch
import torch.nn.functional as F

import threestudio
from threestudio.models.background.base import BaseBackground
from threestudio.models.geometry.base import BaseImplicitGeometry
from threestudio.models.materials.base import BaseMaterial
from threestudio.models.renderers.base import Rasterizer, VolumeRenderer
from threestudio.utils.misc import get_device
from threestudio.utils.rasterize import NVDiffRasterizerContext
from threestudio.utils.typing import *

from PIL import Image
import numpy as np
import _raytracing as _backend

class RayTracer():
    def __init__(self, vertices, triangles):
        # vertices: np.ndarray, [N, 3]
        # triangles: np.ndarray, [M, 3]

        if torch.is_tensor(vertices): vertices = vertices.detach().cpu().numpy()
        if torch.is_tensor(triangles): triangles = triangles.detach().cpu().numpy()

        assert triangles.shape[0] > 8, "BVH needs at least 8 triangles."
        
        # implementation
        self.impl = _backend.create_raytracer(vertices, triangles)

    def trace(self, rays_o, rays_d, inplace=False):
        # rays_o: torch.Tensor, cuda, float, [N, 3]
        # rays_d: torch.Tensor, cuda, float, [N, 3]
        # inplace: write positions to rays_o, face_normals to rays_d

        rays_o = rays_o.float().contiguous()
        rays_d = rays_d.float().contiguous()

        if not rays_o.is_cuda: rays_o = rays_o.cuda()
        if not rays_d.is_cuda: rays_d = rays_d.cuda()

        prefix = rays_o.shape[:-1]
        rays_o = rays_o.view(-1, 3)
        rays_d = rays_d.view(-1, 3)

        N = rays_o.shape[0]

        if not inplace:
            # allocate
            positions = torch.empty_like(rays_o)
            face_normals = torch.empty_like(rays_d)
        else:
            positions = rays_o
            face_normals = rays_d

        depth = torch.empty_like(rays_o[:, 0])
        
        # inplace write intersections back to rays_o
        self.impl.trace(rays_o, rays_d, positions, face_normals, depth) # [N, 3]

        positions = positions.view(*prefix, 3)
        face_normals = face_normals.view(*prefix, 3)
        depth = depth.view(*prefix)

        return positions, face_normals, depth

def xfm_vectors(vectors, matrix):
    '''Transform vectors.
    Args:
        vectors: Tensor containing 3D vectors with shape [minibatch_size, num_vertices, 3] or [1, num_vertices, 3]
        matrix: A 4x4 transform matrix with shape [minibatch_size, 4, 4]
        use_python: Use PyTorch's torch.matmul (for validation)

    Returns:
        Transformed vectors in homogeneous 4D with shape [minibatch_size, num_vertices, 4].
    '''    

    out = torch.matmul(torch.nn.functional.pad(vectors, pad=(0,1), mode='constant', value=0.0), torch.transpose(matrix, 1, 2))[..., 0:3].contiguous()
    if torch.is_anomaly_enabled():
        assert torch.all(torch.isfinite(out)), "Output of xfm_vectors contains inf or NaN"
    return out

@threestudio.register("raytracing-renderer")
class RaytraceRender(Rasterizer):
    @dataclass
    class Config(VolumeRenderer.Config):
        context_type: str = "gl"
        

    cfg: Config

    def configure(
        self,
        geometry: BaseImplicitGeometry,
        material: BaseMaterial,
        background: BaseBackground,
    ) -> None:
        super().configure(geometry, material, background)
        self.ctx = NVDiffRasterizerContext(self.cfg.context_type, get_device())
        self.mesh = self.geometry.isosurface()
        self.ray_tracer = RayTracer(self.mesh.v_pos, self.mesh.t_pos_idx)
        self.material.set_raytracer(lambda o,d: self.trace(o,d))

        self.change_type = 'gaussian'
        self.change_eps = 0.05

    def forward(
        self,  
        env_id,
        rays_o,
        rays_d,
        w2c: Float[Tensor, "B 4 4"],
        mvp_mtx: Float[Tensor, "B 4 4"],
        camera_positions: Float[Tensor, "B 3"],
        light_positions: Float[Tensor, "B 3"],
        height: int,
        width: int,
        **kwargs
    ):
        batch_size = mvp_mtx.shape[0]
        v_pos_clip: Float[Tensor, "B Nv 4"] = self.ctx.vertex_transform(self.mesh.v_pos, mvp_mtx)
        rast, _ = self.ctx.rasterize(v_pos_clip, self.mesh.t_pos_idx, (height, width))
        mask = rast[..., 3:] > 0
        selector = mask[..., 0].reshape(batch_size,width*height)
        mask_aa = self.ctx.antialias(mask.float(), rast, v_pos_clip, self.mesh.t_pos_idx)

        min_val=0.3
        depth = (rast[..., 2:3] ).float()
        depth[mask]=1. / (depth[mask] + 1e-6)
        depth_max = depth[mask].max()
        depth_min = depth[mask].min()
        depth[mask] = (1 - min_val) *(depth[mask] - depth_min) / (depth_max - depth_min + 1e-6) + min_val

        gb_normal, _ = self.ctx.interpolate_one(self.mesh.v_nrm, rast, self.mesh.t_pos_idx)
        gb_normal = F.normalize(gb_normal, dim=-1)

        gb_normal=gb_normal.reshape(batch_size,width*height,3)
        normal_controlnet=self.compute_controlnet_normals(gb_normal[selector],w2c,batch_size)
        background=torch.tensor([0.5,0.5,1.0]).reshape(1,1,1,3).repeat(batch_size,width,height,1).to(self.device)
        #background=torch.ones((batch_size,width,height,3)).to(self.device)
        gb_normal_aa=torch.ones_like(gb_normal).to(self.device)
        gb_normal_aa[selector]=normal_controlnet
        gb_normal_aa=gb_normal_aa.reshape(batch_size,height,width,3)
        gb_normal_aa = torch.lerp(background, gb_normal_aa, mask.float())
        gb_normal_aa = self.ctx.antialias(gb_normal_aa, rast, v_pos_clip, self.mesh.t_pos_idx)
        
        
        gb_pos, _ = self.ctx.interpolate_one(self.mesh.v_pos, rast, self.mesh.t_pos_idx)
        #gb_viewdirs = F.normalize(gb_pos - camera_positions[:, None, None, :], dim=-1)
        gb_viewdirs = -rays_d.reshape(batch_size,-1,3)

        
        gb_pos=gb_pos.reshape(batch_size,width*height,3)
        
        #gb_viewdirs=gb_viewdirs.reshape(batch_size,width*height,3)
        
        positions = gb_pos[selector]

        if self.geometry.cfg.n_input_dims==3:
            x = self.get_orthogonal_directions(gb_normal[selector])
            y = torch.cross(gb_normal[selector],x)
            ang = torch.rand(positions.shape[0],1).to(self.device)*np.pi*2
            if self.change_type=='constant':
                change = (torch.cos(ang) * x + torch.sin(ang) * y) * self.change_eps
            elif self.change_type=='gaussian':
                eps = torch.normal(mean=0.0, std=self.change_eps, size=[x.shape[0], 1]).to(self.device)
                change = (torch.cos(ang) * x + torch.sin(ang) * y) * eps
            else:
                raise NotImplementedError
            
            positions_jitter = gb_pos[selector] + change
            geo_out = self.geometry(positions, output_normal=False)
            geo_out_jitter = self.geometry(positions_jitter,output_normal=False)

        elif self.geometry.cfg.n_input_dims==2:
            gb_texc, _ = self.ctx.interpolate_one(self.mesh.v_tex, rast, self.mesh.t_tex_idx.int())
            gb_texc=gb_texc.reshape(batch_size,width*height,2)
            geo_out = self.geometry(gb_texc[selector], output_normal=False)
            geo_out_jitter = self.geometry(gb_texc[selector] + torch.normal(mean=0, std=0.005, size=gb_texc[selector].shape, device=self.device))
        else:
            raise NotImplementedError()
        
        shade_outputs,mat_reg_loss = self.material(positions, geo_out['features'], geo_out_jitter['features'], gb_viewdirs[selector], gb_normal[selector], env_id)

        

        color=torch.ones((batch_size,height*width,3),requires_grad=True).to(self.device)
        metalness=torch.ones((batch_size,height*width,1)).to(self.device)
        roughness=torch.ones((batch_size,height*width,1)).to(self.device)
        albedo=torch.ones((batch_size,height*width,3)).to(self.device)
        specular_light=torch.ones((batch_size,height*width,3)).to(self.device)
        diffuse_light=torch.ones((batch_size,height*width,3)).to(self.device)
        specular_color=torch.ones((batch_size,height*width,3)).to("cuda")
        diffuse_color=torch.ones((batch_size,height*width,3)).to("cuda")

        color[selector]=shade_outputs['color']
        color_aa = self.ctx.antialias(color.reshape(batch_size,height,width,3), rast, v_pos_clip, self.mesh.t_pos_idx)

        metalness[selector]=shade_outputs['metalness'].detach()
        roughness[selector]=shade_outputs['roughness'].detach()
        albedo[selector]=shade_outputs['albedo'].detach()
        specular_light[selector]=shade_outputs['specular_lights'].detach()
        diffuse_light[selector]=shade_outputs['diffuse_lights'].detach()
        specular_color[selector]=shade_outputs['specular_colors']
        diffuse_color[selector]=shade_outputs['diffuse_colors']

        return {
            "comp_rgb": color_aa,
            "opacity":mask_aa,
            "comp_depth":depth,
            "comp_normal":gb_normal_aa,#normal_controlnet.reshape(batch_size,height,width,3),#
            'albedo':albedo.reshape(batch_size,height,width,3),
            'metalness':metalness.reshape(batch_size,height,width,1),
            'roughness':roughness.reshape(batch_size,height,width,1),
            'specular_light':specular_light.reshape(batch_size,height,width,3),
            'diffuse_light':diffuse_light.reshape(batch_size,height,width,3),
            'specular_color':specular_color.reshape(batch_size,height,width,3),
            'diffuse_color':diffuse_color.reshape(batch_size,height,width,3),
            'loss_mat_reg':mat_reg_loss
            }
    
    def forward__(
        self,
        env_id,
        rays_o,
        rays_d,
        w2c: Float[Tensor, "B 4 4"],
        height: int,
        width: int,
        render_rgb: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        batch_size = w2c.shape[0]
        

        inters, normals, depth, hit_mask = self.trace(rays_o.reshape(-1,3), rays_d.reshape(-1,3)) 

        # depthimg=depth.detach().reshape(height,width,1).cpu().numpy().repeat(3,axis=2)/10
        # depthimg=Image.fromarray((depthimg*255).astype(np.uint8))
        # depthimg.save("depth.png")

        inters, normals, depth, hit_mask = inters.reshape(batch_size,height*width,3), normals.reshape(batch_size,height*width,3), depth.reshape(batch_size,height*width,1), hit_mask.reshape(batch_size,height*width)
        view_dirs_all = -rays_d.reshape(batch_size,-1,3)

        color=torch.ones((batch_size,height*width,3),requires_grad=True).to("cuda")
        metalness=torch.zeros((batch_size,height*width,1)).to("cuda")
        roughness=torch.zeros((batch_size,height*width,1)).to("cuda")
        albedo=torch.zeros((batch_size,height*width,3)).to("cuda")
        specular_light=torch.zeros((batch_size,height*width,3)).to("cuda")
        diffuse_light=torch.zeros((batch_size,height*width,3)).to("cuda")
        specular_color=torch.ones((batch_size,height*width,3)).to("cuda")
        diffuse_color=torch.ones((batch_size,height*width,3)).to("cuda")


        pts = inters[hit_mask]
        view_dirs = view_dirs_all[hit_mask]
        shading_normals = normals[hit_mask]

        x = self.get_orthogonal_directions(shading_normals)
        y = torch.cross(shading_normals,x)
        ang = torch.rand(pts.shape[0],1).to(self.device)*np.pi*2
        if self.change_type=='constant':
            change = (torch.cos(ang) * x + torch.sin(ang) * y) * self.change_eps
        elif self.change_type=='gaussian':
            eps = torch.normal(mean=0.0, std=self.change_eps, size=[x.shape[0], 1]).to(self.device)
            change = (torch.cos(ang) * x + torch.sin(ang) * y) * eps
        else:
            raise NotImplementedError
        
        geo_out_jitter = self.geometry(pts + change,output_normal=False)
        geo_out = self.geometry(pts, output_normal=False)

        shade_outputs,mat_reg_loss  = self.material(pts, geo_out['features'], geo_out_jitter['features'], view_dirs, shading_normals, env_id)

        color[hit_mask]=shade_outputs['color']
        metalness[hit_mask]=shade_outputs['metalness']
        roughness[hit_mask]=shade_outputs['roughness']
        albedo[hit_mask]=shade_outputs['albedo']
        specular_light[hit_mask]=shade_outputs['specular_lights']
        diffuse_light[hit_mask]=shade_outputs['diffuse_lights']
        specular_color[hit_mask]=shade_outputs['specular_colors']
        diffuse_color[hit_mask]=shade_outputs['diffuse_colors']

        normal_controlnet=self.compute_controlnet_normals(normals[hit_mask],w2c,batch_size)
        normals[hit_mask]=normal_controlnet

        depth=self.compute_controlnet_depth(depth,hit_mask)

        return {
            "comp_rgb": color.reshape(batch_size,height,width,3),
            "opacity":hit_mask.reshape(batch_size,height,width,1),
            "comp_depth":depth.reshape(batch_size,height,width,1),
            "comp_normal":normals.reshape(batch_size,height,width,3),#normal_controlnet.reshape(batch_size,height,width,3),#
            'albedo':albedo.reshape(batch_size,height,width,3),
            'metalness':metalness.reshape(batch_size,height,width,1),
            'roughness':roughness.reshape(batch_size,height,width,1),
            'specular_light':specular_light.reshape(batch_size,height,width,3),
            'diffuse_light':diffuse_light.reshape(batch_size,height,width,3),
            'specular_color':specular_color.reshape(batch_size,height,width,3),
            'diffuse_color':diffuse_color.reshape(batch_size,height,width,3),
            'loss_mat_reg':mat_reg_loss
            }

    def get_orthogonal_directions(self, directions):
        x, y, z = torch.split(directions, 1, dim=-1) # pn,1
        otho0 = torch.cat([y,-x,torch.zeros_like(x)],-1)
        otho1 = torch.cat([-z,torch.zeros_like(x),x],-1)
        mask0 = torch.norm(otho0,dim=-1)>torch.norm(otho1,dim=-1)
        mask1 = ~mask0
        otho = torch.zeros_like(directions,device=self.device)
        otho[mask0] = otho0[mask0]
        otho[mask1] = otho1[mask1]
        otho = F.normalize(otho, dim=-1)
        return otho
    
    def trace(self, rays_o, rays_d):
        inters, normals, depth = self.ray_tracer.trace(rays_o, rays_d)
        depth = depth.reshape(*depth.shape, 1)
        normals = F.normalize(normals, dim=-1)
        miss_mask = depth >= 10
        hit_mask = ~miss_mask
        return inters, normals, depth, hit_mask
    
    def compute_controlnet_normals(self,normals,mv,batch_size):
        normal_view  = xfm_vectors(normals.view(batch_size, normals.shape[0], normals.shape[1]), mv).view(*normals.shape)
        normal_view = F.normalize(normal_view)
        normal_controlnet=0.5*(normal_view+1)
        normal_controlnet[..., 0]=1.0-normal_controlnet[..., 0] # Flip the sign on the x-axis to match bae system
        return normal_controlnet
    
    def compute_controlnet_depth(self,depth,hit_mask):
        min_val=0.3
   
        depth_inv = 1. / (depth + 1e-6)
        
        depth_max = depth_inv[hit_mask].max()
        depth_min = depth_inv[hit_mask].min()
                    
        depth[hit_mask] = (1 - min_val) *(depth_inv[hit_mask] - depth_min) / (depth_max - depth_min + 1e-6) + min_val
        depth[~hit_mask]=0.0
        return depth