import os
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import threestudio
from threestudio.models.geometry.base import (
    BaseExplicitGeometry,
    BaseGeometry,
    contract_to_unisphere,
)
from threestudio.models.mesh import Mesh
from threestudio.models.networks import get_encoding, get_mlp
from threestudio.utils.ops import scale_tensor
from threestudio.utils.typing import *
class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x: x)
            out_dim += d

        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']

        if self.kwargs['log_sampling']:
            freq_bands = 2. ** torch.linspace(0., max_freq, N_freqs)
        else:
            freq_bands = torch.linspace(2.**0., 2.**max_freq, N_freqs)

        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)
def get_embedder(multires, input_dims=3):
    embed_kwargs = {
        'include_input': True,
        'input_dims': input_dims,
        'max_freq_log2': multires-1,
        'num_freqs': multires,
        'log_sampling': True,
        'periodic_fns': [torch.sin, torch.cos],
    }

    embedder_obj = Embedder(**embed_kwargs)
    def embed(x, eo=embedder_obj): return eo.embed(x)
    return embed, embedder_obj.out_dim

def make_predictor(feats_dim: object, output_dim: object, weight_norm: object = True) -> object:

    run_dim = 256
    if weight_norm:
        module=nn.Sequential(
            nn.utils.weight_norm(nn.Linear(feats_dim, run_dim)),
            nn.ReLU(),
            nn.utils.weight_norm(nn.Linear(run_dim, run_dim)),
            nn.ReLU(),
            nn.utils.weight_norm(nn.Linear(run_dim, run_dim)),
            nn.ReLU(),
            nn.utils.weight_norm(nn.Linear(run_dim, output_dim)),
        )
    else:
        module=nn.Sequential(
            nn.Linear(feats_dim, run_dim),
            nn.ReLU(),
            nn.Linear(run_dim, run_dim),
            nn.ReLU(),
            nn.Linear(run_dim, run_dim),
            nn.ReLU(),
            nn.Linear(run_dim, output_dim),
        )

    return module
@threestudio.register("dreammat-mesh")
class DreamMatMesh(BaseExplicitGeometry):
    @dataclass
    class Config(BaseExplicitGeometry.Config):
        n_input_dims: int = 3
        n_feature_dims: int = 5
        pos_encoding_config: dict = field(
            default_factory=lambda: {
                "otype": "HashGrid",
                "n_levels": 16,
                "n_features_per_level": 2,
                "log2_hashmap_size": 19,
                "base_resolution": 16,
                "per_level_scale": 1.447269237440378,
            }
        )
        mlp_network_config: dict = field(
            default_factory=lambda: {
                "otype": "VanillaMLP",
                "activation": "ReLU",
                "output_activation": "none",
                "n_neurons": 64,
                "n_hidden_layers": 1,
            }
        )
        shape_init: str = ""
        shape_init_params: Optional[Any] = None
        shape_init_mesh_up: str = "+z"
        shape_init_mesh_front: str = "+x"

    cfg: Config

    def configure(self) -> None:
        super().configure()

        self.encoding = get_encoding(
            self.cfg.n_input_dims, self.cfg.pos_encoding_config
        )
        self.feature_network = get_mlp(
            self.encoding.n_output_dims,
            self.cfg.n_feature_dims,
            self.cfg.mlp_network_config,
        )
        self.pos_enc, pos_dim = get_embedder(10, 3)
        self.metallic_predictor = make_predictor(pos_dim, 1)
        self.roughness_predictor = make_predictor(pos_dim, 1)
        self.albedo_predictor = make_predictor(pos_dim, 3)

        # Initialize custom mesh
        if self.cfg.shape_init.startswith("mesh:"):
            assert isinstance(self.cfg.shape_init_params, float)
            mesh_path = self.cfg.shape_init[5:]
            if not os.path.exists(mesh_path):
                raise ValueError(f"Mesh file {mesh_path} does not exist.")

            import trimesh

            scene = trimesh.load(mesh_path)
            if isinstance(scene, trimesh.Trimesh):
                mesh = scene
            elif isinstance(scene, trimesh.scene.Scene):
                mesh = trimesh.Trimesh()
                for obj in scene.geometry.values():
                    mesh = trimesh.util.concatenate([mesh, obj])
            else:
                raise ValueError(f"Unknown mesh type at {mesh_path}.")
            
            if not isinstance(mesh.visual, trimesh.visual.TextureVisuals):
                mesh=mesh.unwrap()

            # move to center
            centroid = mesh.vertices.mean(0)
            mesh.vertices = mesh.vertices - centroid

            # align to up-z and front-x
            dirs = ["+x", "+y", "+z", "-x", "-y", "-z"]
            dir2vec = {
                "+x": np.array([1, 0, 0]),
                "+y": np.array([0, 1, 0]),
                "+z": np.array([0, 0, 1]),
                "-x": np.array([-1, 0, 0]),
                "-y": np.array([0, -1, 0]),
                "-z": np.array([0, 0, -1]),
            }
            if (
                self.cfg.shape_init_mesh_up not in dirs
                or self.cfg.shape_init_mesh_front not in dirs
            ):
                raise ValueError(
                    f"shape_init_mesh_up and shape_init_mesh_front must be one of {dirs}."
                )
            if self.cfg.shape_init_mesh_up[1] == self.cfg.shape_init_mesh_front[1]:
                raise ValueError(
                    "shape_init_mesh_up and shape_init_mesh_front must be orthogonal."
                )
            z_, x_ = (
                dir2vec[self.cfg.shape_init_mesh_up],
                dir2vec[self.cfg.shape_init_mesh_front],
            )
            y_ = np.cross(z_, x_)
            std2mesh = np.stack([x_, y_, z_], axis=0).T
            mesh2std = np.linalg.inv(std2mesh)

            # scaling
            scale = np.abs(mesh.vertices).max()
            mesh.vertices = mesh.vertices / scale * self.cfg.shape_init_params
            mesh.vertices = np.dot(mesh2std, mesh.vertices.T).T

            v_pos = torch.tensor(mesh.vertices, dtype=torch.float32).to(self.device)
            t_pos_idx = torch.tensor(mesh.faces, dtype=torch.int64).to(self.device)
            v_normal = torch.tensor(np.ascontiguousarray(mesh.vertex_normals), dtype=torch.float32).to(self.device)
            v_tex = torch.tensor(mesh.visual.uv, dtype=torch.float32).to(self.device)
                
            self.mesh = Mesh(v_pos=v_pos,t_pos_idx=t_pos_idx, v_nrm=v_normal, v_tex=v_tex)
            self.register_buffer(
                "v_buffer",
                v_pos,
            )
            self.register_buffer(
                "vnrm_buffer",
                v_normal,
            )
            self.register_buffer(
                "vtex_buffer",
                v_tex,
            )
            self.register_buffer(
                "t_buffer",
                t_pos_idx,
            )

        else:
            raise ValueError(
                f"Unknown shape initialization type: {self.cfg.shape_init}"
            )
        print(self.mesh.v_pos.device)

    def isosurface(self) -> Mesh:
        if hasattr(self, "mesh"):
            return self.mesh
        elif hasattr(self, "v_buffer"):
            self.mesh = Mesh(v_pos=self.v_buffer, t_pos_idx=self.t_buffer, v_nrm=self.vnrm_buffer, v_tex=self.vtex_buffer)
            return self.mesh
        else:
            raise ValueError(f"custom mesh is not initialized")

    def forward(
        self, points: Float[Tensor, "*N Di"], output_normal: bool = False
    ) -> Dict[str, Float[Tensor, "..."]]:
        assert (
            output_normal == False
        ), f"Normal output is not supported for {self.__class__.__name__}"
        points_unscaled = points  # points in the original scale
        if self.cfg.n_input_dims==3:
            points = contract_to_unisphere(points, self.bbox3d)  # points normalized to (0, 1)
        else:
            points = contract_to_unisphere(points, self.bbox2d)  # points normalized to (0, 1)
        enc = self.encoding(points.view(-1, self.cfg.n_input_dims))
        features = self.feature_network(enc).view(
            *points.shape[:-1], self.cfg.n_feature_dims
        )
        return {"features": features}

    def export(self, points: Float[Tensor, "*N Di"], **kwargs) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        if self.cfg.n_feature_dims == 0:
            return out
        points_unscaled = points
        if self.cfg.n_input_dims==3:
            points = contract_to_unisphere(points_unscaled, self.bbox3d)
        else:
            points = contract_to_unisphere(points, self.bbox2d)  # points normalized to (0, 1)
        enc = self.encoding(points.reshape(-1, self.cfg.n_input_dims))
        features = self.feature_network(enc).view(
            *points.shape[:-1], self.cfg.n_feature_dims
        )
        out.update(
            {
                "features": features,
            }
        )
        return out