import random
from dataclasses import dataclass, field

import envlight
import numpy as np
import nvdiffrast.torch as dr
import torch
import torch.nn as nn
import torch.nn.functional as F

import threestudio
from threestudio.models.materials.base import BaseMaterial
from threestudio.utils.ops import get_activation
from threestudio.utils.typing import *
import cv2
import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"

class IdentityActivation(nn.Module):
    def forward(self, x): return x

class ExpActivation(nn.Module):
    def __init__(self, max_light=5.0):
        super().__init__()
        self.max_light=max_light

    def forward(self, x):
        return torch.exp(torch.clamp(x, max=self.max_light))

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

def saturate_dot(v0,v1):
    return torch.clamp(torch.sum(v0*v1,dim=-1,keepdim=True),min=0.0,max=1.0)

def load_hdr_image(fn) ->np.ndarray:
    img = cv2.imread(fn, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def hdr_to_ldr(hdr_color, exposure=1.0, gamma=2.4):
    # Scale with exposure
    ldr_color = hdr_color * exposure
    A,B,C,D,E,F = 0.22,0.3,0.1,0.2,0.01,0.3
    x = ldr_color
    ldr_color = ((x * (A * x + C * B) + D * E) / (x * (A * x + B) + D * F)) - E / F
    # Clamp values to the range [0, 1]
    # ldr_color = torch.clamp(ldr_color, 0, 1)
    
    # Apply gamma correction
    assert(torch.any(ldr_color)>0)
    ldr_color = torch.where(ldr_color<0.0031308, ldr_color*12.92, 1.055 * ldr_color.pow(1/gamma) - 0.055)

    #ldr_color=1.055 * ldr_color.pow(1/gamma) - 0.055
    
    # Scale to 8-bit color range
    
    return ldr_color

def sample_sphere(num_samples,begin_elevation = 0):
    """ sample angles from the sphere
    reference: https://zhuanlan.zhihu.com/p/25988652?group_id=828963677192491008
    """
    ratio = (begin_elevation + 90) / 180
    num_points = int(num_samples // (1 - ratio))
    phi = (np.sqrt(5) - 1.0) / 2.
    azimuths = []
    elevations = []
    for n in range(num_points - num_samples, num_points):
        z = 2. * n / num_points - 1.
        azimuths.append(2 * np.pi * n * phi % (2 * np.pi))
        elevations.append(np.arcsin(z))
    return np.array(azimuths), np.array(elevations)

def az_el_to_points(azimuths, elevations):
    z = np.sin(elevations)
    x = np.cos(azimuths)*np.cos(elevations)
    y = np.sin(azimuths)*np.cos(elevations)
    return np.stack([x,y,z],-1) #

def material_smoothness_grad(material, material_jitter):
    lambda_kd=0.25
    lambda_ks=0.1
    kd_grad=torch.abs(material[..., :3]-material_jitter[..., :3])
    ks_grad=torch.abs(material[..., 3:5]-material_jitter[..., 3:5])
    kd_luma_grad = (kd_grad[..., 0] + kd_grad[..., 1] + kd_grad[..., 2]) / 3
    loss  = torch.mean(kd_luma_grad * kd_grad[..., -1]) * lambda_kd
    loss += torch.mean(ks_grad[..., :-1] * ks_grad[..., -1:]) * lambda_ks

    # loss = loss + torch.sum(torch.clamp(material[..., 4:5] - 0.98, min=0))
    # loss = loss + torch.sum(torch.clamp(0.02 - material[..., 4:5], min=0))
    # loss = loss + torch.sum(torch.clamp(material[..., 3:4] - 0.98, min=0))
    # loss = loss + torch.sum(torch.clamp(0.02 - material[..., 3:4], min=0))
    return loss

def make_predictor(feats_dim: object, output_dim: object, weight_norm: object = True, activation='sigmoid', exp_max=0.0) -> object:
    if activation == 'sigmoid':
        activation = nn.Sigmoid()
    elif activation=='exp':
        activation = ExpActivation(max_light=exp_max)
    elif activation=='none':
        activation = IdentityActivation()
    elif activation=='relu':
        activation = nn.ReLU()
    else:
        raise NotImplementedError

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
            activation,
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
            activation,
        )

    return module

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

def get_ml_array(deg_view):
    """Create a list with all pairs of (l, m) values to use in the encoding."""
    ml_list = []
    for i in range(deg_view):
        l = 2**i
        # Only use nonnegative m values, later splitting real and imaginary parts.
        for m in range(l + 1):
            ml_list.append((m, l))

    # Convert list into a numpy array.
    ml_array = np.array(ml_list).T
    return ml_array

def generalized_binomial_coeff(a, k):
    """Compute generalized binomial coefficients."""
    return np.prod(a - np.arange(k)) / np.math.factorial(k)

def assoc_legendre_coeff(l, m, k):
    """Compute associated Legendre polynomial coefficients.

      Returns the coefficient of the cos^k(theta)*sin^m(theta) term in the
      (l, m)th associated Legendre polynomial, P_l^m(cos(theta)).

      Args:
        l: associated Legendre polynomial degree.
        m: associated Legendre polynomial order.
        k: power of cos(theta).

      Returns:
        A float, the coefficient of the term corresponding to the inputs.
    """
    return ((-1)**m * 2**l * np.math.factorial(l) / np.math.factorial(k) /
          np.math.factorial(l - k - m) *
          generalized_binomial_coeff(0.5 * (l + k + m - 1.0), l))

def sph_harm_coeff(l, m, k):
  """Compute spherical harmonic coefficients."""
  return (np.sqrt(
      (2.0 * l + 1.0) * np.math.factorial(l - m) /
      (4.0 * np.pi * np.math.factorial(l + m))) * assoc_legendre_coeff(l, m, k))

def generate_ide_fn(deg_view):
    """Generate integrated directional encoding (IDE) function.

      This function returns a function that computes the integrated directional
      encoding from Equations 6-8 of arxiv.org/abs/2112.03907.

      Args:
        deg_view: number of spherical harmonics degrees to use.

      Returns:
        A function for evaluating integrated directional encoding.

      Raises:
        ValueError: if deg_view is larger than 5.
    """
    if deg_view > 5:
        raise ValueError('Only deg_view of at most 5 is numerically stable.')

    ml_array = get_ml_array(deg_view)
    l_max = 2**(deg_view - 1)

    # Create a matrix corresponding to ml_array holding all coefficients, which,
    # when multiplied (from the right) by the z coordinate Vandermonde matrix,
    # results in the z component of the encoding.
    mat = np.zeros((l_max + 1, ml_array.shape[1]))
    for i, (m, l) in enumerate(ml_array.T):
        for k in range(l - m + 1):
            mat[k, i] = sph_harm_coeff(l, m, k)

    mat = torch.from_numpy(mat.astype(np.float32)).cuda()
    ml_array = torch.from_numpy(ml_array.astype(np.float32)).cuda()

    def integrated_dir_enc_fn(xyz, kappa_inv):
        """Function returning integrated directional encoding (IDE).

        Args:
          xyz: [..., 3] array of Cartesian coordinates of directions to evaluate at.
          kappa_inv: [..., 1] reciprocal of the concentration parameter of the von
            Mises-Fisher distribution.

        Returns:
          An array with the resulting IDE.
        """
        x = xyz[..., 0:1]
        y = xyz[..., 1:2]
        z = xyz[..., 2:3]

        # Compute z Vandermonde matrix.
        vmz = torch.concat([z**i for i in range(mat.shape[0])], dim=-1)

        # Compute x+iy Vandermonde matrix.
        vmxy = torch.concat([(x + 1j * y)**m for m in ml_array[0, :]], dim=-1)

        # Get spherical harmonics.
        sph_harms = vmxy * torch.matmul(vmz, mat)

        # Apply attenuation function using the von Mises-Fisher distribution
        # concentration parameter, kappa.
        sigma = 0.5 * ml_array[1, :] * (ml_array[1, :] + 1)
        ide = sph_harms * torch.exp(-sigma * kappa_inv)

        # Split into real and imaginary parts and return
        return torch.concat([torch.real(ide), torch.imag(ide)], dim=-1)

    return integrated_dir_enc_fn

def equirectangular_to_cubemap(equirectangular_img, cube_size):
    cube_map = np.zeros((6, cube_size, cube_size, 3), dtype=np.float32)
    def get_directions(face):
        if face == 0: 
            x_coord = np.linspace(-1, 1, cube_size)
            y_coord = np.linspace(-1, 1, cube_size)
            grid = np.meshgrid(x_coord, y_coord)
            u,v = grid
            w = np.full_like(u, 1)
            return np.array([-w, u, -v]) 
        if face == 1:
            x_coord = np.linspace(-1, 1, cube_size)
            y_coord = np.linspace(-1, 1, cube_size)
            grid = np.meshgrid(x_coord, y_coord)
            u,v = grid
            w = np.full_like(u, 1)
            return np.array([w, -u, -v])
        if face == 2:
            x_coord = np.linspace(-1, 1, cube_size)
            y_coord = np.linspace(-1, 1, cube_size)
            grid = np.meshgrid(x_coord, y_coord)
            u,v = grid
            w = np.full_like(u, 1)
            return np.array([-u, -w, -v])
        if face == 3: 
            x_coord = np.linspace(-1, 1, cube_size)
            y_coord = np.linspace(-1, 1, cube_size)
            grid = np.meshgrid(x_coord, y_coord)
            u,v = grid
            w = np.full_like(u, 1)
            return np.array([u, w, -v])
        if face == 4:
            x_coord = np.linspace(-1, 1, cube_size)
            y_coord = np.linspace(-1, 1, cube_size)
            grid = np.meshgrid(x_coord, y_coord)
            u,v = grid
            w = np.full_like(u, 1)
            return np.array([u, v, w])  
        if face == 5: 
            x_coord = np.linspace(-1, 1, cube_size)
            y_coord = np.linspace(-1, 1, cube_size)
            grid = np.meshgrid(x_coord, y_coord)
            u,v = grid
            w = np.full_like(u, 1)
            return np.array([u, -v, -w])  
    for face in range(6):
        directions = get_directions(face)
        directions /= np.linalg.norm(directions,axis=0,ord=2)
        theta = np.arccos(directions[2])
        phi = np.arctan2(directions[1], directions[0])

        width = equirectangular_img.shape[1]
        height = equirectangular_img.shape[0]
        u = -phi / (2 * np.pi)+0.5
        v = theta / np.pi
        x = (u * width) % width  # Ensure x is within image bounds
        y = (v * height) % height
        pixel_values = cv2.remap(equirectangular_img.cpu().numpy(), x.astype(np.float32), y.astype(np.float32), cv2.INTER_LINEAR, cv2.BORDER_WRAP)
        cube_map[face, :, :, :] = pixel_values
        cv2.imwrite(f'cube_{face}.png', (pixel_values * 255))
    return torch.from_numpy(cube_map).to('cuda')

@threestudio.register("dreammat-material")
class DreamMatMaterial(BaseMaterial):
    @dataclass
    class Config(BaseMaterial.Config):
        material_activation: str = "sigmoid"
        environment_texture: str = "load/lights/mud_road_puresky_1k.hdr"
        environment_scale: float = 1.0
        min_metallic: float = 0.0
        max_metallic: float = 0.9
        min_roughness_squre: float = 0.01
        max_roughness_squre: float = 0.9
        min_roughness: float = 0.1
        max_roughness: float = 0.95
        use_bump: bool = True

        diffuse_sample_num: int = 512
        specular_sample_num: int = 256
        geometry_type: str='schlick'
        random_azimuth: bool = True

        use_raytracing: bool = True

    cfg: Config

    def configure(self) -> None:
        self.requires_normal = True
        self.requires_tangent = self.cfg.use_bump

        self.light=[]
        self.envlight=[]
        self.cube_map=[]


        for i in range(5):
            index=str(i+1)
            pathexr = self.cfg.environment_texture+"/map"+index+"/map"+index+'.exr'
            pathhdr = self.cfg.environment_texture+"/map"+index+"/map"+index+'.hdr'
            self.envlight.append(envlight.EnvLight(pathhdr, scale=self.cfg.environment_scale))
            latlong_img = torch.tensor(load_hdr_image(pathexr), dtype=torch.float32, device='cuda')
            print("EnvProbe,", latlong_img.shape, ", min/max", torch.min(latlong_img).item(), torch.max(latlong_img).item())
            self.light.append(latlong_img)
            # self.cube_map.append(equirectangular_to_cubemap(latlong_img,512))

        # predefined diffuse sample directions
        az, el = sample_sphere(self.cfg.diffuse_sample_num, 0)
        az, el = az * 0.5 / np.pi, 1 - 2 * el / np.pi # scale to [0,1]
        self.diffuse_direction_samples = np.stack([az, el], -1)
        self.diffuse_direction_samples = torch.from_numpy(self.diffuse_direction_samples.astype(np.float32)).cuda() # [dn0,2]

        az, el = sample_sphere(self.cfg.specular_sample_num, 0)
        az, el = az * 0.5 / np.pi, 1 - 2 * el / np.pi # scale to [0,1]
        self.specular_direction_samples = np.stack([az, el], -1)
        self.specular_direction_samples = torch.from_numpy(self.specular_direction_samples.astype(np.float32)).cuda() # [dn1,2]

        az, el = sample_sphere(8192, 0)
        light_pts = az_el_to_points(az, el)
        self.register_buffer('light_pts', torch.from_numpy(light_pts.astype(np.float32)))


        FG_LUT = torch.from_numpy(
            np.fromfile("load/lights/bsdf_256_256.bin", dtype=np.float32).reshape(
                1, 256, 256, 2
            )
        )
        self.register_buffer("FG_LUT", FG_LUT)

        self.pos_enc, pos_dim = get_embedder(8, 3)
        #self.illum_enc, illum_dim = get_embedder(10, 1)
        self.inner_light = make_predictor(pos_dim + 72 , 3, activation='exp', exp_max=5.0)
        nn.init.constant_(self.inner_light[-2].bias, np.log(0.5))
        self.sph_enc = generate_ide_fn(5)

    def get_inner_lights(self, points, view_dirs, normals):
        pos_enc = self.pos_enc(points)
        normals = F.normalize(normals,dim=-1)
        view_dirs = F.normalize(view_dirs,dim=-1)
        reflections = torch.sum(view_dirs * normals, -1, keepdim=True) * normals * 2 - view_dirs
        dir_enc = self.sph_enc(reflections, 0)
        return self.inner_light(torch.cat([pos_enc, dir_enc], -1))
    
    def set_raytracer(self,ray_trace_fun):
        self.ray_trace_fun = ray_trace_fun

    def get_envirmentlight(self,directions,env_id):
        latitude = 1.0-(torch.acos(torch.clamp(directions[:,1],-0.999,0.999))*1.0 / torch.pi)
        longitude = 0.5-(torch.atan2(directions[:,2], directions[:,0])*0.5 / torch.pi)
        assert(torch.isnan(latitude).any()==False)
        assert(torch.isnan(longitude).any()==False)
        h,w,_=self.light[env_id].shape
        x=torch.floor(latitude*(w-1))
        y=torch.floor(longitude*(h-1))
        return self.light[env_id][y.long(),x.long(),:]
    
    def get_envirmentlight_blender(self,directions,env_id):

        height,width,_=self.light[env_id].shape
        
        directions = directions / directions.norm(p=2, dim=-1, keepdim=True)  # Normalize the vectors
        x, y, z = directions.unbind(-1)

        # Compute theta and phi
        theta = torch.acos(z)
        phi = torch.atan2(y, x) % (2 * np.pi)
        
        u = -phi / (2 * np.pi)+0.5
        v = theta / np.pi
        x = (u * width) % width  # Ensure x is within image bounds
        y = (v * height) % height
        
        return self.light[env_id][y.long(),x.long(),:]
    
    def get_environment_light_cubemap(self, directions,env_id):
        cube_size = self.cube_map[env_id].shape[1]
        directions = directions / directions.norm(p=2, dim=-1, keepdim=True)  # Normalize the vectors
        x, y, z = directions.unbind(-1)

        # Compute the face of the cube map and the coordinates within the face
        abs_directions = torch.abs(directions)
        max_dim = torch.argmax(abs_directions, dim=-1)
        u = torch.zeros_like(x)
        v = torch.zeros_like(y)
        face = torch.zeros_like(max_dim)

        # Compute the coordinates within the face
        mask = max_dim == 0
        u[mask] = ((y[mask] / abs_directions[mask, 0]) + 1) / 2
        v[mask] = ((-z[mask] /abs_directions[mask, 0]) + 1) / 2
        face[mask] = 0 + (x[mask] > 0).long()

        mask = max_dim == 1
        u[mask] = ((x[mask] / abs_directions[mask, 1]) + 1) / 2
        v[mask] = ((-z[mask] / abs_directions[mask, 1]) + 1) / 2
        face[mask] = 2+ (y[mask] > 0).long()

        mask = max_dim == 2
        u[mask] = ((x[mask] / abs_directions[mask, 2]) + 1) / 2
        v[mask] = ((-y[mask] / abs_directions[mask, 2]) + 1) / 2
        face[mask] = 5- (z[mask] > 0).long()

        # Convert u, v to pixel coordinates
        u = (u * (cube_size - 1)).long()
        v = (v * (cube_size - 1)).long()

        return self.cube_map[env_id][face, v, u, :]
    def get_lights(self, points, directions,  env_id):
        # trace
        shape = points.shape[:-1] # pn,sn
        eps = 1e-5
        inters, normals, depth, hit_mask = self.ray_trace_fun(points.reshape(-1,3)+directions.reshape(-1,3) * eps, directions.reshape(-1,3))
        inters, normals, depth, hit_mask = inters.reshape(*shape,3), normals.reshape(*shape,3), depth.reshape(*shape, 1), hit_mask.reshape(*shape)
        miss_mask = ~hit_mask

        # hit_mask
        lights = torch.zeros((*shape, 3),device=points.device)
        if torch.sum(miss_mask)>0:
            outer_lights = self.get_envirmentlight_blender(directions[miss_mask],env_id)#self.predict_outer_lights(points[miss_mask], directions[miss_mask], enviroment_map)
            # outer_lights = self.get_environment_light_cubemap(directions[miss_mask],env_id)
            lights[miss_mask] = outer_lights
            
        if torch.sum(hit_mask)>0:
            lights[hit_mask] = 0.0#self.get_inner_lights(inters[hit_mask], -directions[hit_mask], normals[hit_mask])  # direct light
        return lights#, inters, normals, hit_mask

    def fresnel_schlick(self, F0, HoV):
        return F0 + (1.0 - F0) * torch.clamp(1.0 - HoV, min=0.0, max=1.0)**5.0

    def fresnel_schlick_directions(self, F0, view_dirs, directions):
        H = (view_dirs + directions) # [pn,sn0,3]
        H = F.normalize(H, dim=-1)
        HoV = torch.clamp(torch.sum(H * view_dirs, dim=-1, keepdim=True), min=0.0, max=1.0) # [pn,sn0,1]
        fresnel = self.fresnel_schlick(F0, HoV) # [pn,sn0,1]
        return fresnel, H, HoV

    def geometry_schlick_ggx(self, NoV, roughness):
        a = roughness # a = roughness**2: we assume the predicted roughness is already squared

        k = a / 2
        num = NoV
        denom = NoV * (1 - k) + k
        return num / (denom + 1e-5)

    def geometry_schlick(self, NoV, NoL, roughness):
        ggx2 = self.geometry_schlick_ggx(NoV, roughness)
        ggx1 = self.geometry_schlick_ggx(NoL, roughness)
        return ggx2 * ggx1

    def geometry_ggx_smith_correlated(self, NoV, NoL, roughness):
        def fun(alpha2, cos_theta):
            # cos_theta = torch.clamp(cos_theta,min=1e-7,max=1-1e-7)
            cos_theta2 = cos_theta**2
            tan_theta2 = (1 - cos_theta2) / (cos_theta2 + 1e-7)
            return 0.5 * torch.sqrt(1+alpha2*tan_theta2) - 0.5

        alpha_sq = roughness ** 2
        return 1.0 / (1.0 + fun(alpha_sq, NoV) + fun(alpha_sq, NoL))
    
    def get_orthogonal_directions(self, directions):
        x, y, z = torch.split(directions, 1, dim=-1) # pn,1
        otho0 = torch.cat([y,-x,torch.zeros_like(x)],-1)
        otho1 = torch.cat([-z,torch.zeros_like(x),x],-1)
        mask0 = torch.norm(otho0,dim=-1)>torch.norm(otho1,dim=-1)
        mask1 = ~mask0
        otho = torch.zeros_like(directions)
        otho[mask0] = otho0[mask0]
        otho[mask1] = otho1[mask1]
        otho = F.normalize(otho, dim=-1)
        return otho

    def sample_diffuse_directions(self, normals, is_train):
        # normals [pn,3]
        z = normals # pn,3
        x = self.get_orthogonal_directions(normals) # pn,3
        y = torch.cross(z, x, dim=-1) # pn,3
        # y = torch.cross(z, x, dim=-1) # pn,3

        # project onto this tangent space
        az, el = torch.split(self.diffuse_direction_samples,1,dim=1) # sn,1
        el, az = el.unsqueeze(0), az.unsqueeze(0)
        az = az * torch.pi * 2
        el_sqrt = torch.sqrt(el+1e-7)
        if is_train and self.cfg.random_azimuth:
            az = (az + torch.rand((z.shape[0], 1, 1),device=az.device) * torch.pi * 2) % (2 * torch.pi)
        coeff_z = torch.sqrt(1 - el + 1e-7)
        coeff_x = el_sqrt * torch.cos(az)
        coeff_y = el_sqrt * torch.sin(az)

        directions = coeff_x * x.unsqueeze(1) + coeff_y * y.unsqueeze(1) + coeff_z * z.unsqueeze(1) # pn,sn,3
        return directions

    def sample_specular_directions(self, reflections, roughness, is_train):
        # roughness [pn,1]
        z = reflections  # pn,3
        x = self.get_orthogonal_directions(reflections)  # pn,3
        y = torch.cross(z, x, dim=-1)  # pn,3
        a = roughness # we assume the predicted roughness is already squared

        az, el = torch.split(self.specular_direction_samples, 1, dim=1)  # sn,1
        phi = np.pi * 2 * az # sn,1
        a, el = a.unsqueeze(1), el.unsqueeze(0) # [pn,1,1] [1,sn,1]
        cos_theta = torch.sqrt((1.0 - el + 1e-6) / (1.0 + (a**2 - 1.0) * el + 1e-6) + 1e-6) # pn,sn,1
        sin_theta = torch.sqrt(1 - cos_theta**2 + 1e-6) # pn,sn,1

        phi = phi.unsqueeze(0) # 1,sn,1
        if is_train and self.cfg.random_azimuth:
            phi = (phi + torch.rand((z.shape[0], 1, 1),device=az.device) * np.pi * 2) % (2 * np.pi)
        coeff_x = torch.cos(phi) * sin_theta # pn,sn,1
        coeff_y = torch.sin(phi) * sin_theta # pn,sn,1
        coeff_z = cos_theta # pn,sn,1

        directions = coeff_x * x.unsqueeze(1) + coeff_y * y.unsqueeze(1) + coeff_z * z.unsqueeze(1) # pn,sn,3
        return directions
        #return reflections.unsqueeze(1).repeat(1,self.specular_direction_samples.shape[0],1)

    def distribution_ggx(self, NoH, roughness):
        a = roughness
        a2 = a**2
        NoH2 = NoH**2
        denom = NoH2 * (a2 - 1.0) + 1.0
        return a2 / (np.pi * denom**2 + 1e-4)

    def geometry(self,NoV, NoL, roughness):
        if self.cfg.geometry_type=='schlick':
            geometry = self.geometry_schlick(NoV, NoL, roughness)
        elif self.cfg.geometry_type=='ggx_smith':
            geometry = self.geometry_ggx_smith_correlated(NoV, NoL, roughness)
        else:
            raise NotImplementedError
        return geometry

    def shade_raytracing(self, pts, normals, view_dirs, env_id, metallic, roughness, albedo, is_train):
        if(torch.isnan(roughness).any()==True):
            import ipdb
            ipdb.set_trace()

        reflections = torch.sum(view_dirs * normals, -1, keepdim=True) * normals * 2 - view_dirs
        F0 = 0.04 * (1 - metallic) + metallic * albedo # [pn,1]

        # sample diffuse directions
        diffuse_directions = self.sample_diffuse_directions(normals, is_train)  # [pn,sn0,3]
        point_num, diffuse_num, _ = diffuse_directions.shape
        # sample specular directions
        specular_directions = self.sample_specular_directions(reflections, roughness, is_train) # [pn,sn1,3]
        
        specular_num = specular_directions.shape[1]

        # diffuse sample prob
        NoL_d = saturate_dot(diffuse_directions, normals.unsqueeze(1))
        diffuse_probability = NoL_d / np.pi * (diffuse_num / (specular_num+diffuse_num))

        # specualr sample prob
        H_s = (view_dirs.unsqueeze(1) + specular_directions) # [pn,sn0,3]
        H_s = F.normalize(H_s, dim=-1)
        NoH_s = saturate_dot(normals.unsqueeze(1), H_s)
        VoH_s = saturate_dot(view_dirs.unsqueeze(1),H_s)
        specular_probability = self.distribution_ggx(NoH_s, roughness.unsqueeze(1)) * NoH_s / (4 * VoH_s + 1e-5) * (specular_num / (specular_num+diffuse_num)) # D * NoH / (4 * VoH)

        # combine
        directions = torch.cat([diffuse_directions, specular_directions], 1)
        probability = torch.cat([diffuse_probability, specular_probability], 1)
        sn = diffuse_num+specular_num

        # specular
        fresnel, H, HoV = self.fresnel_schlick_directions(F0.unsqueeze(1), view_dirs.unsqueeze(1), directions)
        NoV = saturate_dot(normals, view_dirs).unsqueeze(1) # pn,1,3
        NoL = saturate_dot(normals.unsqueeze(1), directions) # pn,sn,3
        geometry = self.geometry(NoV, NoL, roughness.unsqueeze(1))
        NoH = saturate_dot(normals.unsqueeze(1), H)
        distribution = self.distribution_ggx(NoH, roughness.unsqueeze(1))
        pts_ = pts.unsqueeze(1).repeat(1, sn, 1)
        lights = self.get_lights(pts_, directions, env_id) # pn,sn,3
        specular_weights = distribution * geometry / (4 * NoV * probability + 1e-5)
        specular_lights =  lights * specular_weights
        specular_colors = torch.mean(fresnel * specular_lights, 1)
        specular_weights = specular_weights * fresnel

        diffuse_lights = lights[:,:diffuse_num]
        diffuse_colors = albedo.unsqueeze(1) * diffuse_lights 
        diffuse_colors = torch.mean(diffuse_colors, 1)

        colors = diffuse_colors + specular_colors
        colors=get_activation("lin2srgb")(colors)
        outputs={}
        outputs['color']=colors
        outputs['albedo'] = get_activation("lin2srgb")(albedo.detach())
        outputs['roughness'] = torch.sqrt(roughness + 1e-7)
        outputs['metalness'] = metallic
        outputs['specular_lights'] = get_activation("lin2srgb")(torch.mean(lights[:,diffuse_num:,:].detach(),dim=1))
        outputs['diffuse_lights'] = get_activation("lin2srgb")(torch.mean(lights[:,:diffuse_num,:].detach(),dim=1))
        outputs['specular_colors'] = get_activation("lin2srgb")(specular_colors.detach())
        outputs['diffuse_colors'] = get_activation("lin2srgb")(diffuse_colors.detach())

        return outputs

    def shade_splitsum(self,normals, viewdirs, env_id, metallic, roughness, albedo, prefix_shape):
        v = viewdirs
        n_dot_v = (normals * v).sum(-1, keepdim=True)
        reflective = n_dot_v * normals * 2 - v

        diffuse_albedo = albedo #* (1 - metallic)

        fg_uv = torch.cat([n_dot_v, roughness], -1).clamp(0, 1)
        fg = dr.texture(
            self.FG_LUT,
            fg_uv.reshape(1, -1, 1, 2).contiguous(),
            filter_mode="linear",
            boundary_mode="clamp",
        ).reshape(*prefix_shape, 2)
        F0 = (1 - metallic) * 0.04 + metallic * albedo
        specular_albedo = F0 * fg[:, 0:1] + fg[:, 1:2]

        diffuse_light = self.envlight[env_id](normals)
        specular_light = self.envlight[env_id](reflective, roughness)

        color = diffuse_albedo * diffuse_light + specular_albedo * specular_light
        color = color.clamp(0.0, 1.0)

        outputs={}
        outputs['color']=color
        outputs['albedo'] = albedo #get_activation("lin2srgb")(albedo.detach())
        outputs['roughness'] = roughness
        outputs['metalness'] = metallic
        outputs['specular_lights'] = get_activation("lin2srgb")(specular_light)
        outputs['diffuse_lights'] = get_activation("lin2srgb")(diffuse_light)
        outputs['specular_colors'] = get_activation("lin2srgb")(specular_albedo)
        outputs['diffuse_colors'] = get_activation("lin2srgb")(diffuse_albedo)
        return outputs

    def forward(
        self,
        pts: Float[Tensor, "*B 3"],
        features: Float[Tensor, "*B Nf"],
        features_jitter: Float[Tensor, "*B Nf"],
        viewdirs: Float[Tensor, "*B 3"],
        normals: Float[Tensor, "B ... 3"],
        env_id: Float[Tensor,"B 1"],
        **kwargs,
    ) -> Float[Tensor, "*B 3"]:
        prefix_shape = features.shape[:-1]

        
    
        if self.cfg.use_raytracing:
            material: Float[Tensor, "*B Nf"] = get_activation(self.cfg.material_activation)(
                features
            )
            material_jitter: Float[Tensor, "*B Nf"] = get_activation(self.cfg.material_activation)(
                features_jitter
            )
            mat_reg = material_smoothness_grad(material,material_jitter)
            albedo = material[..., :3].clamp(0.0,1.0)
            metallic = (
                material[..., 3:4] * (self.cfg.max_metallic - self.cfg.min_metallic)
                + self.cfg.min_metallic
            )
            roughness = (
                material[..., 4:5] * (self.cfg.max_roughness_squre - self.cfg.min_roughness_squre)
                + self.cfg.min_roughness_squre
            )
            outputs = self.shade_raytracing(pts, normals, viewdirs, env_id, metallic, roughness, albedo,is_train=True)
        else:
            material: Float[Tensor, "*B Nf"] = get_activation(self.cfg.material_activation)(
                features
            )
            material_jitter: Float[Tensor, "*B Nf"] = get_activation(self.cfg.material_activation)(
                features_jitter
            )
            mat_reg = material_smoothness_grad(material,material_jitter)
            albedo = material[..., :3].clamp(0.0,1.0)
            metallic = (
                material[..., 3:4] * (self.cfg.max_metallic - self.cfg.min_metallic)
                + self.cfg.min_metallic
            )
            roughness = (
                material[..., 4:5] * (self.cfg.max_roughness - self.cfg.min_roughness)
                + self.cfg.min_roughness
            )
            outputs = self.shade_splitsum(normals, viewdirs, env_id, metallic, roughness, albedo, prefix_shape)
        return outputs,mat_reg

    def export(self, features: Float[Tensor, "*N Nf"], **kwargs) -> Dict[str, Any]:
        material: Float[Tensor, "*N Nf"] = get_activation(self.cfg.material_activation)(
            features
        )
        albedo = material[..., :3]
        metallic = (
            material[..., 3:4] * (self.cfg.max_metallic - self.cfg.min_metallic)
            + self.cfg.min_metallic
        )
        roughness = (
            material[..., 4:5] * (self.cfg.max_roughness_squre - self.cfg.min_roughness_squre)
            + self.cfg.min_roughness_squre
        )

        out = {
            "albedo": albedo,
            "metallic": metallic,
            "roughness": torch.sqrt(roughness + 1e-7),
        }

        if self.cfg.use_bump:
            perturb_normal = (material[..., 5:8] * 2 - 1) + torch.tensor(
                [0, 0, 1], dtype=material.dtype, device=material.device
            )
            perturb_normal = F.normalize(perturb_normal.clamp(-1, 1), dim=-1)
            perturb_normal = (perturb_normal + 1) / 2
            out.update(
                {
                    "bump": perturb_normal,
                }
            )

        return out
