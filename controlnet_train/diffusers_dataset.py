import json
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import os
import random
from pathlib import Path
import json
def loadpromot(promptpath, use_env=True):
    with open(promptpath, 'r') as file:
        lines = file.readlines()
    if len(lines)==0:
        #print(promptpath)
        prompt = "soft lighting"
        with open('lose.txt', 'a') as file:
            file.write(promptpath+"\n")
            file.close()
    else:
        if use_env:
            prompt0=lines[0].strip()
            prompt1=lines[1].strip()
            prompt = prompt0+", "+prompt1
        else:
            prompt=lines[0].strip()
    return prompt

def loadrgb(imgpath,dim):
    img = cv2.imread(imgpath,cv2.IMREAD_UNCHANGED)
    
    if img is None:
        img=np.zeros((dim[0],dim[1],3))
        print(imgpath)
    if(img.shape[2]==4):
        mask=img[...,3].astype(np.bool_)
        img[~mask]=0
        img=img[...,0:3]
    if img.shape[0]!=dim[0]:
        img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img = img.astype(np.float32) / 255.0
    return img

def loadtarget(imgpath,dim):
    img = cv2.imread(imgpath,cv2.IMREAD_UNCHANGED)
    alpha=False
    if img is None:
        img=np.ones((dim[0],dim[1],3))
        print(imgpath)
    if(img.shape[2]==4):
        mask=img[...,3].astype(np.bool_)
        img[~mask]=255
        img=img[...,0:3]
        alpha=True
    if img.shape[0]!=dim[0]:
        img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = (img.astype(np.float32) / 127.5) - 1.0
    return img,alpha

def loaddepth(imgpath,dim):
    depth = cv2.imread(imgpath, cv2.IMREAD_ANYDEPTH)/1000
    if depth.shape[0]!=dim[0]:
        depth = cv2.resize(depth, dim, interpolation = cv2.INTER_NEAREST)
    object_mask = depth>0
        
    if object_mask.sum()<=0:
        print(imgpath)
        return depth[...,None]

    min_val=0.3
   
    depth_inv = 1. / (depth + 1e-6)
    
    depth_max = depth_inv[object_mask].max()
    depth_min = depth_inv[object_mask].min()
                   
    depth[object_mask] = (1 - min_val) *(depth_inv[object_mask] - depth_min) / (depth_max - depth_min + 1e-6) + min_val
    return depth[...,None],object_mask

class DiffusersDataset(Dataset):
    def __init__(self,datadir,promptfile,size=512, is_use_cfg=False):
        self.obj_info = []
        self.size = size
        self.use_cfg = is_use_cfg
        directory = datadir
        

        with open(promptfile) as json_file:
            json_content = json.load(json_file)
        json_file.close()
        for obj_name,obj_prompt in json_content.items():
            subpath = os.path.join(directory, obj_name)
            if os.path.isdir(subpath):
                self.obj_info.append({"path":subpath,"prompt":obj_prompt})

        self.env_num=5
        self.view_num=16
        self.data_num_perobj=self.env_num*self.view_num
        
        

    def __len__(self):
        return len(self.obj_info)*self.env_num*self.view_num

    def __getitem__(self, idx):
        obj_idx=idx//(self.data_num_perobj)
        objpath=self.obj_info[obj_idx]['path']
        prompt=self.obj_info[obj_idx]['prompt']

        env_idx=(idx%(self.data_num_perobj))//self.view_num+1
        view_idx=(idx%(self.data_num_perobj))%self.view_num

        render_path = objpath+"/color/"+f"{view_idx:03d}_color_env"+str(env_idx)+".png"
        depth_path = objpath+"/depth/"+f"{view_idx:03d}.png"
        normal_path = objpath+"/normal/"+f"{view_idx:03d}.png"

        light_path_m0r0 =    objpath+"/light/"+f"{view_idx:03d}_m0.0r0.0_env"+str(env_idx)+".png"
        light_path_m0rhalf = objpath+"/light/"+f"{view_idx:03d}_m0.0r0.5_env"+str(env_idx)+".png"
        light_path_m0r1 =    objpath+"/light/"+f"{view_idx:03d}_m0.0r1.0_env"+str(env_idx)+".png"
        light_path_m1r0 =    objpath+"/light/"+f"{view_idx:03d}_m1.0r0.0_env"+str(env_idx)+".png"
        light_path_m1rhalf = objpath+"/light/"+f"{view_idx:03d}_m1.0r0.5_env"+str(env_idx)+".png"
        light_path_m1r1 =    objpath+"/light/"+f"{view_idx:03d}_m1.0r1.0_env"+str(env_idx)+".png"
      

        dim=(self.size,self.size)

        target,alpha=loadtarget(render_path,dim)

        depth,mask=loaddepth(depth_path,dim)
        normal=loadrgb(normal_path,dim)
        
        if alpha==False:
            target[~mask]=1.0

        light_m0r0=loadrgb(light_path_m0r0,dim)
        light_m0rhalf=loadrgb(light_path_m0rhalf,dim)
        light_m0r1=loadrgb(light_path_m0r1,dim)
        light_m1r0=loadrgb(light_path_m1r0,dim)
        light_m1rhalf=loadrgb(light_path_m1rhalf,dim)
        light_m1r1=loadrgb(light_path_m1r1,dim)

        source=np.concatenate((depth, normal,light_m0r0,light_m0rhalf,light_m0r1,light_m1r0,light_m1rhalf,light_m1r1), axis=-1)

        # save_condition=np.hstack((depth.repeat(3,-1), normal,light_m0r0,light_m0rhalf,light_m0r1,light_m1r0,light_m1rhalf,light_m1r1))

        if self.use_cfg:
            randomnum=random.random()
            if randomnum<0.05:
                source=np.zeros_like(source)
            elif randomnum<0.1 and randomnum>0.05:
                source[...,0]=np.zeros_like(source[...,0])
            elif randomnum<0.15 and randomnum>0.1:
                source[...,1:4]=np.zeros_like(normal)
            elif randomnum<0.2 and randomnum>0.15:
                source[...,4:]=np.zeros_like(source[...,4:])
            elif randomnum<0.5 and randomnum>0.2:
                prompt=""
        
        return dict(pixel_values=torch.from_numpy(target), input_ids=prompt, conditioning_pixel_values=torch.from_numpy(source))
    
class CombinedDataset(Dataset):
    def __init__(self, train_data_dir,prompt_file, resolution, is_use_cfg):
        self.dataset = DiffusersDataset(train_data_dir,prompt_file,resolution, is_use_cfg)
        self.dataset_size = len(self.dataset)
        self.total_size = self.dataset_size
        print("total size: "+str(self.total_size))
    def __len__(self):
        return self.total_size

    def __getitem__(self, idx):
        data_idx = idx 
        data = self.dataset[data_idx]
        return data

