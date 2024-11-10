import numpy as np
import torch
import os
import cv2
import torch.nn.functional as F
from util.render import MeshRenderer, Pytorch3dRasterizer
import argparse
from . import networks

from .pix2pix.pix2pix_model import Pix2PixModel
from .pix2pix.pix2pix_options import Pix2PixOptions

def process_uv_one(uv_coords):
    uv_coords[:,0] = uv_coords[:,0]
    uv_coords[:,1] = uv_coords[:,1] # 1 - uv_coords[:,1]
    uv_coords = np.hstack((uv_coords, np.zeros((uv_coords.shape[0], 1)))) # add z
    return uv_coords

def vertex_normals(vertices, faces):
    """
    :param vertices: [batch size, number of vertices, 3]
    :param faces: [batch size, number of faces, 3]
    :return: [batch size, number of vertices, 3]
    """
    assert (vertices.ndimension() == 3)
    assert (faces.ndimension() == 3)
    assert (vertices.shape[0] == faces.shape[0])
    assert (vertices.shape[2] == 3)
    assert (faces.shape[2] == 3)
    bs, nv = vertices.shape[:2]
    bs, nf = faces.shape[:2]
    device = vertices.device
    normals = torch.zeros(bs * nv, 3).to(device)

    faces = faces + (torch.arange(bs, dtype=torch.int32).to(device) * nv)[:, None, None] # expanded faces
    vertices_faces = vertices.reshape((bs * nv, 3))[faces.long()]

    faces = faces.reshape(-1, 3)
    vertices_faces = vertices_faces.reshape(-1, 3, 3)

    normals.index_add_(0, faces[:, 1].long(), 
                       torch.cross(vertices_faces[:, 2] - vertices_faces[:, 1], vertices_faces[:, 0] - vertices_faces[:, 1]))
    normals.index_add_(0, faces[:, 2].long(), 
                       torch.cross(vertices_faces[:, 0] - vertices_faces[:, 2], vertices_faces[:, 1] - vertices_faces[:, 2]))
    normals.index_add_(0, faces[:, 0].long(),
                       torch.cross(vertices_faces[:, 1] - vertices_faces[:, 0], vertices_faces[:, 2] - vertices_faces[:, 0]))

    normals = F.normalize(normals, eps=1e-6, dim=1)
    normals = normals.reshape((bs, nv, 3))
    # pytorch only supports long and byte tensors for indexing
    return normals

def process_uv(uv_coords, uv_h = 224, uv_w = 224):
    uv_coords[:,0] = uv_coords[:,0] * (uv_w - 1)
    uv_coords[:,1] = uv_coords[:,1] * (uv_h - 1)
    # uv_coords[:,1] = uv_h - uv_coords[:,1] - 1
    uv_coords = np.hstack((uv_coords, np.zeros((uv_coords.shape[0], 1)))) # add z
    return uv_coords

def bilinear_interpolate(img, x, y):
    
    x0 = torch.floor(x).long()
    x1 = x0 + 1
    y0 = torch.floor(y).long()
    y1 = y0 + 1

    x0 = torch.clamp(x0, 0, img.shape[1] - 1)
    x1 = torch.clamp(x1, 0, img.shape[1] - 1)
    y0 = torch.clamp(y0, 0, img.shape[0] - 1)
    y1 = torch.clamp(y1, 0, img.shape[0] - 1)

    i_a = img[y0, x0]
    i_b = img[y1, x0]
    i_c = img[y0, x1]
    i_d = img[y1, x1]

    wa = (x1 - x) * (y1 - y)
    wb = (x1 - x) * (y - y0)
    wc = (x - x0) * (y1 - y)
    wd = (x - x0) * (y - y0)

    return wa.unsqueeze(-1) * i_a + wb.unsqueeze(-1) * i_b + wc.unsqueeze(-1) * i_c + wd.unsqueeze(-1) * i_d

class face_model:
    def __init__(self, args):

        self.args = args

        self.device = self.args.device
        model = np.load("./TDDFAv3/assets/face_model.npy",allow_pickle=True).item()

        # mean shape, size (107127, 1)
        self.u = torch.tensor(model['u'], requires_grad=False, dtype=torch.float32, device=self.device)
        # face identity bases, size (107127, 80)
        self.id = torch.tensor(model['id'], requires_grad=False, dtype=torch.float32, device=self.device)
        # face expression bases, size (107127, 64)
        self.exp = torch.tensor(model['exp'], requires_grad=False, dtype=torch.float32, device=self.device)
        # mean albedo, size (107127, 1)
        self.u_alb = torch.tensor(model['u_alb'], requires_grad=False, dtype=torch.float32, device=self.device)
        # face albedo bases, size (107127, 80)
        self.alb = torch.tensor(model['alb'], requires_grad=False, dtype=torch.float32, device=self.device)
        # for computing vertex normals, size (35709, 8), see https://github.com/sicxu/Deep3DFaceRecon_pytorch/issues/132
        self.point_buf = torch.tensor(model['point_buf'], requires_grad=False, dtype=torch.int64, device=self.device)
        # triangle faces, size (70789, 3)
        self.tri = torch.tensor(model['tri'], requires_grad=False, dtype=torch.int64, device=self.device)
        # vertex uv coordinates, size (35709, 3), range (0, 1.)
        self.uv_coords = torch.tensor(model['uv_coords'], requires_grad=False, dtype=torch.float32, device=self.device)

        temp = process_uv(model['uv_coords'].copy(), 1024, 1024)
        self.uv_coords_torch = (torch.tensor(temp, requires_grad=False, dtype=torch.float32, device=self.device) / 1023 - 0.5) * 2

        # vertex indices for 68 landmarks, size (68,)
        if self.args.ldm68:
            self.ldm68 = torch.tensor(model['ldm68'], requires_grad=False, dtype=torch.int64, device=self.device)
        # vertex indices for 106 landmarks, size (106,)
        if self.args.ldm106 or self.args.ldm106_2d:
            self.ldm106 = torch.tensor(model['ldm106'], requires_grad=False, dtype=torch.int64, device=self.device)
        # vertex indices for 134 landmarks, size (134,)
        if self.args.ldm134:
            self.ldm134 = torch.tensor(model['ldm134'], requires_grad=False, dtype=torch.int64, device=self.device)

        # segmentation annotation indices for 8 parts, [right_eye, left_eye, right_eyebrow, left_eyebrow, nose, up_lip, down_lip, skin]
        if self.args.seg_visible:
            self.annotation = model['annotation']

        # segmentation triangle faces for 8 parts
        if self.args.seg:
            self.annotation_tri = [torch.tensor(i, requires_grad=False, dtype=torch.int64, device=self.device) for i in model['annotation_tri']]

        # face profile parallel, list
        if self.args.ldm106_2d:
            self.parallel = model['parallel']
            # parallel for profile matching
            self.v_parallel = - torch.ones(35709, device=self.device).type(torch.int64)
            for i in range(len(self.parallel)):
                self.v_parallel[self.parallel[i]]=i

        # focal = 1015, center = 112
        self.persc_proj = torch.tensor([1015.0, 0, 112.0, 0, 1015.0, 112.0, 0, 0, 1], requires_grad=False, dtype=torch.float32, device=self.device).reshape([3, 3]).transpose(0,1)
        self.camera_distance = 10.0

        self.renderer = MeshRenderer(
                    rasterize_fov=2 * np.arctan(112. / 1015) * 180 / np.pi, znear=5., zfar=15., rasterize_size=int(2 * 112.)
        )

        self.render_deca = Pytorch3dRasterizer()

        # Base Reconstruction
        self.net_recon = networks.define_net_recon(
            net_recon='resnet50', use_last_fc=False, init_path=None
        )
        self.net_recon.load_state_dict(torch.load("./assets/sketch_recon.pth", map_location=torch.device('cpu'))['sketch_base'])
        self.net_recon = self.net_recon.to(self.device)
        self.net_recon.eval()

        self.input_img = None

        # Detail Reconstruction
        self.high_opt = Pix2PixOptions()
        self.high_opt.input_nc = 6
        self.high_opt.output_nc = 1
        self.high_opt.name = 'high_net'
        self.high_net = Pix2PixModel(self.high_opt).netG
        self.high_net.load_state_dict(torch.load("./assets/sketch_recon.pth", map_location=torch.device('cpu'))['sketch_high'])
        self.high_net = self.high_net.to(self.device)
        self.high_net.eval()

        self.beta = torch.load("./assets/sketch_recon.pth", map_location=torch.device('cpu'))['sketch_beta'].to(self.device)
        self.beta.requires_grad = False

        g_uv_coords = np.load('./assets/bfm_uvs2.npy').astype(np.float32)
        self.uv_coords_one_numpy = process_uv_one(g_uv_coords.copy())

        self.gray_tex = torch.ones((1, 3, 256, 256)).to(self.device) * 156 / 255
        self.gray_tex[:,0,:,:] = 190 / 255


    def compute_shape(self, alpha_id, alpha_exp):
        """
        Return:
            face_shape       -- torch.tensor, size (B, N, 3), face vertice without rotation or translation

        Parameters:
            alpha_id         -- torch.tensor, size (B, 80), identity parameter
            alpha_exp        -- torch.tensor, size (B, 64), expression parameter
        """
        batch_size = alpha_id.shape[0]
        face_shape = torch.einsum('ij,aj->ai', self.id, alpha_id) + torch.einsum('ij,aj->ai', self.exp, alpha_exp) + self.u.reshape([1, -1])
        return face_shape.reshape([batch_size, -1, 3])

    def compute_albedo(self, alpha_alb, normalize=True):
        """
        Return:
            face_albedo     -- torch.tensor, size (B, N, 3), in RGB order, range (0, 1.), without lighting

        Parameters:
            alpha_alb        -- torch.tensor, size (B, 80), albedo parameter
        """
        batch_size = alpha_alb.shape[0]
        face_albedo = torch.einsum('ij,aj->ai', self.alb, alpha_alb) + self.u_alb.reshape([1, -1])
        if normalize:
            face_albedo = face_albedo / 255.
        return face_albedo.reshape([batch_size, -1, 3])

    def compute_norm(self, face_shape):
        """
        Return:
            vertex_norm      -- torch.tensor, size (B, N, 3)

        Parameters:
            face_shape       -- torch.tensor, size (B, N, 3)
        """
        v1 = face_shape[:, self.tri[:, 0]]
        v2 = face_shape[:, self.tri[:, 1]]
        v3 = face_shape[:, self.tri[:, 2]]
        e1 = v1 - v2
        e2 = v2 - v3
        face_norm = torch.cross(e1, e2, dim=-1)
        face_norm = F.normalize(face_norm, dim=-1, p=2)
        face_norm = torch.cat([face_norm, torch.zeros(face_norm.shape[0], 1, 3).to(self.device)], dim=1)
        
        vertex_norm = torch.sum(face_norm[:, self.point_buf], dim=2)
        vertex_norm = F.normalize(vertex_norm, dim=-1, p=2)
        return vertex_norm

    def compute_rotation(self, angles):
        """
        Return:
            rot              -- torch.tensor, size (B, 3, 3), pts @ trans_mat

        Parameters:
            angles           -- torch.tensor, size (B, 3), use radian
        """
        batch_size = angles.shape[0]
        ones = torch.ones([batch_size, 1]).to(self.device)
        zeros = torch.zeros([batch_size, 1]).to(self.device)
        x, y, z = angles[:, :1], angles[:, 1:2], angles[:, 2:],
        
        rot_x = torch.cat([
            ones, zeros, zeros,
            zeros, torch.cos(x), -torch.sin(x), 
            zeros, torch.sin(x), torch.cos(x)
        ], dim=1).reshape([batch_size, 3, 3])
        
        rot_y = torch.cat([
            torch.cos(y), zeros, torch.sin(y),
            zeros, ones, zeros,
            -torch.sin(y), zeros, torch.cos(y)
        ], dim=1).reshape([batch_size, 3, 3])

        rot_z = torch.cat([
            torch.cos(z), -torch.sin(z), zeros,
            torch.sin(z), torch.cos(z), zeros,
            zeros, zeros, ones
        ], dim=1).reshape([batch_size, 3, 3])

        rot = rot_z @ rot_y @ rot_x
        return rot.permute(0, 2, 1)

    def to_camera(self, face_shape):
        face_shape[..., -1] = self.camera_distance - face_shape[..., -1]
        return face_shape

    def to_image(self, face_shape):
        """
        Return:
            face_proj        -- torch.tensor, size (B, N, 2)

        Parameters:
            face_shape       -- torch.tensor, size (B, N, 3)
        """
        # to image_plane
        face_proj = face_shape @ self.persc_proj
        face_proj = face_proj[..., :2] / face_proj[..., 2:]
        return face_proj

    def transform(self, face_shape, rot, trans):
        """
        Return:
            face_shape       -- torch.tensor, size (B, N, 3) pts @ rot + trans

        Parameters:
            face_shape       -- torch.tensor, size (B, N, 3)
            rot              -- torch.tensor, size (B, 3, 3)
            trans            -- torch.tensor, size (B, 3)
        """
        return face_shape @ rot + trans.unsqueeze(1)

    def split_alpha(self, alpha):
        """
        Return:
            alpha_dict     -- a dict of torch.tensors

        Parameters:
            alpha          -- torch.tensor, size (B, 256)
        """
        alpha_id = alpha[:, :80]
        alpha_exp = alpha[:, 80: 144]
        alpha_alb = alpha[:, 144: 224]
        alpha_a = alpha[:, 224: 227]
        alpha_sh = alpha[:, 227: 254]
        alpha_t = alpha[:, 254:]
        return {
            'id': alpha_id,
            'exp': alpha_exp,
            'alb': alpha_alb,
            'angle': alpha_a,
            'sh': alpha_sh,
            'trans': alpha_t
        }

    def get_landmarks_68(self, v2d):
        """
        Return:
            landmarks_68_3d         -- torch.tensor, size (B, 68, 2)

        Parameters:
            v2d                     -- torch.tensor, size (B, N, 2)
        """
        return v2d[:, self.ldm68]

    def get_landmarks_106(self, v2d):
        """
        Return:
            landmarks_106_3d         -- torch.tensor, size (B, 106, 2)

        Parameters:
            v2d                      -- torch.tensor, size (B, N, 2)
        """
        return v2d[:, self.ldm106]

    def get_landmarks_134(self, v2d):
        """
        Return:
            landmarks_134            -- torch.tensor, size (B, 134, 2)

        Parameters:
            v2d                      -- torch.tensor, size (B, N, 2)
        """
        return v2d[:, self.ldm134]

    def get_landmarks_106_2d(self, v2d, face_shape, alpha_dict):
        """
        Return:
            landmarks_106_2d         -- torch.tensor, size (B, 106, 2)

        Parameters:
            v2d                     -- torch.tensor, size (B, N, 2)
            face_shape              -- torch.tensor, size (B, N, 3), face vertice without rotation or translation
            alpha_dict              -- a dict of torch.tensors
        """

        temp_angle = alpha_dict['angle'].clone()
        temp_angle[:,2] = 0
        rotation_without_roll = self.compute_rotation(temp_angle)
        v2d_without_roll = self.to_image(self.to_camera(self.transform(face_shape, rotation_without_roll, alpha_dict['trans'])))

        visible_parallel = self.v_parallel.clone()
        # visible_parallel[visible_idx == 0] = -1

        ldm106_dynamic=self.ldm106.clone()
        for i in range(16):
            temp=v2d_without_roll.clone()[:,:,0]
            temp[:,visible_parallel!=i] = 1e5
            ldm106_dynamic[i]=torch.argmin(temp)

        for i in range(17,33):
            temp=v2d_without_roll.clone()[:,:,0]
            temp[:,visible_parallel!=i] = -1e5
            ldm106_dynamic[i]=torch.argmax(temp)

        return v2d[:, ldm106_dynamic]

    def add_directionlight(self, normals, lights):
        '''
        see https://github.com/yfeng95/DECA/blob/master/decalib/utils/renderer.py
        '''
        light_direction = lights[:,:,:3]; light_intensities = lights[:,:,3:]
        directions_to_lights = F.normalize(light_direction[:,:,None,:].expand(-1,-1,normals.shape[1],-1), dim=3)
        normals_dot_lights = torch.clamp((normals[:,None,:,:]*directions_to_lights).sum(dim=3), 0., 1.)
        shading = normals_dot_lights[:,:,:,None]*light_intensities[:,:,None,:]
        return shading.mean(1)

    def compute_gray_shading_with_directionlight(self, face_texture, normals):
        '''
        see https://github.com/yfeng95/DECA/blob/master/decalib/utils/renderer.py
        '''
        batch_size = normals.shape[0]
        light_positions = torch.tensor(
                [
                [-1,1,1],
                [1,1,1],
                [-1,-1,1],
                [1,-1,1],
                [0,0,1]
                ]
        )[None,:,:].expand(batch_size, -1, -1).float()
        light_intensities = torch.ones_like(light_positions).float()*1.7
        lights = torch.cat((light_positions, light_intensities), 2).to(face_texture.device)

        shading = self.add_directionlight(normals, lights)
        texture =  face_texture*shading
        return texture

    def segmentation(self, v3d):

        seg = torch.zeros(224,224,8).to(v3d.device)
        for i in range(8):
            mask, _, _, _ = self.renderer(v3d.clone(), self.annotation_tri[i])
            seg[:,:,i] = mask.squeeze()
        return seg

    def segmentation_visible(self, v3d, visible_idx):

        seg = torch.zeros(224,224,8).to(v3d.device)
        for i in range(8):
            temp = torch.zeros_like(v3d)
            temp[:,self.annotation[i],:] = 1
            temp[:,visible_idx == 0,:] = 0
            _, _, temp_image, _ = self.renderer(v3d.clone(), self.tri, temp.clone())
            temp_image = temp_image.mean(axis=1)
            mask = torch.where(temp_image >= 0.5, torch.tensor(1.0).to(v3d.device), torch.tensor(0.0).to(v3d.device))
            seg[:,:,i] = mask.squeeze()
        return seg

    def get_texture_map(self, face_vertex, input_img):
        batch_size = input_img.shape[0]
        h, w = input_img.shape[2:]
        face_vertex_uv = self.render_deca.world2uv(face_vertex)  # (B, 3, 256, 256)
        face_vertex_uv = face_vertex_uv.reshape(batch_size, 3, -1).permute(0, 2, 1)  # (B, N, 3), N: 256*256
        face_vertex_uv_proj = self.to_image(face_vertex_uv)  # (B, N, 2) , project to image (size=224)
        face_vertex_uv_proj[..., 0] *= w / 224
        face_vertex_uv_proj[..., 1] *= h / 224
        face_vertex_uv_proj[torch.isnan(face_vertex_uv_proj)] = 0

        face_vertex_uv_proj[..., -1] = h - 1 - face_vertex_uv_proj[..., -1]

        input_img = input_img.permute(0, 2, 3, 1)  # (B, h, w, 3)

        face_vertex_uv_proj_int = torch.floor(face_vertex_uv_proj)
        face_vertex_uv_proj_float = face_vertex_uv_proj - face_vertex_uv_proj_int  # (B, N, 2)
        face_vertex_uv_proj_float = face_vertex_uv_proj_float.reshape(-1, 2)  # (B * N, 2)
        face_vertex_uv_proj_int = face_vertex_uv_proj_int.long()  # (B, N, 2)

        batch_indices = torch.arange(0, batch_size)[:, None, None].repeat(1, face_vertex_uv_proj_int.shape[1], 1).long().to(face_vertex_uv_proj_int.device)  # (B, N, 1)
        indices = torch.cat([face_vertex_uv_proj_int, batch_indices], dim=2)
        indices = indices.reshape(-1, 3)  # (B * N, 3)

        face_vertex_uv_proj_lt = input_img[indices[:, 2], indices[:, 1].clamp(0, h-1), indices[:, 0].clamp(0, w-1)]  # (B * N, 3)
        face_vertex_uv_proj_lb = input_img[indices[:, 2], (indices[:, 1]+1).clamp(0, h-1), indices[:, 0].clamp(0, w-1)]
        face_vertex_uv_proj_rt = input_img[indices[:, 2], indices[:, 1].clamp(0, h-1), (indices[:, 0]+1).clamp(0, w-1)]
        face_vertex_uv_proj_rb = input_img[indices[:, 2], (indices[:, 1]+1).clamp(0, h-1), (indices[:, 0]+1).clamp(0, w-1)]

        texture_map = face_vertex_uv_proj_lt * (1 - face_vertex_uv_proj_float[:, :1]) * face_vertex_uv_proj_float[:, 1:] + \
                       face_vertex_uv_proj_lb * (1 - face_vertex_uv_proj_float[:, :1]) * (1 - face_vertex_uv_proj_float[:, 1:]) + \
                       face_vertex_uv_proj_rt * face_vertex_uv_proj_float[:, :1] * face_vertex_uv_proj_float[:, 1:] + \
                       face_vertex_uv_proj_rb * face_vertex_uv_proj_float[:, :1] * (1 - face_vertex_uv_proj_float[:, 1:])  # (B * N, 3)

        texture_map = texture_map.reshape(batch_size, self.render_deca.uv_size, self.render_deca.uv_size, -1)  # (B, 256, 256, 3)

        return texture_map

    def displacement2normal(self, uv_z, coarse_verts, coarse_normals):
        ''' Convert displacement map into detail normal map
        '''
        batch_size = uv_z.shape[0]
        uv_coarse_vertices = self.render_deca.world2uv(coarse_verts)
        uv_coarse_normals = self.render_deca.world2uv(coarse_normals)

        uv_detail_vertices = uv_coarse_vertices
        dense_vertices = uv_detail_vertices.permute(0, 2, 3, 1).reshape([batch_size, -1, 3])
        uv_detail_normals = vertex_normals(dense_vertices, self.render_deca.dense_faces.expand(batch_size, -1, -1))
        uv_detail_normals = uv_detail_normals.reshape(
            [batch_size, uv_coarse_vertices.shape[2], uv_coarse_vertices.shape[3], 3]).permute(0, 3, 1, 2)
        uv_detail_normals[:, :2, ...] = -uv_detail_normals[:, :2, ...]
        offset = uv_coarse_normals - uv_detail_normals

        uv_detail_vertices = uv_coarse_vertices + uv_z * uv_coarse_normals
        dense_vertices = uv_detail_vertices.permute(0, 2, 3, 1).reshape([batch_size, -1, 3])
        uv_detail_normals = vertex_normals(dense_vertices, self.render_deca.dense_faces.expand(batch_size, -1, -1))

        uv_detail_normals = uv_detail_normals.reshape(
            [batch_size, uv_coarse_vertices.shape[2], uv_coarse_vertices.shape[3], 3]).permute(0, 3, 1, 2)
        uv_detail_normals[:, :2, ...] = -uv_detail_normals[:, :2, ...]
        uv_detail_normals = uv_detail_normals + offset

        return uv_detail_normals

    def compute_color_with_displacement_directionlight(self, face_texture_uv, verts, normals, displacement_uv, intensities = 2):
        batch_size = verts.shape[0]
        light_positions = torch.tensor(
                [
                [-1,1,1],
                [1,1,1],
                [-1,-1,1],
                [1,-1,1],
                [0,0,1]
                ]
        )[None,:,:].expand(batch_size, -1, -1).float()
        light_intensities = torch.ones_like(light_positions).float()*intensities
        lights = torch.cat((light_positions, light_intensities), 2).to(face_texture_uv.device)

        uv_detail_normals = self.displacement2normal(displacement_uv, verts, normals)

        shading = self.add_directionlight(uv_detail_normals.permute(0,2,3,1).reshape([batch_size, -1, 3]), lights)
        shading_images = shading.reshape([batch_size, face_texture_uv.shape[2], face_texture_uv.shape[3], 3]).permute(0,3,1,2)
        uv_texture =  face_texture_uv*shading_images
        return uv_texture

    def get_dense_mesh(self, uv_z, coarse_verts, coarse_normals, face_albedo_map = None):
        ''' Convert displacement map into detail normal map
        '''
        batch_size = uv_z.shape[0]
        uv_coarse_vertices = self.render_deca.world2uv(coarse_verts)
        uv_coarse_normals = self.render_deca.world2uv(coarse_normals)
        if face_albedo_map != None:
            dense_texture = face_albedo_map.permute(0, 2, 3, 1).reshape([batch_size, -1, 3])

        uv_detail_vertices = uv_coarse_vertices + uv_z * uv_coarse_normals
        dense_vertices = uv_detail_vertices.permute(0, 2, 3, 1).reshape([batch_size, -1, 3])
        dense_faces = self.render_deca.dense_faces.expand(batch_size, -1, -1)
        dense_mesh = {
            'vertices': dense_vertices,
            'faces': dense_faces,
            'texture': dense_texture,
        }
        
        return dense_mesh


    def get_albedo_from_controlnet(self, face_albedo_map, visible_idx_renderer, face_norm_roted, v2d):

        visible_idx = torch.zeros(35709).type(torch.int64).to(self.v3d.device)
        visible_idx[visible_idx_renderer.type(torch.int64)] = 1
        visible_idx[(face_norm_roted[..., 2] < 0)[0]] = 0

        img_colors = bilinear_interpolate(face_albedo_map.permute(0, 2, 3, 1).detach()[0], v2d[0, :, 0].detach(), 223 - v2d[0, :, 1].detach())
        _, _, uv_color_img, _ = self.renderer.get_uv(self.uv_coords_torch.unsqueeze(0).clone(), self.tri, img_colors.unsqueeze(0).clone())
        _, _, uv_weight, _ = self.renderer.get_uv(self.uv_coords_torch.unsqueeze(0).clone(), self.tri, (1 - torch.stack((visible_idx,)*3, axis=-1).unsqueeze(0).type(torch.float32).to(self.tri.device)).clone())
        
        uv_color_img = uv_color_img.detach().cpu().permute(0, 2, 3, 1).numpy()[0]
        
        # use fliped img-texture for texture blending
        res_colors = cv2.seamlessClone((uv_color_img*255).astype(np.uint8), cv2.flip((uv_color_img*255).astype(np.uint8), 1), ((1 - uv_weight.detach().cpu().permute(0, 2, 3, 1).numpy()[0])*255)[:,:,0].astype(np.uint8), (512, 512), cv2.NORMAL_CLONE) / 255.
        res_colors = cv2.resize(res_colors, (256, 256))
        return torch.from_numpy(res_colors).permute(2,0,1).unsqueeze(0).to(self.tri.device)

    def forward_texture(self, result_dict, face_albedo_map_list, face_albedo_map_in_uv_list, light_intensities = 1.5):
        result_dict['render_texture'] = []
        result_dict['dense_mesh'] = []
        result_dict['face_albedo_map_list'] = []
        for face_albedo_map_ind in range(len(face_albedo_map_list)):

            face_albedo_map = face_albedo_map_list[face_albedo_map_ind]
            face_albedo_map_in_uv = face_albedo_map_in_uv_list[face_albedo_map_ind]

            if face_albedo_map_in_uv == False:
                face_albedo_map = self.get_albedo_from_controlnet(face_albedo_map, self.visible_idx_renderer, self.face_norm_roted, self.v2d).float()
            
            face_color_map_high = self.compute_color_with_displacement_directionlight(face_albedo_map, self.face_shape_transformed, self.face_norm_roted, self.displacement_map, intensities = light_intensities) # , self.pred_coeffs_dict['gamma'])

            self.pred_mask, _, self.pred_face = self.renderer.render_uv_texture(self.v3d,
                                                                            self.tri,
                                                                            self.uv_coords.clone(),
                                                                            face_color_map_high)
            result_dict['render_texture'].append(self.pred_face.detach().cpu().permute(0, 2, 3, 1).numpy())
            dense_mesh = self.get_dense_mesh(self.displacement_map, self.face_shape_transformed, self.face_norm_roted, face_albedo_map)
            result_dict['dense_mesh'].append(dense_mesh)

            result_dict['face_albedo_map_list'].append(face_albedo_map)
        return result_dict

    def forward(self, beta_magnitude = 0.12):
        assert self.net_recon.training == False
        alpha = self.net_recon(self.input_img)

        alpha_dict = self.split_alpha(alpha)
        face_shape = self.compute_shape(alpha_dict['id'], alpha_dict['exp'])
        rotation = self.compute_rotation(alpha_dict['angle'])
        self.face_shape_transformed = self.transform(face_shape, rotation, alpha_dict['trans'])

        # face vertice in 3d
        self.v3d = self.to_camera(self.face_shape_transformed.clone())
        v3d_noTrans = self.to_camera(face_shape.clone())

        # face vertice in 2d image plane
        self.v2d = self.to_image(self.v3d)

        face_norm = self.compute_norm(face_shape)
        self.face_norm_roted = face_norm @ rotation

        _, _, _, self.visible_idx_renderer = self.renderer(self.v3d.clone(), self.tri, visible_vertice = True)

        position_map = self.render_deca.world2uv(face_shape)
        
        texture_map_input_high = self.get_texture_map(self.v3d, self.input_img)
        texture_map_input_high = texture_map_input_high.permute(0, 3, 1, 2).detach()  # (1, 3, 256, 256)
        texture_map_input_high = (texture_map_input_high - 0.5) * 2

        position_map_input_high = position_map
        input_high = torch.cat([position_map_input_high, texture_map_input_high], dim=1)

        self.position_map_input_high = position_map_input_high.permute(0, 2, 3, 1).detach().cpu().numpy()
        self.texture_map_input_high = texture_map_input_high.permute(0, 2, 3, 1).detach().cpu().numpy()
    
        with torch.no_grad():
            assert self.high_net.training == False
            self.displacement_map = self.high_net(input_high) * beta_magnitude * self.beta
        self.displacement_map_numpy = self.displacement_map.permute(0, 2, 3, 1).repeat(1, 1, 1, 3).detach().cpu().numpy()

        result_dict = {
            'v3d': self.v3d.detach().cpu().numpy(),
            'v2d': self.v2d.detach().cpu().numpy(),

            'tri': self.tri.detach().cpu().numpy(),
            'uv_coords': self.uv_coords_one_numpy, # self.uv_coords.detach().cpu().numpy(),
            'displacement_map_numpy': self.displacement_map_numpy,
            'position_map_input_high': self.position_map_input_high,
            'texture_map_input_high': self.texture_map_input_high,

        }

        # compute visible vertice according to normal and renderer
        if self.args.seg_visible or self.args.extractTex:
            visible_idx = torch.zeros(35709).type(torch.int64).to(self.v3d.device)
            visible_idx[self.visible_idx_renderer.type(torch.int64)] = 1
            visible_idx[(self.face_norm_roted[..., 2] < 0)[0]] = 0
            # result_dict['visible_idx'] = visible_idx

        # landmarks 68 3d
        if self.args.ldm68:
            v2d_68 = self.get_landmarks_68(self.v2d)
            result_dict['ldm68'] = v2d_68.detach().cpu().numpy()

        # landmarks 106 3d
        if self.args.ldm106:
            v2d_106 = self.get_landmarks_106(self.v2d)
            result_dict['ldm106'] = v2d_106.detach().cpu().numpy()

        # landmarks 106 2d
        if self.args.ldm106_2d:
            # v2d_106_2d = self.get_landmarks_106_2d(v2d, face_shape, alpha_dict, visible_idx)
            v2d_106_2d = self.get_landmarks_106_2d(self.v2d, face_shape, alpha_dict)
            result_dict['ldm106_2d'] = v2d_106_2d.detach().cpu().numpy()

        # landmarks 134
        if self.args.ldm134:
            v2d_134 = self.get_landmarks_134(self.v2d)
            result_dict['ldm134'] = v2d_134.detach().cpu().numpy()

        # segmentation in 2d without visible mask
        if self.args.seg:
            seg = self.segmentation(self.v3d)
            result_dict['seg'] = seg.detach().cpu().numpy()

        # segmentation in 2d with visible mask
        if self.args.seg_visible:
            seg_visible = self.segmentation_visible(self.v3d, visible_idx)
            result_dict['seg_visible'] = seg_visible.detach().cpu().numpy()

        # alpha_dict['alb'] and alpha_dict['sh'] are not used during inference.
        gray_tex = self.gray_tex

        tex_high_gray = self.compute_color_with_displacement_directionlight(gray_tex.detach(), self.face_shape_transformed, self.face_norm_roted, self.displacement_map)
        self.pred_mask, _, self.pred_face_gray = self.renderer.render_uv_texture(self.v3d,
                                                                        self.tri,
                                                                        self.uv_coords.clone(),
                                                                        tex_high_gray)

        tex_base_gray = self.compute_color_with_displacement_directionlight(gray_tex.detach(), self.face_shape_transformed, self.face_norm_roted, torch.zeros_like(self.displacement_map))
        _, _, self.pred_face_gray_base = self.renderer.render_uv_texture(self.v3d,
                                                                        self.tri,
                                                                        self.uv_coords.clone(),
                                                                        tex_base_gray)

        result_dict['render_mask'] = self.pred_mask.detach().cpu().permute(0, 2, 3, 1).numpy()
        result_dict['render_shape_detail'] = self.pred_face_gray.detach().cpu().permute(0, 2, 3, 1).numpy()
        result_dict['render_shape_base'] = self.pred_face_gray_base.detach().cpu().permute(0, 2, 3, 1).numpy()

        return result_dict

