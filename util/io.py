import argparse
import cv2
import os
import sys
import torch
import numpy as np
from PIL import Image

def plot_kpts(image, kpts, color = 'g'):

    if color == 'r':
        c = (255, 0, 0)
    elif color == 'g':
        c = (0, 255, 0)
    elif color == 'b':
        c = (255, 0, 0)
    image = image.copy()
    kpts = kpts.copy()
    radius = max(int(min(image.shape[0], image.shape[1])/200), 1)
    for i in range(kpts.shape[0]):
        st = kpts[i, :]
        image = cv2.circle(image,(int(st[0]), int(st[1])), radius, c, radius*2)

    return image

def show_seg_visble(new_seg_visible_one, img):

    img = img.copy()
    new_seg_visible_one = new_seg_visible_one.copy()
    mask2=np.stack((new_seg_visible_one[:,:,0],)*3, axis=-1).astype(np.uint8)

    back2=np.full_like(mask2,0)

    colormap = label_colormap(9)
    alphas = np.linspace(0.75, 0.25, num=9)

    dst2=np.full_like(back2,0)
    for i, mask in enumerate(mask2[:,:,0][None,:,:]):
        alpha = alphas[i]
        index = mask > 0
        res = colormap[mask]
        dst2[index] = (1 - alpha) * back2[index].astype(float) + alpha * res[index].astype(float)
    dst2 = np.clip(dst2.round(), 0, 255).astype(np.uint8)

    return ((dst2[:,:,::-1]*0.5+img*0.5)).astype(np.uint8)

def label_colormap(n_label=9):
    """Label colormap.
    Parameters
    ----------
    n_labels: int
        Number of labels (default: 9).
    value: float or int
        Value scale or value of label color in HSV space.
    Returns
    -------
    cmap: numpy.ndarray, (N, 3), numpy.uint8
        Label id to colormap.
    """
    if n_label == 9:  # helen, ibugmask
        cmap = np.array(
            [
                (0, 0, 0),
                (0, 205, 0),
                (0, 138, 0),
                (139, 76, 57),
                (139, 54, 38),
                (154, 50, 205),
                (72, 118, 255),
                (22, 22, 139),
                (255, 255, 0),
            ],
            dtype=np.uint8,
        )
    else:

        def bitget(byteval, idx):
            return (byteval & (1 << idx)) != 0

        cmap = np.zeros((n_label, 3), dtype=np.uint8)
        for i in range(n_label):
            id = i
            r, g, b = 0, 0, 0
            for j in range(8):
                r = np.bitwise_or(r, (bitget(id, 0) << 7 - j))
                g = np.bitwise_or(g, (bitget(id, 1) << 7 - j))
                b = np.bitwise_or(b, (bitget(id, 2) << 7 - j))
                id = id >> 3
            cmap[i, 0] = r
            cmap[i, 1] = g
            cmap[i, 2] = b

    return cmap

def back_resize_crop_img(img, trans_params, ori_img, resample_method = Image.BICUBIC):
    
    w0, h0, s, t, target_size = trans_params[0], trans_params[1], trans_params[2], [trans_params[3],trans_params[4]], 224
    
    img=Image.fromarray(img)
    ori_img=Image.fromarray(ori_img)
    w = (w0*s).astype(np.int32)
    h = (h0*s).astype(np.int32)
    left = (w/2 - target_size/2 + float((t[0] - w0/2)*s)).astype(np.int32)
    right = left + target_size
    up = (h/2 - target_size/2 + float((h0/2 - t[1])*s)).astype(np.int32)
    below = up + target_size

    old_img = ori_img
    old_img = old_img.resize((w, h), resample=resample_method)

    old_img.paste(img, (left, up, right, below))
    old_img = old_img.resize((int(w0), int(h0)), resample=resample_method)

    old_img = np.array(old_img)
    return old_img

def back_resize_ldms(ldms, trans_params):
    
    w0, h0, s, t, target_size = trans_params[0], trans_params[1], trans_params[2], [trans_params[3],trans_params[4]], 224

    w = (w0*s).astype(np.int32)
    h = (h0*s).astype(np.int32)
    left = (w/2 - target_size/2 + float((t[0] - w0/2)*s)).astype(np.int32)
    right = left + target_size
    up = (h/2 - target_size/2 + float((h0/2 - t[1])*s)).astype(np.int32)
    below = up + target_size

    ldms[:, 0] = ldms[:, 0] + left
    ldms[:, 1] = ldms[:, 1] + up

    ldms[:, 0] = ldms[:, 0] / w * w0
    ldms[:, 1] = ldms[:, 1] / h * h0

    return ldms

def write_obj_with_colors(obj_name, vertices, triangles, colors):
    ''' Save 3D face model with texture represented by colors.
    Args:
        obj_name: str
        vertices: shape = (nver, 3)
        triangles: shape = (ntri, 3)
        colors: shape = (nver, 3)
    '''
    triangles = triangles.copy()
    triangles += 1 # obj start with 1
    
    if obj_name.split('.')[-1] != 'obj':
        obj_name = obj_name + '.obj'
        
    # write obj
    with open(obj_name, 'w') as f:
        s = '# Results of S2TD-Face, https://github.com/wang-zidu/S2TD-Face\n'
        f.write(s)

        for i in range(vertices.shape[0]):
            s = 'v {} {} {} {} {} {}\n'.format(vertices[i, 0], vertices[i, 1], vertices[i, 2], colors[i, 0], colors[i, 1], colors[i, 2])
            f.write(s)

        [k, ntri] = triangles.shape
        for i in range(triangles.shape[0]):
            s = 'f {} {} {}\n'.format(triangles[i, 0], triangles[i, 1], triangles[i, 2])
            f.write(s)

def write_obj_with_uv_no_texture(obj_name, vertices, triangles, uv_coords):
    ''' Save 3D face model with texture represented by colors.
    Args:
        obj_name: str
        vertices: shape = (nver, 3)
        triangles: shape = (ntri, 3)
        colors: shape = (nver, 3)
    '''
    triangles = triangles.copy()
    triangles += 1 # meshlab start with 1
    
    if obj_name.split('.')[-1] != 'obj':
        obj_name = obj_name + '.obj'
        
    # write obj
    with open(obj_name, 'w') as f:
        s = '# Results of S2TD-Face, https://github.com/wang-zidu/S2TD-Face\n'
        f.write(s)

        # write vertices & colors
        for i in range(vertices.shape[0]):
            s = 'v {} {} {} \n'.format(vertices[i, 0], vertices[i, 1], vertices[i, 2])
            # s = 'v {} {} {} {} {} {}\n'.format(vertices[i, 0], vertices[i, 1], vertices[i, 2], colors[i, 0], colors[i, 1], colors[i, 2])
            f.write(s)

            s = 'vt {} {}\n'.format(uv_coords[i, 0], uv_coords[i, 1])
            f.write(s)

        # write f: ver ind/ uv ind
        [k, ntri] = triangles.shape
        for i in range(triangles.shape[0]):
            # s = 'f {} {} {}\n'.format(triangles[i, 0], triangles[i, 1], triangles[i, 2])
            # s = 'f {}/{} {}/{} {}/{}\n'.format(triangles[i, 2], triangles[i, 2], triangles[i, 1], triangles[i, 1], triangles[i, 0], triangles[i, 0])
            s = 'f {}/{} {}/{} {}/{}\n'.format(triangles[i, 0], triangles[i, 0], triangles[i, 1], triangles[i, 1], triangles[i, 2], triangles[i, 2])

            f.write(s)

def crop_mesh(mesh, keep_vert_inds):
    vertices = mesh['vertices'].copy()
    faces = mesh['faces']
    new_vertices = vertices[keep_vert_inds].copy()
    # faces -= 1

    inds_mapping = dict()
    for i in range(len(keep_vert_inds)):
        inds_mapping[keep_vert_inds[i]] = i

    new_faces = []
    keep_face_inds = []
    for ind, face in enumerate(faces):
        if face[0] in inds_mapping and face[1] in inds_mapping and face[2] in inds_mapping:
            new_face = [inds_mapping[face[0]], inds_mapping[face[1]], inds_mapping[face[2]]]
            new_faces.append(new_face)
            keep_face_inds.append(ind)
    new_faces = np.array(new_faces)
    # new_faces += 1
    keep_face_inds = np.array(keep_face_inds)

    new_mesh = mesh.copy()
    new_mesh['vertices'] = new_vertices
    new_mesh['faces'] = new_faces
    if 'colors' in new_mesh:
        new_mesh['colors'] = new_mesh['colors'][keep_vert_inds]
    if 'faces_uv' in new_mesh:
        new_mesh['faces_uv'] = new_mesh['faces_uv'][keep_face_inds]
    if 'faces_normal' in new_mesh:
        new_mesh['faces_normal'] = new_mesh['faces_normal'][keep_face_inds]

    return new_mesh, keep_face_inds

def cv2_text(img_list, title_list):
    position1 = (8, 18)
    position2 = (8, 42)
    font = cv2.FONT_HERSHEY_DUPLEX
    font_scale = 0.5
    color = (0, 0, 0)
    thickness = 1
    res_list = []
    for i in range(len(img_list)):
        temp = cv2.putText(img_list[i], title_list[i].split('-')[0], position1, font, font_scale, color, thickness)
        if len(title_list[i].split('-'))>1:
            temp = cv2.putText(temp, title_list[i].split('-')[1], position2, font, font_scale, color, thickness)
        res_list.append(temp)
    return res_list

class visualize:
    def __init__(self, result_dict, args):

        self.items = ['render_shape_base','render_shape_detail','render_texture']
        self.result_dict = result_dict
        option_list = ['ldm68', 'ldm106', 'ldm106_2d', 'ldm134', 'seg', 'seg_visible']
        for i in option_list:
            if i in self.result_dict.keys():
                self.items.append(i)
        
        self.visualize_dict = []
        self.save_dict = {}
        self.args = args

    def visualize_and_output(self, trans_params, img, save_path, img_name, face_albedo_map_name_list, use_img_background = False):
        # assert batch_size = 1
        self.visualize_dict.append(img)

        render_shape = (self.result_dict['render_shape_base'][0]*255).astype(np.uint8)
        render_shape_detail = (self.result_dict['render_shape_detail'][0]*255).astype(np.uint8)
        render_mask  = (np.stack((self.result_dict['render_mask'][0][:,:,0],)*3, axis=-1)*255).astype(np.uint8)
        render_face = [(self.result_dict['render_texture'][k][0]*255).astype(np.uint8) for k in range(len(self.result_dict['render_texture']))]

        if trans_params is not None:
            render_shape = back_resize_crop_img(render_shape, trans_params, np.zeros_like(img), resample_method = Image.BICUBIC)
            render_shape_detail = back_resize_crop_img(render_shape_detail, trans_params, np.zeros_like(img), resample_method = Image.BICUBIC)
            render_face = [back_resize_crop_img(render_face[k], trans_params, np.zeros_like(img), resample_method = Image.BICUBIC) for k in  range(len(render_face))]
            render_mask = back_resize_crop_img(render_mask, trans_params, np.zeros_like(img), resample_method = Image.NEAREST)

        if use_img_background:
            render_shape = ((render_shape/255. * render_mask/255. + img[:,:,::-1]/255. * (1 - render_mask/255.))*255).astype(np.uint8)
            render_shape_detail = ((render_shape_detail/255. * render_mask/255. + img[:,:,::-1]/255. * (1 - render_mask/255.))*255).astype(np.uint8)
            render_face = [((render_face[k]/255. * render_mask/255. + img[:,:,::-1]/255. * (1 - render_mask/255.))*255).astype(np.uint8) for k in  range(len(render_face))]
        else:
            render_shape = ((render_shape/255. * render_mask/255. + np.ones_like(img) * (1 - render_mask/255.))*255).astype(np.uint8)
            render_shape_detail = ((render_shape_detail/255. * render_mask/255. + np.ones_like(img) * (1 - render_mask/255.))*255).astype(np.uint8)
            render_face = [((render_face[k]/255. * render_mask/255. + np.ones_like(img) * (1 - render_mask/255.))*255).astype(np.uint8) for k in  range(len(render_face))]

        render_face = cv2_text(render_face, face_albedo_map_name_list)

        for k in range(len(self.result_dict['face_albedo_map_list'])):
            cv2.imwrite(os.path.join(save_path, img_name + '_uv_' + face_albedo_map_name_list[k] + '.png'), (self.result_dict['face_albedo_map_list'][k][0].permute(1,2,0).cpu().numpy()*255).astype(np.uint8))

        self.visualize_dict.append(render_shape)
        self.visualize_dict.append(render_shape_detail)
        self.visualize_dict += render_face

        if 'ldm68' in self.items:
            ldm68=self.result_dict['ldm68'][0]
            ldm68[:, 1] = 224 -1 - ldm68[:, 1]
            if trans_params is not None:
                ldm68 = back_resize_ldms(ldm68, trans_params)
            img_ldm68 = plot_kpts(img, ldm68)
            self.visualize_dict.append(img_ldm68)
            self.save_dict['ldm68'] = ldm68

        if 'ldm106' in self.items:
            ldm106=self.result_dict['ldm106'][0]
            ldm106[:, 1] = 224 -1 - ldm106[:, 1]
            if trans_params is not None:
                ldm106 = back_resize_ldms(ldm106, trans_params)
            img_ldm106 = plot_kpts(img, ldm106)
            self.visualize_dict.append(img_ldm106)
            self.save_dict['ldm106'] = ldm106

        if 'ldm106_2d' in self.items:
            ldm106_2d=self.result_dict['ldm106_2d'][0]
            ldm106_2d[:, 1] = 224 -1 - ldm106_2d[:, 1]
            if trans_params is not None:
                ldm106_2d = back_resize_ldms(ldm106_2d, trans_params)
            img_ldm106_2d = plot_kpts(img, ldm106_2d)
            self.visualize_dict.append(img_ldm106_2d)
            self.save_dict['ldm106_2d'] = ldm106_2d

        if 'ldm134' in self.items:
            ldm134=self.result_dict['ldm134'][0]
            ldm134[:, 1] = 224 -1 - ldm134[:, 1]
            if trans_params is not None:
                ldm134 = back_resize_ldms(ldm134, trans_params)
            img_ldm134 = plot_kpts(img, ldm134)
            self.visualize_dict.append(img_ldm134)
            self.save_dict['ldm134'] = ldm134

        if 'seg_visible' in self.items:
            seg_visible = self.result_dict['seg_visible']
            new_seg_visible = np.zeros((img.shape[0],img.shape[1],8))
            for i in range(8):
                temp = np.stack((seg_visible[:,:,i],)*3, axis=-1)
                if trans_params is not None:
                    temp = back_resize_crop_img((temp).astype(np.uint8), trans_params, np.zeros_like(img), resample_method = Image.NEAREST)[:,:,::-1]
                new_seg_visible[:,:,i] = temp[:,:,0]*255

            new_seg_visible_one = np.zeros((img.shape[0],img.shape[1],1))
            for i in range(8):
                new_seg_visible_one[new_seg_visible[:,:,i]==255]=i+1
            self.visualize_dict.append(show_seg_visble(new_seg_visible_one, img))
            self.save_dict['seg_visible'] = new_seg_visible_one

        if 'seg' in self.items:
            seg = self.result_dict['seg']
            new_seg = np.zeros((img.shape[0],img.shape[1],8))
            for i in range(8):
                temp = np.stack((seg[:,:,i],)*3, axis=-1)
                if trans_params is not None:
                    temp = back_resize_crop_img((temp).astype(np.uint8), trans_params, np.zeros_like(img), resample_method = Image.NEAREST)[:,:,::-1]
                new_seg[:,:,i] = temp[:,:,0]*255

                temp2 = img.copy()
                temp2[new_seg[:,:,i]==255]=np.array([200,200,100])
                self.visualize_dict.append(temp2)
            
            self.save_dict['seg'] = new_seg

        np.save(os.path.join(save_path, img_name + '_displacement_map.npy'), self.result_dict['displacement_map_numpy'])
        temp = (self.result_dict['displacement_map_numpy'] - self.result_dict['displacement_map_numpy'].min()) / (self.result_dict['displacement_map_numpy'].max() - self.result_dict['displacement_map_numpy'].min())
        cv2.imwrite(os.path.join(save_path, img_name + '_displacement_map.png'), (temp[0]*255).astype(np.uint8))

        temp = (self.result_dict['position_map_input_high'] - self.result_dict['position_map_input_high'].min()) / (self.result_dict['position_map_input_high'].max() - self.result_dict['position_map_input_high'].min())
        cv2.imwrite(os.path.join(save_path, img_name + '_position_map.png'), (temp[0]*255).astype(np.uint8))

        temp = self.result_dict['texture_map_input_high'] / 2 + 0.5
        cv2.imwrite(os.path.join(save_path, img_name + '_uv_sketch.png'), (temp[0]*255).astype(np.uint8))
        

        v3d_new = self.result_dict['v3d'][0].copy()
        v3d_new[..., -1] = 10 - v3d_new[..., -1]
        write_obj_with_uv_no_texture(os.path.join(save_path, img_name + '_shape_base.obj'), v3d_new, self.result_dict['tri'], self.result_dict['uv_coords'])

        assert len(self.result_dict['dense_mesh']) == len(face_albedo_map_name_list)
        for k in range(len(self.result_dict['dense_mesh'])):
            v3d_dense_new = self.result_dict['dense_mesh'][k]['vertices'][0].detach().cpu().numpy().copy()
            dense_faces = self.result_dict['dense_mesh'][k]['faces'][0].detach().cpu().numpy().copy()
            dense_texture = self.result_dict['dense_mesh'][k]['texture'][0].detach().cpu().numpy().copy()[:,::-1]
            vertices_zero = v3d_dense_new == 0.0
            keep_inds = np.where((vertices_zero[:, 0] * vertices_zero[:, 1] * vertices_zero[:, 2]) == False)[0]
            dense_mesh = {'vertices': v3d_dense_new, 'faces': dense_faces, 'colors': dense_texture}
            dense_mesh, _ = crop_mesh(dense_mesh, keep_inds)  # remove the redundant vertices and faces
            write_obj_with_colors(os.path.join(save_path, img_name + '_shape_detail_' + face_albedo_map_name_list[k] + '.obj'), dense_mesh['vertices'], dense_mesh['faces'], dense_mesh['colors'])

        len_visualize_dict = len(self.visualize_dict)
        if len(self.visualize_dict) < 4:
            img_res = np.ones((img.shape[0], len(self.visualize_dict) * img.shape[1], 3), dtype=np.uint8) * 255
        else:
            img_res = np.ones((np.ceil(len_visualize_dict/4).astype(np.int32) * img.shape[0], 4 * img.shape[1], 3), dtype=np.uint8) * 255
        for i, image in enumerate(self.visualize_dict):
            row = i // 4
            col = i % 4
            x_start = col * img.shape[1]
            y_start = row * img.shape[0]
            x_end = x_start + img.shape[1]
            y_end = y_start + img.shape[0]
            img_res[y_start:y_end, x_start:x_end] = image

        cv2.imwrite(os.path.join(save_path, img_name + '_res.png'), img_res)
        # np.save(os.path.join(save_path, img_name + '.npy'), self.save_dict)






