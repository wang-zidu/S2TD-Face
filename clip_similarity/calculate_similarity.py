import os
import torch
from tqdm import tqdm
import clip
from glob import glob
from PIL import Image

def save_feature(image_path):
    image_list = glob(os.path.join(image_path,'*.png')) + glob(os.path.join(image_path,'*.jpg'))
    feature_path = 'clip_similarity/clip_texture_feature.pth'
    feature = {}
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load('ViT-B/32', device)
    for image_name in tqdm(image_list):
        image = Image.open(image_name)
        with torch.no_grad():
            image_input = preprocess(image).unsqueeze(0).to(device)
            image_feature = model.encode_image(image_input)
            image_feature /= image_feature.norm(dim=-1, keepdim=True)
            feature[image_name] = image_feature

    torch.save(feature, feature_path)
    return

def load_features():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    feature_path = 'clip_similarity/clip_texture_feature.pth'
    feature = torch.load(feature_path)
    image_paths = list(feature.keys())
    image_features = []
    for key in image_paths:
        image_features.append(torch.tensor(feature[key])) 
    return image_paths, torch.cat(image_features).to(device)

def match_image_list(description, top_k = 3):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _ = clip.load('ViT-B/32', device)
    with torch.no_grad():
        text = clip.tokenize(description).to(device)
        image_paths, image_features = load_features()
        text_feature = model.encode_text(text)
        text_feature /= text_feature.norm(dim=-1, keepdim=True)
        similarity = (image_features @ text_feature.T).T#.softmax(dim=-1)

        values, indices = similarity[0].topk(top_k)

    return [image_paths[i] for i in indices.cpu().numpy().tolist()]


if __name__=='__main__':
    path = 'texture_library/'
    save_feature(path)
    