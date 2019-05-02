import os
from fastai.vision import *
from fastai.callbacks import *

from glob import glob
from tqdm import tqdm

bs = 64
res = 64
os.environ["CUDA_VISIBLE_DEVICES"]="0"
path = "./char_imgs_cleaned_case_sensitive"

#cls = glob("./char_imgs_cleaned2/char_imgs_cleaned_case_sensitive/*/")

def main():
    src = ImageList.from_folder(path).split_by_rand_pct(0.1, seed=430)
    tfms = get_transforms(max_rotate=20,  do_flip=False, max_zoom=1.03, p_affine=0.5, xtra_tfms=[cutout(length=(res//2, res//2), p=0.5)])
    
    def get_data(data, size, bs, padding_mode='zeros'):
        return (data.label_from_folder()
               .transform(tfms, size=size, padding_mode=padding_mode)
               .databunch(bs=bs).normalize(imagenet_stats))
    
    data = get_data(src, res, bs, 'zeros')
    
    # Define class weights
    #dict_num_files = {}
    #for c in tqdm(cls):
    #    label = c[-2]
    #    num_files = len(glob(c+"*.jpg"))
    #    dict_num_files[label] = num_files
    #    
    #list_class_weights = []
    #assert src.train.y.classes == src.valid.y.classes
    #for c in tqdm(src.train.y.classes):
    #    list_class_weights.append(np.clip(400/dict_num_files[c], 0.01, 10))
    #class_weights = torch.FloatTensor(list_class_weights).cuda()
    
    # STAGE 1
    gc.collect()
    #learn = cnn_learner(data, models.resnet50, metrics=[error_rate], loss_func=nn.CrossEntropyLoss(weight=class_weights))
    learn = cnn_learner(data, models.resnet50, metrics=[error_rate], loss_func=LabelSmoothingCrossEntropy)
    
    learn.model_dir = "./fastai_model"
    learn.fit_one_cycle(3, slice(1e-3), pct_start=0.8)
    learn.save('stage-1', return_path=True)
    
    # STAGE 2        
    learn = cnn_learner(data, models.resnet50, metrics=[error_rate], loss_func=LabelSmoothingCrossEntropy())
    learn.model_dir = "C:/Users/shaoa/conda_env_tensorflow/AdvancedEAST-DenseNet121/fastai_model"
    learn.load('stage-2')
    
    learn.unfreeze()
    for _ in range(8):
        learn.fit_one_cycle(1, max_lr=slice(1e-6,3e-4))
        learn.save('stage-2', return_path=True)
    
    data = get_data(128, bs, 'zeros')
    learn.data = data    
    for _ in range(8):
        learn.fit_one_cycle(1, max_lr=slice(1e-6,1e-4))
        learn.save('stage-2', return_path=True)
    
    ## STAGE 3    
    data = get_data(224, bs, 'zeros')
    learn.data = data    
    for _ in range(8):
        learn.fit_one_cycle(1, max_lr=slice(1e-6,1e-4))
        learn.save('stage-3', return_path=True)

if __name__ == '__main__':
    main()