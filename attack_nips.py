from utils.vlm import VLM, VLMName
from utils.transfer_attack import AttackConfig, launch_attack
from utils.utils import get_device, get_memory_consumption, plot_images

from datasets import load_dataset
from torchvision import transforms as T
from pathlib import Path
import os
import torch
from PIL import Image

import csv
import random
from utils.dataset import Dataset, DatasetSource
from torch.utils.data import DataLoader
from enum import Enum, auto
from pip import *
from mirrorcheck import *
from nearside import *
#from utils.diffpure import ConfigYami, ConfigCus, RevGuidedDiffusion, default_args_cfg



import numpy as np
import pickle
import json
from tqdm import tqdm
import argparse
import copy


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_gradient_steps",default=1000,type=int,help="attack steps for attack stage")# (cyclic opt)")
    parser.add_argument("--cycles",default=30,type=int,help="full cycle attack steps for adaptive attack")
    parser.add_argument("--cyclic",action='store_true',help="do cyclic optimization")
    parser.add_argument("--naive",action='store_true',help="non-adaptive attack!")
    parser.add_argument("--start",default=0,type=int,help="strating idx to consider")
    parser.add_argument("--stop",default=1,type=int,help="stopping idx to consider")
    parser.add_argument("--npatches",default=0,type=int,help="adv. patches")
    parser.add_argument('--method', default="gait", type=str,help="defense")

    parser.add_argument("--vlm1",default=0,type=int,help="VICTIM VLM")

    parser.add_argument('--max_pert', type=float, default=8.0, help='adversarial perturbation budget (l_inf norm for norm-bounded, total_area% for patches)')
    parser.add_argument('--lr', type=float, default=1.0, help='attacker learning rate')
    parser.add_argument('--topk', type=float, default=0.05, help='maskable region')
    #parser.add_argument('--lr', default="ezz", type=str,help="PGD attacker learning rate")
    parser.add_argument('--user_query', default="What is in the below image?", type=str,help="path to data to avoid")
    parser.add_argument("--perz",default=0,type=int,help="Training Data FPR%")
    parser.add_argument("--save",action='store_true',help="save generated adv. images + attack statistics")

    #
    #PIP
    parser.add_argument("--occ",action='store_true',help="one-class SVM")

    #MirrorCheck
    parser.add_argument("--mc_gsteps",default=1,type=int,help="MirrorCheck sampling steps")

    parser.add_argument('--strparam', default="nutnet_results/", type=str,help="path to data to avoid")

    args = parser.parse_args()



    cls_file = "text_imagenet/imagenet1000_clsidx_to_labels.txt"
    relative_home = "./"#cv_transfer/src/"
    device  = get_device()#prefer_mps=True)
    #input(device)
    with open(cls_file, "r") as f:
        idx2label = eval(f.read())

    VLMS = [
        VLMName.SMOLVLM_1_256M,#OG (300M)
        VLMName.SMOLVLM_1_500M,#500M
        VLMName.SMOLVLM_1_2B,#2B
        VLMName.QWEN_2p5_VL_3B,#4B
        VLMName.LLAVA_ONEVISION_0p5B,#0.9B
        VLMName.INTERNVL_3_2B,
        #VLMName.INTERNVL_2p5_2B,
    ]
    #model_name_clf = CLASSIFIERS[args.clf1] if args.clf1!=-1 else CLASSIFIERS[1]#ClassifierName.VIT_B_P16_224
    model_name_vlm = VLMS[args.vlm1] if args.vlm1!=-1 else VLMS[0]
    user_query = args.user_query

    #dataset = load_dataset("vibhamasti/imagenet-subset-40", split='validation')
    method=args.method
    occ=args.occ
    caseno=args.vlm1
    refname = VLMS[caseno]
    refmod=VLM(refname, 'cuda')
    pipname = f"pipsvm_occ_{caseno}.joblib" if occ else f"pipsvm_{caseno}.joblib"
    nsname=f'nearside_adv_dir_vlm_{caseno}.pt'

    lr=args.lr

    p_defense, m_defense, n_defense, purifier = None, None, None, None
    if method=='mc':
        #from mirrorcheck import *
        config = MirrorCheckConfig(
            image_to_text_name = refname,
            user_query = user_query,
            n_diffusion_steps=args.mc_gsteps,
            device=device,
        )
        m_defense = MirrorCheck(config, victim_model=refmod)
        #p_defense=None

    elif method=='pip':
        #from pip import *
        config = PIPConfig(
            vlm_name=refname,
            occ=args.occ
        )
        p_defense = PIPDefense(config, victim_model=refmod)
        clf = load_or_train_svm(None, defense=p_defense, attack_config=None, file=pipname)
        p_defense.set_svm(clf)

    elif method=='ns':
        config = NearsideConfig(
            vlm_name=refname,#Name.SMOLVLM_1_256M,
        )
        n_defense = NearSideDefense(config, victim_model=refmod)
        adv_direction = load_or_compute_adv_direction(None, defense=n_defense, filename=nsname)
        n_defense.set_adv_direction(adv_direction)

    #elif method in ['diffpure', 'cider']:
    #    args_difp, cfg_difp = default_args_cfg()
    #    purifier = RevGuidedDiffusion(args_difp,cfg_difp)
    else:
        p_defense = m_defense = n_defense = purifier = None #pointless


    Nstep=1
    print_every=50
    perz =  args.perz
    #speed...
    mag = args.max_pert
    gradsteps = args.n_gradient_steps
    npatches=args.npatches
    patchst = '_patch' if npatches else ''

    attack_config = AttackConfig(
        model_name_vlm=refname,
        n_gradient_steps=gradsteps,
        lr=lr,#64/255,#255*(5e-4),
        lambda_vlm=1,
        cycles=1,
        user_query=user_query,
        target_vlm_answer="",
        target_clf_idx=0,
        max_perturbation=mag,
    )

    #gc.collect()
    #torch.cuda.empty_cache()
    print("MEMORY CONS. NOW")
    print(get_memory_consumption('cuda'))

    # important for choosing the same target labels each time
    random.seed(42)
    ds = Dataset(DatasetSource.NIPS_17)
    # create train and test splits
    ds_train, ds_test = ds.split_train_test(train_ratio=0.1)
    print(len(ds_train), len(ds_test))
    # create random subset and loop over it
    #ds_subset = ds.create_susbset(subset_len=10, is_random=False)
    dataloader = DataLoader(ds_train, batch_size=1, shuffle=False)

    vlmbol = []#can be populated to fool more than 1 victim model at once...

    for sampled_idx, (img, lbl, tgt) in enumerate(dataloader):
        if sampled_idx not in range(args.start, args.stop):
            continue
        #print(img.shape, lbl, tgt)

        attack_config.target_clf_idx = tgt.item()
        attack_config.target_vlm_answer = idx2label[tgt.item()].split(",")[-1].strip()
        #if row['label']==850:
        #    continue
        image_tensor = (img.squeeze(0)*255).to(torch.uint8)
        #image_tensor = T.PILToTensor()(sampled_image)
        image_tensor = T.Resize((224,224))(image_tensor).float().squeeze(0)

        if image_tensor.shape[0]==1:
            image_tensor = T.PILToTensor()(Image.merge("RGB", (sampled_image, sampled_image, sampled_image)))
            image_tensor = T.Resize((224,224))(image_tensor).float()


        bonanza, itertimes = launch_attack(
                                image=image_tensor.cuda(),
                                vlmi=refmod,
                                config=attack_config,
                                print_every=print_every,
                                device=device,
                                morevlm=vlmbol,
                                untargeted=False,#for untargeted only!
                                patch_attack=npatches>0,
                                npatches=npatches,
                                pipdef=p_defense,
                                mcdef=m_defense,
                                nsdef=n_defense,
                                dpdef=purifier,
                                nmc=args.mc_gsteps,
                                naib=method=='naib',
                                #gait=gait,
                                #refsims = torch.tensor(),
                                topk=args.topk,
                                cycles=1,
                                perz=perz,

                            )

        meinevs= np.array(itertimes)

        if args.save:
            torch.save(bonanza[0], (f'out_script/FPR{perz}_{method}_{args.topk}_adv_{patchst}_IDX_{sampled_idx}_VLM_{caseno}_BUDGET_{mag}_{npatches}.pt'))

            for ni, nam in enumerate(['obj', 'all', 'stealth_1', 'stealth_2', 'stealth_all',
                                  'esc_obj', 'esc_obj_big', 'esc_stealth', 'esc_stealth_big']):
                if bonanza[ni+1]!=None:
                    torch.save(bonanza[ni+1], (f'out_script/FPR{perz}_{method}_{args.topk}_adv_{patchst}_{nam}_IDX_{sampled_idx}_VLM_{caseno}_BUDGET_{mag}_{npatches}.pt'))


            with open(f'out_script/FPR{perz}_{method}_{args.topk}_adv_{patchst}_IDX_{sampled_idx}_VLM_{caseno}_BUDGET_{mag}_{npatches}_itertimes.npy', 'wb') as f:
                np.save(f, meinevs)

        #ASR_S-based success
        if meinevs[-6]!=-1:
            print(f"SUCCESSFUL ADAPTIVE ATTACK AFTER {np.round(meinevs[-6], 2)} SECONDS (image index {sampled_idx})")
        else:
            print(f"UNSUCCESFUL ATTACK... (image index {sampled_idx})")
