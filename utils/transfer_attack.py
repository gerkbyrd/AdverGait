"""
This file provides a class for jointly training an andversrial attack against an image classifier and a VLM
By default we use PGD
"""
from dataclasses import dataclass, asdict
import torch
from .vlm import VLM, VLMName
#from .diffpure import purify
from .utils import get_memory_consumption
import gc
import string
import numpy as np
import time

#DETECTION THRESHOLDS BASED ON STATISTICS OVER 5 RUNS ON FATIMA DATASET:
det_thresh_dict = {
    0.05:{

      VLMName.SMOLVLM_1_2B:{
        "gait":{0:[0.9140625, 3.328125 ],1:[1.1953125, 2.09375],2:[1.2654687500000001, 1.743203124999976],5:[1.3515625, 1.640625],10:[1.4609375, 1.4542968749999936],20:[1.55390625,1.2273437499999993]},
        "gait_bounds":{"ldif_lo":-0.1875, "mloss_lo":1.40625, "mloss_hi":4.8125, "norm_lo":0.0027971118688583374, "norm_hi":0.14907875657081604, "cosim_lo":-0.6841363},
        "pip":{0:1.2481015897403132 ,1:-0.0037356366210736926, 2:-0.3577426308358191, 5:-0.9122822744325021, 10:-0.9996689564389931, 20:-0.9998123992352288},
        #"pip_occ":{0: 133.28019770618985, 1:72.63934315026646 , 2:59.95085215437579 , 5:38.39318047275097, 10:25.475288747585406, 20:11.97756719032835},
        "mc":{0:0.43678259853839324, 1:0.5025724411010742, 2:0.5183742475509644, 5:0.5766531437635422, 10:0.6274253964424134, 20:0.684480357170105},
        "ns":{0: -40.25000007484991, 1: -43.75, 2: -45.25, 5: -47.0, 10: -48.25, 20: -49.75}
        },

    VLMName.QWEN_2p5_VL_3B:{
        "gait":{0:[0.2324219335703125, 1.16796875],1:[0.29296875, 0.9765625],2:[0.30859375, 0.90625],5:[0.396484375, 0.7578125],10:[0.7146484375, 0.6796875],20:[0.91796875, 0.5703125]},
        "gait_bounds":{"ldif_lo":-0.22265608893164063, "mloss_lo":0.265625, "mloss_hi":2.5312499998438227, "norm_lo":0.0020795960217502274, "norm_hi":4.856465023223778, "cosim_lo":-0.8402335},
        "pip":{0:0.2763563433199534 ,1:-0.5918927960800316 , 2:-0.8079542437125616 , 5:-0.9996385997628311, 10:-0.9997248250231398, 20:-0.9999670221567097},
        #"pip_occ":{0: 46.335923265250685, 1:37.861092019514984 , 2:31.906905492132736 , 5:23.488855241797463, 10:18.088539998632463, 20:11.380382072488855},
        "mc":{0:0.3255233793918484, 1:0.46928912937641143, 2:0.5107188951969147, 5:0.5603158265352249, 10:0.6149698078632355, 20:0.6799704074859619},
        "ns":{0: -152.00000009979988, 1: -161.99, 2: -168.0, 5: -172.0, 10: -174.0, 20: -176.0}
    },


    VLMName.INTERNVL_3_2B:{
        "gait":{0:[0.388671875, 2.242187480476566],1:[0.44921875, 1.3985937500000034],2:[0.4842578125, 1.2266406249999982],5:[0.5078125, 1.0703125],10:[0.5662109375, 0.8984375],20:[0.632421875, 0.71875]},
        "gait_bounds":{"ldif_lo":-0.12890622071484376, "mloss_lo":0.47460938476171877, "mloss_hi":3.046875, "norm_lo":0.002694146243546235, "norm_hi":0.388117220754051, "cosim_lo":-0.5787542216028684},
        "pip":{0:1.074510078721194,1:0.12206671603481829 , 2:-0.09526149823362705 , 5:-0.3169100668987158, 10:-0.6358529592265089, 20:-0.9891253985614739},
        #"pip_occ":{0: 46.335923265250685, 1:37.861092019514984 , 2:31.906905492132736 , 5:23.488855241797463, 10:18.088539998632463, 20:11.380382072488855},#MUST UPDATE
        "mc":{0:0.39889872319693187, 1:0.5485487711429596, 2:0.5629990184307099, 5:0.6149919182062149, 10:0.6703250288963318, 20:0.7207682967185974},
        "ns":{0: -56.500000498999384, 1: -81.495, 2: -89.98000000000002, 5: -99.97500000000002, 10: -105.94999999999999, 20: -111.5}
    }
    },
    0.025:{
    VLMName.QWEN_2p5_VL_3B:{
        "gait":{0:[0.28515625, 1.1640625],1:[0.3046484375, 0.7462109374999999],2:[0.31052734375, 0.6875781249999999],5:[0.391455078125, 0.5859375],10:[0.8388671875, 0.5235351562499999],20:[0.933203125, 0.3914062500000002]},
        "gait_bounds":{"ldif_lo":-0.0546875, "mloss_lo":0.25390625, "mloss_hi":2.140625, "norm_lo":0.0013182901311665773, "norm_hi":1.555273413658142, "cosim_lo":-0.8270035982131958},
        "pip":{0:0.2763563433199534 ,1:-0.5918927960800316 , 2:-0.8079542437125616 , 5:-0.9996385997628311, 10:-0.9997248250231398, 20:-0.9999670221567097},
        #"pip_occ":{0: 46.335923265250685, 1:37.861092019514984 , 2:31.906905492132736 , 5:23.488855241797463, 10:18.088539998632463, 20:11.380382072488855},
        "mc":{0:0.3255233793918484, 1:0.46928912937641143, 2:0.5107188951969147, 5:0.5603158265352249, 10:0.6149698078632355, 20:0.6799704074859619},
        "ns":{0: -152.00000009979988, 1: -161.99, 2: -168.0, 5: -172.0, 10: -174.0, 20: -176.0}
    }
    },
    0.1:{
    VLMName.QWEN_2p5_VL_3B:{
        "gait":{0:[0.298828125, 1.5078125],1:[0.309521484375, 1.179765625],2:[0.339609375, 1.1021093749999995],5:[0.5046386718750001, 0.9494140624999998],10:[0.8708984375, 0.8828125],20:[0.94140625, 0.7421875]},
        "gait_bounds":{"ldif_lo":-0.05078125, "mloss_lo":0.314453125, "mloss_hi":2.546875, "norm_lo":0.0009021197911351919, "norm_hi":3.4451279640197754, "cosim_lo":-0.8712633848190308},
        "pip":{0:0.2763563433199534 ,1:-0.5918927960800316 , 2:-0.8079542437125616 , 5:-0.9996385997628311, 10:-0.9997248250231398, 20:-0.9999670221567097},
        #"pip_occ":{0: 46.335923265250685, 1:37.861092019514984 , 2:31.906905492132736 , 5:23.488855241797463, 10:18.088539998632463, 20:11.380382072488855},
        "mc":{0:0.3255233793918484, 1:0.46928912937641143, 2:0.5107188951969147, 5:0.5603158265352249, 10:0.6149698078632355, 20:0.6799704074859619},
        "ns":{0: -152.00000009979988, 1: -161.99, 2: -168.0, 5: -172.0, 10: -174.0, 20: -176.0}
    }
    },
    0.2:{
    VLMName.QWEN_2p5_VL_3B:{
        "gait":{0:[0.298828125, 1.7890625],1:[0.309521484375, 1.4494921874999998],2:[0.339609375, 1.414140625],5:[0.5046386718750001, 1.296875],10:[0.8708984375, 1.1875],20:[0.94140625, 1.0242187500000002]},
        "gait_bounds":{"ldif_lo":-0.0546875, "mloss_lo":0.34765625, "mloss_hi":2.828125, "norm_lo":0.0009021197911351919, "norm_hi":3.4451279640197754, "cosim_lo":-0.8712633848190308},
        "pip":{0:0.2763563433199534 ,1:-0.5918927960800316 , 2:-0.8079542437125616 , 5:-0.9996385997628311, 10:-0.9997248250231398, 20:-0.9999670221567097},
        #"pip_occ":{0: 46.335923265250685, 1:37.861092019514984 , 2:31.906905492132736 , 5:23.488855241797463, 10:18.088539998632463, 20:11.380382072488855},
        "mc":{0:0.3255233793918484, 1:0.46928912937641143, 2:0.5107188951969147, 5:0.5603158265352249, 10:0.6149698078632355, 20:0.6799704074859619},
        "ns":{0: -152.00000009979988, 1: -161.99, 2: -168.0, 5: -172.0, 10: -174.0, 20: -176.0}
    }
    },
    0.5:{
    VLMName.QWEN_2p5_VL_3B:{
        "gait":{0:[0.298828125, 2.0390625],1:[0.309521484375, 1.6604296874999998],2:[0.339609375, 1.6015625],5:[0.5046386718750001, 1.4925781249999996],10:[0.8708984375, 1.3363281249999996],20:[0.94140625, 1.1574218750000003]},
        "gait_bounds":{"ldif_lo":-0.078125, "mloss_lo":0.3046875, "mloss_hi":3.078125, "norm_lo":0.0009021197911351919, "norm_hi":3.4451279640197754, "cosim_lo":-0.8712633848190308},
        "pip":{0:0.2763563433199534 ,1:-0.5918927960800316 , 2:-0.8079542437125616 , 5:-0.9996385997628311, 10:-0.9997248250231398, 20:-0.9999670221567097},
        #"pip_occ":{0: 46.335923265250685, 1:37.861092019514984 , 2:31.906905492132736 , 5:23.488855241797463, 10:18.088539998632463, 20:11.380382072488855},
        "mc":{0:0.3255233793918484, 1:0.46928912937641143, 2:0.5107188951969147, 5:0.5603158265352249, 10:0.6149698078632355, 20:0.6799704074859619},
        "ns":{0: -152.00000009979988, 1: -161.99, 2: -168.0, 5: -172.0, 10: -174.0, 20: -176.0}
    }
    }

}

@dataclass
class AttackConfig:
    model_name_vlm: str
    n_gradient_steps: int
    lr: float
    lambda_vlm: float
    user_query: str
    target_vlm_answer: str
    target_clf_idx: int
    max_perturbation: float
    cycles: int

    def to_dict(self,):
        return asdict(self)

def launch_attack(
    image: torch.tensor,
    #
    vlmi: VLM,
    config: AttackConfig,
    print_every: int,
    device: str,
    clfi=None,
    moreclf=None,
    morevlm=None,
    evol=False,
    untargeted=False,
    patch_attack=False,
    npatches=1,
    pipdef=None,
    mcdef=None,
    nsdef=None,
    dpdef=None,
    nmc=10,
    naib=False,
    refsims=None,
    cycles=1,
    ez=False,
    rmask=False,
    perz=0,
    exclude=True,
    topk=0.05,
):
    vlms=[vlmi]
    if morevlm is not None:
        vlms = vlms + morevlm
    """
    launches the attack
    """
    user_query = config.user_query
    target_clf_idx = config.target_clf_idx
    target_vlm_answer = config.target_vlm_answer
    lambda_vlm = config.lambda_vlm
    n_gradient_steps = config.n_gradient_steps
    lr = config.lr
    max_perturbation_pixels = config.max_perturbation if not patch_attack else 255

    if evol:
        list_evol=[]

    initial_image = image.clone().float() if device=="cuda" else image.clone()
    #roco = initial_image.clone().detach()
    if lambda_vlm:
        ftvs, tts=[],[]
        for vlm in vlms:
        #mock_images = [initial_image]
            full_text_vlm_prompt, target_tokens = vlm.get_training_prompt(user_query, target_vlm_answer)
            ftvs.append(full_text_vlm_prompt)
            tts.append(target_tokens)


    loss_vlm = torch.tensor([0]).to(device)

    # attack iterations
    #input(n_gradient_steps)
    if patch_attack:
        dx,dy=int(( (config.max_perturbation/npatches)**0.5)*image.shape[1]), int(( (config.max_perturbation/npatches)**0.5)*image.shape[2])
        mask= torch.zeros_like(image)
        #mask_s = torch.zeros_like(image)
        check=mask[0].clone()
        check[-dx:,:] = 1.0
        check[:, -dy:] = 1.0
        for n in range(npatches):

            fail=True
            print("looking for patch placement")
            while fail:
                opt = torch.where(check==0)

                #xmsk, ymsk = torch.randint(low=dx, high=image.shape[1]-dx, size=(1,1)).item(), torch.randint(low=dy, high=image.shape[2]-dy, size=(1,1)).item()

                #xmsk, ymsk = , torch.randint(low=0, high=image.shape[2]-dy, size=(1,1)).item()
                xmsk=torch.randint(low=0, high=opt[0].shape[0], size=(1,1)).item()
                xmsk, ymsk = opt[0][xmsk], opt[1][xmsk]

                #if xmsk < dx or xmsk > image.shape[1] - dx or ymsk < dy or ymsk > image.shape[2] - dy:# or torch.sum(check[xmsk:xmsk+dx, ymsk:ymsk+dy])>0.0:
                #    continue
                if True:#not n:
                    tcheck = check.clone()
                    check[max(0,xmsk-dx):xmsk+dx, max(0,ymsk-dy):ymsk+dy]=1.0
                    if n+1<npatches and check.sum() >= image.shape[2]*image.shape[1]:# - check.sum() < dx*dy:
                        print(f"not possible {check.sum().item()}")
                        check = tcheck#mask[0].clone()
                        del tcheck
                        continue
                    mask[:, xmsk:xmsk+dx, ymsk:ymsk+dy]=1.0
                #else:
                #    mask_s[:, xmsk:xmsk+dx, ymsk:ymsk+dy]=1.0
                fail=False
                print(f"done ({n}): {xmsk}, {ymsk} to {xmsk+dx}, {ymsk+dy}")
        image=image*(1-mask) + (mask*0.5*255).to(torch.uint8).to(torch.float)
        #image=image*(1-mask_s) + mask_s*0.5*255

    best_loss, best_det, best_dist, best_total, best_full_det = torch.inf, torch.inf, -torch.inf, torch.inf, torch.inf
    best, soon, fifty = None, None, None
    best_big, stealth_1, stealth_2, stealth_big = None, None, None, None
    best_esc, best_big_esc, stealth_esc, stealth_big_esc = None, None, None, None
    best_exit, best_exit_det, best_exit_big, best_exit_det_big = torch.inf, torch.inf, torch.inf, torch.inf
    times={"success":-1, "lowest":-1, "lowest_glob":-1, "lowest_und":-1, "lowest_glob_und":-1}
    ims=[]
    #cycles=cycles if (pipdef is not None) else 1

    itertimes=[]
    totime=0
    if mcdef is not None:
        print(f"Generating {nmc} MirrorCheck examples")
        starter = time.time()
        mcimgs = []
        for mci in range(nmc):
            mcimgs.append(mcdef.generate(target_vlm_answer))
        totime += time.time() - starter
        print("ready")
    if not ez:
        threshold=det_thresh_dict[topk][vlmi.name]["gait"][perz]
        bounds=det_thresh_dict[topk][vlmi.name]["gait_bounds"]
        pipth=det_thresh_dict[topk][vlmi.name]["pip"][perz]
        mcth=det_thresh_dict[topk][vlmi.name]["mc"][perz]
        nsth=det_thresh_dict[topk][vlmi.name]["ns"][perz]

    for h in range(cycles):#this was for the adaptive attacks that alternate between misleading the model and misleading the attack detector... for now cycles is always 1.
        exit=False
        momentum=None
        if ez and exclude:#this means the current attack is only doing the advergait defense, so we compute the first two steps and produce the mask
            #print("ke say")
            for i in range(2):
                image.requires_grad = True
                loss_vlm=0
                for full_text_vlm_prompt, target_tokens, vlm in zip(ftvs, tts, vlms):
                    out = vlm.forward(image, full_text_vlm_prompt, overwrite=True)
                    loss_vlm += vlm.compute_gen_loss(out, target_tokens)

                total_loss = loss_vlm
                if i==0:
                    iniloss=total_loss.item()
                #print(f"RAM usage -> {get_memory_consumption(device):.2f}")
                #input(i)

                obj_grads = torch.autograd.grad(total_loss, image, retain_graph=True, create_graph=True)[0]
                #print(f"RAM usage -> {get_memory_consumption(device):.2f}")
                #input("psotgen")
                adapt_grads = 0

                with torch.no_grad():
                    momentum = obj_grads.detach() #if momentum is None else momentum*0.5 + grads
                    image = attack_step_pgd(image, momentum + adapt_grads, 8/255, 255, initial_image)
                if not i:
                    lastgrads = momentum.sum(0)
                    ims=[image.detach().clone()]
                    #obj_grads=obj_grads.detach()
                del obj_grads
                #print(f"RAM usage -> {get_memory_consumption(device):.2f}")
                #input("postrik")

            goal = lastgrads*momentum.sum(0)
            cosim = goal/(torch.linalg.norm(lastgrads) * torch.linalg.norm(momentum.sum(0)))
            if not rmask:
                emancipator, _ = torch.sort((goal).flatten())
                emancipator=emancipator[-int(224*224*topk)]
            else:
                goal=torch.rand(goal.shape).cuda()
                emancipator, _ = torch.sort((goal).flatten())
                emancipator=emancipator[-int(224*224*topk)]

            if exclude==2:
                emancipator = 1.0*(goal < emancipator)
            elif exclude==1:
                emancipator = 1.0*(goal >= emancipator)

            zamn = emancipator

            #note that the mask is increase to include neighboring pixels of any that is in the original mask, maybe cleaner to implement with a 2D convolution...
            zamno = torch.zeros(emancipator.shape).cuda() +  zamn
            zamno[:-1]+=zamn[1:]#right neighbour
            zamno[1:]+=zamn[:-1]#left neigh
            zamno[:, :-1]+=zamn[:, 1:]#down neigh
            zamno[:, 1:]+=zamn[:, :-1]#up neigh
            zamno[:-1, :-1]+=zamn[1:, 1:]#SE neigh
            zamno[1:, 1:]+=zamn[:-1, :-1]#NW neigh
            zamno[1:, :-1]+=zamn[:-1, 1:]#SW neigh
            zamno[:-1, 1:]+=zamn[1:, :-1]#NE neigh
            emancipator = 1.0*(zamno > 0)

            image=initial_image.clone().detach()
            image=image*(1-emancipator) + (emancipator*0.5*255)#.to(torch.uint8).to(torch.float)
            list_evol.append([cosim.sum(), iniloss, torch.linalg.norm(lastgrads).item()])#None])

        for i in range(n_gradient_steps):
            starter = time.time()
            image.requires_grad = True
            if dpdef is not None:
                purified = purify(dpdef, image.unsqueeze(0)).squeeze(0)
                loss_vlm=0
                for full_text_vlm_prompt, target_tokens, vlm in zip(ftvs, tts, vlms):
                    out = vlm.forward(purified, full_text_vlm_prompt, overwrite=True)
                    #olout=out
                    loss_vlm += vlm.compute_gen_loss(out, target_tokens)
            else:
                if lambda_vlm > 0:
                    loss_vlm=0
                    for full_text_vlm_prompt, target_tokens, vlm in zip(ftvs, tts, vlms):
                        out = vlm.forward(image, full_text_vlm_prompt, overwrite=True)
                        #olout=out
                        loss_vlm += vlm.compute_gen_loss(out, target_tokens)


            total_loss = lambda_vlm * loss_vlm

            #input(total_loss)
            if i==0 or (i+1)%print_every==0 or dpdef is not None:
                print(f"Iter {i+1:4d}/{n_gradient_steps}, RAM usage -> {get_memory_consumption(device):.2f} GB, Losses -> VLM: {loss_vlm.item():.8f}, Total: {total_loss.item():.8f}")


            obj_grads = torch.autograd.grad(total_loss, image, create_graph=not naib and pipdef is None and mcdef is None and dpdef is None and nsdef is None, retain_graph=not naib and pipdef is None and mcdef is None and dpdef is None and nsdef is None)[0]
            #perz% FPR in TRAINING DATA



            #adaptive attack for PIP
            if pipdef is not None:
                ini_curr = image.clone().detach()

                inatt = pipdef.get_attention(image/255).flatten()
                #FOR RBF KERNEL:
                """
                gamma = pipdef.svm._gamma
                svec, scoef, sbias = torch.tensor(pipdef.svm.support_vectors_).cuda(), torch.tensor(pipdef.svm.dual_coef_).cuda(), torch.tensor(pipdef.svm.intercept_).cuda()
                K = torch.exp(-gamma * torch.linalg.norm(inatt - svec, dim=-1)**2)
                piploss = (K * scoef).sum() + sbias
                """
                #FOR LINEAR KERNEL (like in PIP paper)
                scoef, sbias = torch.tensor(pipdef.svm.coef_).cuda(), torch.tensor(pipdef.svm.intercept_).cuda()
                piploss = (scoef*inatt).sum() + sbias
                #input(piploss)
                #FOR A ONE-CLASS SVM:
                """
                piploss = torch.clip(-piploss - pipth, min=0)
                """
                #FOR A BINARY SVM (like in PIP paper)
                piploss = torch.clip(piploss - pipth, min=0)

                adapt_grads = torch.autograd.grad(piploss, image)[0]

            #adaptive attack for MirrorCheck
            elif mcdef is not None:
                ini_curr = image.clone().detach()
                zimz=torch.cat([mcdef.similarity(genimg, image/255).unsqueeze(0) for genimg in mcimgs])
                mcloss = torch.min(zimz)
                mcloss = torch.clip(mcth - mcloss, min=0)
                adapt_grads = torch.autograd.grad(mcloss, image)[0]

            #adaptive attack for Nearside
            elif nsdef is not None:
                ini_curr = image.clone().detach()
                _, detns = nsdef.detect(image)
                nsloss = torch.clip(detns - nsth, min=0)
                adapt_grads = torch.autograd.grad(nsloss, image)[0]

            #adaptive attack for diffpure (maybe needed if we do the detection adapter...)
            elif dpdef is not None:
                ini_curr = image.clone().detach()
                adapt_grads=0#the "adapt grads" are already the normal grads for the image with purification
                """
                purified = purify(dpdef, image.unsqueeze(0)).squeeze(0)
                lossdp=0
                for full_text_vlm_prompt, target_tokens, vlm in zip(ftvs, tts, vlms):
                    out = vlm.forward(image, full_text_vlm_prompt, overwrite=True)
                    #olout=out
                    lossdp += vlm.compute_gen_loss(out, target_tokens)

                adapt_grads = torch.autograd.grad(lossdp, image)[0]
                """




            #adaptive attack for AdverGait
            elif not ez and not naib:#adaptive attack

                if i==0 or (i+1)%print_every==0:
                    vlm_answer_b = vlmi.generate_e2e(image.clone().detach(), user_query, temperature=0)[0]
                    print(vlm_answer_b)
                obj_loss=total_loss
                grads=obj_grads
                #FEATURE PHI_0 (LOSS)
                loss_loss = torch.clip(obj_loss - threshold[0], min=0) + torch.clip(threshold[0] - obj_loss, min=0)#loss as detection score
                grad_loss = torch.linalg.norm(obj_grads.sum(0))#grad norm as detection score
                gnorm = grad_loss.clone().detach().item()

                #FEATURE PHI_1 (GRADIENT NORM)
                grad_loss = torch.clip(bounds["norm_lo"] - grad_loss, min=0) + torch.clip(grad_loss - bounds["norm_hi"], min=0)

                current=image#.clone().detach()
                ini_curr = image.clone().detach()
                lastgrads=grads.sum(0)#None

                with torch.no_grad():
                    current = attack_step_pgd(current, grads, 8/255, 255, ini_curr)

                out = vlmi.forward(current, full_text_vlm_prompt, overwrite=True)
                loss = vlmi.compute_gen_loss(out, target_tokens)
                grads = torch.autograd.grad(loss, current, create_graph=True, retain_graph=True)[0]
                #gs = grads.clone().detach()
                cosim_un = lastgrads*grads.sum(0)
                cosim=cosim_un/(torch.linalg.norm(lastgrads) * torch.linalg.norm(grads.sum(0)))#cosine sim. as detection score

                #FEATURE PHI_2 (COSINE SIMILARITY)
                cosim_loss = torch.clip(bounds["cosim_lo"] - cosim.sum(), min= 0)#only matters if cosine similarity is too low

                #potential enhancements for adaptive attack...
                if True:
                    cosloss = torch.tensor([0]).cuda()#for now let's focus on the masked loss directly...
                elif patch_attack:
                    cosloss =  - cosim_un*(1-mask) + cosim_un*(mask)#similarity contribution inside the patch should be lower than outside
                    cosloss = cosloss.sum()
                else:
                    cosloss = cosim_un.std()#similarity should not vary much across the image (is this good?)

                bound_loss = grad_loss.unsqueeze(0) + cosim_loss #cos. sim. and gradient norm should be within bounds from clean data stats


                #GET MASK FOR MASKED LOSS
                dmask, _ = torch.sort((cosim_un).flatten())
                dmask=dmask[-int(224*224*topk)]
                dmask = 1.0*(cosim_un >= dmask)



                zamn = dmask
                zamno = torch.zeros(zamn.shape).cuda() +  zamn
                zamno[:-1]+=zamn[1:]#right neighbour
                zamno[1:]+=zamn[:-1]#left neigh
                zamno[:, :-1]+=zamn[:, 1:]#down neigh
                zamno[:, 1:]+=zamn[:, :-1]#up neigh
                #plotee(zamno)
                zamno[:-1, :-1]+=zamn[1:, 1:]#SE neigh
                zamno[1:, 1:]+=zamn[:-1, :-1]#NW neigh
                #plotee(zamno)
                zamno[1:, :-1]+=zamn[:-1, 1:]#SW neigh
                zamno[:-1, 1:]+=zamn[1:, :-1]#NE neigh

                dmask = 1.0*(zamno > 0)

                ris = ini_curr.clone().detach() - current.clone().detach()
                with torch.no_grad():
                    current+=ris#ini_curr.clone().detach()
                    current*=(1-dmask)# + dmask*0.5*255
                    current+= (dmask*0.5*255)#.to(torch.uint8).to(torch.float)

                dout = vlmi.forward(current, full_text_vlm_prompt, overwrite=True)
                dloss = vlmi.compute_gen_loss(dout, target_tokens)#masked loss

                #FEATURE PHI_3 (MASKED LOSS)
                bound_loss += torch.clip(bounds["mloss_lo"] - dloss, min=0) + torch.clip(dloss - bounds["mloss_hi"], min=0)#masked loss should be within bounds (AGG) (QWEN)#remoed lasttttt

                ldif = dloss - obj_loss

                #FEATURE PHI_4 (LOSS DIFFERENCE)
                bound_loss += torch.clip(ldif - threshold[1], min=0) + torch.clip(bounds["ldif_lo"] - ldif, min=0)#loss difference should be within bounds (AGG) (QWEN)


                det_score = cosloss#does not matter at the moment and is = Zero

                bound_loss += loss_loss
                full_det_score = bound_loss + det_score
                adapt_grads = torch.autograd.grad(full_det_score, current)[0]# + grads_large

                #remove the masking from the image...
                image=ini_curr.clone().detach()
            else:
                adapt_grads = 0


            #EZ WAY (gait defense without adaptive attacker)
            if ez:
                out = vlmi.forward(image.clone().detach(), full_text_vlm_prompt, overwrite=True)
                det_score = vlmi.compute_gen_loss(out, target_tokens)
                grads = torch.zeros_like(image)
                if evol:
                    list_evol.append([obj_grads.clone().detach(), det_score.item(), torch.linalg.norm(obj_grads.clone().detach())])#dima_x ])


            #Naive attacks (no defenses to bypass)
            if naib:
                ini_curr = image.clone().detach()

            #ADAPTIVE ATTACK STEP
            with torch.no_grad():
                if patch_attack:
                    obj_grads=obj_grads*mask
                    adapt_grads=adapt_grads*mask#_s
                momentum = obj_grads if (naib or ez or pipdef !=None or mcdef != None or dpdef != None or nsdef != None) else 0#if momentum is None else momentum*0.5 + grads
                image = attack_step_pgd(image, momentum + adapt_grads, lr, max_perturbation_pixels, initial_image)
            itertimes.append(time.time() - starter)
            if ez:
                del obj_grads




            print_this=False
            bypass = False
            if naib or dpdef is not None:#naive attacks por purifier-only defense...
                to_see = purified if dpdef is not None else ini_curr
                bypass=True
                if loss_vlm < best_loss:
                    print_this=True
                    vlm_answer_b = vlmi.generate_e2e(to_see.clone().detach(), user_query, temperature=0)[0]
                    best=ini_curr.clone().detach()
                    best_loss  = loss_vlm
                if i==0 or (i+1)%print_every==0 or dpdef is not None:
                    vlm_answer_b = vlmi.generate_e2e(to_see.clone().detach(), user_query, temperature=0)[0]
                    print(vlm_answer_b)

            if not naib and not ez and pipdef is None and mcdef is None and dpdef is None and nsdef is None:# and bound_loss <= 0:# and loss_large <=0:
                #print_this=False
                #print("INT. CHECK")
                #print(torch.prod(ini_curr == ini_curr.to(torch.uint8).to(torch.float)))
                if loss_vlm < best_loss:
                    print("lowest loss!")
                    vlm_answer_b = vlmi.generate_e2e(ini_curr.clone().detach(), user_query, temperature=0)[0]
                    print(vlm_answer_b)
                    best = ini_curr.detach().clone().cpu()
                    best_loss = loss_vlm
                    print_this=True
                    times["lowest"] = i
                if (bound_loss - loss_loss) <=0 and obj_loss >= threshold[0] and loss_vlm < best_det:#det_score < best_det:
                    print("lowest loss! (undetectable only)")
                    vlm_answer_b = vlmi.generate_e2e(ini_curr.clone().detach(), user_query, temperature=0)[0]
                    print(vlm_answer_b)
                    stealth_1 = ini_curr.detach().clone().cpu()
                    best_det = loss_vlm#det_score
                    print_this=True
                    bypass = True
                    times["lowest_und"] = i
                if (bound_loss - loss_loss) <=0 and obj_loss >= threshold[0] and bound_loss < best_full_det:#det_score < best_det:
                    print("best overall! (undetectable only)")
                    vlm_answer_b = vlmi.generate_e2e(ini_curr.clone().detach(), user_query, temperature=0)[0]
                    print(vlm_answer_b)
                    stealth_2 = ini_curr.detach().clone().cpu()
                    best_full_det = bound_loss#det_score
                    print_this=True
                    bypass = True
                    times["lowest_glob_und"] = i
                if det_score + bound_loss < best_total:#det_score - dist_score - loss_large < best_total:
                    print("best overall!")
                    vlm_answer_b = vlmi.generate_e2e(ini_curr.clone().detach(), user_query, temperature=0)[0]
                    print(vlm_answer_b)
                    best_big = ini_curr.detach().clone().cpu()
                    best_total = det_score + bound_loss# - dist_score - loss_large#full_det_score + loss_vlm
                    print_this=True
                    times["lowest_glob"] = i
                if print_this:
                    #print("way out heah")
                    #input(torch.prod(chek==ini_curr))
                    print(f'obj/self loss: {obj_loss.item()}')
                    #print(f'self loss: {loss.item()}')
                    print(f'masked loss: {dloss.item()}')
                    print(f'dif: {ldif.item()}')
                    print(f'cosloss: {cosloss.item()}')
                del ini_curr


            if pipdef is not None or mcdef is not None or nsdef is not None:
                #print_this=False
                if pipdef is not None:
                    detloss = piploss
                elif mcdef is not None:
                     detloss = mcloss
                elif nsdef is not None:
                    detloss = nsloss
                if total_loss < best_loss:
                    print("lowest loss!")
                    vlm_answer_b = vlmi.generate_e2e(ini_curr.clone().detach(), user_query, temperature=0)[0]
                    print(vlm_answer_b)
                    best = ini_curr.detach().clone().cpu()
                    best_loss = total_loss
                    print_this=True
                    times["lowest"] = i


                if total_loss + detloss < best_total:#det_score - dist_score - loss_large < best_total:
                    print("best overall!")
                    vlm_answer_b = vlmi.generate_e2e(ini_curr.clone().detach(), user_query, temperature=0)[0]
                    print(vlm_answer_b)
                    best_big = ini_curr.detach().clone().cpu()
                    best_total = total_loss + detloss
                    print_this=True
                    times["lowest_glob"] = i

                if total_loss < best_det and detloss<=0:
                    print("lowest loss! (undetectable)")
                    vlm_answer_b = vlmi.generate_e2e(ini_curr.clone().detach(), user_query, temperature=0)[0]
                    print(vlm_answer_b)
                    stealth_1 = ini_curr.detach().clone().cpu()
                    best_det = total_loss
                    print_this=True
                    bypass =True
                    times["lowest_und"] = i

                if total_loss + detloss < best_full_det and detloss<=0:#det_score - dist_score - loss_large < best_total:
                    print("best overall! (undetectable)")
                    vlm_answer_b = vlmi.generate_e2e(ini_curr.clone().detach(), user_query, temperature=0)[0]
                    print(vlm_answer_b)
                    stealth_2 = ini_curr.detach().clone().cpu()
                    best_full_det = total_loss + detloss
                    print_this=True
                    bypass =True
                    times["lowest_glob_und"] = i

                if print_this:
                    #print("way out heah")
                    #input(torch.prod(chek==ini_curr))
                    print(f'obj loss: {total_loss.item()}')
                    print(f'pip/ns/mc loss: {detloss.item()}')

            if print_this and target_vlm_answer == vlm_answer_b.lower().translate(str.maketrans('', '', string.punctuation)) and times["success"] == -1 and bypass:
                times["success"] = i
                if naib or dpdef is not None:
                    stealth_1 = ini_curr.detach().clone().cpu()
                break#SUCCESSFUL ADAPTIVE ATTACK


    if evol:
        return list_evol, ims
    print("ALL OVER")
    print(h)
    #input()
    for k in times.keys():
        adon=-1
        if times[k] != -1:
            adon = sum(itertimes[:times[k]])
        itertimes.append(adon)
    itertimes.append(totime)
    return [image.clone(), best, best_big, stealth_1, stealth_2, stealth_big, best_esc, best_big_esc, stealth_esc, stealth_big_esc], itertimes##soon, fifty




def attack_step_pgd(
    image: torch.tensor,
    grads: torch.tensor,
    lr: float,
    max_perturbation_pixels: int,
    initial_image: torch.tensor,
    additive=False
):
    """
    which direction should we go, given the computed loss gradient
    """
    # take step
    image -= lr * torch.sign(grads)
    if not additive:
        # clip according to attack budget
        if max_perturbation_pixels < 255:
            torch.clip(image, min=initial_image-max_perturbation_pixels, max=initial_image+max_perturbation_pixels, out=image)
        # clip to make sure we stay within allowed RGB values
        torch.clip(image, min=0, max=255, out=image)
    else:
        # clip according to attack budget
        #input(max_perturbation_pixels)
        torch.clip(image, min=initial_image-max_perturbation_pixels*2/255, max=initial_image+max_perturbation_pixels*2/255, out=image)
        # clip to make sure we stay within allowed RGB values
        torch.clip(image, min=-1, max=1, out=image)

    return image

def dif_entropy(x, window=torch.tensor(1)):
    x=torch.sort(x).values
    m=window#window_length
    n=torch.tensor(x.shape[-1])
    dif=x[1:] - x[:-1]
    term1 = 1/(n-m) * torch.sum(torch.log((n+1)/m * dif), axis=-1)
    k = torch.arange(m, n+1, dtype=term1.dtype)
    return term1 + torch.sum(1/k) + torch.log(m) - torch.log(n+1)
