# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Train and eval functions used in main.py
"""

import math
import os
import sys
from typing import Iterable
import json
import matplotlib.pyplot as plt
import cv2

from util.utils import slprint, to_device, convert_boxes_to_normalized

import torch

import util.misc as utils
from datasets.coco_eval import CocoEvaluator
from datasets.sgg_eval import SggEvaluator 
from datasets.panoptic_eval import PanopticEvaluator

from util.vis_utils import plot_raw_img2, add_box_to_img

def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0, 
                    wo_class_error=False, lr_scheduler=None, args=None, 
                    logger=None, ema_m=None, wandb_logger=None, 
                    model_t=None):

    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    teacher_update_interval = getattr(args, "teacher_update_interval", 1000)
    momentum_t = getattr(args, "teacher_momentum", 0.999)
    teacher_update = getattr(args, "teacher_update", False)


    try:
        need_tgt_for_training = args.use_dn
    except:
        need_tgt_for_training = False

    use_text_labels = getattr(args, "use_text_labels", False)
    if use_text_labels:
        need_tgt_for_training = True 


    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    #
    try:
        criterion.ind_to_predicates = data_loader.dataset.ind_to_predicates
    except:
        pass

    if not wo_class_error:
        metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    _cnt = 0

    for samples, targets in metric_logger.log_every(data_loader, print_freq, header, logger=logger):
        args.global_iter += 1

        samples = samples.to(device)
        targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]

        with torch.cuda.amp.autocast(enabled=args.amp):
            if need_tgt_for_training:
                outputs = model(samples, targets, global_iter=args.global_iter)
            else:
                outputs = model(samples, global_iter=args.global_iter)
            
            if model_t is not None:
                with torch.no_grad():
                    if need_tgt_for_training:
                        outputs_t = model_t(samples, targets, global_iter=args.global_iter)
                    else:
                        outputs_t = model_t(samples, global_iter=args.global_iter)
            else:
                outputs_t = None 
                    
            loss_dict = criterion(outputs, targets, outputs_t=outputs_t, 
                                  global_iter=args.global_iter)

            weight_dict = criterion.weight_dict
            losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

               

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)


        # amp backward function
        if args.amp:
            optimizer.zero_grad()
            scaler.scale(losses).backward()
            if max_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            # original backward function
            optimizer.zero_grad()
            losses.backward()
            if max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            optimizer.step()

        if model_t is not None and teacher_update and args.global_iter % teacher_update_interval == 0:
            if utils.get_rank() == 0:
                print("*"*10, "global iter:", args.global_iter, 
                  " update teacher's weights with momentum:%s!" % momentum_t)
            for name, p in model_t.named_parameters():
                state = model.state_dict()
                keys = list(state.keys())
                if 'module.' in keys[0]:
                    name = 'module.' + name

                with torch.no_grad():
                    p.data = momentum_t * p.data + (1-momentum_t) * state[name]


        if args.onecyclelr:
            lr_scheduler.step()
        if args.use_ema:
            if epoch >= args.ema_epoch:
                ema_m.update(model)

        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        if 'class_error' in loss_dict_reduced:
            metric_logger.update(class_error=loss_dict_reduced['class_error'])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])


        if wandb_logger is not None:
            dict_info = loss_dict_reduced_scaled
            dict_info["lr"] = optimizer.param_groups[0]["lr"]
            if 'rel_batch' in loss_dict:
                dict_info['rel_batch'] = loss_dict['rel_batch'].item()
            wandb_logger.log(dict_info)
            

        _cnt += 1
        if args.debug:
            if _cnt % 15 == 0:
                print("BREAK!"*5)
                break


    if getattr(criterion, 'loss_weight_decay', False):
        criterion.loss_weight_decay(epoch=epoch)
    if getattr(criterion, 'tuning_matching', False):
        criterion.tuning_matching(epoch)


    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    resstat = {k: meter.global_avg for k, meter in metric_logger.meters.items() if meter.count > 0}
    if getattr(criterion, 'loss_weight_decay', False):
        resstat.update({f'weight_{k}': v for k,v in criterion.weight_dict.items()})
    return resstat


@torch.no_grad()
def evaluate(model, criterion, postprocessors, data_loader, base_ds, device, output_dir, wo_class_error=False, args=None, logger=None):
    try:
        need_tgt_for_training = args.use_dn
    except:
        need_tgt_for_training = False

    model.eval()
    criterion.eval()
    postprocessors['bbox'].eval()
    try:
        criterion.ind_to_predicates = data_loader.dataset.ind_to_predicates
    except:
        pass

    metric_logger = utils.MetricLogger(delimiter="  ")
    if not wo_class_error:
        metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Test:'

    iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessors.keys())
    useCats = True
    try:
        useCats = args.useCats
    except:
        useCats = True
    if not useCats:
        print("useCats: {} !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!".format(useCats))

    try:
        assert args.dataset_file == "coco"
        #coco_evaluator = CocoEvaluator(base_ds, iou_types, useCats=useCats)
        # coco_evaluator.coco_eval[iou_types[0]].params.iouThrs = [0, 0.1, 0.5, 0.75]
        coco_evaluator = None 
    except:
        coco_evaluator = None 

    sgg_evaluator = None 
    if args.dataset_file in ["vg", "oicap", "coco", "coco_flickr30k", "coco_flickr30k_sbucaptions"]:
        iou_types = list(iou_types)
        if args is not None and getattr(args, "iou_types", None):
            iou_types = list(args.iou_types)
        else:
            iou_types.append("relation")

        iou_types = tuple(set(iou_types))
        try:
            num_rel_category = args.num_rel_category
        except:
            num_rel_category = 51

        sgg_evaluator = SggEvaluator(data_loader.dataset, iou_types,
                                     mode="sgdet", num_rel_category=num_rel_category,
                                     multiple_preds=False, iou_thres=0.5,
                                     output_folder=os.path.join(output_dir, "sgg_eval"),
                                     ovd_enabled=getattr(args, "sg_ovd_mode", False),
                                     ovr_enabled=getattr(args, "sg_ovr_mode", False)
                                     )
        postprocessors['bbox'].eval()


    use_text_labels = getattr(args, "use_text_labels", False)
    if use_text_labels:
        need_tgt_for_training = True 

        postprocessors['bbox'].name2classes = data_loader.dataset.name2classes
        do_sgg = getattr(postprocessors['bbox'], 'do_sgg', False)
        if do_sgg:
            postprocessors['bbox'].name2predicates = data_loader.dataset.name2predicates



    panoptic_evaluator = None
    if 'panoptic' in postprocessors.keys():
        panoptic_evaluator = PanopticEvaluator(
            data_loader.dataset.ann_file,
            data_loader.dataset.ann_folder,
            output_dir=os.path.join(output_dir, "panoptic_eval"),
        )

    _cnt = 0
    output_state_dict = {} # for debug only

    # For getting stats about relations 
    relations_info = dict()
    relations_info['number_rel'] = 0
    relations_info['all_rel'] = []
    relations_info['min_rel'] = float('inf') 
    relations_info['max_rel'] = float('-inf')
    relations_info['mean_rel'] = 0

    vis_dir = os.path.join(args.output_dir, "visualization")
    os.makedirs(vis_dir, exist_ok=True)

    for samples, targets in metric_logger.log_every(data_loader, 10, header, logger=logger):
        samples = samples.to(device)
        targets = [{k: to_device(v, device) for k, v in t.items()} for t in targets]

        with torch.cuda.amp.autocast(enabled=args.amp):
            if need_tgt_for_training:
                outputs = model(samples, targets)
            else:
                outputs = model(samples)

            loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}


        metric_logger.update(loss=sum(loss_dict_reduced_scaled.values()),
                             **loss_dict_reduced_scaled,
                             **loss_dict_reduced_unscaled)
        if 'class_error' in loss_dict_reduced:
            metric_logger.update(class_error=loss_dict_reduced['class_error'])

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        
        if postprocessors['bbox'].use_gt_box:
            match_ids = postprocessors['bbox'].matcher(outputs, targets)
            tgt_dicts = [{'ids': ids, 'gt_boxes': target['boxes'],
                          'gt_labels': target['labels']}
                          for target, ids in zip(targets, match_ids)]

            results = postprocessors['bbox'](outputs, orig_target_sizes, 
                                             gt_dicts=tgt_dicts)
        else:
            results = postprocessors['bbox'](outputs, orig_target_sizes)

        # [scores: [100], labels: [100], boxes: [100, 4]] x B
        if 'segm' in postprocessors.keys():
            target_sizes = torch.stack([t["size"] for t in targets], dim=0)
            results = postprocessors['segm'](results, outputs, orig_target_sizes, target_sizes)

        res = {target['image_id']: output for target, output in zip(targets, results)}

        #torch.save(res, os.path.join(output_dir, "%s.pt"%_cnt ))
        #res = torch.load(os.path.join(output_dir, "%s.pt"%_cnt))

        if coco_evaluator is not None:
            coco_evaluator.update(res)

        if sgg_evaluator is not None:
            sgg_evaluator.update(res)

        _cnt += 1
        if panoptic_evaluator is not None:
            res_pano = postprocessors["panoptic"](outputs, target_sizes, orig_target_sizes)
            for i, target in enumerate(targets):
                image_id = target["image_id"]
                file_name = f"{image_id:012d}.png"
                res_pano[i]["image_id"] = image_id
                res_pano[i]["file_name"] = file_name

            panoptic_evaluator.update(res_pano)

        # For mapping the classes and predicates 
        idx2classes = {v: k for k, v in postprocessors['bbox'].name2classes.items()}
        idx2predicates = {v: k for k, v in postprocessors['bbox'].name2predicates.items()}

        save_graphs = True

        if args.save_results:
            # res_score = outputs['res_score']
            # res_label = outputs['res_label']
            # res_bbox = outputs['res_bbox']
            # res_idx = outputs['res_idx']

            for i, (tgt, res, outbbox) in enumerate(zip(targets, results, outputs['pred_boxes'])):
                """
                pred vars:
                    K: number of bbox pred
                    score: Tensor(K),
                    label: list(len: K),
                    bbox: Tensor(K, 4)
                    idx: list(len: K)
                tgt: dict.

                """
                # import pdb; pdb.set_trace()
                image_id = tgt['image_id'].item() if torch.is_tensor(tgt['image_id']) else tgt['image_id']
                img_dir = os.path.join(vis_dir, f"{image_id}")
                os.makedirs(img_dir, exist_ok=True)
                triplets_dir = os.path.join(img_dir, "triplets")
                os.makedirs(triplets_dir, exist_ok=True)

                # compare gt and res (after postprocess)
                gt_bbox = tgt['boxes']
                gt_label = tgt['labels']
                gt_info = torch.cat((gt_bbox, gt_label.unsqueeze(-1)), 1)
                
                img_h, img_w = tgt['orig_size'].unbind()
                scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=0)
                _res_bbox = res['boxes'] / scale_fct
                # _res_bbox = outbbox
                _res_prob = res['scores']
                _res_label = res['labels']
                res_info = torch.cat((_res_bbox, _res_prob.unsqueeze(-1), _res_label.unsqueeze(-1)), 1)

                if 'gt_info' not in output_state_dict:
                    output_state_dict['gt_info'] = []
                output_state_dict['gt_info'].append(gt_info.cpu())

                if 'res_info' not in output_state_dict:
                    output_state_dict['res_info'] = []
                output_state_dict['res_info'].append(res_info.cpu())

                ### ADDITIONAL INFO ABOUT RELATIONS
                num_pairs = len(res['graph']['all_node_pairs'])        
                relations_info['number_rel'] += num_pairs
                relations_info['all_rel'].append(num_pairs) 
                if num_pairs < relations_info['min_rel']: relations_info['min_rel'] = num_pairs 
                if num_pairs > relations_info['max_rel']: relations_info['max_rel'] = num_pairs 
                relations_info['mean_rel'] = sum(relations_info['all_rel']) / len(relations_info['all_rel'])
                print("----INFO ABOUT RELATIONS ----\n", relations_info)

                ### GETTING IMAGE WITH BOUNDING BOXES
                # import pdb; pdb.set_trace()
                imgg = samples.tensors
                gt_img = plot_raw_img2(imgg[0], gt_bbox, gt_label, idx2classes)
                pred_img = plot_raw_img2(imgg[0], _res_bbox, _res_label, idx2classes)

                save_gt = os.path.join(img_dir, f"gt_bbox.png")
                save_pred = os.path.join(img_dir, f"pred_bbox.png")
                cv2.imwrite(save_gt, gt_img) 
                cv2.imwrite(save_pred, pred_img) 

                ### SAVING THE TRIPLETS
                if save_graphs:
                    
                    object_labels = res['labels']  # shape [K]

                    pairs = res['graph']['all_node_pairs']         # shape [N, 2]
                    all_rels = res['graph']['all_relation']        # shape [N, R]
                    all_bbox = res['graph']['pred_boxes']
                    max_scores, max_indices = all_rels.max(dim=1)  # max per row; max_scores: [N], max_indices: [N]

                    # import pdb; pdb.set_trace()
                    triplets = []
                    threshold = 0.50
                    for (sub_idx, obj_idx), rel_score, rel_idx in zip(pairs, max_scores, max_indices):
                        if rel_score.item() > threshold:
                            # Extract subject and object bounding boxes
                            subject_bbox = all_bbox[sub_idx].tolist() 
                            object_bbox = all_bbox[obj_idx].tolist()  

                            # Validate bounding boxe
                            # print(f"Subject BBox: {subject_bbox}, Object BBox: {object_bbox}")

                            # Extract labels
                            subject_cls_id = object_labels[sub_idx].item()
                            object_cls_id = object_labels[obj_idx].item()
                            subject_label = idx2classes[subject_cls_id]
                            object_label = idx2classes[object_cls_id]
                            rel_label = idx2predicates[rel_idx.item()]

                            # Append triplet
                            triplet = {
                                "source": f"{subject_label}.{sub_idx.item()}",
                                "target": f"{object_label}.{obj_idx.item()}",
                                "relation": rel_label,
                                "score": rel_score.item()
                            }
                            triplets.append(triplet)

                            # Generate triplet visualization
                            print(f"Triplets for image {image_id}")
                            triplet_boxes = torch.tensor([subject_bbox, object_bbox])
                            triplet_boxes = convert_boxes_to_normalized(triplet_boxes, img_w, img_h)
                            triplet_labels = torch.tensor([subject_cls_id, object_cls_id])

                            triplet_img = plot_raw_img2(
                                imgg[0],
                                triplet_boxes,
                                triplet_labels,
                                idx2classes
                            )

                            # Save triplet visualization
                            save_triplet_path = os.path.join(
                                triplets_dir,
                                f"{subject_label}_{rel_label}_{object_label}.png"
                            )
                            cv2.imwrite(save_triplet_path, triplet_img)

                    # writing the jsons
                    json_path = os.path.join(img_dir, f"{image_id}_triplets.json")

                    with open(json_path, "w") as f:
                        json.dump(triplets, f, indent=2)

            # # for debug only
            # import random
            # if random.random() > 0.7:
            #     print("Now let's break")
            #     break

        if args.debug:
            if _cnt % 15 == 0:
                print("BREAK!"*5)
                break

    relations_info['mean_rel'] = sum(relations_info['all_rel']) / len(relations_info['all_rel'])
    print("The mean number of relations is ", relations_info['mean_rel'])

    infos_path = os.path.join(args.output_dir, "1-info_about_relations.txt")

    with open(infos_path, "w") as f:
        json.dump(relations_info, f, indent=2) 

    if args.save_results:
        import os.path as osp
        
        # output_state_dict['gt_info'] = torch.cat(output_state_dict['gt_info'])
        # output_state_dict['res_info'] = torch.cat(output_state_dict['res_info'])
        savepath = osp.join(args.output_dir, 'results-{}.pkl'.format(utils.get_rank()))
        print("Saving res to {}".format(savepath))
        torch.save(output_state_dict, savepath)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()
    if panoptic_evaluator is not None:
        panoptic_evaluator.synchronize_between_processes()

    if sgg_evaluator is not None:
        sgg_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    if coco_evaluator is not None:
        coco_evaluator.accumulate()
        coco_evaluator.summarize()

    sgg_res = None
    if sgg_evaluator is not None:
        sgg_res = sgg_evaluator.accumulate()
        sgg_evaluator.summarize()
        sgg_evaluator.reset()

        
    panoptic_res = None
    if panoptic_evaluator is not None:
        panoptic_res = panoptic_evaluator.summarize()
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items() if meter.count > 0}
    if coco_evaluator is not None:
        if 'bbox' in postprocessors.keys():
            stats['coco_eval_bbox'] = coco_evaluator.coco_eval['bbox'].stats.tolist()
        if 'segm' in postprocessors.keys():
            stats['coco_eval_masks'] = coco_evaluator.coco_eval['segm'].stats.tolist()
    if panoptic_res is not None:
        stats['PQ_all'] = panoptic_res["All"]
        stats['PQ_th'] = panoptic_res["Things"]
        stats['PQ_st'] = panoptic_res["Stuff"]

    if sgg_evaluator is not None:
        if 'bbox' in postprocessors.keys():
            try:
                stats['coco_eval_bbox'] = sgg_evaluator.coco_eval['bbox'].stats.tolist()
            except:
                stats['coco_eval_bbox'] = sgg_evaluator.coco_eval['bbox'].stats

        if sgg_res is not None:
            stats.update(sgg_res)



    return stats, coco_evaluator


@torch.no_grad()
def test(model, criterion, postprocessors, data_loader, base_ds, device, output_dir, wo_class_error=False, args=None, logger=None):
    model.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    # if not wo_class_error:
    #     metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Test:'

    iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessors.keys())
    # coco_evaluator = CocoEvaluator(base_ds, iou_types)
    # coco_evaluator.coco_eval[iou_types[0]].params.iouThrs = [0, 0.1, 0.5, 0.75]

    panoptic_evaluator = None
    if 'panoptic' in postprocessors.keys():
        panoptic_evaluator = PanopticEvaluator(
            data_loader.dataset.ann_file,
            data_loader.dataset.ann_folder,
            output_dir=os.path.join(output_dir, "panoptic_eval"),
        )

    final_res = []
    for samples, targets in metric_logger.log_every(data_loader, 10, header, logger=logger):
        samples = samples.to(device)

        # targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        targets = [{k: to_device(v, device) for k, v in t.items()} for t in targets]

        outputs = model(samples)
        # loss_dict = criterion(outputs, targets)
        # weight_dict = criterion.weight_dict

        # # reduce losses over all GPUs for logging purposes
        # loss_dict_reduced = utils.reduce_dict(loss_dict)
        # loss_dict_reduced_scaled = {k: v * weight_dict[k]
        #                             for k, v in loss_dict_reduced.items() if k in weight_dict}
        # loss_dict_reduced_unscaled = {f'{k}_unscaled': v
        #                               for k, v in loss_dict_reduced.items()}
        # metric_logger.update(loss=sum(loss_dict_reduced_scaled.values()),
        #                      **loss_dict_reduced_scaled,
        #                      **loss_dict_reduced_unscaled)
        # if 'class_error' in loss_dict_reduced:
        #     metric_logger.update(class_error=loss_dict_reduced['class_error'])

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = postprocessors['bbox'](outputs, orig_target_sizes, not_to_xyxy=True)
        # [scores: [100], labels: [100], boxes: [100, 4]] x B
        if 'segm' in postprocessors.keys():
            target_sizes = torch.stack([t["size"] for t in targets], dim=0)
            results = postprocessors['segm'](results, outputs, orig_target_sizes, target_sizes)
        res = {target['image_id']: output for target, output in zip(targets, results)}
        for image_id, outputs in res.items():
            _scores = outputs['scores'].tolist()
            _labels = outputs['labels'].tolist()
            _boxes = outputs['boxes'].tolist()
            for s, l, b in zip(_scores, _labels, _boxes):
                assert isinstance(l, int)
                itemdict = {
                        "image_id": image_id, 
                        "category_id": l, 
                        "bbox": b, 
                        "score": s,
                        }
                final_res.append(itemdict)

    if args.output_dir:
        import json
        with open(args.output_dir + f'/results{args.rank}.json', 'w') as f:
            json.dump(final_res, f)        

    return final_res