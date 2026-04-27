import json
import logging
import math
import os
import time
from typing import Optional
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.parallel.distributed import DistributedDataParallel
from collections import defaultdict, Counter
import gc
try:
    import wandb
except ImportError:
    wandb = None

from open_clip import get_input_dtype, CLIP, CustomTextCLIP
from open_clip.loss import multi_positive_cross_entropy_loss
from open_clip.transform import VolumeNormalizationError
from open_clip_train.distributed import is_master
from open_clip_train.zero_shot import zero_shot_eval
from open_clip_train.precision import get_autocast
from transformers import AutoTokenizer
from PIL import Image

# Load your tokenizer (replace 'your-model-name' with the actual model name)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def postprocess_clip_output(model_out):
    return {
        "image_features": model_out[0],
        "text_features": model_out[1],
        "logit_scale": model_out[2]
    }


def unwrap_model(model):
    if hasattr(model, 'module'):
        return model.module
    else:
        return model


def backward(total_loss, scaler):
    if scaler is not None:
        scaler.scale(total_loss).backward()
    else:
        total_loss.backward()


def train_one_epoch(model, data, loss, epoch, optimizer, scaler, scheduler, dist_model, args, tb_writer=None):
    device = torch.device(args.device)
    autocast = get_autocast(args.precision, device_type=device.type)
    input_dtype = get_input_dtype(args.precision)
    model.train()
    if args.distill:
        dist_model.eval()

    data['train'].set_epoch(epoch)  # set epoch in process safe manner via sampler or shared_epoch
    dataloader = data['train'].dataloader
    num_batches_per_epoch = dataloader.num_batches // args.accum_freq
    sample_digits = math.ceil(math.log(dataloader.num_samples + 1, 10))

    if args.accum_freq > 1:
        accum_images, accum_texts, accum_features = [], [], {}
    if args.freeze:
                   # Freeze all layers except the last few
        for param in unwrap_model(model).text.transformer.parameters():
            param.requires_grad = False
                # Unfreeze the last 2 layers
        for param in list(unwrap_model(model).text.transformer.encoder.layer[-args.freezelast:].parameters()):
            param.requires_grad = True   
    losses_m = {}
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    skipped_volumes = 0
    end = time.time()
    i = 0
    dataloader_iter = iter(dataloader)
    while True:
        try:
            batch = next(dataloader_iter)
        except StopIteration:
            break
        except VolumeNormalizationError as e:
            skipped_volumes += 1
            if is_master(args) and skipped_volumes % 10 == 0:
                logging.warning(f"Skipped {skipped_volumes} volumes due to normalization errors. Latest: {str(e)}")
            continue
        
        try:
            i_accum = i // args.accum_freq
            step = num_batches_per_epoch * epoch + i_accum

            if not args.skip_scheduler:
                scheduler(step)
            if args.distance:
                images, texts, labels,echotime,repetitiontime = batch
                echotime = echotime.to(device=device, non_blocking=True)
                repetitiontime = repetitiontime.to(device=device, non_blocking=True)
            else:   
                images, texts, labels = batch
            images = images.to(device=device, dtype=input_dtype, non_blocking=True)

            texts = texts.to(device=device, non_blocking=True)
            labels = labels.to(device=device, non_blocking=True)
            data_time_m.update(time.time() - end)
            optimizer.zero_grad()

            if args.accum_freq == 1:
                with autocast():
                    model_out = model(images, texts)
                    logit_scale = model_out["logit_scale"]
                    if args.distill:
                        with torch.no_grad():
                            dist_model_out = dist_model(images, texts)
                        model_out.update({f'dist_{k}': v for k, v in dist_model_out.items()})
                    #logging.info(f'Calculating loss for epoch {epoch} and batch {i}.')    
                    if args.multipositiveloss and args.distance is False:
                        losses = loss(**model_out, tokenized_texts=labels,delta=args.delta,output_dict=True)
                    elif args.multipositiveloss and args.distance is True:
                        losses = loss(**model_out, tokenized_texts=labels,delta=args.delta,echotime=echotime,repetitiontime=repetitiontime,output_dict=True)
                    else:
                        losses = loss(**model_out, output_dict=True)

                    total_loss = sum(losses.values())
                    losses["loss"] = total_loss

                backward(total_loss, scaler)
            else:
                # First, cache the features without any gradient tracking.
                with torch.no_grad():
                    with autocast():
                        model_out = model(images, texts)

                        for f in ("logit_scale", "logit_bias"):
                            model_out.pop(f, None)

                        for key, val in model_out.items():
                            if key in accum_features:
                                accum_features[key].append(val)
                            else:
                                accum_features[key] = [val]

                    accum_images.append(images)
                    accum_texts.append(texts)

                # If (i + 1) % accum_freq is not zero, move on to the next batch.
                if ((i + 1) % args.accum_freq) > 0:
                    # FIXME this makes data time logging unreliable when accumulating
                    continue

                # Now, ready to take gradients for the last accum_freq batches.
                # Re-do the forward pass for those batches, and use the cached features from the other batches as negatives.
                # Call backwards each time, but only step optimizer at the end.
                optimizer.zero_grad()
                for j in range(args.accum_freq):
                    images = accum_images[j]
                    texts = accum_texts[j]
                    with autocast():
                        model_out = model(images, texts)

                        inputs_no_accum = {}
                        inputs_no_accum["logit_scale"] = logit_scale = model_out.pop("logit_scale")
                        if "logit_bias" in model_out:
                            inputs_no_accum["logit_bias"] = model_out.pop("logit_bias")

                        inputs = {}
                        for key, val in accum_features.items():
                            accumulated = accum_features[key]
                            inputs[key] = torch.cat(accumulated[:j] + [model_out[key]] + accumulated[j + 1:])

                        losses = loss(**inputs, **inputs_no_accum,delta=args.delta, output_dict=True)
                        del inputs
                        del inputs_no_accum
                        total_loss = sum(losses.values())
                        losses["loss"] = total_loss

                    backward(total_loss, scaler)

            if scaler is not None:
                if args.horovod:
                    optimizer.synchronize()
                    scaler.unscale_(optimizer)
                    if args.grad_clip_norm is not None:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
                    with optimizer.skip_synchronize():
                        scaler.step(optimizer)
                else:
                    if args.grad_clip_norm is not None:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
                    scaler.step(optimizer)
                scaler.update()
            else:
                if args.grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
                optimizer.step()

            # reset gradient accum, if enabled
            if args.accum_freq > 1:
                accum_images, accum_texts, accum_features = [], [], {}

            # Note: we clamp to 4.6052 = ln(100), as in the original paper.
            with torch.no_grad():
                unwrap_model(model).logit_scale.clamp_(0, math.log(100))

            batch_time_m.update(time.time() - end)
            end = time.time()
            batch_count = i_accum + 1
            if is_master(args) and (i_accum % args.log_every_n_steps == 0 or batch_count == num_batches_per_epoch):
                batch_size = len(images)
                num_samples = batch_count * batch_size * args.accum_freq * args.world_size
                samples_per_epoch = dataloader.num_samples
                percent_complete = 100.0 * batch_count / num_batches_per_epoch

                # NOTE loss is coarsely sampled, just master node and per log update
                for key, val in losses.items():
                    if key not in losses_m:
                        losses_m[key] = AverageMeter()
                    losses_m[key].update(val.item(), batch_size)

                logit_scale_scalar = logit_scale.item()
                loss_log = " ".join(
                    [
                        f"{loss_name.capitalize()}: {loss_m.val:#.5g} ({loss_m.avg:#.5g})" 
                        for loss_name, loss_m in losses_m.items()
                    ]
                )
                samples_per_second = args.accum_freq * args.batch_size * args.world_size / batch_time_m.val
                samples_per_second_per_gpu = args.accum_freq * args.batch_size / batch_time_m.val
                logging.info(
                    f"Train Epoch: {epoch} [{num_samples:>{sample_digits}}/{samples_per_epoch} ({percent_complete:.0f}%)] "
                    f"Data (t): {data_time_m.avg:.3f} "
                    f"Batch (t): {batch_time_m.avg:.3f}, {samples_per_second:#g}/s, {samples_per_second_per_gpu:#g}/s/gpu "
                    f"LR: {optimizer.param_groups[0]['lr']:5f} "
                    f"Logit Scale: {logit_scale_scalar:.3f} " + loss_log
                )

                # Save train loss / etc. Using non avg meter values as loggers have their own smoothing
                log_data = {
                    "data_time": data_time_m.val,
                    "batch_time": batch_time_m.val,
                    "samples_per_second": samples_per_second,
                    "samples_per_second_per_gpu": samples_per_second_per_gpu,
                    "scale": logit_scale_scalar,
                    "lr": optimizer.param_groups[0]["lr"]
                }            
                log_data.update({name:val.val for name,val in losses_m.items()})

                log_data = {"train/" + name: val for name, val in log_data.items()}

                if tb_writer is not None:
                    for name, val in log_data.items():
                        tb_writer.add_scalar(name, val, step)
                
                if args.wandb:
                    assert wandb is not None, 'Please install wandb.'
                    log_data['step'] = step  # for backwards compatibility
                    wandb.log(log_data, step=step)
                
                # resetting batch / data time meters per log window
                batch_time_m.reset()
                data_time_m.reset()
        except Exception as e:
            # Catch any other unexpected errors during batch processing
            if is_master(args):
                logging.error(f"Unexpected error during batch processing: {str(e)}")
            raise
        
        i += 1
    # end while
    if skipped_volumes > 0 and is_master(args):
        logging.info(f"Training epoch {epoch}: Skipped {skipped_volumes} volumes due to normalization errors.")
    gc.collect()


def evaluate(model, data, epoch, args, tb_writer=None, tokenizer=None):
    metrics = {}
    if not is_master(args):
        return metrics
    device = torch.device(args.device)
    model.eval()

    zero_shot_metrics = zero_shot_eval(model, data, epoch, args, tokenizer=tokenizer)
    metrics.update(zero_shot_metrics)

    autocast = get_autocast(args.precision, device_type=device.type)
    input_dtype = get_input_dtype(args.precision)

    if 'val' in data and (args.val_frequency and ((epoch % args.val_frequency) == 0 or epoch == args.epochs)):
        dataloader = data['val'].dataloader
        num_samples = 0
        samples_per_val = dataloader.num_samples

        cumulative_loss = 0.0
        image_to_text_loss = 0.0
        text_to_image_loss = 0.0
        cumulative_gen_loss = 0.0
        all_image_features, all_text_features = [], []
        all_unique_labels = {}  
        all_labels_list = []
        all_labels_list_unique = []
        with torch.inference_mode():
            for i, batch in enumerate(dataloader):
                if args.distance:
                    images, texts, labels,echotime,repetitiontime = batch
                    echotime = echotime.to(device=device, non_blocking=True)
                    repetitiontime = repetitiontime.to(device=device, non_blocking=True)
                else:   
                    images, texts, labels = batch
                images = images.to(device=device, dtype=input_dtype, non_blocking=True)
                texts = texts.to(device=device, non_blocking=True)
                labels = labels.to(device=device, non_blocking=True)

                with autocast():
                    model_out = model(images, texts)
                    image_features = model_out["image_features"]
                    text_features = model_out["text_features"]
                    logit_scale = model_out["logit_scale"]
                    if args.metrics or (args.metrics is False and len(all_image_features) * args.batch_size < 25000):
                        all_image_features.append(image_features.cpu())
                        all_text_features.append(text_features.cpu())
                        for tokens in texts:
                            key = tuple(tokens.tolist()) if hasattr(tokens, "tolist") else tuple(tokens)
                            if key not in all_unique_labels:
                                all_unique_labels[key] = len(all_unique_labels)
                            all_labels_list_unique.append(all_unique_labels[key])
                        all_labels_list.extend([label.item() for label in labels])

                    logit_scale = logit_scale.mean()
                    logits_per_image = logit_scale * image_features @ text_features.t()
                    logits_per_text = logits_per_image.t()
                    batch_size = images.shape[0]
                    if args.multipositiveloss:
                        batch_labels = labels
                        pos_mask = torch.eq(batch_labels.unsqueeze(1), batch_labels.unsqueeze(0)).float().to(device)
                        loss_img = multi_positive_cross_entropy_loss(logits_per_image, pos_mask)
                        loss_txt = multi_positive_cross_entropy_loss(logits_per_text, pos_mask)
                        total_loss = args.delta * loss_img + (1 - args.delta) * loss_txt
                    else:
                        labels = torch.arange(batch_size, device=device).long()
                        loss_img = F.cross_entropy(logits_per_image, labels)
                        loss_txt = F.cross_entropy(logits_per_text, labels)
                        total_loss = args.delta * loss_img + (1 - args.delta) * loss_txt

                    gen_loss = maybe_compute_generative_loss(model_out)

                cumulative_loss += total_loss * batch_size
                image_to_text_loss += loss_img * batch_size
                text_to_image_loss += loss_txt * batch_size
                num_samples += batch_size
                if is_master(args) and (i % 100) == 0:
                    logging.info(
                        f"Eval Epoch: {epoch} [{num_samples} / {samples_per_val}]\t"
                        f"Clip Loss: {cumulative_loss / num_samples:.6f}\t")

                    if gen_loss is not None:
                        cumulative_gen_loss += gen_loss * batch_size
                        logging.info(
                            f"Generative Loss: {cumulative_gen_loss / num_samples:.6f}\t")
            if args.unique:
                val_metrics, vocabulary = get_clip_metrics(
                    image_features=torch.cat(all_image_features),
                    text_features=torch.cat(all_text_features),
                    logit_scale=logit_scale.cpu(),
                    ground_truth_general=all_labels_list,
                    ground_truth_unique=all_labels_list_unique,
                    trace=args.tracepreds
                )
                if args.tracepreds:
                    # Decode the vocabulary using the real tokens stored as keys in all_unique_labels
                    reverse_unique_labels = {v: k for k, v in all_unique_labels.items()}
                    decoded_vocabulary = {
                        name: {
                            values["anchor"]: {
                                #"anchor_decoded": tokenizer.decode(list(reverse_unique_labels[values["gt"]])),
                                "anchor": dataloader.dataset.captions[values["anchor"]],
                                #"decoded": [tokenizer.decode(list(reverse_unique_labels[val])) for val in values["labels"] if val in reverse_unique_labels],
                                "captions": [dataloader.dataset.captions[idx] for idx in values["indices"]],  # Add image file paths here
                                "labels": values["labels"],
                                "indices": values["indices"],
                                "gt":values["gt"],
                                "image_paths": [dataloader.dataset.images[idx] for idx in values["indices"]]  # Add image file paths here
                            }
                            for key, values in vocab.items()
                        }
                        for name, vocab in vocabulary.items()
                    }
                    # Load existing vocabulary if it exists
                    vocab_file_path = os.path.join(args.checkpoint_path, "vocabulary.json")
                    if os.path.exists(vocab_file_path):
                        with open(vocab_file_path, "r") as f:
                            existing_vocab = json.load(f)
                    else:
                        existing_vocab = {}

                    # Update existing vocabulary with new entries
                    for name, vocab in decoded_vocabulary.items():
                        if name in existing_vocab:
                            existing_vocab[name].update(vocab)
                        else:
                            existing_vocab[name] = vocab
                    existing_vocab["epoch"] = epoch
                    # Save the updated vocabulary to a JSON file
                    with open(vocab_file_path, "w") as f:
                        json.dump(existing_vocab, f, indent=4)

                    # Save images based on indices
                    image_save_path = os.path.join(args.checkpoint_path, f"epoch_{epoch}_images")
                    os.makedirs(image_save_path, exist_ok=True)
                    for name, vocab in vocabulary.items():
                        for key, values in vocab.items():
                            anchor_idx = values["anchor"]
                            anchor_image_path = dataloader.dataset.images[anchor_idx]
                            anchor_image = Image.open(anchor_image_path)
                            anchor_image_save_file = os.path.join(image_save_path, f"{name}_{key}_anchor_{anchor_idx}_GT_{values['gt']}.png")
                            anchor_image.save(anchor_image_save_file)
                            for idx, label in enumerate(values["labels"]):
                                image_path = dataloader.dataset.images[values['indices'][idx]]
                                image = Image.open(image_path)
                                image_save_file = os.path.join(image_save_path, f"{name}_{key}_anchor_{anchor_idx}_label_{label}_idx_{values['indices'][idx]}.png")
                                image.save(image_save_file)
            else:
                val_metrics = get_clip_metrics(
                    image_features=torch.cat(all_image_features),
                    text_features=torch.cat(all_text_features),
                    logit_scale=logit_scale.cpu(),
                    ground_truth_general=all_labels_list,
                )
            loss = cumulative_loss / num_samples
            image_loss_cumulative = image_to_text_loss / num_samples
            text_loss_cumulative = text_to_image_loss / num_samples
            metrics.update(
                {**val_metrics, "clip_val_loss": loss.item(), "image_to_text_loss": image_loss_cumulative.item(), "text_to_image_loss": text_loss_cumulative.item(), "epoch": epoch, "num_samples": num_samples}
            )

    if not metrics:
        return metrics

    logging.info(
        f"Eval Epoch: {epoch} "
        + "\t".join([f"{k}: {round(v, 4):.4f}" for k, v in metrics.items()])
    )

    log_data = {"val/" + name: val for name, val in metrics.items()}

    if args.save_logs:
        if tb_writer is not None:
            for name, val in log_data.items():
                tb_writer.add_scalar(name, val, epoch)

        with open(os.path.join(args.checkpoint_path, "results.jsonl"), "a+") as f:
            f.write(json.dumps(metrics))
            f.write("\n")

    if args.wandb:
        assert wandb is not None, 'Please install wandb.'
        if 'train' in data:
            dataloader = data['train'].dataloader
            num_batches_per_epoch = dataloader.num_batches // args.accum_freq
            step = num_batches_per_epoch * epoch
        else:
            step = None
        log_data['epoch'] = epoch
        wandb.log(log_data, step=step)

    return metrics


def get_clip_metrics(image_features, text_features, logit_scale, ground_truth_general, ground_truth_unique=None, trace=False):
    """
    Computes retrieval metrics for CLIP in a multi-class setting.

    Args:
        image_features (Tensor): (batch_size, feature_dim)
        text_features (Tensor): (batch_size, feature_dim)
        logit_scale (Tensor): Scaling factor for logits
        ground_truth (Tensor): Ground-truth labels (batch_size,)
        trace (bool): If True, returns a vocabulary with top 10 highest ranked ground_truth_unique values.

    Returns:
        dict: Dictionary with mean rank, median rank, Recall@K metrics, and optionally a vocabulary.
    """
    metrics = {}
    vocabulary = {}

    # Compute similarity logits
    logits_per_image = (logit_scale * image_features @ text_features.t()).detach().cpu()
    logits_per_text = logits_per_image.t().detach().cpu()
    logits = {"image_to_text": logits_per_image, "text_to_image": logits_per_text}

    for ground_truth in [ground_truth_general, ground_truth_unique]:
        if ground_truth is None:
            continue
        for name, logit in logits.items():
            ranking = torch.argsort(logit, descending=True)  # Sort predictions
            name = f"{name}_{'general' if ground_truth == ground_truth_general else 'unique'}"
            preds = []
            preds_mean = []
            for i, gt in enumerate(ground_truth):
                # Get all indices in batch that have the same label as the current ground truth
                gt_indices = (torch.tensor(ground_truth) == gt).nonzero(as_tuple=True)[0]  # Indices of same-class samples
                
                # Find the rank position of these correct samples
                rank_positions = torch.nonzero(torch.isin(ranking[i], gt_indices), as_tuple=True)[0]

                # Store the best (minimum) rank
                preds.append(rank_positions.min().item())  # Use the minimum rank for recall calculation
                preds_mean.append(rank_positions.float().mean().item())
            preds = np.array(preds)
            preds_mean = np.array(preds_mean)
            metrics[f"{name}_meanofmean_rank"] = preds_mean.mean() + 1
            metrics[f"{name}_mean_rank"] = preds.mean() + 1
            metrics[f"{name}_median_rank"] = np.floor(np.median(preds)) + 1

            for k in [1, 5, 10]:
                metrics[f"{name}_R@{k}"] = np.mean(preds < k)

            if trace and ground_truth is ground_truth_general:
                count = 0
                for i, gt in enumerate(ground_truth):
                    if count > 200:
                        break
                    count += 1
                    top_10_indices = ranking[i, :10].tolist()
                    if i in vocabulary.setdefault(name, {}):
                        vocabulary[name][i]["indices"].extend(top_10_indices)
                        vocabulary[name][i]["labels"].extend([ground_truth[j] for j in top_10_indices])
                    else:
                        vocabulary[name][i] = {
                            "anchor": i,
                            "gt": gt,
                            "indices": top_10_indices,
                            "labels": [ground_truth[j] for j in top_10_indices]
                        }

    if trace:
        return metrics, vocabulary
    return metrics


def maybe_compute_generative_loss(model_out):
    if "logits" in model_out and "labels" in model_out:
        token_logits = model_out["logits"]
        token_labels = model_out["labels"]
        return F.cross_entropy(token_logits.permute(0, 2, 1), token_labels)

def train_one_epoch_vision_only(model, data, loss, epoch, optimizer, scaler, scheduler, dist_model, args, tb_writer=None):
    device = torch.device(args.device)
    autocast = get_autocast(args.precision, device_type=device.type)
    input_dtype = get_input_dtype(args.precision)
    model.train()
    if args.distill:
        dist_model.eval()

    data['train'].set_epoch(epoch)  # set epoch in process safe manner via sampler or shared_epoch
    dataloader = data['train'].dataloader
    num_batches_per_epoch = dataloader.num_batches // args.accum_freq
    sample_digits = math.ceil(math.log(dataloader.num_samples + 1, 10))

    if args.accum_freq > 1:
        accum_images, accum_texts, accum_features = [], [], {}

    losses_m = {}
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    skipped_volumes = 0
    end = time.time()
    i = 0
    dataloader_iter = iter(dataloader)
    while True:
        try:
            batch = next(dataloader_iter)
        except StopIteration:
            break
        except VolumeNormalizationError as e:
            skipped_volumes += 1
            if is_master(args) and skipped_volumes % 10 == 0:
                logging.warning(f"Skipped {skipped_volumes} volumes due to normalization errors. Latest: {str(e)}")
            continue
        
        try:
            i_accum = i // args.accum_freq
            step = num_batches_per_epoch * epoch + i_accum

            if not args.skip_scheduler:
                scheduler(step)

            images, texts,labels = batch
            images = images.to(device=device, dtype=input_dtype, non_blocking=True)
            texts = texts.to(device=device, non_blocking=True)
            labels = labels.to(device=device, non_blocking=True)

            data_time_m.update(time.time() - end)
            optimizer.zero_grad()

            if args.accum_freq == 1:
                with autocast():
                    model_out = model(images)
                    logit_scale = model_out["logit_scale"]
                    if args.distill:
                        with torch.no_grad():
                            dist_model_out = dist_model(images)
                        model_out.update({f'dist_{k}': v for k, v in dist_model_out.items()})
                    if args.multipositiveloss:
                        losses = loss(**model_out, tokenized_texts=labels,output_dict=True)
                    else:
                        losses = loss(**model_out, output_dict=True)

                    total_loss = sum(losses.values())
                    losses["loss"] = total_loss

                backward(total_loss, scaler)
            else:
                # First, cache the features without any gradient tracking.
                with torch.no_grad():
                    with autocast():
                        model_out = model(images)

                        for f in ("logit_scale", "logit_bias"):
                            model_out.pop(f, None)

                        for key, val in model_out.items():
                            if key in accum_features:
                                accum_features[key].append(val)
                            else:
                                accum_features[key] = [val]

                    accum_images.append(images)
                    accum_texts.append(texts)

                # If (i + 1) % accum_freq is not zero, move on to the next batch.
                if ((i + 1) % args.accum_freq) > 0:
                    # FIXME this makes data time logging unreliable when accumulating
                    continue

                # Now, ready to take gradients for the last accum_freq batches.
                # Re-do the forward pass for those batches, and use the cached features from the other batches as negatives.
                # Call backwards each time, but only step optimizer at the end.
                optimizer.zero_grad()
                for j in range(args.accum_freq):
                    images = accum_images[j]
                    texts = accum_texts[j]
                    with autocast():
                        model_out = model(images)

                        inputs_no_accum = {}
                        inputs_no_accum["logit_scale"] = logit_scale = model_out.pop("logit_scale")
                        if "logit_bias" in model_out:
                            inputs_no_accum["logit_bias"] = model_out.pop("logit_bias")

                        inputs = {}
                        for key, val in accum_features.items():
                            accumulated = accum_features[key]
                            inputs[key] = torch.cat(accumulated[:j] + [model_out[key]] + accumulated[j + 1:])

                        losses = loss(**inputs, **inputs_no_accum, output_dict=True)
                        del inputs
                        del inputs_no_accum
                        total_loss = sum(losses.values())
                        losses["loss"] = total_loss

                    backward(total_loss, scaler)

            if scaler is not None:
                if args.horovod:
                    optimizer.synchronize()
                    scaler.unscale_(optimizer)
                    if args.grad_clip_norm is not None:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
                    with optimizer.skip_synchronize():
                        scaler.step(optimizer)
                else:
                    if args.grad_clip_norm is not None:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
                    scaler.step(optimizer)
                scaler.update()
            else:
                if args.grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
                optimizer.step()

            # reset gradient accum, if enabled
            if args.accum_freq > 1:
                accum_images, accum_texts, accum_features = [], [], {}

            # Note: we clamp to 4.6052 = ln(100), as in the original paper.
            with torch.no_grad():
                unwrap_model(model).logit_scale.clamp_(0, math.log(100))

            batch_time_m.update(time.time() - end)
            end = time.time()
            batch_count = i_accum + 1
            if is_master(args) and (i_accum % args.log_every_n_steps == 0 or batch_count == num_batches_per_epoch):
                batch_size = len(images)
                num_samples = batch_count * batch_size * args.accum_freq * args.world_size
                samples_per_epoch = dataloader.num_samples
                percent_complete = 100.0 * batch_count / num_batches_per_epoch

                # NOTE loss is coarsely sampled, just master node and per log update
                for key, val in losses.items():
                    if key not in losses_m:
                        losses_m[key] = AverageMeter()
                    losses_m[key].update(val.item(), batch_size)

                logit_scale_scalar = logit_scale.item()
                loss_log = " ".join(
                    [
                        f"{loss_name.capitalize()}: {loss_m.val:#.5g} ({loss_m.avg:#.5g})" 
                        for loss_name, loss_m in losses_m.items()
                    ]
                )
                samples_per_second = args.accum_freq * args.batch_size * args.world_size / batch_time_m.val
                samples_per_second_per_gpu = args.accum_freq * args.batch_size / batch_time_m.val
                logging.info(
                    f"Train Epoch: {epoch} [{num_samples:>{sample_digits}}/{samples_per_epoch} ({percent_complete:.0f}%)] "
                    f"Data (t): {data_time_m.avg:.3f} "
                    f"Batch (t): {batch_time_m.avg:.3f}, {samples_per_second:#g}/s, {samples_per_second_per_gpu:#g}/s/gpu "
                    f"LR: {optimizer.param_groups[0]['lr']:5f} "
                    f"Logit Scale: {logit_scale_scalar:.3f} " + loss_log
                )

                # Save train loss / etc. Using non avg meter values as loggers have their own smoothing
                log_data = {
                    "data_time": data_time_m.val,
                    "batch_time": batch_time_m.val,
                    "samples_per_second": samples_per_second,
                    "samples_per_second_per_gpu": samples_per_second_per_gpu,
                    "scale": logit_scale_scalar,
                    "lr": optimizer.param_groups[0]["lr"]
                }            
                log_data.update({name:val.val for name,val in losses_m.items()})

                log_data = {"train/" + name: val for name, val in log_data.items()}

                if tb_writer is not None:
                    for name, val in log_data.items():
                        tb_writer.add_scalar(name, val, step)
                
                if args.wandb:
                    assert wandb is not None, 'Please install wandb.'
                    log_data['step'] = step  # for backwards compatibility
                    wandb.log(log_data, step=step)
                
                # resetting batch / data time meters per log window
                batch_time_m.reset()
                data_time_m.reset()
        except Exception as e:
            # Catch any other unexpected errors during batch processing
            if is_master(args):
                logging.error(f"Unexpected error during batch processing: {str(e)}")
            raise
        
        i += 1
    # end while
    if skipped_volumes > 0 and is_master(args):
        logging.info(f"Training epoch {epoch}: Skipped {skipped_volumes} volumes due to normalization errors.")
    gc.collect()


def evaluate_vision_only(model, data, epoch, args, tb_writer=None, tokenizer=None):
    metrics = {}
    if not is_master(args):
        return metrics
    device = torch.device(args.device)
    model.eval()

    zero_shot_metrics = zero_shot_eval(model, data, epoch, args, tokenizer=tokenizer)
    metrics.update(zero_shot_metrics)

    autocast = get_autocast(args.precision, device_type=device.type)
    input_dtype = get_input_dtype(args.precision)

    if 'val' in data and (args.val_frequency and ((epoch % args.val_frequency) == 0 or epoch == args.epochs)):
        dataloader = data['val'].dataloader
        num_samples = 0
        samples_per_val = dataloader.num_samples

        # FIXME this does not scale past small eval datasets
        # all_image_features @ all_text_features will blow up memory and compute very quickly
        cumulative_loss = 0.0
        image_to_text_loss = 0.0
        cumulative_gen_loss = 0.0
        all_image_features = []
        all_unique_labels = {}
        all_labels_list = []
        all_labels_list_unique = []
        with torch.inference_mode():
            for i, batch in enumerate(dataloader):
                images, texts, labels = batch
                images = images.to(device=device, dtype=input_dtype, non_blocking=True)
                texts = texts.to(device=device, non_blocking=True)
                labels = labels.to(device=device, non_blocking=True)

                with autocast():
                    model_out = model(images)
                    image_features = model_out["image_features"]
                    logit_scale = model_out["logit_scale"]
                    # features are accumulated in CPU tensors, otherwise GPU memory exhausted quickly
                    # however, system RAM is easily exceeded and compute time becomes problematic
                    if args.metrics or (args.metrics is False and len(all_image_features)*args.batch_size < 10000):
                        all_image_features.append(image_features.cpu())
                        for tokens in texts:
                            # Convert tokenized representation to a tuple (using .tolist() if it's a tensor).
                            key = tuple(tokens.tolist()) if hasattr(tokens, "tolist") else tuple(tokens)
                            if key not in all_unique_labels:
                                all_unique_labels[key] = len(all_unique_labels)
                            all_labels_list_unique.append(all_unique_labels[key])  
                        all_labels_list.extend([label.item() for label in labels])  # extend the list with label items  # append as a list of label items

                    logit_scale = logit_scale.mean()
                    logits_per_image = logit_scale * image_features @ image_features.t()
                    batch_size = images.shape[0]
                    if args.multipositiveloss:
                        unique_labels = {}
                        labels_list = []
                        for tokens in texts:
                            # Convert tokenized representation to a tuple (using .tolist() if it's a tensor).
                            key = tuple(tokens.tolist()) if hasattr(tokens, "tolist") else tuple(tokens)
                            if key not in unique_labels:
                                unique_labels[key] = len(unique_labels)
                            labels_list.append(unique_labels[key])
                        batch_labels = torch.tensor(labels_list, device=device).long()
                        pos_mask = torch.eq(batch_labels.unsqueeze(1), batch_labels.unsqueeze(0)).float().to(device)
                        pos_mask.diagonal().zero_()
                        loss_img = multi_positive_cross_entropy_loss(logits_per_image, pos_mask)
                        total_loss = (loss_img ) 
                    else:                      
                        labels = torch.arange(batch_size, device=device).long()
                        loss_img = F.cross_entropy(logits_per_image, labels)
                        total_loss = loss_img
                         

                    gen_loss = maybe_compute_generative_loss(model_out)

                cumulative_loss += total_loss * batch_size
                image_to_text_loss += loss_img * batch_size
                num_samples += batch_size
                if is_master(args) and (i % 100) == 0:
                    logging.info(
                        f"Eval Epoch: {epoch} [{num_samples} / {samples_per_val}]\t"
                        f"Clip Loss: {cumulative_loss / num_samples:.6f}\t")

                    if gen_loss is not None:
                        cumulative_gen_loss += gen_loss * batch_size
                        logging.info(
                            f"Generative Loss: {cumulative_gen_loss / num_samples:.6f}\t")

            loss = cumulative_loss / num_samples
            image_loss_cumulative = image_to_text_loss / num_samples
            metrics.update(
                {"clip_val_loss": loss.item(),"image_to_text_loss": image_loss_cumulative.item(), "epoch": epoch, "num_samples": num_samples}
            )

    if not metrics:
        return metrics

    logging.info(
        f"Eval Epoch: {epoch} "
        + "\t".join([f"{k}: {round(v, 4):.4f}" for k, v in metrics.items()])
    )

    log_data = {"val/" + name: val for name, val in metrics.items()}

    if args.save_logs:
        if tb_writer is not None:
            for name, val in log_data.items():
                tb_writer.add_scalar(name, val, epoch)

        with open(os.path.join(args.checkpoint_path, "results.jsonl"), "a+") as f:
            f.write(json.dumps(metrics))
            f.write("\n")

    if args.wandb:
        assert wandb is not None, 'Please install wandb.'
        if 'train' in data:
            dataloader = data['train'].dataloader
            num_batches_per_epoch = dataloader.num_batches // args.accum_freq
            step = num_batches_per_epoch * epoch
        else:
            step = None
        log_data['epoch'] = epoch
        wandb.log(log_data, step=step)

    return metrics
import pickle

def test_metrics(model, data, start_epoch, args, tb_writer=None, tokenizer=None):
    """
    Computes overall retrieval metrics (image-to-text and text-to-image) in two separate blocks:
      Block 1: Precompute text features, then for each image compute its similarity to all texts.
      Block 2: Precompute image features, then for each text compute its similarity to all images.
    Large matrices are deleted after use to free memory.
    """
    metrics = {}
    if not is_master(args):
        return metrics
    device = torch.device(args.device)
    model.eval()

    # Mixed precision setup and proper input dtype.
    autocast = get_autocast(args.precision, device_type=device.type)
    input_dtype = get_input_dtype(args.precision)

    batch_size = args.batch_size
    if "val" not in data:
        return metrics

    dataloader = data["val"].dataloader
    dataset = dataloader.dataset
    num_samples = len(dataset.images)

    overall_metrics = {}
    vocabulary = {}

    # Compute logit_scale once.
    with torch.no_grad():
        with autocast():
            logit_scale = model.logit_scale.exp().detach().cpu()

    # -------------------- Block 1: Image-to-Text Metrics -------------------- #
    logging.info(
        "test_metrics: IMPORTANT — the default test metrics below use only UNIQUE caption texts "
        "(same labels but slightly different captions are removed). This usually makes results lower than "
        "from normal evaluation metrics."
    )
    # Precompute text features (NxD) in batches.
    # Global dictionary to track unique texts and their corresponding labels
    global_text_map = {}
    global_label_map = {}
    text_features_list = []
    # Always populated: used for general N×N metrics and save_embeddings.
    text_features_list_all = []
    with torch.no_grad():
        for i in range(0, num_samples, batch_size):
            # Initialize batch_texts with the captions for the current batch
            batch_texts = dataset.captions[i : i + batch_size]
            batch_labels = dataset.labels[i : i + batch_size]

            # Process batch and update global maps
            unique_indices_in_batch = []
            for bi, (text, label) in enumerate(zip(batch_texts, batch_labels)):
                if text not in global_text_map:
                    global_text_map[text] = len(global_text_map)  # Assign a unique index
                    global_label_map[text] = label
                    unique_indices_in_batch.append(bi)

            # Encode all batch captions once; derive deduped + full views from same forward pass.
            batch_tokenized = tokenizer(batch_texts).to(device)
            with autocast():
                batch_feats_all = model.encode_text(batch_tokenized)
            batch_feats_all = batch_feats_all.detach().cpu()
            text_features_list_all.append(batch_feats_all)
            if unique_indices_in_batch:
                text_features_list.append(batch_feats_all[unique_indices_in_batch])
    # Concatenate all text features
    text_features = torch.cat(text_features_list, dim=0)  # shape: (N, D)

    # Full per-row matrix for general N×N metrics (same candidate pool as evaluate()).
    text_features_full = (
        torch.cat(text_features_list_all, dim=0)
        if text_features_list_all
        else text_features
    )

    # Map global indices to filtered labels
    filtered_labels = [global_label_map[text] for text in global_text_map.keys()]
    dataset.filtered_labels = filtered_labels  # Save filtered labels separately
    logging.info(f"Filtered labels: {len(filtered_labels)}")
    logging.info(f"Unique texts: {text_features.size(0)}")

    i2t_ranks = []  # To store best rank for each image (0-indexed)
    count=0
    analysis_3d = {}
    with torch.no_grad():
        for img_idx in range(num_samples):
            # Use dataset's __getitem__ to get the image (transforms applied inside)
            image, _, _, _, _ = dataset[img_idx]
            image = image.unsqueeze(0).to(device, dtype=input_dtype)
            with autocast():
                image_feature = model.encode_image(image)  # shape: (1, D)

            # Compute similarity with all text features (NxD) → (1, N)
            sim_scores = logit_scale * (image_feature.cpu() @ text_features.t())
            sim_scores = sim_scores.squeeze(0)  # shape: (N,)
            ranking = torch.argsort(sim_scores, descending=True)

            # Get the original label for the image
            original_label = dataset.labels[img_idx]
            # Find the corresponding index in the filtered labels
            gt_indices = [j for j, lbl in enumerate(dataset.filtered_labels) if lbl == original_label]

            # If no matching label exists (e.g., due to filtering), skip this image
            if not gt_indices:
                continue

            # Convert gt_indices to a tensor and calculate the best rank
            gt_tensor = torch.tensor(gt_indices, device=ranking.device)
            mask = torch.isin(ranking, gt_tensor)
            top_10_indices = ranking[:10].tolist()
            if img_idx not in analysis_3d:
                analysis_3d[img_idx] = []

            # Append a new sample as a dictionary
            analysis_3d[img_idx].append({
                'filename': dataset.images[img_idx],
                'gt': original_label,
                'top_10_labels': [filtered_labels[j] for j in top_10_indices]
            })
            name='i2t'            
            if count < 200:                    
                count += 1
                
                if count in vocabulary.setdefault(name, {}):
                    vocabulary[name][count]["indices"].extend(top_10_indices)
                    vocabulary[name][count]["labels"].extend([filtered_labels[j] for j in top_10_indices])
                else:
                    vocabulary[name][count] = {
                        "anchor": img_idx,
                        "gt": original_label,
                        "indices": top_10_indices,
                        "labels": [filtered_labels[j] for j in top_10_indices]
                    }

            best_rank = torch.nonzero(mask, as_tuple=False).min().item()  # Best (lowest) rank
            i2t_ranks.append(best_rank)
    # Function to extract 3D image ID from filename
    def extract_3d_image_id(filename):
        """Extracts the 3D image identifier from the filename by removing slice number."""
        base_name = os.path.basename(filename)  # Get 'ur_sub-XXXXX_ses-YYYY_run-Z_MODALITY_stripped_axial_sliceNNN.png'
        parts = base_name.split('_')
        slice_part = parts[-1]  # "slice140.png"
        three_d_id = base_name.replace(f"_{slice_part}", "")  # Remove slice info
        return three_d_id

    # Group slices by 3D image
    grouped_analysis_3d = defaultdict(lambda: {'gt': None, 'slices': [], 'top_10_labels': []})

    for idx in analysis_3d:
        slice_data=analysis_3d[idx][0]
        filename = slice_data['filename']
        gt_label = slice_data['gt']
        top_10_labels = slice_data['top_10_labels']

        # Extract 3D image identifier
        three_d_id = extract_3d_image_id(filename)

        # Store ground truth once per 3D image
        if grouped_analysis_3d[three_d_id]['gt'] is None:
            grouped_analysis_3d[three_d_id]['gt'] = gt_label

        # Store slice predictions
        grouped_analysis_3d[three_d_id]['slices'].append(top_10_labels)

    # Compute performance metrics
    correct_all_votes = 0
    correct_first_label = 0
    correct_top_1_most_voted = 0  # New metric for top 1 most voted
    correct_top_5_most_voted = 0  # New metric for top 5 most voted
    correct_top_10_most_voted = 0  # New metric for top 10 most voted
    total_images = len(grouped_analysis_3d)

    # Store results per 3D image
    results = {}

    for three_d_id, data in grouped_analysis_3d.items():
        gt_label = data['gt']
        
        # Flatten all votes from all slices
        all_votes = [label for top_10 in data['slices'] for label in top_10]
        first_label_votes = [top_10[0] for top_10 in data['slices']]  # First label per slice

        # Majority vote for all votes
        all_vote_counts = Counter(all_votes)
        top_label_all, _ = all_vote_counts.most_common(1)[0]
        top_10_labels_all = [label for label, _ in all_vote_counts.most_common(10)]
        grouped_analysis_3d[three_d_id]['top_10_labels'] = top_10_labels_all
        top_5_labels_all = top_10_labels_all[:5]  # Top 5 most voted labels
        top_1_labels_all = top_10_labels_all[:1]  # Top 1 most voted label

        # Check if gt_label is in top 10, top 5, and top 1 most voted labels
        if gt_label in top_10_labels_all:
            correct_top_10_most_voted += 1
        if gt_label in top_5_labels_all:
            correct_top_5_most_voted += 1
        if gt_label in top_1_labels_all:
            correct_top_1_most_voted += 1

        # Check if gt_label is in top 10 labels (all votes approach)
        recall_success_all = gt_label in top_10_labels_all
        correct_all_votes += recall_success_all

        # Majority vote for first label approach
        first_vote_counts = Counter(first_label_votes)
        top_label_first, _ = first_vote_counts.most_common(1)[0]
        recall_success_first = (top_label_first == gt_label)
        correct_first_label += recall_success_first

    # Compute overall performance
    accuracy_all_votes = correct_all_votes / total_images
    accuracy_first_label = correct_first_label / total_images
    accuracy_top_1_most_voted = correct_top_1_most_voted / total_images
    accuracy_top_5_most_voted = correct_top_5_most_voted / total_images
    accuracy_top_10_most_voted = correct_top_10_most_voted / total_images

    # Print dataset-wide performance
    logging.info(f"Dataset Performance:")
    logging.info(f"  Accuracy (All Votes Approach): {accuracy_all_votes:.4f}")
    logging.info(f"  Accuracy (First Label Approach): {accuracy_first_label:.4f}")
    logging.info(f"  Accuracy (Top 1 Most Voted): {accuracy_top_1_most_voted:.4f}")
    logging.info(f"  Accuracy (Top 5 Most Voted): {accuracy_top_5_most_voted:.4f}")
    logging.info(f"  Accuracy (Top 10 Most Voted): {accuracy_top_10_most_voted:.4f}")
    json_file_path = os.path.join(args.checkpoint_path, "grouped_3d_analysis.json")
    with open(json_file_path, "w") as f:
        json.dump(grouped_analysis_3d, f, indent=4)
    # Aggregate image-to-text metrics.
    i2t_ranks = np.array(i2t_ranks)
    overall_metrics["image_to_text_mean_rank"] = i2t_ranks.mean() + 1  # convert to 1-indexed
    overall_metrics["image_to_text_median_rank"] = np.floor(np.median(i2t_ranks)) + 1
    for k in [1, 5, 10]:
        overall_metrics[f"image_to_text_R@{k}"] = np.mean(i2t_ranks < k)
    if args.save_embeddings:
        # Save text embeddings, labels, and filepaths
        # Save text embeddings, labels, and captions (filepaths)
        text_emb_dict = {
            "embeddings": text_features_full.numpy(),
            "labels": filtered_labels,
            "captions": list(global_text_map.keys())
        }
        # Save path for the pickle file
        save_path = os.path.join(args.checkpoint_path, "text_embeddings.pkl")

        # Write to file
        with open(save_path, "wb") as f:
            pickle.dump(text_emb_dict, f)

        print(f"Saved embeddings, labels, and captions to {save_path}")
        # Free text_features to reclaim memory.
        del text_features

    # -------------------- Block 2: Text-to-Image Metrics -------------------- #
    # Precompute image features (NxD) in batches.
    image_features_list = []
    with torch.no_grad():
        for i in range(0, num_samples, batch_size):
            batch_images = [dataset[j][0] for j in range(i, min(i + batch_size, num_samples))]
            batch_images = torch.stack(batch_images).to(device, dtype=input_dtype)
            with autocast():
                batch_feats = model.encode_image(batch_images)
            image_features_list.append(batch_feats.detach().cpu())
    image_features = torch.cat(image_features_list, dim=0)  # shape: (N, D)

    t2i_ranks = []  # to store best rank for each text (0-indexed)
    count=0 
    processed_texts = set()  # Set to track processed texts
    with torch.no_grad():
        for txt_idx in range(num_samples):
            if dataset.captions[txt_idx] not in processed_texts:
                processed_texts.add(dataset.captions[txt_idx])
                tokenized_text = tokenizer([dataset.captions[txt_idx]]).to(device)
                with autocast():
                    text_feature = model.encode_text(tokenized_text)  # shape: (1, D)
                sim_scores = logit_scale * (text_feature.cpu() @ image_features.t())
                sim_scores = sim_scores.squeeze(0)  # shape: (N,)
                ranking = torch.argsort(sim_scores, descending=True)
                
                gt_label = dataset.labels[txt_idx]
                gt_indices = [j for j, lbl in enumerate(dataset.labels) if lbl == gt_label]
                gt_tensor = torch.tensor(gt_indices, device=ranking.device)
                mask = torch.isin(ranking, gt_tensor)
                name='t2i'    
                    
                # if count < 500:                    
                #     count += 1
                #     top_10_indices = ranking[:10].tolist()
                #     if i in vocabulary.setdefault(name, {}):
                #         vocabulary[name][count]["indices"].extend(top_10_indices)
                #         vocabulary[name][count]["labels"].extend([dataset.labels[j] for j in top_10_indices])
                #     else:
                #         vocabulary[name][count] = {
                #             "anchor": txt_idx,
                #             "gt": gt_label,
                #             "indices": top_10_indices,
                #             "labels": [dataset.labels[j] for j in top_10_indices]
                #         }
                best_rank = torch.nonzero(mask, as_tuple=False).min().item()
                t2i_ranks.append(best_rank)
                if best_rank > 0:
                    count += 1
                    top_10_indices = ranking[:10].tolist()
                    if i in vocabulary.setdefault(name, {}):
                        vocabulary[name][count]["indices"].extend(top_10_indices)
                        vocabulary[name][count]["labels"].extend([dataset.labels[j] for j in top_10_indices])
                    else:
                        vocabulary[name][count] = {
                            "anchor": txt_idx,
                            "gt": gt_label,
                            "indices": top_10_indices,
                            "labels": [dataset.labels[j] for j in top_10_indices]
                        }
            #print(txt_idx, best_rank)

    # Aggregate text-to-image metrics.
    t2i_ranks = np.array(t2i_ranks)
    # Save t2i_ranks and i2t_ranks as NumPy arrays
    np.save(os.path.join(args.checkpoint_path, "t2i_ranks.npy"), t2i_ranks)
    np.save(os.path.join(args.checkpoint_path, "i2t_ranks.npy"), i2t_ranks)

    logging.info(f"Saved t2i_ranks to {os.path.join(args.checkpoint_path, 't2i_ranks.npy')}")
    logging.info(f"Saved i2t_ranks to {os.path.join(args.checkpoint_path, 'i2t_ranks.npy')}")
    overall_metrics["text_to_image_mean_rank"] = t2i_ranks.mean() + 1
    overall_metrics["text_to_image_median_rank"] = np.floor(np.median(t2i_ranks)) + 1
    for k in [1, 5, 10]:
        overall_metrics[f"text_to_image_R@{k}"] = np.mean(t2i_ranks < k)
    if args.save_embeddings:
        # Save image embeddings, labels, and filepaths
        image_emb_dict = {
            "embeddings": image_features.numpy(),
            "labels": [dataset.labels[i] for i in range(num_samples)],
            "filepaths": [dataset.images[i] for i in range(num_samples)]
        }
            # Save path for the pickle file
        save_path = os.path.join(args.checkpoint_path, "image_embeddings.pkl")

        # Write to file
        with open(save_path, "wb") as f:
            pickle.dump(image_emb_dict, f)

        print(f"Saved embeddings, labels, and captions to {save_path}")
    # -------------------- Block 3: General N×N retrieval metrics -------------------- #
    # Mirrors evaluate()'s `_general` protocol:
    #   - full per-row text pool (N rows, duplicates kept)
    #   - GT for row i: all rows with same class label
    logging.info(
        "test_metrics: IMPORTANT — now computing *_general_* metrics using ALL dataset rows "
        "(same labels but slightly different captions are kept). This is the protocol followed in the paper."
    )
    if num_samples > 0 and text_features_full.size(0) == num_samples:
        labels_list = [dataset.labels[k] for k in range(num_samples)]
        label_to_indices = defaultdict(list)
        for k, lbl in enumerate(labels_list):
            label_to_indices[lbl].append(k)

        i2t_general_ranks = []
        t2i_general_ranks = []
        with torch.no_grad():
            for k in range(num_samples):
                gt_indices = torch.tensor(label_to_indices[labels_list[k]], dtype=torch.long)
                if gt_indices.numel() == 0:
                    continue

                sim_i2t = (logit_scale * (image_features[k:k + 1] @ text_features_full.t())).squeeze(0)
                ranking_i2t = torch.argsort(sim_i2t, descending=True)
                rank_pos = torch.nonzero(torch.isin(ranking_i2t, gt_indices), as_tuple=True)[0]
                i2t_general_ranks.append(rank_pos.min().item())

                sim_t2i = (logit_scale * (text_features_full[k:k + 1] @ image_features.t())).squeeze(0)
                ranking_t2i = torch.argsort(sim_t2i, descending=True)
                rank_pos = torch.nonzero(torch.isin(ranking_t2i, gt_indices), as_tuple=True)[0]
                t2i_general_ranks.append(rank_pos.min().item())

        i2t_general_ranks = np.array(i2t_general_ranks)
        t2i_general_ranks = np.array(t2i_general_ranks)
        np.save(os.path.join(args.checkpoint_path, "i2t_general_ranks.npy"), i2t_general_ranks)
        np.save(os.path.join(args.checkpoint_path, "t2i_general_ranks.npy"), t2i_general_ranks)

        if i2t_general_ranks.size > 0:
            overall_metrics["image_to_text_general_mean_rank"] = float(i2t_general_ranks.mean() + 1)
            overall_metrics["image_to_text_general_median_rank"] = float(np.floor(np.median(i2t_general_ranks)) + 1)
            for k in [1, 5, 10]:
                overall_metrics[f"image_to_text_general_R@{k}"] = float(np.mean(i2t_general_ranks < k))

        if t2i_general_ranks.size > 0:
            overall_metrics["text_to_image_general_mean_rank"] = float(t2i_general_ranks.mean() + 1)
            overall_metrics["text_to_image_general_median_rank"] = float(np.floor(np.median(t2i_general_ranks)) + 1)
            for k in [1, 5, 10]:
                overall_metrics[f"text_to_image_general_R@{k}"] = float(np.mean(t2i_general_ranks < k))

    else:
        logging.warning(
            "test_metrics: skipping general N×N metrics — text_features_full rows=%s, num_samples=%s",
            text_features_full.size(0) if text_features_full is not None else None,
            num_samples,
        )

    # Free image/text features to reclaim memory.
    del image_features
    del text_features_full

    # -------------------- Logging and Saving Metrics -------------------- #
    if overall_metrics:
        logging.info("Test " + "\t".join([f"{k}: {round(v, 4):.4f}" for k, v in overall_metrics.items()]))
        log_data = {"test/" + name: val for name, val in overall_metrics.items()}
    if args.tracepreds:
        # Decode the vocabulary using the real tokens stored as keys in all_unique_labels
        #reverse_unique_labels = {v: k for k, v in all_unique_labels.items()}
        decoded_vocabulary = {
            name: {
                values["anchor"]: {
                    #"anchor_decoded": tokenizer.decode(list(reverse_unique_labels[values["gt"]])),
                    "anchor": dataloader.dataset.captions[values["anchor"]],
                    #"decoded": [tokenizer.decode(list(reverse_unique_labels[val])) for val in values["labels"] if val in reverse_unique_labels],
                    "captions": [dataloader.dataset.captions[idx] for idx in values["indices"]],  # Add image file paths here
                    "labels": values["labels"],
                    "indices": values["indices"],
                    "gt":values["gt"],
                    "image_paths": [dataloader.dataset.images[idx] for idx in values["indices"]]  # Add image file paths here
                }
                for key, values in vocab.items()
            }
            for name, vocab in vocabulary.items()
        }
        # Load existing vocabulary if it exists
        vocab_file_path = os.path.join(args.checkpoint_path, "vocabulary.json")
        if os.path.exists(vocab_file_path):
            with open(vocab_file_path, "r") as f:
                existing_vocab = json.load(f)
        else:
            existing_vocab = {}

        # Update existing vocabulary with new entries
        for name, vocab in decoded_vocabulary.items():
            if name in existing_vocab:
                existing_vocab[name].update(vocab)
            else:
                existing_vocab[name] = vocab
        existing_vocab["epoch"] = 'test'
        # Save the updated vocabulary to a JSON file
        with open(vocab_file_path, "w") as f:
            json.dump(existing_vocab, f, indent=4)

        # Save images based on indices
        image_save_path = os.path.join(args.checkpoint_path, f"test_images")
        os.makedirs(image_save_path, exist_ok=True)
        for name, vocab in vocabulary.items():
            for key, values in vocab.items():
                anchor_idx = values["anchor"]
                anchor_image_path = dataloader.dataset.images[anchor_idx]
                anchor_image = Image.open(anchor_image_path)
                anchor_image_save_file = os.path.join(image_save_path, f"{name}_{key}_anchor_{anchor_idx}_GT_{values['gt']}.png")
                anchor_image.save(anchor_image_save_file)
                for idx, label in enumerate(values["labels"]):
                    image_path = dataloader.dataset.images[values['indices'][idx]]
                    image = Image.open(image_path)
                    image_save_file = os.path.join(image_save_path, f"{name}_{key}_anchor_{anchor_idx}_label_{label}_idx_{values['indices'][idx]}.png")
                    image.save(image_save_file)
    return overall_metrics
