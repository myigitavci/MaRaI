from typing import Optional
import logging

import torch
import torch.nn as nn
from torch.nn import functional as F
from statsmodels.distributions.empirical_distribution import ECDF
try:
    import torch.distributed.nn
    from torch import distributed as dist

    has_distributed = True
except ImportError:
    has_distributed = False

try:
    import horovod.torch as hvd
except ImportError:
    hvd = None


def gather_features(
        image_features,
        text_features,
        local_loss=False,
        gather_with_grad=False,
        rank=0,
        world_size=1,
        use_horovod=False
):
    assert has_distributed, 'torch.distributed did not import correctly, please use a PyTorch version with support.'
    if use_horovod:
        assert hvd is not None, 'Please install horovod'
        if gather_with_grad:
            all_image_features = hvd.allgather(image_features)
            all_text_features = hvd.allgather(text_features)
        else:
            with torch.no_grad():
                all_image_features = hvd.allgather(image_features)
                all_text_features = hvd.allgather(text_features)
            if not local_loss:
                # ensure grads for local rank when all_* features don't have a gradient
                gathered_image_features = list(all_image_features.chunk(world_size, dim=0))
                gathered_text_features = list(all_text_features.chunk(world_size, dim=0))
                gathered_image_features[rank] = image_features
                gathered_text_features[rank] = text_features
                all_image_features = torch.cat(gathered_image_features, dim=0)
                all_text_features = torch.cat(gathered_text_features, dim=0)
    else:
        # We gather tensors from all gpus
        if gather_with_grad:
            all_image_features = torch.cat(torch.distributed.nn.all_gather(image_features), dim=0)
            all_text_features = torch.cat(torch.distributed.nn.all_gather(text_features), dim=0)
        else:
            gathered_image_features = [torch.zeros_like(image_features) for _ in range(world_size)]
            gathered_text_features = [torch.zeros_like(text_features) for _ in range(world_size)]
            dist.all_gather(gathered_image_features, image_features)
            dist.all_gather(gathered_text_features, text_features)
            if not local_loss:
                # ensure grads for local rank when all_* features don't have a gradient
                gathered_image_features[rank] = image_features
                gathered_text_features[rank] = text_features
            all_image_features = torch.cat(gathered_image_features, dim=0)
            all_text_features = torch.cat(gathered_text_features, dim=0)

    return all_image_features, all_text_features


class ClipLoss(nn.Module):

    def __init__(
            self,
            local_loss=False,
            gather_with_grad=False,
            cache_labels=False,
            rank=0,
            world_size=1,
            use_horovod=False,
    ):
        super().__init__()
        self.local_loss = local_loss
        self.gather_with_grad = gather_with_grad
        self.cache_labels = cache_labels
        self.rank = rank
        self.world_size = world_size
        self.use_horovod = use_horovod

        # cache state
        self.prev_num_logits = 0
        self.labels = {}

    def get_ground_truth(self, device, num_logits) -> torch.Tensor:
        # calculated ground-truth and cache if enabled
        if self.prev_num_logits != num_logits or device not in self.labels:
            labels = torch.arange(num_logits, device=device, dtype=torch.long)
            if self.world_size > 1 and self.local_loss:
                labels = labels + num_logits * self.rank
            if self.cache_labels:
                self.labels[device] = labels
                self.prev_num_logits = num_logits
        else:
            labels = self.labels[device]
        return labels

    def get_logits(self, image_features, text_features, logit_scale):
        if self.world_size > 1:
            all_image_features, all_text_features = gather_features(
                image_features,
                text_features,
                local_loss=self.local_loss,
                gather_with_grad=self.gather_with_grad,
                rank=self.rank,
                world_size=self.world_size,
                use_horovod=self.use_horovod,
            )

            if self.local_loss:
                logits_per_image = logit_scale * image_features @ all_text_features.T
                logits_per_text = logit_scale * text_features @ all_image_features.T
            else:
                logits_per_image = logit_scale * all_image_features @ all_text_features.T
                logits_per_text = logits_per_image.T
        else:
            logits_per_image = logit_scale * image_features @ text_features.T
            logits_per_text = logit_scale * text_features @ image_features.T
        
        return logits_per_image, logits_per_text

    def forward(self, image_features, text_features, logit_scale, output_dict=False):
        device = image_features.device
        logits_per_image, logits_per_text = self.get_logits(image_features, text_features, logit_scale)

        labels = self.get_ground_truth(device, logits_per_image.shape[0])

        total_loss = (
            F.cross_entropy(logits_per_image, labels) +
            F.cross_entropy(logits_per_text, labels)
        ) / 2

        return {"contrastive_loss": total_loss} if output_dict else total_loss


class CoCaLoss(ClipLoss):
    def __init__(
            self,
            caption_loss_weight,
            clip_loss_weight,
            pad_id=0,  # pad_token for open_clip custom tokenizer
            local_loss=False,
            gather_with_grad=False,
            cache_labels=False,
            rank=0,
            world_size=1,
            use_horovod=False,
    ):
        super().__init__(
            local_loss=local_loss,
            gather_with_grad=gather_with_grad,
            cache_labels=cache_labels,
            rank=rank,
            world_size=world_size,
            use_horovod=use_horovod
        )

        self.clip_loss_weight = clip_loss_weight
        self.caption_loss_weight = caption_loss_weight
        self.caption_loss = nn.CrossEntropyLoss(ignore_index=pad_id)

    def forward(self, image_features, text_features, logits, labels, logit_scale, output_dict=False):
        if self.clip_loss_weight:
            clip_loss = super().forward(image_features, text_features, logit_scale)
            clip_loss = self.clip_loss_weight * clip_loss
        else:
            clip_loss = torch.tensor(0, device=logits.device)

        caption_loss = self.caption_loss(
            logits.permute(0, 2, 1),
            labels,
        )
        caption_loss = caption_loss * self.caption_loss_weight

        if output_dict:
            return {"contrastive_loss": clip_loss, "caption_loss": caption_loss}

        return clip_loss, caption_loss


class DistillClipLoss(ClipLoss):

    def dist_loss(self, teacher_logits, student_logits):
        return -(teacher_logits.softmax(dim=1) * student_logits.log_softmax(dim=1)).sum(dim=1).mean(dim=0)

    def forward(
            self,
            image_features,
            text_features,
            logit_scale,
            dist_image_features,
            dist_text_features,
            dist_logit_scale,
            output_dict=False,
    ):
        logits_per_image, logits_per_text = \
            self.get_logits(image_features, text_features, logit_scale)

        dist_logits_per_image, dist_logits_per_text = \
            self.get_logits(dist_image_features, dist_text_features, dist_logit_scale)

        labels = self.get_ground_truth(image_features.device, logits_per_image.shape[0])

        contrastive_loss = (
            F.cross_entropy(logits_per_image, labels) +
            F.cross_entropy(logits_per_text, labels)
        ) / 2

        distill_loss = (
            self.dist_loss(dist_logits_per_image, logits_per_image) +
            self.dist_loss(dist_logits_per_text, logits_per_text)
        ) / 2

        if output_dict:
            return {"contrastive_loss": contrastive_loss, "distill_loss": distill_loss}

        return contrastive_loss, distill_loss


def neighbour_exchange(from_rank, to_rank, tensor, group=None):
    tensor_recv = torch.zeros_like(tensor)
    send_op = torch.distributed.P2POp(
        torch.distributed.isend,
        tensor,
        to_rank,
        group=group,
    )
    recv_op = torch.distributed.P2POp(
        torch.distributed.irecv,
        tensor_recv,
        from_rank,
        group=group,
    )
    reqs = torch.distributed.batch_isend_irecv([send_op, recv_op])
    for req in reqs:
        req.wait()
    return tensor_recv


def neighbour_exchange_bidir(left_rank, right_rank, tensor_to_left, tensor_to_right, group=None):
    tensor_from_left = torch.zeros_like(tensor_to_right)
    tensor_from_right = torch.zeros_like(tensor_to_left)
    send_op_left = torch.distributed.P2POp(
        torch.distributed.isend,
        tensor_to_left,
        left_rank,
        group=group,
    )
    send_op_right = torch.distributed.P2POp(
        torch.distributed.isend,
        tensor_to_right,
        right_rank,
        group=group,
    )
    recv_op_left = torch.distributed.P2POp(
        torch.distributed.irecv,
        tensor_from_left,
        left_rank,
        group=group,
    )
    recv_op_right = torch.distributed.P2POp(
        torch.distributed.irecv,
        tensor_from_right,
        right_rank,
        group=group,
    )
    reqs = torch.distributed.batch_isend_irecv([send_op_right, send_op_left, recv_op_right, recv_op_left])
    for req in reqs:
        req.wait()
    return tensor_from_right, tensor_from_left


class NeighbourExchange(torch.autograd.Function):
    @staticmethod
    def forward(ctx, from_rank, to_rank, group, tensor):
        ctx.group = group
        ctx.from_rank = from_rank
        ctx.to_rank = to_rank
        return neighbour_exchange(from_rank, to_rank, tensor, group=group)

    @staticmethod
    def backward(ctx, grad_output):
        return (None, None, None) + (NeighbourExchange.apply(ctx.to_rank, ctx.from_rank, ctx.group, grad_output),)


def neighbour_exchange_with_grad(from_rank, to_rank, tensor, group=None):
    return NeighbourExchange.apply(from_rank, to_rank, group, tensor)


class NeighbourExchangeBidir(torch.autograd.Function):
    @staticmethod
    def forward(ctx, left_rank, right_rank, group, tensor_to_left, tensor_to_right):
        ctx.group = group
        ctx.left_rank = left_rank
        ctx.right_rank = right_rank
        return neighbour_exchange_bidir(left_rank, right_rank, tensor_to_left, tensor_to_right, group=group)

    @staticmethod
    def backward(ctx, *grad_outputs):
        return (None, None, None) + \
            NeighbourExchangeBidir.apply(ctx.right_rank, ctx.left_rank, ctx.group, *grad_outputs)


def neighbour_exchange_bidir_with_grad(left_rank, right_rank, tensor_to_left, tensor_to_right, group=None):
    return NeighbourExchangeBidir.apply(left_rank, right_rank, group, tensor_to_left, tensor_to_right)


class SigLipLoss(nn.Module):
    """ Sigmoid Loss for Language Image Pre-Training (SigLIP) - https://arxiv.org/abs/2303.15343

    @article{zhai2023sigmoid,
      title={Sigmoid loss for language image pre-training},
      author={Zhai, Xiaohua and Mustafa, Basil and Kolesnikov, Alexander and Beyer, Lucas},
      journal={arXiv preprint arXiv:2303.15343},
      year={2023}
    }
    """
    def __init__(
            self,
            cache_labels: bool = False,
            rank: int = 0,
            world_size: int = 1,
            dist_impl: Optional[str] = None,
    ):
        super().__init__()
        self.cache_labels = cache_labels
        self.rank = rank
        self.world_size = world_size
        self.dist_impl = dist_impl or 'bidir'  # default to bidir exchange for now, this will likely change
        assert self.dist_impl in ('bidir', 'shift', 'reduce', 'gather')

        # cache state FIXME cache not currently used, worthwhile?
        self.prev_num_logits = 0
        self.labels = {}

    def get_ground_truth(self, device, dtype, num_logits, negative_only=False) -> torch.Tensor:
        labels = -torch.ones((num_logits, num_logits), device=device, dtype=dtype)
        if not negative_only:
            labels = 2 * torch.eye(num_logits, device=device, dtype=dtype) + labels
        return labels

    def get_logits(self, image_features, text_features, logit_scale, logit_bias=None):
        logits = logit_scale * image_features @ text_features.T
        if logit_bias is not None:
            logits += logit_bias
        return logits

    def _loss(self, image_features, text_features, logit_scale, logit_bias=None, negative_only=False):
        logits = self.get_logits(image_features, text_features, logit_scale, logit_bias)
        labels = self.get_ground_truth(
            image_features.device,
            image_features.dtype,
            image_features.shape[0],
            negative_only=negative_only,
        )
        loss = -F.logsigmoid(labels * logits).sum() / image_features.shape[0]
        return loss

    def forward(self, image_features, text_features, logit_scale, logit_bias, output_dict=False):
        loss = self._loss(image_features, text_features, logit_scale, logit_bias)

        if self.world_size > 1:
            if self.dist_impl == 'bidir':
                right_rank = (self.rank + 1) % self.world_size
                left_rank = (self.rank - 1 + self.world_size) % self.world_size
                text_features_to_right = text_features_to_left = text_features
                num_bidir, remainder = divmod(self.world_size - 1, 2)
                for i in range(num_bidir):
                    text_features_recv = neighbour_exchange_bidir_with_grad(
                        left_rank,
                        right_rank,
                        text_features_to_left,
                        text_features_to_right,
                    )
                    for f in text_features_recv:
                        loss += self._loss(
                            image_features,
                            f,
                            logit_scale,
                            logit_bias,
                            negative_only=True,
                        )
                    text_features_to_left, text_features_to_right = text_features_recv

                if remainder:
                    text_features_recv = neighbour_exchange_with_grad(
                        left_rank,
                        right_rank,
                        text_features_to_right
                    )
                    loss += self._loss(
                        image_features,
                        text_features_recv,
                        logit_scale,
                        logit_bias,
                        negative_only=True,
                    )
            elif self.dist_impl == "shift":
                right_rank = (self.rank + 1) % self.world_size
                left_rank = (self.rank - 1 + self.world_size) % self.world_size
                text_features_to_right = text_features
                for i in range(self.world_size - 1):
                    text_features_from_left = neighbour_exchange_with_grad(
                        left_rank,
                        right_rank,
                        text_features_to_right,
                    )
                    loss += self._loss(
                        image_features,
                        text_features_from_left,
                        logit_scale,
                        logit_bias,
                        negative_only=True,
                    )
                    text_features_to_right = text_features_from_left
            elif self.dist_impl == "reduce":
                for i in range(self.world_size):
                    text_from_other = torch.distributed.nn.all_reduce(
                        text_features * (self.rank == i),
                        torch.distributed.ReduceOp.SUM,
                    )
                    loss += float(i != self.rank) * self._loss(
                        image_features,
                        text_from_other,
                        logit_scale,
                        logit_bias,
                        negative_only=True,
                    )
            elif self.dist_impl == "gather":
                all_text = torch.distributed.nn.all_gather(text_features)
                for i in range(self.world_size):
                    loss += float(i != self.rank) * self._loss(
                        image_features,
                        all_text[i],
                        logit_scale,
                        logit_bias,
                        negative_only=True,
                    )
            else:
                assert False

        return {"contrastive_loss": loss} if output_dict else loss
    
def gather_features_with_tokens(
        image_features,
        text_features,
        text_tokens=None,
        local_loss=False,
        gather_with_grad=False,
        rank=0,
        world_size=1,
        use_horovod=False
):
    assert has_distributed, 'torch.distributed did not import correctly, please use a PyTorch version with support.'
    if use_horovod:
        assert hvd is not None, 'Please install horovod'
        if gather_with_grad:
            all_image_features = hvd.allgather(image_features)
            all_text_features = hvd.allgather(text_features)
            all_text_tokens = hvd.allgather(text_tokens) if text_tokens is not None else None
        else:
            with torch.no_grad():
                all_image_features = hvd.allgather(image_features)
                all_text_features = hvd.allgather(text_features)
                all_text_tokens = hvd.allgather(text_tokens) if text_tokens is not None else None
            if not local_loss:
                # ensure grads for local rank when all_* features don't have a gradient
                gathered_image_features = list(all_image_features.chunk(world_size, dim=0))
                gathered_text_features = list(all_text_features.chunk(world_size, dim=0))
                gathered_image_features[rank] = image_features
                gathered_text_features[rank] = text_features
                all_image_features = torch.cat(gathered_image_features, dim=0)
                all_text_features = torch.cat(gathered_text_features, dim=0)
                if text_tokens is not None:
                    gathered_text_tokens = list(all_text_tokens.chunk(world_size, dim=0))
                    gathered_text_tokens[rank] = text_tokens
                    all_text_tokens = torch.cat(gathered_text_tokens, dim=0)
    else:
        # We gather tensors from all gpus
        if gather_with_grad:
            all_image_features = torch.cat(torch.distributed.nn.all_gather(image_features), dim=0)
            all_text_features = torch.cat(torch.distributed.nn.all_gather(text_features), dim=0)
            all_text_tokens = torch.cat(torch.distributed.nn.all_gather(text_tokens), dim=0) if text_tokens is not None else None
        else:
            gathered_image_features = [torch.zeros_like(image_features) for _ in range(world_size)]
            gathered_text_features = [torch.zeros_like(text_features) for _ in range(world_size)]
            dist.all_gather(gathered_image_features, image_features)
            dist.all_gather(gathered_text_features, text_features)
            if text_tokens is not None:
                gathered_text_tokens = [torch.zeros_like(text_tokens) for _ in range(world_size)]
                dist.all_gather(gathered_text_tokens, text_tokens)
            if not local_loss:
                # ensure grads for local rank when all_* features don't have a gradient
                gathered_image_features[rank] = image_features
                gathered_text_features[rank] = text_features
                if text_tokens is not None:
                    gathered_text_tokens[rank] = text_tokens
            all_image_features = torch.cat(gathered_image_features, dim=0)
            all_text_features = torch.cat(gathered_text_features, dim=0)
            if text_tokens is not None:
                all_text_tokens = torch.cat(gathered_text_tokens, dim=0)

    return all_image_features, all_text_features, all_text_tokens

def gather_features_with_echotime_repetitiontime(
        image_features,
        text_features,
        text_tokens=None,
        echotime=None,
        repetitiontime=None,
        local_loss=False,
        gather_with_grad=False,
        rank=0,
        world_size=1,
        use_horovod=False
):
    assert has_distributed, 'torch.distributed did not import correctly, please use a PyTorch version with support.'
    if use_horovod:
        assert hvd is not None, 'Please install horovod'
        if gather_with_grad:
            all_image_features = hvd.allgather(image_features)
            all_text_features = hvd.allgather(text_features)
            all_text_tokens = hvd.allgather(text_tokens) if text_tokens is not None else None
            all_echotime = hvd.allgather(echotime) if echotime is not None else None
            all_repetitiontime = hvd.allgather(repetitiontime) if repetitiontime is not None else None
        else:
            with torch.no_grad():
                all_image_features = hvd.allgather(image_features)
                all_text_features = hvd.allgather(text_features)
                all_text_tokens = hvd.allgather(text_tokens) if text_tokens is not None else None
                all_echotime = hvd.allgather(echotime) if echotime is not None else None
                all_repetitiontime = hvd.allgather(repetitiontime) if repetitiontime is not None else None
            if not local_loss:
                # ensure grads for local rank when all_* features don't have a gradient
                gathered_image_features = list(all_image_features.chunk(world_size, dim=0))
                gathered_text_features = list(all_text_features.chunk(world_size, dim=0))
                gathered_image_features[rank] = image_features
                gathered_text_features[rank] = text_features
                all_image_features = torch.cat(gathered_image_features, dim=0)
                all_text_features = torch.cat(gathered_text_features, dim=0)
                if text_tokens is not None:
                    gathered_text_tokens = list(all_text_tokens.chunk(world_size, dim=0))
                    gathered_text_tokens[rank] = text_tokens
                    all_text_tokens = torch.cat(gathered_text_tokens, dim=0)
                if echotime is not None:
                    gathered_echotime = list(all_echotime.chunk(world_size, dim=0))
                    gathered_echotime[rank] = echotime
                    all_echotime = torch.cat(gathered_echotime, dim=0)
                if repetitiontime is not None:
                    gathered_repetitiontime = list(all_repetitiontime.chunk(world_size, dim=0))
                    gathered_repetitiontime[rank] = repetitiontime
                    all_repetitiontime = torch.cat(gathered_repetitiontime, dim=0)
    else:
        # We gather tensors from all gpus
        if gather_with_grad:
            all_image_features = torch.cat(torch.distributed.nn.all_gather(image_features), dim=0)
            all_text_features = torch.cat(torch.distributed.nn.all_gather(text_features), dim=0)
            all_text_tokens = torch.cat(torch.distributed.nn.all_gather(text_tokens), dim=0) if text_tokens is not None else None
            all_echotime = torch.cat(torch.distributed.nn.all_gather(echotime), dim=0) if echotime is not None else None
            all_repetitiontime = torch.cat(torch.distributed.nn.all_gather(repetitiontime), dim=0) if repetitiontime is not None else None
        else:
            gathered_image_features = [torch.zeros_like(image_features) for _ in range(world_size)]
            gathered_text_features = [torch.zeros_like(text_features) for _ in range(world_size)]
            dist.all_gather(gathered_image_features, image_features)
            dist.all_gather(gathered_text_features, text_features)
            if text_tokens is not None:
                gathered_text_tokens = [torch.zeros_like(text_tokens) for _ in range(world_size)]
                dist.all_gather(gathered_text_tokens, text_tokens)
            if echotime is not None:
                gathered_echotime = [torch.zeros_like(echotime) for _ in range(world_size)]
                dist.all_gather(gathered_echotime, echotime)
            if repetitiontime is not None:
                gathered_repetitiontime = [torch.zeros_like(repetitiontime) for _ in range(world_size)]
                dist.all_gather(gathered_repetitiontime, repetitiontime)
            if not local_loss:
                # ensure grads for local rank when all_* features don't have a gradient
                gathered_image_features[rank] = image_features
                gathered_text_features[rank] = text_features
                if text_tokens is not None:
                    gathered_text_tokens[rank] = text_tokens
                if echotime is not None:
                    gathered_echotime[rank] = echotime
                if repetitiontime is not None:
                    gathered_repetitiontime[rank] = repetitiontime
            all_image_features = torch.cat(gathered_image_features, dim=0)
            all_text_features = torch.cat(gathered_text_features, dim=0)
            if text_tokens is not None:
                all_text_tokens = torch.cat(gathered_text_tokens, dim=0)
            if echotime is not None:
                all_echotime = torch.cat(gathered_echotime, dim=0)
            if repetitiontime is not None:
                all_repetitiontime = torch.cat(gathered_repetitiontime, dim=0)

    return all_image_features, all_text_features, all_text_tokens, all_echotime, all_repetitiontime

def multi_positive_cross_entropy_loss(logits, pos_mask):
    """
    Computes a cross-entropy loss with multiple positive targets.
    Normalizes by the number of positives to prevent class imbalance.
    """
    logits_max, _ = torch.max(logits, dim=1, keepdim=True)
    logits = logits - logits_max.detach()  # Numerical stability
    exp_logits = torch.exp(logits)

    # Sum over positive pairs
    pos_exp_sum = (exp_logits * pos_mask).sum(dim=1)

    # Sum over all pairs (denominator of softmax)
    all_exp_sum = exp_logits.sum(dim=1)

    # Compute per-sample loss
    loss_per_sample = -torch.log((pos_exp_sum / (all_exp_sum + 1e-12)) + 1e-12)

    # Normalize by number of positives per sample
    num_positives = pos_mask.sum(dim=1)  # Count of positives per row
    loss_per_sample = loss_per_sample / num_positives.clamp(min=1)  # Avoid division by zero

    return loss_per_sample.mean()


# New loss class that inherits from ClipLoss and uses multi-positive loss.
class MultiPositiveClipLoss(ClipLoss):
    def get_logits_custom(self, image_features, text_features, text_tokens, logit_scale):
        if self.world_size > 1:
            all_image_features, all_text_features, text_tokens2 = gather_features_with_tokens(
                image_features,
                text_features,
                text_tokens,
                local_loss=self.local_loss,
                gather_with_grad=self.gather_with_grad,
                rank=self.rank,
                world_size=self.world_size,
                use_horovod=self.use_horovod,
            )
            if self.local_loss:
                logits_per_image = logit_scale * image_features @ all_text_features.T
                logits_per_text = logit_scale * text_features @ all_image_features.T
            else:
                logits_per_image = logit_scale * all_image_features @ all_text_features.T
                logits_per_text = logits_per_image.T
            return logits_per_image, logits_per_text, text_tokens2
        else:
            logits_per_image = logit_scale * image_features @ text_features.T
            logits_per_text = logit_scale * text_features @ image_features.T
            return logits_per_image, logits_per_text, text_tokens

    def forward(self, image_features, text_features, logit_scale, delta=0.5, tokenized_texts=None, output_dict=False):
        """
        If tokenized_texts is provided, derive labels from the tokenized representations.
        Samples with identical tokenized texts get the same label.
        Then compute the multi-positive contrastive loss.
        """
        device = image_features.device
        # Get logits using the parent's get_logits (which takes into account gather_with_grad and local_loss)
        if self.world_size > 1:
            logits_per_image, logits_per_text, tokenized_texts2 = self.get_logits_custom(image_features, text_features, tokenized_texts, logit_scale)
        else:
            logits_per_image, logits_per_text = self.get_logits(image_features, text_features, logit_scale)
            tokenized_texts2 = None

        # Print shapes for debugging
        # print(f"image_features shape: {image_features.shape}")
        # print(f"text_features shape: {text_features.shape}")
        # print(f"logits_per_image shape: {logits_per_image.shape}")
        # print(f"logits_per_text shape: {logits_per_text.shape}")
        if tokenized_texts is not None:
            # print(f"tokenized_texts shape: {tokenized_texts.shape}")
            pass
        if tokenized_texts2 is not None:
            # print(f"tokenized_texts2 shape: {tokenized_texts2.shape}")
            pass

        # Derive labels from tokenized texts if provided.

        batch_labels = tokenized_texts
        pos_mask = torch.eq(batch_labels.unsqueeze(1), batch_labels.unsqueeze(0)).float().to(device)

        if tokenized_texts2 is not None:
            batch_labels_all = tokenized_texts2
            # Create positive mask: pos_mask[i, j] = 1 if samples i and j share the same label.
            pos_mask = torch.eq(batch_labels.unsqueeze(1), batch_labels_all.unsqueeze(0)).float().to(device)

        # Print shapes for debugging
        # print(f"batch_labels shape: {batch_labels.shape}")
        if tokenized_texts2 is not None:
            # print(f"batch_labels_all shape: {batch_labels_all.shape}")
            pass
        # print(f"pos_mask shape: {pos_mask.shape}")
        # Detach and move to CPU for printing
        # print(f"pos_mask: {pos_mask.detach().cpu()}")
        # print(f"tokenized_texts2: {tokenized_texts2.detach().cpu() if tokenized_texts2 is not None else None}")
        # print(f"tokenized_texts: {tokenized_texts.detach().cpu() if tokenized_texts is not None else None}")

        # Compute multi-positive loss for image-to-text and text-to-image.
        loss_img = multi_positive_cross_entropy_loss(logits_per_image, pos_mask)
        loss_txt = multi_positive_cross_entropy_loss(logits_per_text, pos_mask)
        total_loss = delta*loss_img + (1-delta)*loss_txt
        return {"multi contrastive_loss": total_loss} if output_dict else total_loss
def empirical_cdf_scaling_gpu(distances):
    """
    Scales a distance matrix between 0 and 1 using the empirical cumulative distribution function (ECDF).
    
    Args:
        distances (torch.Tensor): (n, n) pairwise distance matrix.

    Returns:
        torch.Tensor: (n, n) scaled distances in [0,1].
    """
    flat_distances = distances.flatten()  # Convert to 1D
    _, indices = torch.sort(flat_distances)  # Sort values
    ranks = torch.arange(len(flat_distances), dtype=torch.float, device=distances.device)
    ecdf_scaled = ranks / (len(flat_distances) - 1)  # Normalize ranks to [0,1]
    
    # Reconstruct the original shape
    scaled_distances = torch.zeros_like(flat_distances, dtype=torch.float)
    scaled_distances[indices] = ecdf_scaled  # Assign ECDF values in original order
    return scaled_distances.view(distances.shape)

def multi_positive_cross_entropy_loss_with_distance(logits, pos_mask, distance):
    """
    Computes a cross-entropy loss with multiple positive targets.
    Normalizes by the number of positives to prevent class imbalance.
    """
    #logging.info(f"Calculating ECDF")
    # dist_ecdf = empirical_cdf_scaling_gpu(distance)  # Use the GPU-based ECDF scaling

    # dist_ecdf = 2 * torch.clamp(dist_ecdf, 0, 2)
    # dist_ecdf = dist_ecdf * (torch.ones_like(pos_mask) - pos_mask)
    # logging.info(f"Logits: {logits[0]}")
    # logging.info(f"Distance: {distance[0]}")
    # logging.info(f"Dist ECDF: {dist_ecdf[0]}")

    dist_ecdf = distance * (torch.ones_like(pos_mask) - pos_mask)

    logits_max, _ = torch.max(logits+dist_ecdf, dim=1, keepdim=True)
    logits = logits - logits_max.detach()  # Numerical stability
    exp_logits = torch.exp(logits)

    # Sum over positive pairs
    pos_exp_sum = (exp_logits * pos_mask).sum(dim=1)

    # Sum over all pairs (denominator of softmax)
    all_exp_sum = exp_logits.sum(dim=1)

    # Compute per-sample loss
    loss_per_sample = -torch.log((pos_exp_sum / (all_exp_sum + 1e-12)) + 1e-12)

    # Normalize by number of positives per sample
    num_positives = pos_mask.sum(dim=1)  # Count of positives per row
    loss_per_sample = loss_per_sample / num_positives.clamp(min=1)  # Avoid division by zero

    return loss_per_sample.mean()

# New loss class that inherits from ClipLoss and uses multi-positive loss.
class MultiPositiveClipLossWithDistance(ClipLoss):
    def get_logits_custom(self, image_features, text_features, text_tokens, echotime, repetitiontime, logit_scale):
        if self.world_size > 1:
            all_image_features, all_text_features, text_tokens2, all_echotime, all_repetitiontime = gather_features_with_echotime_repetitiontime(
                image_features,
                text_features,
                text_tokens,
                echotime,
                repetitiontime,
                local_loss=self.local_loss,
                gather_with_grad=self.gather_with_grad,
                rank=self.rank,
                world_size=self.world_size,
                use_horovod=self.use_horovod,
            )
            if self.local_loss:
                logits_per_image = logit_scale * image_features @ all_text_features.T
                logits_per_text = logit_scale * text_features @ all_image_features.T
            else:
                logits_per_image = logit_scale * all_image_features @ all_text_features.T
                logits_per_text = logits_per_image.T
            return logits_per_image, logits_per_text, text_tokens2, all_echotime, all_repetitiontime
        else:
            logits_per_image = logit_scale * image_features @ text_features.T
            logits_per_text = logit_scale * text_features @ image_features.T
            return logits_per_image, logits_per_text, text_tokens, echotime, repetitiontime

    def forward(self, image_features, text_features, logit_scale,echotime, repetitiontime,  delta=0.5, tokenized_texts=None, output_dict=False):
        """
        If tokenized_texts is provided, derive labels from the tokenized representations.
        Samples with identical tokenized texts get the same label.
        Then compute the multi-positive contrastive loss.
        """
        device = image_features.device
        # Get logits using the parent's get_logits (which takes into account gather_with_grad and local_loss)
        if self.world_size > 1:
            logits_per_image, logits_per_text, tokenized_texts2, all_echotime, all_repetitiontime = self.get_logits_custom(
                image_features, text_features, tokenized_texts, echotime, repetitiontime, logit_scale)
        else:
            logits_per_image, logits_per_text, tokenized_texts2, all_echotime, all_repetitiontime = self.get_logits_custom(
                image_features, text_features, tokenized_texts, echotime, repetitiontime, logit_scale)
            tokenized_texts2 = None
        # Derive labels from tokenized texts if provided.

        batch_labels = tokenized_texts
        pos_mask = torch.eq(batch_labels.unsqueeze(1), batch_labels.unsqueeze(0)).float().to(device)

        # print(f"logits_per_image shape: {logits_per_image.shape}")
        if tokenized_texts2 is not None:
            batch_labels_all = tokenized_texts2
            # Create positive mask: pos_mask[i, j] = 1 if samples i and j share the same label.
            pos_mask = torch.eq(batch_labels.unsqueeze(1), batch_labels_all.unsqueeze(0)).float().to(device)
                # Print shapes for debugging
        #logging.info(f"Calculating distance")
        #distance=mahalanobis_distance_batchwise(echotime,repetitiontime,all_echotime,all_repetitiontime)
        distance=weighted_euclidean_distance_batchwise(echotime,repetitiontime,all_echotime,all_repetitiontime)
        # print(f"allechotime shape: {all_echotime.shape}")
        # print(f"all_repetitiontime shape: {all_repetitiontime.shape}")
        # print(f"pos_mask shape: {pos_mask.shape}")
        # print(f"distance shape: {distance.shape}")
        # Compute multi-positive loss for image-to-text and text-to-image.
        #logging.info(f"Calculating loss")
        loss_img = multi_positive_cross_entropy_loss_with_distance(logits_per_image, pos_mask,distance)
        loss_txt = multi_positive_cross_entropy_loss_with_distance(logits_per_text, pos_mask,distance)
        total_loss = delta*loss_img + (1-delta)*loss_txt
        return {"multi contrastive_loss": total_loss} if output_dict else total_loss
# New loss class that inherits from ClipLoss and uses multi-positive loss.
class MultiPositiveClipLossVisionOnly(MultiPositiveClipLoss):
    def get_logits_custom(self, image_features, text_features, text_tokens, logit_scale):
        if self.world_size > 1:
            all_image_features, all_text_features, text_tokens2 = gather_features_with_tokens(
                image_features,
                text_features,
                text_tokens,
                local_loss=self.local_loss,
                gather_with_grad=self.gather_with_grad,
                rank=self.rank,
                world_size=self.world_size,
                use_horovod=self.use_horovod,
            )
            if self.local_loss:
                logits_per_image = logit_scale * image_features @ all_text_features.T
            else:
                logits_per_image = logit_scale * all_image_features @ all_text_features.T
            return logits_per_image, text_tokens2
        else:
            logits_per_image = logit_scale * image_features @ text_features.T
            return logits_per_image, text_tokens
        
    def forward(self, image_features, logit_scale, tokenized_texts=None, output_dict=False):
        """
        If tokenized_texts is provided, derive labels from the tokenized representations.
        Samples with identical tokenized texts get the same label.
        Then compute the multi-positive contrastive loss.
        """
        device = image_features.device
        # Get logits using the parent's get_logits (which takes into account gather_with_grad and local_loss)
        if self.world_size > 1:
            logits_per_image, tokenized_texts2 = self.get_logits_custom(image_features, image_features, tokenized_texts, logit_scale)
        else:
            logits_per_image, _ = self.get_logits(image_features, image_features, logit_scale)
            tokenized_texts2 = None

        # Print shapes for debugging
        # print(f"image_features shape: {image_features.shape}")
        # print(f"text_features shape: {text_features.shape}")
        # print(f"logits_per_image shape: {logits_per_image.shape}")

        batch_labels = torch.tensor(tokenized_texts, device=device).long()
        pos_mask = torch.eq(batch_labels.unsqueeze(1), batch_labels.unsqueeze(0)).float().to(device)

        if tokenized_texts2 is not None:
            batch_labels_all = torch.tensor(tokenized_texts2, device=device).long()
            # Create positive mask: pos_mask[i, j] = 1 if samples i and j share the same label.
            pos_mask = torch.eq(batch_labels.unsqueeze(1), batch_labels_all.unsqueeze(0)).float().to(device)
            
        # Print shapes for debugging
        # print(f"batch_labels shape: {batch_labels.shape}")
        if tokenized_texts2 is not None:
            # print(f"batch_labels_all shape: {batch_labels_all.shape}")
            pass
        # print(f"pos_mask shape: {pos_mask.shape}")
        # Detach and move to CPU for printing
        # print(f"pos_mask: {pos_mask.detach().cpu()}")
        # print(f"tokenized_texts2: {tokenized_texts2.detach().cpu() if tokenized_texts2 is not None else None}")
        # print(f"tokenized_texts: {tokenized_texts.detach().cpu() if tokenized_texts is not None else None}")
        pos_mask.diagonal().zero_()
        # Compute multi-positive loss for image-to-text and text-to-image.
        loss_img = multi_positive_cross_entropy_loss(logits_per_image, pos_mask)
        total_loss = (loss_img )
        return {"multi contrastive_loss": total_loss} if output_dict else total_loss

class MultiPositiveClipLosswithVision(ClipLoss):
    def get_logits_custom(self, image_features, text_features, text_tokens, logit_scale):
        if self.world_size > 1:
            all_image_features, all_text_features, text_tokens2 = gather_features_with_tokens(
                image_features,
                text_features,
                text_tokens,
                local_loss=self.local_loss,
                gather_with_grad=self.gather_with_grad,
                rank=self.rank,
                world_size=self.world_size,
                use_horovod=self.use_horovod,
            )
            if self.local_loss:
                logits_per_image = logit_scale * image_features @ all_text_features.T
                logits_per_text = logit_scale * text_features @ all_image_features.T
            else:
                logits_per_image = logit_scale * all_image_features @ all_text_features.T
                logits_per_text = logits_per_image.T
            return logits_per_image, logits_per_text, text_tokens2
        else:
            logits_per_image = logit_scale * image_features @ text_features.T
            logits_per_text = logit_scale * text_features @ image_features.T
            return logits_per_image, logits_per_text, text_tokens

    def forward(self, image_features, text_features, logit_scale, lam=0.3, tokenized_texts=None, output_dict=False):
        """
        If tokenized_texts is provided, derive labels from the tokenized representations.
        Samples with identical tokenized texts get the same label.
        Then compute the multi-positive contrastive loss.
        """
        device = image_features.device
        # Get logits using the parent's get_logits (which takes into account gather_with_grad and local_loss)
        if self.world_size > 1:
            logits_per_image, logits_per_text, tokenized_texts2 = self.get_logits_custom(image_features, text_features, tokenized_texts, logit_scale)
            logits_per_image_image, tokenized_texts2 = MultiPositiveClipLossVisionOnly.get_logits_custom(image_features, image_features, tokenized_texts, logit_scale)
        else:
            logits_per_image, logits_per_text = self.get_logits(image_features, text_features, logit_scale)
            logits_per_image_image, _ = self.get_logits(image_features, image_features, logit_scale)

            tokenized_texts2 = None

        # Print shapes for debugging
        # print(f"image_features shape: {image_features.shape}")
        # print(f"text_features shape: {text_features.shape}")
        # print(f"logits_per_image shape: {logits_per_image.shape}")
        # print(f"logits_per_text shape: {logits_per_text.shape}")
        if tokenized_texts is not None:
            # print(f"tokenized_texts shape: {tokenized_texts.shape}")
            pass
        if tokenized_texts2 is not None:
            # print(f"tokenized_texts2 shape: {tokenized_texts2.shape}")
            pass

        # Derive labels from tokenized texts if provided.
        batch_labels = torch.tensor(tokenized_texts, device=device).long()
        pos_mask = torch.eq(batch_labels.unsqueeze(1), batch_labels.unsqueeze(0)).float().to(device)

        if tokenized_texts2 is not None:
            batch_labels_all = torch.tensor(tokenized_texts2, device=device).long()
            # Create positive mask: pos_mask[i, j] = 1 if samples i and j share the same label.
            pos_mask = torch.eq(batch_labels.unsqueeze(1), batch_labels_all.unsqueeze(0)).float().to(device)

        # Print shapes for debugging
        # print(f"batch_labels shape: {batch_labels.shape}")
        if tokenized_texts2 is not None:
            # print(f"batch_labels_all shape: {batch_labels_all.shape}")
            pass
        # print(f"pos_mask shape: {pos_mask.shape}")
        # Detach and move to CPU for printing
        # print(f"pos_mask: {pos_mask.detach().cpu()}")
        # print(f"tokenized_texts2: {tokenized_texts2.detach().cpu() if tokenized_texts2 is not None else None}")
        # print(f"tokenized_texts: {tokenized_texts.detach().cpu() if tokenized_texts is not None else None}")

        # Compute multi-positive loss for image-to-text and text-to-image.
        loss_img = multi_positive_cross_entropy_loss(logits_per_image, pos_mask)
        loss_txt = multi_positive_cross_entropy_loss(logits_per_text, pos_mask)
        
        pos_mask.diagonal().zero_()
        loss_img_to_img = multi_positive_cross_entropy_loss(logits_per_image_image, pos_mask)

        #total_loss = (loss_img + loss_txt)/2 + lam*loss_img_to_img
        return {"loss_img": loss_img,"loss_txt": loss_txt,"loss_img_to_img": loss_img_to_img} if output_dict else loss_img,loss_txt,loss_img_to_img

def weighted_euclidean_distance_batchwise(te, tr, all_te, all_tr, w_te=0.2, w_tr=10.0):
    """
    Computes pairwise weighted Euclidean distances between local batch and all gathered batches.
    
    Args:
        te (torch.Tensor): (local_batch_size,) tensor of local echo times.
        tr (torch.Tensor): (local_batch_size,) tensor of local repetition times.
        all_te (torch.Tensor): (global_batch_size,) tensor of all gathered echo times.
        all_tr (torch.Tensor): (global_batch_size,) tensor of all gathered repetition times.
        w_te (float): Weight for echo time differences.
        w_tr (float): Weight for repetition time differences.
    
    Returns:
        torch.Tensor: (local_batch_size, global_batch_size) distance matrix.
    """
    te_diff = te[:, None] - all_te[None, :]  # Compute pairwise differences
    tr_diff = tr[:, None] - all_tr[None, :]
    
    distances = torch.sqrt(te_diff**2 / w_te + tr_diff**2 / w_tr)  # Compute normalized Euclidean distance
    return distances

def mahalanobis_distance_batchwise(te, tr, all_te, all_tr, eps=1e-6):
    """
    Computes Mahalanobis distance between local batch and all gathered batches.
    
    Args:
        te (torch.Tensor): (local_batch_size,) tensor of local echo times.
        tr (torch.Tensor): (local_batch_size,) tensor of local repetition times.
        all_te (torch.Tensor): (global_batch_size,) tensor of all gathered echo times.
        all_tr (torch.Tensor): (global_batch_size,) tensor of all gathered repetition times.
        eps (float): Small value for numerical stability.
    
    Returns:
        torch.Tensor: (local_batch_size, global_batch_size) Mahalanobis distance matrix.
    """
    local_X = torch.stack([te, tr], dim=1)  # Shape (local_batch_size, 2)
    global_X = torch.stack([all_te, all_tr], dim=1)  # Shape (global_batch_size, 2)

    # Compute covariance matrix of the global batch
    cov_matrix = torch.cov(global_X.T) + eps * torch.eye(2, device=global_X.device)  # Regularization
    inv_cov = torch.linalg.inv(cov_matrix)

    # Compute pairwise differences
    diffs = local_X[:, None, :] - global_X[None, :, :]  # Shape (local_batch_size, global_batch_size, 2)

    # Compute Mahalanobis distance
    distances = torch.sqrt(torch.einsum('bij,jk,bik->bi', diffs, inv_cov, diffs))
    
    return distances
