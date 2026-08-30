# mp.set_start_method('spawn')
from copy import copy, deepcopy
from itertools import product
from pathlib import Path
import time
import os
import resource
import math

import torch
import multiprocessing as mp
from torch import nn
import numpy as np


import efficient_cross_entropy_modules
import modules


class FusedProjectionPlusCrossEntropyLoss(nn.Module):
    # Wraps the class from https://github.com/mgmalek/efficient_cross_entropy/blob/main/modules.py
    def __init__(self, dim: int, n_classes: int, **kwargs):
        super().__init__()
        dtype = kwargs.pop("dtype")
        device = kwargs.pop("device")
        # assert kwargs.pop("bias") == False
        assert kwargs.pop("label_smoothing") == 0.0
        module = efficient_cross_entropy_modules.FusedProjectionPlusCrossEntropyLoss(
            dim, n_classes, **kwargs
        )
        if device == "cuda":
            module = module.cuda()
        if dtype is not None:
            module = module.to(dtype)
        self.module = module

    def forward(self, x, targ):
        assert targ.dtype == torch.int64, targ.dtype
        return self.module(x, targ)


class PyTorchProjectionPlusCrossEntropyLoss(nn.Module):
    # Same as https://github.com/mgmalek/efficient_cross_entropy/blob/main/modules.py
    # Used as a reference to LinearCrossEntropyLoss benchmarks.
    # Not implemented features: loss_dims, weight

    def __init__(
        self,
        dim: int,
        n_classes: int,
        ignore_index: int = -100,
        reduction: str = "mean",
        label_smoothing: float = 0.0,
        # bias: bool = True,
        device=None,
        dtype=None,
    ):
        super().__init__()
        bias = False
        self.proj = nn.Linear(dim, n_classes, bias=bias, device=device, dtype=dtype)
        self.loss = nn.CrossEntropyLoss(
            ignore_index=ignore_index,
            reduction=reduction,
            label_smoothing=label_smoothing,
        )

    def forward(self, x, targ):
        logits = self.proj(x)
        return self.loss(logits, targ)


config_defaults = dict(
    num_classes=[32768, 8192][1], num_tokens=8192, in_features=[2048, 4096, 8192][1]
)


def pair2str(key, value):
    if key == "max_numel" and isinstance(value, int):
        e = math.log(value, 2)
        if e.is_integer():
            e = int(e)
            return f"{key}=2**{e}"
    return f"{key}={value}"


def configs(default_force=False):
    params = dict(
        dtype=[torch.float32, torch.bfloat16, torch.float16][-1:],
        token_dtype=[torch.long, torch.float32][:1],
        device=["cuda", "cpu"][:1],
        in_features=[
            1024,
            2048,
            4096,
            8192,
            (8192 + 16384) // 2,
            16384,
            16384 + 4096,
            16384 + 2 * 4096,
        ][:-3],
        num_classes=[
            2048,
            4096,
            8192,
            16384,
            (16384 + 32768) // 2,
            32768,
            32768 + 8192,
            32768 + 2 * 8192,
            65536,
        ][:-2],
        num_tokens=[
            1024,
            2048,
            4096,
            8192,
            16384,
            (16384 + 32768) // 2,
            *(32768 + k * 8192 for k in range(10))
        ][:7],
        bias=[False],
        reduction=["mean", "sum"][:1],
        ignore_index=[-100],
        label_smoothing=[0.0],
        module_extra_kwargs=[
            # *((modules.MyLinearCrossEntropyLoss, dict(force=False, splits=(i, j, 1))) for i in [1,2,4,8] for j in [1,2,4,8]),
            # *((modules.MyLinearCrossEntropyLoss, dict(force=not True, splits=(None, j, k * 1024))) for j in [1,2,4,8] for k in [1, 2, 4, 8]),
            # *((modules.MyLinearCrossEntropyLoss, dict(force=not True, splits=(i, None, k * 1024))) for i in [1,2,4,8] for k in [1, 2, 4, 8]),
            # *((modules.MyLinearCrossEntropyLoss, dict(force=not True, splits=(None, None, k * 1024))) for k in [1, 2, 4, 8, 16, 32]),
            # *((modules.MyLinearCrossEntropyLoss, dict(force=not True, splits=(None, None, k * 1024), inplace_grad=True)) for k in [1, 2, 4, 8, 16, 32]),
            # (modules.MyLinearCrossEntropyLoss, dict(force=not True, splits=(None, None, None))),
            *(
                (
                    modules.MyLinearCrossEntropyLoss,
                    dict(
                        force=not True,
                        params=dict(
                            chunk_size=512 * k,
                            grad_inplace=True,
                            chunks_classes=chunks_classes,
                        ),
                    ),
                )
                for k in [1, 2, 4, 8]
                for chunks_classes in [1, 2][:0]
            ),
            *(
                (
                    modules.MyLinearCrossEntropyLoss,
                    dict(
                        force=not True,
                        params=dict(
                            max_numel=(1024 * k) ** 2,
                            grad_inplace=True,
                        ),
                    ),
                )
                for k in [1, 2, 4, 8, 16][:0]
            ),
            *(
                (
                    modules.MyLinearCrossEntropyLoss,
                    dict(
                        force=not True,
                        params=dict(
                            grad_inplace=True,
                            max_memory_gb=0.5 * k,
                            min_chunk_size=min_chunk_size,
                        ),
                    ),
                )
                for k in [1, 2, 3, 4, 5, 6][1:2]
                for min_chunk_size in [256, 512, 1024][:0]
            ),
            # (modules.MyLinearCrossEntropyLoss, dict(force=True, num_chunks=2)),
            # (modules.MyLinearCrossEntropyLoss, dict(force=True, num_chunks=4)),
            # (modules.MyLinearCrossEntropyLoss, dict(force=True, num_chunks=8)),
            # (modules.MyLinearCrossEntropyLoss, dict(force=True, num_chunks=16)),
            # (modules.MyLinearCrossEntropyLoss2, dict(force=True)),
            #(nn.LinearCrossEntropyLoss, dict(options=None)),
            #(PyTorchProjectionPlusCrossEntropyLoss, dict()),
            # (FusedProjectionPlusCrossEntropyLoss, dict(n_loop_iters=1)),
            # (FusedProjectionPlusCrossEntropyLoss, dict(n_loop_iters=2)),
            # (FusedProjectionPlusCrossEntropyLoss, dict(n_loop_iters=4)),
            # (FusedProjectionPlusCrossEntropyLoss, dict(n_loop_iters=8)),
            # (FusedProjectionPlusCrossEntropyLoss, dict(n_loop_iters=16)),
            # (modules.VoLinearCrossEntropyLoss, dict()),
            (modules.LiLinearCrossEntropyLoss, dict(force=not True)),
            #(modules.NewLinearCrossEntropyLoss, dict(force=not True, options=torch.nn.functional.LinearCrossEntropyOptions(grad_inplace=True, chunking_method="liger"))),
            #(modules.NewLinearCrossEntropyLoss, dict(force=not True, options=torch.nn.functional.LinearCrossEntropyOptions(grad_inplace=False, chunking_method="liger"))),
            #(modules.NewLinearCrossEntropyLoss, dict(force=not True, options=torch.nn.functional.LinearCrossEntropyOptions())),
            #(modules.NewLinearCrossEntropyLoss, dict(force=not True, options=torch.nn.functional.LinearCrossEntropyOptions(grad_inplace=True))),
            #*((modules.NewLinearCrossEntropyLoss, dict(force=not True, options=torch.nn.functional.LinearCrossEntropyOptions(grad_inplace=True, batch_chunk_size=1024 * k ))) for k in [1, 2, 4]),
            #*((modules.NewLinearCrossEntropyLoss, dict(force=not True, options=torch.nn.functional.LinearCrossEntropyOptions(grad_inplace=True, batch_chunk_size=1024 * k ))) for k in [1, 2, 4]),
            *((modules.NewLinearCrossEntropyLoss, dict(
                force=not True,
                options=torch.nn.LinearCrossEntropyOptions(
                acc_dtype=torch.float32, acc_policy=a
            ))) for a in ["auto", "accurate", "compact"][1:2]),
            # (modules.NewLinearCrossEntropyLoss, dict(force=not True, options=dict(max_memory_gb=1))),
            # (modules.NewLinearCrossEntropyLoss,dict(force=not True, options=dict(max_memory_gb=1.5, grad_inplace=True))),
            # (modules.NewLinearCrossEntropyLoss, dict(force=not True, options=dict(max_memory_gb=1, grad_inplace=True, features_chunk_size=8192))),
        ],
    )

    for field in config_defaults:
        config_ = dict(
            (k, (params[k] if field == k else config_defaults[k]))
            for k in config_defaults
        )
        names, values_lst = zip(
            *[(name, value) for name, value in params.items() if name not in config_]
        )
        for values in product(*values_lst):
            config = copy(config_)
            for i, name in enumerate(names):
                config[name] = values[i]
            key = [k for k, v in config.items() if isinstance(v, list)][0]
            for value in config[key]:
                kwargs = deepcopy(config)
                cls, extra_kwargs = kwargs.pop("module_extra_kwargs")
                extra_kwargs.pop("bias", None)
                force = extra_kwargs.pop("force", default_force)
                num_tokens = kwargs.pop("num_tokens")
                token_dtype = kwargs.pop("token_dtype")
                skip_keys = ["params"]
                if "options" in extra_kwargs and extra_kwargs["options"] is None:
                    params_ = extra_kwargs.get(
                        "params", dict()
                    )  # leave `options=None` alone
                else:
                    params_ = extra_kwargs.get(
                        "options", extra_kwargs.get("params", dict())
                    )
                    if isinstance(params_, torch.nn.LinearCrossEntropyOptions):
                        import dataclasses
                        params_ = dataclasses.asdict(params_)
                    skip_keys.append("options")
                if key == "num_tokens":
                    num_tokens = value
                elif key == "token_dtype":
                    token_dtype = value
                else:
                    kwargs[key] = value
                label = " ".join(
                    [f"{k}={v}" for k, v in extra_kwargs.items() if k not in skip_keys]
                    + [pair2str(k, v) for k, v in params_.items() if (v is not None or k not in {"batch_chunk_size", "chunking_method"}) and k not in {"acc_dtype", "acc_policy"}]
                )
                if params_.get('acc_dtype', None) is not None:
                    label += f" {params_['acc_policy']}"
                label = f"{cls.__name__}[{label}]" if label else cls.__name__
                if (
                    issubclass(cls, FusedProjectionPlusCrossEntropyLoss)
                    or issubclass(cls, modules.MyLinearCrossEntropyLoss)
                    or issubclass(cls, modules.NewLinearCrossEntropyLoss)
                ) and token_dtype != torch.int64:
                    continue
                yield label, cls, kwargs, extra_kwargs, num_tokens, token_dtype, force


def measure(queue, cls, args, kwargs, num_tokens, token_dtype):
    fwd_only = kwargs.pop("fwd_only", False)
    torch.manual_seed(321)
    in_features, num_classes = args
    device = torch.device(kwargs["device"])
    cpu_initial_peak_mem_bytes = (
        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1e3
    )

    if token_dtype.is_floating_point:
        tokens = torch.randn(
            (num_tokens, num_classes), device=device, dtype=token_dtype
        ).softmax(dim=1)
    else:
        tokens = torch.randint(
            0, num_classes, (num_tokens,), device=device, dtype=token_dtype
        )
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        initial_peak_mem_bytes = torch.cuda.max_memory_allocated(device=device)
    elif device.type == "cpu":
        initial_peak_mem_bytes = 0
    else:
        assert 0, device.type

    dtype = kwargs["dtype"]
    message = []
    try:
        features = torch.randn(
            (num_tokens, in_features),
            device=device,
            dtype=dtype,
            requires_grad=not fwd_only,
        )
        tokens = tokens.clone()
        module = cls(*args, **kwargs)
        if fwd_only:
            module.linear.weight.requires_grad_(not fwd_only)
        loss = module(features, tokens).mean()
        if not fwd_only:
            _ = loss.backward()
        loss = loss.detach().item()
    except torch.OutOfMemoryError as msg:
        message.append("OOM")
        loss = None

    if device.type == "cuda":
        final_peak_mem_bytes = torch.cuda.max_memory_allocated(device=device)
    elif device.type == "cpu":
        final_peak_mem_bytes = 0
    else:
        assert 0, device.type

    cpu_final_peak_mem_bytes = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1e3

    if loss is not None and not fwd_only:
        grad_features = features.grad.clone().detach()
        if isinstance(module, modules.LiLinearCrossEntropyLoss):
            grad_linear_weight = module.projection.grad.clone().detach()
        else:
            grad_linear_weight = module.linear.weight.grad.clone().detach()

    repeat = 10

    if device.type == "cuda":
        if fwd_only:
            module.linear.weight.requires_grad_(not fwd_only)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        if "OOM" not in message:
            lst = []  # avoid gc until cuda sync is done
            for _ in range(repeat):
                try:
                    m = module(features, tokens)
                    if fwd_only:
                        lst.append(m.mean())
                    else:
                        lst.append(m.mean().backward())
                except torch.OutOfMemoryError as msg:
                    repeat = max(1, len(lst))
                    break

        end.record()
        torch.cuda.synchronize()
        time_ms = start.elapsed_time(end) / repeat
    elif device.type == "cpu":
        start = time.process_time()
        if "OOM" not in message:
            for _ in range(repeat):
                _ = module(features, tokens).mean().backward()
        end = time.process_time()
        time_ms = (end - start) * 1e3 / repeat
    else:
        assert 0, device.type

    if loss is not None and 0:
        # gradcheck sanity check
        in_features, num_tokens, num_classes = (2, 64, 64)
        args = (in_features, num_classes)
        features = torch.randn(
            (num_tokens, in_features),
            device=device,
            dtype=torch.float64,
            requires_grad=True,
        )
        if token_dtype.is_floating_point:
            tokens = torch.randn(
                (num_tokens, num_classes), device=device, dtype=torch.float64
            ).softmax(dim=1)
        else:
            tokens = torch.randint(
                0, num_classes, (num_tokens,), device=device, dtype=token_dtype
            )
        module = cls(*args, **kwargs).to(torch.float64)
        try:
            torch.autograd.gradcheck(module, (features, tokens))
        except RuntimeError as msg:
            print(f"gradcheck failed: {msg}")
            message.append("GRADCHECKFAILED")
    else:
        print("gradcheck skipped")
        message.append("GRADCHECKSKIPPED")

    loss_error = None
    grad_features_error = None
    grad_linear_weight_error = None
    if fwd_only:
        pass
    elif cls in {modules.NewLinearCrossEntropyLoss, modules.LiLinearCrossEntropyLoss} and dtype != torch.float64 and loss is not None:
        ref_dtype = torch.float64
        eta = torch.finfo(dtype).eps
        feps = torch.finfo(dtype).eps * 2

        def grad_error(grad, expected):
            # return relative error of two jacobian tensors
            if grad.dim() < 2:
                grad = grad.reshape(-1, 1)
                expected = expected.reshape(-1, 1)
            abserr = torch.linalg.matrix_norm(grad - expected, ord="fro")
            norm = torch.linalg.matrix_norm(expected, ord="fro")
            return abserr / (eta + norm)

        ref_features = features.detach().to(ref_dtype).requires_grad_(True)

        ref_kwargs = kwargs.copy()
        ref_kwargs['dtype'] = ref_dtype
        ref_kwargs['options'] = torch.nn.LinearCrossEntropyOptions()
        #ref_kwargs['options'] = None
        ref_module = nn.LinearCrossEntropyLoss(*args, **ref_kwargs)
        if isinstance(module, modules.NewLinearCrossEntropyLoss):
            ref_module.linear.weight.detach().copy_(module.linear.weight.detach().clone()).requires_grad_(True)
        elif isinstance(module, modules.LiLinearCrossEntropyLoss):
            ref_module.linear.weight.detach().copy_(module.projection.detach().clone()).requires_grad_(True)
        else:
            raise NotImplementedError(cls)
        ref_l = ref_module(ref_features, tokens).mean()
        ref_l.backward()
        ref_grad_features = ref_features.grad.detach().clone().to(dtype)
        ref_grad_linear_weight = ref_module.linear.weight.grad.detach().clone().to(dtype)
        loss_error = float(abs(ref_l.to(dtype).item() - loss))
        grad_features_error = float(grad_error(grad_features, ref_grad_features).item())
        grad_linear_weight_error = float(grad_error(grad_linear_weight, ref_grad_linear_weight).item())
    elif (
        cls in {modules.VoLinearCrossEntropyLoss, modules.LiLinearCrossEntropyLoss}
        or dtype != torch.float64
    ):
        message.append("SKIPSANITYCHECK")
    elif cls is not nn.LinearCrossEntropyLoss:
        # sanity check against reference
        in_features, num_tokens, num_classes = (2, 64, 64)
        args = (in_features, num_classes)
        features = torch.randn(
            (num_tokens, in_features),
            device=device,
            dtype=torch.float64,
            requires_grad=True,
        )
        if token_dtype.is_floating_point:
            tokens = torch.randn(
                (num_tokens, num_classes), device=device, dtype=torch.float64
            ).softmax(dim=1)
        else:
            tokens = torch.randint(
                0, num_classes, (num_tokens,), device=device, dtype=token_dtype
            )

        torch.manual_seed(678)
        module = cls(*args, **kwargs).to(torch.float64)
        torch.manual_seed(678)
        kwargs = deepcopy(kwargs)
        kwargs.pop("n_loop_iters", None)
        kwargs.pop("num_chunks", None)
        kwargs.pop("splits", None)
        kwargs.pop("inplace_grad", None)
        kwargs.pop("params", None)
        kwargs.pop("bias", None)
        kwargs.pop("options", None)
        ref_module = nn.LinearCrossEntropyLoss(*args, **kwargs).to(torch.float64)

        l = module(features, tokens).sum()
        l.backward()
        ref_features = features.detach().requires_grad_(True)
        ref_l = ref_module(ref_features, tokens).sum()
        ref_l.backward()
        l_err = abs(ref_l - l)
        atol, rtol = 1e-12, 1e-9
        if not (l_err <= atol):
            message.append(f"LOSSABSERROR[{l_err}>{atol}]")
        g_err = (abs(features.grad - ref_features.grad)).max()
        if not (g_err <= atol):
            message.append(f"GRADABSERROR[{g_err}>{atol}]")

    message = "-".join(message)
    if message:
        print(message)

    cuda_mem_gb = (final_peak_mem_bytes - initial_peak_mem_bytes) / 1e9
    # print(f"{int(cuda_mem_gb * 1e9 / dtype.itemsize)=}")
    queue.put(
        dict(
            message=message,
            cpu_mem_gb=(cpu_final_peak_mem_bytes - cpu_initial_peak_mem_bytes) / 1e9,
            cuda_mem_gb=(final_peak_mem_bytes - initial_peak_mem_bytes) / 1e9,
            time_ms=time_ms,
            loss=loss,
            loss_error=loss_error, 
            grad_features_error=grad_features_error,
            grad_linear_weight_error=grad_linear_weight_error
        )
    )


measurement_fields = ["cpu_mem_gb", "cuda_mem_gb", "time_ms", "loss", "message", "loss_error", "grad_features_error", "grad_linear_weight_error"]
string_fields = ["gpu_name", "label", "message", "device", "reduction", "device_name"]
int_fields = ["in_features", "num_classes", "ignore_index", ""]
float_fields = ["label_smoothing", "cpu_mem_gb", "cuda_mem_gb", "time_ms", "loss", "loss_error", "grad_features_error", "grad_linear_weight_error"]
bool_fields = ["bias"]


def unserialize(fieldname, rawvalue):
    if fieldname in string_fields:
        return rawvalue
    if isinstance(rawvalue, str) and rawvalue == "None":
        return None
    if fieldname in int_fields:
        return int(rawvalue)
    if fieldname in float_fields:
        if rawvalue == "str:":
            return 0.0
        if isinstance(rawvalue, str) and rawvalue.startswith("float:"):
            return float(rawvalue[6:])
        return float(rawvalue)
    if fieldname in bool_fields:
        return {"False": False, "True": True, "": "N/A", "N/A": "N/A"}[rawvalue]
    if ":" not in rawvalue:
        return rawvalue
    typename, value = rawvalue.split(":", 1)
    if typename == "float":
        return float(value)
    elif typename == "int":
        return int(value)
    elif typename == "bool":
        return {"False": False, "True": True}[value]
    if typename == "dtype":
        assert value.startswith("torch."), value
        return getattr(torch, value.split(".", 1)[1])
    print(f"TODO: unserialize({fieldname}, {value!r})")
    return value


def serialize(fieldname, value):
    if fieldname in string_fields:
        assert isinstance(value, str), (type(value), value)
        return value
    if fieldname in int_fields + float_fields + bool_fields:
        return str(value)
    return f"{type(value).__name__}:{value}"


def load_measurements(path):
    data = []
    if path.exists():
        import csv

        with open(path, newline="") as csvfile:
            reader = csv.DictReader(csvfile, delimiter=";")
            for row in reader:
                row = dict([(k, unserialize(k, v)) for k, v in row.items()])
                if row.get("num_classes", 0) <= 32768 or 1:
                    data.append(row)
    return data


def get_fieldnames(data):
    fieldnames = None
    for row in data:
        if fieldnames is None:
            fieldnames = list(row)
        else:
            for name in row:
                if name not in fieldnames:
                    fieldnames.append(name)
    return fieldnames


def save_measurements(path, data):
    import csv

    with open(path, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=get_fieldnames(data), delimiter=";")
        writer.writeheader()
        for row in data:
            writer.writerow(dict([(k, serialize(k, v)) for k, v in row.items()]))


def filter_measurements(data, match, with_indices=False):
    result = []
    indices = []
    for i, row in enumerate(data):
        for k, v in match.items():
            r = row.get(k)
            if isinstance(v, (list, set, tuple)) and type(r) != type(v):
                if r not in v:
                    break
            else:
                if r != v:
                    break
        else:
            result.append(row)
            indices.append(i)
    if with_indices:
        return result, indices
    return result


def select_measurements(data, match):
    result = []
    for row in data:
        for k, v in match.items():
            if row.get(k) != v:
                break
        else:
            result.append(row)
    return result


def compute(data, force=False):
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties()
        gpu_name = f"{p.name}-CC{p.major}.{p.minor}-{p.total_memory // 2**20}MB"
    else:
        gpu_name = "N/A"
    p = open("/proc/cpuinfo").read()
    cpu_name = p[p.index("model name") :].split("\n", 1)[0].split(":")[1].strip()
    print(f"{gpu_name=} {cpu_name=}")
    mp_ctx = mp.get_context("spawn")
    not_force = False
    try:
        for i, (label, cls, kwargs, extra_kwargs, num_tokens, token_dtype, force) in enumerate(configs(
            default_force=force
        )):
            if not_force:
                force = False
            if force and 0:
                not_force = True
            device_name = dict(cpu=cpu_name, cuda=gpu_name)[kwargs["device"]]
            match_dct = dict(
                device_name=device_name,
                label=label,
                num_tokens=num_tokens,
                token_dtype=token_dtype,
                **kwargs,
            )
            row, indices = filter_measurements(data, match_dct, with_indices=True)
            if row:
                if not force:
                    continue
                for i in reversed(indices):
                    del data[i]
            args = (kwargs.pop("in_features"), kwargs.pop("num_classes"))
            kwargs = dict(kwargs, **extra_kwargs)
            if cls in {
                modules.NewLinearCrossEntropyLoss,
                torch.nn.modules.loss.LinearCrossEntropyLoss,
                PyTorchProjectionPlusCrossEntropyLoss,
                FusedProjectionPlusCrossEntropyLoss,
                modules.VoLinearCrossEntropyLoss,
                modules.LiLinearCrossEntropyLoss,
            }:
                assert kwargs.pop("bias", False) in {False, "N/A", ""}
            q = mp_ctx.Queue()
            # executing measure in mp.Process enables correct CPU memory
            # usage measurement
            p = mp_ctx.Process(
                target=measure, args=(q, cls, args, kwargs, num_tokens, token_dtype)
            )
            p.start()
            r = q.get()
            p.join()
            cpu_mem_gb, cuda_mem_gb, time_ms, message = (
                r["cpu_mem_gb"],
                r["cuda_mem_gb"],
                r["time_ms"],
                r["message"],
            )

            print(
                f'{label}[{kwargs["device"]}], {args=} {num_tokens=} {cpu_mem_gb, cuda_mem_gb, time_ms=} loss={r["loss"]} loss/_error={r.get("loss_error")} grad_features/linear_weight_error={r.get("grad_features_error")}/{r.get("grad_linear_weight_error")}'
            )

            if message:
                print(message)
            data.append(dict(match_dct, **r))
    except KeyboardInterrupt:
        pass
    return data


def get_values(data, field):
    return list(set([row[field] for row in data if field in row]))


def get_series(data, field, ignore=["device_name", "device"]):
    fieldnames = get_fieldnames(data)
    field_values = dict()
    for f in fieldnames:
        if f == field or f in measurement_fields:
            continue
        if f not in field_values:
            field_values[f] = set()
        field_values[f].update(get_values(data, f))

    result = []

    for p in product(*[tuple((k, v) for v in s) for k, s in field_values.items()]):
        m = filter_measurements(data, dict(p))
        if len(m) > 1:
            x, i = list(zip(*sorted([[row[field], i] for i, row in enumerate(m)])))
            m = [m[i_] for i_ in i]
            x = list(x)
            values = dict()
            params = dict()
            for k in measurement_fields:
                values[k] = [r[k] for r in m]
            for k in fieldnames:
                if not (k == field or k in measurement_fields):
                    v = get_values(m, k)
                    assert len(v) == 1, v
                    params[k] = v[0]
            result.append((params, (field, x), values))

    series_lst = []
    series_params = dict()
    for p, x, y in result:
        for k, v in p.items():
            if k not in series_params:
                series_params[k] = set()
            series_params[k].add(v)
        series_lst.append((p, x, y))

    common_params = dict(
        [(k, list(v)[0]) for k, v in series_params.items() if len(v) == 1]
    )
    series = dict()
    for p, x, y in series_lst:
        label = tolabel(p, ignore=common_params)
        series[label] = (x, y)
    return common_params, series


def tolabel(data, ignore=[]):
    defaults = dict(ignore_index=-100, label_smoothing=0.0)
    names = [
        "label",
        "dtype",
        "token_dtype",
        "features_in",
        "num_classes",
        "num_tokens",
        "bias",
        "reduction",
        "ignore_index",
        "label_smoothing",
    ]
    lst = []

    def tostr(n, v):
        if n in ["dtype", "token_dtype"]:
            return str(v).split(".", 1)[-1]
        elif n == "label":
            return v
        else:
            return f"{n}={v}"

    for n in names:
        if n not in data or n in ignore:
            continue
        elif n in defaults and data[n] == defaults[n]:
            continue
        else:
            lst.append(tostr(n, data[n]))
    for n in data:
        if n in names or n in ignore:
            continue
        elif n in defaults and data[n] == defaults[n]:
            continue
        else:
            lst.append(tostr(n, data[n]))

    return " ".join(lst)


def toxlabel(field, reference=None):
    if field in ["cuda_mem_gb", "cpu_mem_gb"]:
        if reference is not None:
            return "Peak Memory Usage - Reference (GB)"
        return "Peak Memory Usage (GB)"
    if field == "time_ms":
        if reference is not None:
            return "Average Timing - Reference (ms)"
        return "Average Timing (ms)"
    return field


def plot(data, plot_params, reference_label=None):
    print(f'{plot_params=}')
    import matplotlib.pyplot as plt

    paths = []
    root_path = Path("images")
    root_path.mkdir(exist_ok=True, parents=True)

    for device_name in get_values(data, "device_name"):
        print(f"{device_name=}")
        device_data = filter_measurements(
            data, dict(device_name=device_name, **plot_params)
        )
        if not device_data:
            print(f'no data: {plot_params=}')
            continue
        device = get_values(device_data, "device")
        assert len(device) == 1, device
        device = device[0]

        xfields = ["in_features", "num_classes", "num_tokens"]
        xfields = list(config_defaults)
        xfields_complementary = dict(
            (k, dict((k1, config_defaults[k1]) for k1 in xfields if k1 != k))
            for k in xfields
        )
        yfields = [
            dict(cuda="cuda_mem_gb", cpu="cpu_mem_gb")[device],
            "time_ms",
            "loss_error",
            "grad_features_error",
            "grad_linear_weight_error",
        ]

        all_series = []
        common_params = dict()
        existing_xfields = []
        for xfield in xfields:
            params, series = get_series(
                filter_measurements(device_data, xfields_complementary[xfield]), xfield
            )
            for k, v in params.items():
                if k not in common_params:
                    common_params[k] = set()
                common_params[k].add(v)

            if series:
                all_series.append((params, series))
                existing_xfields.append(xfield)
        xfields = existing_xfields

        common_params = dict(
            [
                (k, list(v)[0])
                for k, v in common_params.items()
                if len(v) == 1 and k not in xfields
            ]
        )
        suptitle = tolabel(
            common_params, ignore=["device_name", "device", "token_dtype"]
        )
        fig, axes = plt.subplots(len(yfields), len(xfields), figsize=(len(yfields) * 6, len(xfields) * 7), dpi=300)

        def base_model(in_features, num_tokens, num_classes, dtype):
            input_numel = num_tokens * in_features
            L_numel = num_classes * in_features
            target_numel = num_tokens
            weight_numel = num_classes
            grad_input_numel = input_numel
            grad_L_numel = L_numel
            X_numel = num_tokens * num_classes
            return (
                (
                    input_numel
                    + L_numel
                    + target_numel
                    + weight_numel
                    + grad_input_numel
                    + grad_L_numel
                )
                * dtype.itemsize
                / 1e9
            )

        for xi, (params, series) in enumerate(all_series):
            title = tolabel(params, ignore=common_params)
            for yi, yfield in enumerate(yfields):
                if len(xfields) == 1 and xi == 0:
                    ax = axes[yi]
                else:
                    ax = axes[yi][xi]
                if reference_label is not None:
                    ((xfield, x)), ydata = series[reference_label]
                    z = np.polyfit(x, ydata[yfield], 1)
                    reference = np.poly1d(z)
                else:
                    reference = None
                for label, ((xfield, x), ydata) in sorted(series.items()):
                    if yfield not in ydata:
                        continue
                    y = ydata[yfield]
                    x, y = zip(
                        *[
                            (x_, y_)
                            for x_, y_, l in zip(x, y, ydata["loss"])
                            if l is not None
                        ]
                    )
                    if reference is not None:
                        y = np.array(y) - reference(np.array(x))
                    ax.plot(x, y, "-o", label=label)
                if yfield == "cuda_mem_gb":
                    x1, x2 = min(x) * 1, max(x) * 1
                    if xfield == "in_features":
                        y1 = base_model(
                            x1,
                            params["num_tokens"],
                            params["num_classes"],
                            params["dtype"],
                        )
                        y2 = base_model(
                            x2,
                            params["num_tokens"],
                            params["num_classes"],
                            params["dtype"],
                        )
                    elif xfield == "num_tokens":
                        y1 = base_model(
                            params["in_features"],
                            x1,
                            params["num_classes"],
                            params["dtype"],
                        )
                        y2 = base_model(
                            params["in_features"],
                            x2,
                            params["num_classes"],
                            params["dtype"],
                        )
                    elif xfield == "num_classes":
                        y1 = base_model(
                            params["in_features"],
                            params["num_tokens"],
                            x1,
                            params["dtype"],
                        )
                        y2 = base_model(
                            params["in_features"],
                            params["num_tokens"],
                            x2,
                            params["dtype"],
                        )
                    else:
                        assert 0, xfield
                    if reference is not None:
                        y1, y2 = np.array([y1, y2]) - reference(np.array([x1, x2]))
                    ax.plot([x1, x2], [y1, y2], "k-.", label="Theoretical infima")
                if yfield.endswith("_error"):
                    ax.set_ylim(bottom=0)
                ax.set_xlabel(toxlabel(xfield))
                ax.set_ylabel(toxlabel(yfield, reference=reference))
                ax.set_title(title)
                if xi == 0 and yi == 0:
                    ax.legend(loc="upper left")

        suffix = [""]
        if "reduction" in common_params:
            suffix.append(common_params["reduction"])
        if "token_dtype" in common_params:
            if common_params["token_dtype"].is_floating_point:
                suffix.append("probabilities")
                suptitle = f"{suptitle}; probability targets"
            else:
                suffix.append("indices")
                suptitle = f"{suptitle}; class indices targets"
        if "dtype" in plot_params:
            suffix.append(str(plot_params["dtype"]).split(".")[-1])
        suffix = "-".join(suffix)

        filename = device_name.replace(" ", "_") + f"{suffix}.png"
        path = root_path / filename
        paths.append(path)

        fig.suptitle(f"{device_name}\n{suptitle}\n\n")
        plt.tight_layout()
        plt.savefig(path)
        print(f"created {path}")
        plt.close()

    return paths


def main():
    measurements_path = Path("measurements.csv")
    data = load_measurements(measurements_path)
    if 1:
        data = compute(data, force=not True)
        if 1:
            save_measurements(measurements_path, data)

    if 1:
        data = filter_measurements(
            data,
            dict(
                label=[
                    # *(f"MyLinearCrossEntropyLoss[splits=({i}, {j}, 1)]" for i in [1, 4, 8] for j in [1, 4, 8]),
                    # *(f"MyLinearCrossEntropyLoss[splits=(None, {j}, {1024 * k})]" for j in [1, 4, 8] for k in [1, 2, 4, 8]),
                    # *(f"MyLinearCrossEntropyLoss[splits=({i}, None, {1024 * k})]" for i in [1, 4, 8] for k in [1, 2, 4, 8]),
                    # *(f"MyLinearCrossEntropyLoss[splits=(None, None, {1024 * k})]" for k in [1, 2, 4, 8, 16, 32]),
                    # *(f"MyLinearCrossEntropyLoss[splits=(None, None, {1024 * k}) inplace_grad=True]" for k in [1, 4, 8, 16, 32][:-2]),
                    *(
                        f"MyLinearCrossEntropyLoss[chunk_size={512 * k} grad_inplace=True chunks_classes={chunks_classes}]"
                        for k in [1, 2, 4, 8]
                        for chunks_classes in [1, 2][:0]
                    ),
                    *(
                        f"MyLinearCrossEntropyLoss[{pair2str('max_numel', (512 * k) ** 2)} grad_inplace=True]"
                        for k in [1, 2, 4, 8, 16, 32][:0]
                    ),
                    *(
                        f"MyLinearCrossEntropyLoss[grad_inplace=True max_memory_gb={0.5 * k} {min_chunk_size=}]"
                        for k in [1, 2, 3, 4, 5, 6][1:2]
                        for min_chunk_size in [256, 512, 1024][:0]
                    ),
                    # "MyLinearCrossEntropyLoss[splits=(1, 1, 1)]",
                    # "MyLinearCrossEntropyLoss[splits=(1, 8, 1)]",
                    # "MyLinearCrossEntropyLoss[splits=(8, 8, 1)]",
                    # "MyLinearCrossEntropyLoss[splits=(4, 1, 1)]",
                    # "MyLinearCrossEntropyLoss[splits=(None, None, None)]",
                    # "MyLinearCrossEntropyLoss[splits=(None, None, None) inplace_grad=True]",
                    # "MyLinearCrossEntropyLoss[num_chunks=2]",
                    # "MyLinearCrossEntropyLoss[num_chunks=4]",
                    # "MyLinearCrossEntropyLoss[num_chunks=8]",
                    # "MyLinearCrossEntropyLoss[num_chunks=16]",
                    # "LinearCrossEntropyLoss",
                    #"LinearCrossEntropyLoss[options=None]",
                    # "VoLinearCrossEntropyLoss",
                    "LiLinearCrossEntropyLoss",
                    # "FusedProjectionPlusCrossEntropyLoss[n_loop_iters=1]",
                    # "FusedProjectionPlusCrossEntropyLoss[n_loop_iters=2]",
                    # "FusedProjectionPlusCrossEntropyLoss[n_loop_iters=4]",
                    # "FusedProjectionPlusCrossEntropyLoss[n_loop_iters=8]",
                    # "FusedProjectionPlusCrossEntropyLoss[n_loop_iters=16]",
                    # "PyTorchProjectionPlusCrossEntropyLoss",
                    # "NewLinearCrossEntropyLoss",
                    # "NewLinearCrossEntropyLoss[max_memory_gb=1]",
                    # "NewLinearCrossEntropyLoss[max_memory_gb=1.5 grad_inplace=True]",
                    # "NewLinearCrossEntropyLoss[max_memory_gb=1 grad_inplace=True features_chunk_size=8192]",
                    # "NewLinearCrossEntropyLoss[max_memory_gb=2]",
                    # "NewLinearCrossEntropyLoss[grad_inplace=False batch_chunk_size=None]",
                    # "NewLinearCrossEntropyLoss[grad_inplace=True batch_chunk_size=None]",
                    #"NewLinearCrossEntropyLoss[grad_inplace=True chunking_method=liger]",
                    #"NewLinearCrossEntropyLoss[grad_inplace=False batch_chunk_size=liger]",
                    #*(f"NewLinearCrossEntropyLoss[grad_inplace=True batch_chunk_size={1024*k}]" for k in [1, 2][:0]),
                    #*(f"NewLinearCrossEntropyLoss[grad_inplace=True chunking_method=liger {a}]" for a in ["AAAAA", "AATAA", "TATAA", "TTTAA", "TTTTA", "TTTTTA",
                    #                                                                                      "TTTAAA", "TTTATA"][-3:-2]),
                    *(f"NewLinearCrossEntropyLoss[grad_inplace=True chunking_method=liger:-1 {a}]" for a in ["AAAAA", "AATAA", "TATAA", "TTTAA", "TTTTA", "TTTTTA",
                                                                                                          "TTTAAA", "TTTATA"][-3:-2]),
                ]
            ),
        )

    if 0:
        plot(data, dict(reduction="mean", token_dtype=torch.float32))
    if 0:
        plot(data, dict(reduction="mean", token_dtype=torch.int64))
    if 1:
        plot(
            data,
            dict(reduction="mean", token_dtype=torch.int64, dtype=torch.float16),
        )


if __name__ == "__main__":
    main()
