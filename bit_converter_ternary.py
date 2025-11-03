import torch
import numpy as np
from bitnet_jetson import OneBitViT
import torch.nn as nn
from dataset import build_dataset
from bitnet_jetson.bitlinear import weight_inference_quant
from medmnist import Evaluator
import argparse
from MedViT import MedViT_large, MedViT
from kernels.pack_weight import convert_weight_int8_to_int2
import time 

# ---- Ternary Quantization (STE-style) ----
def ternary_ste(w, threshold=0.05):
    # Center and threshold
    w, scale = weight_inference_quant(w)
    # e = w.mean()
    # centered = w - e
    # w_tern = torch.zeros_like(centered)
    # w_tern[centered > threshold] = 1
    # w_tern[centered <= -threshold] = -1
    # Else remains zero
    return w, scale

def hard_ternary_quantize_model(model, n_channels, threshold=0.05):
    
    with torch.no_grad():        
        for name, module in model.named_modules():
            
            if module.__class__.__name__ == "BitLinear" and hasattr(module, "weight"):
                w = module.weight
                w_q, scale = ternary_ste(w, threshold)
                module.weight.copy_(w_q.to(torch.bfloat16))
                w_q = convert_weight_int8_to_int2(w_q.to(torch.int8)).to('cuda')

                orig_shape = w_q.shape
                N = orig_shape[0]
                K = orig_shape[1] * 4

                weight_compressed = w_q.reshape([N // 16 // 2, 2, K // 16, 2, 8, 4])
                weight_compressed = weight_compressed.permute([0, 2, 3, 1, 4, 5])
                weight_compressed = weight_compressed.reshape([N // 16 // 2, K // 16, 4, 8, 4]).contiguous()

                n_cols = weight_compressed[:, :, [0, 1]]
                n_cols = n_cols.reshape([N // 16 // 2, K // 16 // 2, 4, 8, 4])

                k_cols = weight_compressed[:, :, [2, 3]]
                k_cols = k_cols.reshape([N // 16 // 2, K // 16 // 2, 4, 8, 4])

                weight_compressed = torch.concat([n_cols, k_cols], dim = 2).contiguous()
                w_q = weight_compressed.view(N, K // 4).contiguous()
                
                scale = scale.to(torch.bfloat16)

                module.weight_quant = w_q
                module.weight_scale = scale
                module.quant = True
                module.qconfig = None
                # print(f"{name}: unique ternary values {np.unique(w_q.detach().cpu().numpy())}")     
            # print(module.qconfig)
    
    # model_fp32_prepared = torch.ao.quantization.prepare(model)

    # input_fp32 = torch.randn(4, n_channels, 224, 224) * (0.5 ** 0.5) + 0.5
    # model_fp32_prepared(input_fp32.to('cuda'))

    # model = torch.ao.quantization.convert(model_fp32_prepared)

def evaluate_model(model, test_loader):
    y_score = torch.tensor([])
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            with torch.amp.autocast(device_type='cuda'):
                if(type(model) == MedViT):
                    logits, _ = model(images)
                else:
                    logits, _ = model(images[:, :n_channels])
                                    
                preds= torch.softmax(logits, dim=-1)

            y_score = torch.cat((y_score, preds.cpu()), 0)

    y_score[torch.isnan(y_score)] = 0.0
    y_score = y_score.detach().numpy()
    metrics = test_evaluator.evaluate(y_score)
    val_auc, val_acc = metrics
    return val_auc, val_acc
    
# ---- Config ----
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=4)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--T', type=float, default=2)
parser.add_argument('--L', type=int, default=3)
parser.add_argument('--H', type=int, default=8)
parser.add_argument('--dim', type=int, default=512)
parser.add_argument('--alpha', type=float, default=0.5)
parser.add_argument('--dataset', type=str, default='breastmnist')
parser.add_argument('--gamma', type=float, default=0.5)
parser.add_argument('--multi_query', type=int, default=1)
parser.add_argument('--fp_attn', type=int, default=1)
parser.add_argument('--save_dir', type=str, default="./runs")
parser.add_argument('--data_dir', type=str, default="./data")
parser.add_argument('--teacher_dir', type=str, default="./models")
args = parser.parse_args()

_, test_dataset, _, nb_classes, n_channels = build_dataset(args=args)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)
test_evaluator = Evaluator(args.dataset, 'val', size=224, root=args.data_dir)

layers = args.L
dim = args.dim
heads = args.H

mlp_dim = dim * 4
patch_size = 7

base_dir = args.save_dir + ("/multi_query" if args.multi_query else "/multi_attn") + ("/fp_attn" if args.fp_attn else "/bit_attn")
base_dir = base_dir + f'/{args.dataset}/L{args.L}dim{args.dim}H{args.H}'

MODEL_PATH = base_dir + "/bitnet_student.pth"
OUT_PATH = base_dir + "/bitnet_student_ternary.pth"

# teacher = MedViT_large(num_classes = nb_classes)
# checkpoint = torch.load(f"../models/MedViT_large_{args.dataset}.pth", weights_only=False)
# teacher.load_state_dict(checkpoint["model"])
# teacher.eval().to(device)

# val_auc, val_acc = evaluate_model(teacher, test_loader)
# print(f"Teacher: Auc: {val_auc} | ACC: {val_acc}")
# del teacher

student = OneBitViT(
    image_size=224,
    patch_size=patch_size,
    num_classes=nb_classes,
    dim=dim,
    depth=layers,
    heads=heads,
    mlp_dim=mlp_dim,
    channels=n_channels,
    dim_head=dim / heads,
    multi_query = args.multi_query,
    full_precision_attn = args.fp_attn
).to(device)

state_dict = torch.load(MODEL_PATH, map_location=device)
student.load_state_dict(state_dict)
student = student.to(device)

student.eval()
hard_ternary_quantize_model(student, n_channels, threshold=0.05)

for name, module in student.named_modules():
    if module.__class__.__name__ == "BitLinear" and hasattr(module, "weight"):
        module.quant = False

start = time.time()
val_auc, val_acc = evaluate_model(student, test_loader)
mean_time = (time.time() - start) / len(test_loader)
print(f"Quantized Student: Auc: {val_auc} | ACC: {val_acc} | Latency: {mean_time}")

for name, module in student.named_modules():
    if module.__class__.__name__ == "BitLinear" and hasattr(module, "weight"):
        module.quant = True

start = time.time()
val_auc, val_acc = evaluate_model(student, test_loader)
mean_time = (time.time() - start) / len(test_loader)
print(f"Optimized Quantized Student: Auc: {val_auc} | ACC: {val_acc} | Latency: {mean_time}")

# torch.save(model.state_dict(), OUT_PATH)
# print(f"Ternary quantized model saved to {OUT_PATH}")