import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
from MedViT import MedViT_large

from transformers import ViTForImageClassification
from bitnet.one_bit_vision_transformers import OneBitViT, pair, posemb_sincos_2d
from bitnet import BitLinear
from dataset import build_dataset
from medmnist import Evaluator
from torch.optim.lr_scheduler import CosineAnnealingLR

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import itertools
from einops.layers.torch import Rearrange

class EmbeddingLayer(nn.Module):
    
    def __init__(self,
            img_size=224,
            patch_size=7,
            in_chans=3,
            embed_dim=1024,
            bias=True,  # disable bias if pre-norm is used (e.g. CLIP)
            dynamic_img_pad=False,
            **embed_args
            ):
        super().__init__() 
        image_height, image_width = pair(img_size)
        patch_height, patch_width = pair(patch_size)
        self.num_patches = (image_height // patch_height) * (image_width // patch_width)

        assert (
            image_height % patch_height == 0 and image_width % patch_width == 0
        ), "Image dimensions must be divisible by the patch size."

        patch_dim = in_chans * patch_height * patch_width
        
        self.to_patch_embedding = nn.Sequential(
            Rearrange(
                "b c (h p1) (w p2) -> b (h w) (p1 p2 c)",
                p1=patch_height,
                p2=patch_width,
            ),
            nn.LayerNorm(patch_dim),
            # nn.Linear(patch_dim, dim),
            BitLinear(patch_dim, embed_dim),
            nn.LayerNorm(embed_dim),
        )
        self.pos_embedding = posemb_sincos_2d(
            h=image_height // patch_height,
            w=image_width // patch_width,
            dim=dim,
        )
    def forward(self, x):
        x = self.to_patch_embedding(x)
        x += self.pos_embedding.to(device, dtype=x.dtype)
        return x
    

# --- Device ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=16)
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

train_dataset, test_dataset, _, nb_classes, n_channels = build_dataset(args=args)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=4, persistent_workers=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size*2, shuffle=False, pin_memory=True, num_workers=4, persistent_workers=True)

# --- Load Teacher Model (pretrained, frozen) ---
print("Loading pretrained teacher model...")
teacher = MedViT_large(num_classes = nb_classes)
checkpoint = torch.load(f"{args.teacher_dir}/MedViT_large_{args.dataset}.pth", weights_only=False)
teacher.load_state_dict(checkpoint["model"])
teacher.eval().to(device)

# --- Student Model Configuration ---
layers = args.L
dim = args.dim
heads = args.H

mlp_dim = dim * 4
patch_size = 7

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
print(student)
# student = VisionTransformer(
#     img_size=224,
#     in_chans=n_channels,
#     patch_size=patch_size,
#     num_classes=nb_classes,
#     mlp_ratio=4,
#     embed_dim=dim,
#     num_heads=heads,
#     depth=layers,
#     embed_layer=EmbeddingLayer
# )
student = student.to(device)

# --- Loss, Optimizer, AMP Scaler, TensorBoard ---
if("chest" in args.dataset):
    ce_loss = nn.BCEWithLogitsLoss()
else:
    ce_loss = nn.CrossEntropyLoss()

base_dir = args.save_dir + ("/multi_query" if args.multi_query else "/multi_attn") + ("/fp_attn" if args.fp_attn else "/bit_attn")
base_dir = base_dir + f'/{args.dataset}/L{args.L}dim{args.dim}H{args.H}'

kl_loss = nn.KLDivLoss(reduction='batchmean')
feat_loss = nn.MSELoss()
optimizer = optim.AdamW(student.parameters(), lr=args.lr)
writer = SummaryWriter(log_dir=base_dir + '/bitnet_distill')
scaler = torch.amp.GradScaler(enabled=(device.type == 'cuda'))
scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

# --- Utility Functions ---
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def log_weight_histograms(model, step, prefix='student'):
    for name, param in model.named_parameters():
        if param.requires_grad and param.ndim > 1:
            writer.add_histogram(f"{prefix}/{name}", param.data.cpu().numpy(), step)

def log_weight_distributions_per_layer(model, step, prefix='student'):
    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            writer.add_histogram(f"{prefix}/LinearOrConv/{name}", module.weight.data.cpu().flatten().numpy(), step)

print("Teacher parameters:", count_parameters(teacher))
print("Student parameters:", count_parameters(student))
writer.add_scalar('params/teacher', count_parameters(teacher), 0)
writer.add_scalar('params/student', count_parameters(student), 0)

for param in teacher.parameters():
    param.requires_grad = False

log_interval = 200

train_evaluator = Evaluator(args.dataset, 'train', size=224, root=args.data_dir)
test_evaluator = Evaluator(args.dataset, 'val', size=224, root=args.data_dir)

# --- Training Loop ---
teacher.distilling=True
for epoch in range(args.epochs):
    student.train()
    total_loss, total_acc = 0, 0
    y_score = torch.tensor([])
    y_labels = torch.tensor([])

    timebar = tqdm(train_loader)
    num_batches = len(train_loader)
    for step, (images, labels) in enumerate(timebar):
        
        images = images.to(device, non_blocking=True)
        if("chest" in args.dataset):
            labels = labels.to(device, non_blocking=True).to(torch.float32)
        else:
            labels = labels.to(device, non_blocking=True).view(-1).long()

        
        with torch.no_grad():
            teacher_outputs, teacher_layers = teacher(images)

        optimizer.zero_grad()

        with torch.amp.autocast(device_type='cuda'):            

            student_logits, match_layers = student(images[:, :n_channels])

            loss_ce = ce_loss(student_logits, labels)
            teacher_feat = teacher_layers[-1].flatten(2)
            loss_feat = feat_loss(match_layers, teacher_feat)

            student_log_probs = nn.functional.log_softmax(student_logits / args.T, dim=1)
            teacher_probs = nn.functional.softmax(teacher_outputs / args.T, dim=1)

            # Clamp for numerical stability
            teacher_probs = torch.clamp(teacher_probs, min=1e-8, max=1.0)
            student_log_probs = torch.clamp(student_log_probs, min=-100, max=0)

            loss_kl = kl_loss(student_log_probs, teacher_probs) * (args.T ** 2)

            loss_kd = (1 - args.gamma) * loss_kl + args.alpha * loss_feat

            loss = (1 - args.alpha) * loss_ce + args.alpha * loss_kd
        
        timebar.desc = f"train epoch[{epoch + 1}/{args.epochs}] loss:{loss:.3f}"
            
        # print(teacher_layers[0].flatten(2).shape, student_layers[0].flatten(2).shape)
        # print(teacher_layers[-1].flatten(2).shape, student_layers[-1].flatten(2).shape)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        # torch.nn.utils.clip_grad_norm_(student.parameters(), 0.5)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        total_loss += loss.item() * images.size(0)
        preds = torch.softmax(student_logits, dim=-1)

        y_score = torch.cat((y_score, preds.cpu()), 0)
        y_labels = torch.cat((y_labels, labels.cpu()), 0)

        if step % log_interval == 0:
            global_step = epoch * len(train_loader) + step
            log_weight_histograms(student, global_step)
            log_weight_distributions_per_layer(student, global_step)

    train_evaluator.labels = y_labels.detach().numpy()

    avg_loss = total_loss / len(train_loader.dataset)
    y_score[torch.isnan(y_score)] = 0.0
    np_score = y_score.detach().numpy()
    metrics = train_evaluator.evaluate(np_score)
    avg_auc, avg_acc = metrics

    writer.add_scalar('train/loss', avg_loss, epoch)
    writer.add_scalar('train/accuracy', avg_acc, epoch)
    writer.add_scalar('train/auc', avg_auc, epoch)
    print(f"Epoch {epoch} | Loss: {avg_loss:.4f} | Acc: {avg_acc:.4f} | Auc: {avg_auc:.4f}")

    # --- Validation ---
    student.eval()
    y_score = torch.tensor([])
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device, non_blocking=True)
            # labels = labels.to(device, non_blocking=True)
            with torch.amp.autocast(device_type='cuda'):
                logits = student(images[:, :n_channels])
                preds = torch.softmax(logits, dim=-1)
            y_score = torch.cat((y_score, preds.cpu()), 0)

    y_score[torch.isnan(y_score)] = 0.0
    np_score = y_score.detach().numpy()
    metrics = test_evaluator.evaluate(np_score)
    val_auc, val_acc = metrics

    writer.add_scalar('val/accuracy', val_acc, epoch)
    writer.add_scalar('val/auc', val_auc, epoch)
    print(f"Validation accuracy: {val_acc:.4f} | Validation AuC: {val_auc:.4f}")

# --- Save Final Model ---
torch.save(student.state_dict(), base_dir + "/bitnet_student.pth")