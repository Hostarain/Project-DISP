"""Plot training curves from scalars.json. Usage: python plot_loss.py <work_dir_subdir>"""
import json, sys, os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

work_dir = sys.argv[1] if len(sys.argv) > 1 else 'mmdetection_github/work_dirs/my_train'

# Find newest timestamp dir
subdirs = [d for d in os.listdir(work_dir) if d.startswith('20') and os.path.isdir(os.path.join(work_dir, d))]
if not subdirs:
    print(f'No timestamp dirs found in {work_dir}')
    sys.exit(1)
rundir = os.path.join(work_dir, sorted(subdirs)[-1])

# Try vis_data/scalars.json first, then root
json_path = os.path.join(rundir, 'vis_data', 'scalars.json')
if not os.path.exists(json_path):
    json_path = os.path.join(rundir, 'scalars.json')
if not os.path.exists(json_path):
    print(f'scalars.json not found in {rundir}')
    sys.exit(1)

with open(json_path) as f:
    lines = f.readlines()

iters, loss, loss_cls, loss_bbox = [], [], [], []
val_iters, val_map = [], []
for line in lines:
    d = json.loads(line)
    if 'loss' in d:
        iters.append(d.get('iter', len(iters)))
        loss.append(d['loss'])
        loss_cls.append(d['loss_cls'])
        loss_bbox.append(d['loss_bbox'])
    if 'pascal_voc/mAP' in d:
        val_map.append(d['pascal_voc/mAP'])

fig, axes = plt.subplots(1, 2 if val_map else 1, figsize=(12 if val_map else 8, 4))
if not val_map: axes = [axes]

ax1 = axes[0]
ax1.plot(iters, loss, label='Total', color='#1f77b4', linewidth=1.5)
ax1.plot(iters, loss_cls, label='Cls', color='#ff7f0e', alpha=0.7)
ax1.plot(iters, loss_bbox, label='Bbox', color='#2ca02c', alpha=0.7)
ax1.set_xlabel('Iteration')
ax1.set_ylabel('Loss')
ax1.set_title(f'Training Loss  ({loss[0]:.2f} → {loss[-1]:.2f})')
ax1.legend()
ax1.grid(True, alpha=0.3)

if val_map:
    ax2 = axes[1]
    ax2.bar(range(len(val_map)), val_map, color='#d62728', alpha=0.7)
    ax2.set_xticks(range(len(val_map)))
    ax2.set_xticklabels([f'ep{i+1}' for i in range(len(val_map))])
    ax2.set_ylabel('mAP')
    ax2.set_title(f'Validation mAP  (best: {max(val_map):.4f})')
    ax2.grid(True, alpha=0.3)

out = os.path.join(rundir, 'training_curves.png')
plt.tight_layout()
plt.savefig(out, dpi=150)
print(f'Saved: {out}')
print(f'Total loss:  {loss[0]:.3f} → {loss[-1]:.3f}')
print(f'Cls  loss:   {loss_cls[0]:.3f} → {loss_cls[-1]:.3f}')
print(f'Bbox loss:   {loss_bbox[0]:.3f} → {loss_bbox[-1]:.3f}')
if val_map:
    print(f'Best mAP:    {max(val_map):.4f} (epoch {val_map.index(max(val_map))+1})')
