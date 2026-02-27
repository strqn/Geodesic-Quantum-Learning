import matplotlib.pyplot as plt
import numpy as np
import os

# ==========================================
# 1. إعدادات الرسوم البيانية للأبحاث (Publication Quality)
# ==========================================
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'legend.fontsize': 10,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'font.family': 'serif', # خطوط كلاسيكية تناسب الأبحاث
})

# ==========================================
# 2. البيانات التجريبية
# ==========================================
noise_labels = ['Ideal', 'Depol 0.001', 'Depol 0.005', 'Depol 0.01', 'Depol 0.02', 'Depol 0.05', 'AmpDamp 0.01']
gqe_means = [-14.6102, -14.8436, -12.7117, -13.3896, -12.1424, -8.7926, -13.7893]
gqe_stds = [0.4159, 0.6252, 0.8623, 1.2673, 1.2134, 0.7208, 1.1955]

spsa_means = [-3.2682, -2.0736, -2.3547, -3.1391, -2.5873, -2.4090, -3.7385]
spsa_stds = [0.7789, 1.0372, 1.7033, 1.1868, 0.8016, 1.1057, 0.6670]

exact_energy = -16.2677

# ==========================================
# 3. إعداد الرسم البياني (Bar Chart)
# ==========================================
x = np.arange(len(noise_labels))
width = 0.35  # عرض العمود

fig, ax = plt.subplots(figsize=(14, 7))

# رسم أعمدة GQE و SPSA مع أشرطة الخطأ (Error Bars)
rects1 = ax.bar(x - width/2, gqe_means, width, yerr=gqe_stds, 
                label='Symplectic GQE (Ours)', color='#E24A33', capsize=5, alpha=0.9, edgecolor='black')
rects2 = ax.bar(x + width/2, spsa_means, width, yerr=spsa_stds, 
                label='Standard SPSA', color='#348ABD', capsize=5, alpha=0.9, edgecolor='black')

# رسم خط الحالة الأرضية الدقيقة (Exact Ground State Energy)
ax.axhline(y=exact_energy, color='black', linestyle='--', 
           label=f'Exact Ground State ({exact_energy} Ha)', linewidth=2)

# ==========================================
# 4. التنسيق والجماليات
# ==========================================
ax.set_ylabel('Energy (Hartree)', fontweight='bold')
ax.set_xlabel('Noise Channels and Strengths ($\gamma$)', fontweight='bold')
ax.set_title('Robustness Benchmark: GQE vs SPSA (100 Steps, 5 Seeds)', fontweight='bold', pad=15)
ax.set_xticks(x)
ax.set_xticklabels(noise_labels, rotation=15, ha="right")
ax.legend(loc='lower right', framealpha=0.9, edgecolor='black')
ax.grid(axis='y', linestyle=':', alpha=0.7)

# إضافة القيم الرقمية فوق أعمدة GQE لإبراز مدى اقترابها من القاع
for i, mean in enumerate(gqe_means):
    ax.annotate(f'{mean:.2f}',
                xy=(x[i] - width/2, mean),
                xytext=(0, -20),  # إزاحة النص للأسفل قليلاً داخل العمود
                textcoords="offset points",
                ha='center', va='top', color='white', fontweight='bold', fontsize=10)

plt.tight_layout()

# ==========================================
# 5. الحفظ بصيغة SVG للأوراق العلمية
# ==========================================
# يمكنك تغيير المسار إلى المجلد الذي تريده
save_path = 'gqe_vs_spsa_noise_benchmark.svg'
plt.savefig(save_path, format='svg', dpi=1200, bbox_inches='tight')

print(f"✓ تم حفظ الرسم البياني بنجاح كملف متجهي عالي الدقة في: {save_path}")

plt.show()
