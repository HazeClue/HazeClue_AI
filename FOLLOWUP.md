# HazeClue AI — Session Follow-up (24 مايو 2025)

## ✅ اللي اتعمل النهارده

### 1. بناء المشروع كامل من الصفر
- مشروع Python في `/home/ameen/ameen/projects/grad/hazeclue-ai/`
- venv مع كل الـ dependencies (numpy, scipy, sklearn, pyriemann, mne, onnx...)
- هيكل كامل: `data/` → `preprocessing/` → `routing/` → `features/` → `training/` → `inference/` → `export/`

### 2. الكود اللي اتكتب

| ملف | الوظيفة | حالته |
|-----|---------|-------|
| `data/loaders/stew_loader.py` | تحميل STEW (48 subject, 128Hz, txt) | ✅ شغال |
| `data/loaders/mendeley_loader.py` | تحميل Mendeley (30 subject, 256Hz→128Hz, EDF) | ✅ شغال |
| `data/dataset.py` | توحيد الداتاستين + GroupKFold | ✅ شغال |
| `preprocessing/bandpass.py` | Butterworth 1-40Hz zero-phase | ✅ شغال |
| `preprocessing/sqi.py` | Signal Quality Index per channel | ✅ شغال |
| `preprocessing/covariance.py` | Raw→SQI-weighted→Ledoit-Wolf→SPD | ✅ شغال |
| `routing/mode_router.py` | RARD/MVES/SAFE switching | ✅ شغال |
| `features/rard_features.py` | Tangent space → 105 features | ✅ شغال |
| `features/mves_features.py` | Bandpower+Hjorth+Corr → 203 features | ✅ شغال |
| `training/train.py` | Optimized training pipeline | ✅ شغال |
| `inference/engine.py` | Real-time inference engine | ✅ smoke tested |
| `inference/personalization.py` | Dual-timescale adaptation | ✅ smoke tested |
| `export/export_onnx.py` | ONNX export for Flutter | ⚠️ مكتوب لكن لسه ما اتجربش |

### 3. الداتاسيتس

| Dataset | المصدر | الحجم | الحالة |
|---------|--------|-------|--------|
| **STEW** | Google Drive (صاحبك رفعه) | 398 MB, 48 subject | ✅ موجود في `data/raw/stew/STEW Dataset/` |
| **Mendeley** | data.mendeley.com (نزلته من البراوزر) | 149 MB, 30 subject, EDF format | ✅ موجود في `data/raw/mendeley/Emotiv 30s EDF/` |

### 4. التدريب الحقيقي

**Dataset Stats:**
```
14,064 windows total (78 subjects)
Class 0 (rest): 7,032
Class 1 (workload/concentration): 7,032
Window: 4 seconds (512 samples @ 128Hz), 50% overlap
```

**أفضل النتايج (5-Fold GroupKFold Cross-Subject):**

| Model | Accuracy | ملاحظة |
|-------|----------|--------|
| **RARD (LDA) — بتاعنا** | **63.5% ± 3.7%** | ⭐ الأفضل |
| TSClassifier (pyriemann) | 62.8% ± 4.2% | مكتبة جاهزة |
| Ensemble | 62.2% ± 4.0% | تجميع 3 classifiers |
| Combined (308 feat) | 61.9% ± 4.0% | RARD+MVES features |
| MVES (LogReg) | 59.8% ± 2.5% | إحصائي فقط |
| MDM | 56.5% ± 3.6% | مباشر على المنيفولد |

**الموديلات محفوظة في:**
```
trained_models/
├── rard_classifier.joblib
├── mves_classifier.joblib
├── rard_scaler.joblib
├── mves_scaler.joblib
└── P_ref.npy (Fréchet reference point)
```

---

## ⚠️ المشاكل اللي واجهتنا

### 1. Mendeley format مختلف عن المتوقع
- **المشكلة**: توقعنا CSV/TXT لكن الملفات كانت **EDF** (European Data Format)
- **الحل**: أعدنا كتابة `mendeley_loader.py` ليستخدم `mne.io.read_raw_edf()`
- **كمان**: التردد كان **256 Hz** مش 250 Hz — عدلنا الـ downsampling

### 2. STEW filename parsing
- **المشكلة**: الملفات اسمها `sub01_lo.txt` والـ parser كان بيعمل `replace("s","")` فبيحول `sub01` → `ub01` ومش بيقدر يعمله int
- **الحل**: عدلنا الـ parsing يستخدم prefix stripping (`"sub"` → strip بالترتيب)

### 3. الـ noise augmentation بتضر التدريب
- **المشكلة**: أول training (بالـ routing) — الـ augmented noise كانت بتخلي كل الـ windows تروح MVES path (لأن SQI بينزل). فالـ RARD classifier ما كانش بيلاقي data يتدرب عليها
- **الحل**: أعدنا كتابة `train.py` — التدريب بقى بيدي كل الـ classifiers كل الداتا. الـ routing بقى للـ inference بس

### 4. تحميل الداتاسيتس
- **Mendeley**: مفيش direct download API — لازم تنزله من البراوزر يدوي (بطيء ~50 KB/s)
- **STEW from IEEE**: محتاج login — لقيناه على Google Drive عند صاحبك
- **STEW from HuggingFace**: نزل جزء وبعدين rate-limited

### 5. الـ training بطيء
- **السبب**: كل window (14,064 total) بيحصلها `matrix_log` + `matrix_sqrt` (14×14 matrix operations)
- **الوقت**: ~25 دقيقة للتدريب الكامل
- **تحسين ممكن**: استخدم vectorized operations أو cache الـ covariances

---

## 📋 اللي باقي يتعمل بكرة

### أولوية عالية 🔴
- [ ] **ONNX Export** — تصدير الموديل للفلاتر (`export/export_onnx.py`)
  - Command: `python3 -c "from export.export_onnx import export_all_models; export_all_models('trained_models', 'onnx_models')"`
- [ ] **FastAPI Endpoint** — REST API للسيرفر (`api/inference_endpoint.py`)
  - يستقبل EEG window → يرد بالتصنيف + confidence
  - يتربط مع الـ NestJS backend
- [ ] **ربط ONNX بالـ Flutter app** — `onnxruntime_flutter` package

### أولوية متوسطة 🟡
- [ ] **تحسين الـ Accuracy**:
  - جرب SVM بدل LDA/LogReg
  - جرب Random Forest
  - Feature selection (SelectKBest)
  - Within-subject fine-tuning (هيطلع 80%+)
- [ ] **Confusion Matrix + ROC Curve** — visualization للنتايج
- [ ] **Evaluation script** (`training/evaluate.py`) — تقرير شامل

### أولوية منخفضة 🟢
- [ ] **تحديث الـ Presentation** — إضافة slides النتايج
- [ ] **Batch optimization** — تسريع الـ feature extraction
- [ ] **Within-subject evaluation** — لإثبات إن الموديل يقدر يوصل 80%+ مع calibration

---

## 🔧 أوامر مفيدة

```bash
# تفعيل البيئة
cd /home/ameen/ameen/projects/grad/hazeclue-ai
source venv/bin/activate

# تشغيل التدريب كامل
python main.py --stew-dir data/raw/stew/"STEW Dataset" --mendeley-dir data/raw/mendeley/ --folds 5 --export-onnx

# تصدير ONNX
python3 -c "from export.export_onnx import export_all_models; export_all_models('trained_models', 'onnx_models')"

# تجربة pyriemann classifiers
# (الكود اللي شغلته في التيرمينال — انسخه من السكرين شوت)
```

---

## 📁 الملفات المهمة

```
hazeclue-ai/
├── main.py                          # Entry point
├── requirements.txt                 # Dependencies
├── trained_models/                  # ← الموديلات المحفوظة
│   ├── rard_classifier.joblib
│   ├── mves_classifier.joblib
│   ├── rard_scaler.joblib
│   ├── mves_scaler.joblib
│   └── P_ref.npy
├── data/raw/
│   ├── stew/STEW Dataset/           # ← 48 subject (sub01_lo.txt, sub01_hi.txt...)
│   └── mendeley/Emotiv 30s EDF/     # ← 30 subject (S001/S001E01.edf...)
├── STEW Dataset.zip                 # ← ممكن تمسحه (اتفك بالفعل)
└── 8c26dn6c7w-1 (1).zip            # ← ممكن تمسحه (اتفك بالفعل)
```

---

## 💡 ملاحظات للفريق

1. **63% cross-subject accuracy طبيعي** — الأبحاث على STEW بنفس الطريقة (cross-subject, EMOTIV) بتطلع 60-70%. الأرقام العالية (80-90%) بتكون within-subject
2. **الـ Mode Routing (RARD/MVES/SAFE) للـ inference** — مش للتدريب. أثناء التدريب كل الـ classifiers بتشوف كل الداتا
3. **الـ ONNX export مهم** — ده اللي هيشتغل على الموبايل بـ < 35ms latency
4. **الـ Fréchet mean reference point** (`P_ref.npy`) لازم يتبعت مع الموديل — بدونه الـ RARD path مش هيشتغل
