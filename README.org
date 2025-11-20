* PyTorch Toys
A grab bag of small PyTorch experiments covering clustering, quantization, noiser-based regularization, attention blocks, and distillation losses.

** Supported environments
- *Python*: 3.9 - 3.11 have been tested locally. Earlier versions may miss typing features used across the repo.
- *PyTorch*: 1.13+ (with torchvision 0.14+) is recommended so that eager-mode quantization APIs (``torch.quantization``) and CUDA half precision paths behave as expected.
- *CUDA*: Optional but strongly encouraged for the quantization and clustering demos. Every script falls back to CPU execution when CUDA is unavailable, albeit slowly.

** Dependency installation
All projects rely on a small shared set of packages. Install them either with ``pip`` or ``conda``:

*** pip
#+begin_src bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch torchvision torchaudio \
    --extra-index-url https://download.pytorch.org/whl/cu118  # drop flag for CPU-only
pip install matplotlib scipy tqdm
#+end_src

*** conda
#+begin_src bash
conda create -n pytorch-toys python=3.10 pytorch torchvision torchaudio cudatoolkit=11.8 -c pytorch -c nvidia
conda activate pytorch-toys
pip install matplotlib scipy tqdm
#+end_src

** Dataset prerequisites
- *ImageNet 1k validation (50k imgs)* – required by ``post_quant`` and ``official_quantization``. The scripts expect torchvision ``ImageFolder`` layouts such as:
  - ``/data/ILSVRC2012_img_val/<class_name>/<img>.JPEG`` (full 50k validation split).
  - ``/data/imagenet_1k/val/<class_name>/<img>.JPEG`` (1k-sample calibration subset used in ``official_quantization``).
  Update the hard-coded paths in ``official_quantization/main.py`` (variables ``test_db`` and ``image_net_db``) or symlink your dataset root to those folder names.
- *Calibration subsets* – ``post_quant/main.py`` accepts two positional arguments: the calibration ImageFolder root and the validation ImageFolder root. Reuse ImageNet folders or point to any compatible dataset with ImageNet normalization.
- *CIFAR-10* – ``overfit/main.py`` downloads the dataset automatically into ``./data`` if it is not present.
- *Synthetic data* – ``cluster/test.py`` and ``cluster/pq_test.py`` create random blobs internally, so no external dataset is needed.

** Project index & runnable examples
Each subsection lists the entry-point command(s), noteworthy configuration flags, and the main console outputs you can expect.

*** [[./cluster][cluster]] – K-means & Product Quantization
- *What it does*: ``cluster/kmeans.py`` implements batched k-means along with product quantization helpers (PQ distance books, asymmetric search tables, etc.).
- *Example commands*:
  - Visualize 2D clustering speed/quality and open a matplotlib scatter plot:
    #+begin_src bash
    python cluster/test.py
    #+end_src
  - Benchmark PQ recall on synthetic vectors, printing timing for table construction and distance lookups:
    #+begin_src bash
    python cluster/pq_test.py
    #+end_src
- *Key knobs*: edit ``pts`` / ``batch_size`` in ``test.py`` to stress-test GPU memory; change ``dim``, ``subv_size``, and ``num_centers`` inside ``pq_test.py`` to match your target PQ configuration.
- *Expected output*: both scripts print timing statistics (e.g., ``Time: 2.31`` or ``Recall: 0.92 [M=8, bits=8 @ top 32]``). ``test.py`` also displays the learned centroids overlayed on the synthetic data cloud.

*** [[./loss][loss]] – Distillation loss helper
- ``loss/distill.py`` exposes ``DistillLoss(alpha, temperature, k=None)`` that mixes KL divergence with standard cross entropy.
- *Usage snippet*:
  #+begin_src python
  from loss.distill import DistillLoss
  criterion = DistillLoss(alpha=0.7, temperature=4.0, k=32)
  loss = criterion(student_logits, teacher_logits, labels)
  loss.backward()
  #+end_src
- *Flags*: ``alpha`` balances distillation vs. CE, ``temperature`` scales the soft logits (``temperature**2`` is applied automatically), and ``k`` optionally drops the smallest teacher logits for sparse distillation.

*** [[./post_quant][post_quant]] – Fake-quantization pipeline
- *Entry point*: quantize a torchvision ``resnet101`` with calibration/evaluation datasets supplied on the CLI.
  #+begin_src bash
  python post_quant/main.py /data/imagenet_calibration /data/ILSVRC2012_img_val
  #+end_src
- *Flow*: ``main.py``
  1. Loads the pretrained model, moves it to CUDA if available, and builds a calibration dataloader (resize 256 → center crop 224 → normalize).
  2. Runs ``fake_quant`` to collect activation/weight ranges, then saves ``model.quant``.
  3. Reloads the quantized weights via ``load_fake_quant_model`` and fuses eligible modules (``post_quant/fusion.py``).
  4. Evaluates the fake-quant model via ``post_quant/accuracy_test.validate``; expect ``Acc@1``/``Acc@5`` logs every ~10 batches plus a final summary line.
- *Configuration flags*: adjust ``batch_size``, ``num_workers``, model architecture, or normalization values by editing ``main.py``. ``fake_quant`` also accepts parameters such as bit width via kwargs inside ``post_quant/fake_quantization.py``.
- *Outputs*: ``model.quant`` (state dict with per-tensor scale/zero points) and stdout accuracy metrics similar to `` * Acc@1 76.2 Acc@5 92.8``.

*** [[./official_quantization][official_quantization]] – PTQ reference
- *Entry point*: ``python official_quantization/main.py`` after setting up the ImageNet folders described above.
- *Flow*: the script
  1. Creates a pretrained ``resnet18`` and fuses modules using ``torch.quantization.fuse_modules``.
  2. Wraps the model with ``QuantStub``/``DeQuantStub`` (see ``replace_forward``) and assigns ``qconfig`` from ``qnnpack``.
  3. Builds a small calibration dataloader from ``imagenet_1k/val`` and the full validation loader from ``ILSVRC2012_img_val``.
  4. Evaluates float accuracy, calibrates, converts to quantized weights, prints model size (MB), then evaluates the quantized model on CPU.
- *Knobs*: change ``q_backend`` (``qnnpack`` vs. ``fbgemm``), tweak the module fusion lists, or modify ``batch_size`` for calibration/eval loaders.
- *Expected output*: dotted progress indicators for calibration/evaluation, float accuracy summary (``Evaluation accuracy 69.76``), quantized model size, and final accuracy printed via ``evaluate``.

*** [[./overfit][overfit]] – Random noiser regularization on CIFAR-10
- *Entry point*: train the TallerLeNet baseline with weight noise injection and optional augmentation.
  #+begin_src bash
  python overfit/main.py --aug --batch-size 512 --epochs 120 --save-model
  #+end_src
- *Important flags*:
  - ``--aug`` toggles the random horizontal flip + grayscale pipeline (on by default).
  - ``--batch-size``, ``--test-batch-size``, ``--epochs`` control dataloader sizes and loop length.
  - ``--no-cuda`` forces CPU training; ``--save-model`` writes ``latest.pt`` plus ``best_dict_state.bin`` when higher accuracy is observed.
- *What happens*: ``Noiser`` perturbs model parameters each forward pass (``anchor=True`` keeps a copy), ``CosineAnnealingLR`` schedules SGD, and accuracy/loss statistics are printed every epoch (see ``overfit/README.org`` for historical numbers).
- *Outputs*: console logs for every training batch via ``tqdm``, epoch-wise test accuracy (``Test set: Average loss ...``), and final summary ``Final best acc: xx.x``.

*** [[./utils][utils]] – Conv2d cross-attention building block
- *Module*: ``utils/modules.py`` defines ``Conv2dCrossAttention`` for 2D query/value feature maps.
- *Usage snippet*:
  #+begin_src python
  from utils.modules import Conv2dCrossAttention
  attn = Conv2dCrossAttention(D_query_space=64, D_vk_space=256, D_emb=64, D_output=128)
  bev = torch.randn(8, 64, 16, 10)
  img = torch.randn(8, 256, 36, 64)
  fused = attn(img, bev)
  #+end_src
- *Customization*: tune ``D_emb`` for capacity and observe GPU memory in the self-test block at the bottom of the module (``python utils/modules.py`` prints tensor shapes and per-call latency).

** Troubleshooting tips
- If you see ``RuntimeError: Could not run 'quantized::conv2d'`` double-check that ``torch.backends.quantized.engine`` matches your PyTorch build (``qnnpack`` for ARM/Android or CPU, ``fbgemm`` for x86 AVX2).
- ``cluster/test.py`` displays plots via matplotlib; when running on a headless server set ``MPLBACKEND=Agg`` to avoid GUI errors.
- Calibration/eval dataloaders use ``num_workers=8`` and ``pin_memory=True``. Lower these values if you encounter ``Too many open files``.
