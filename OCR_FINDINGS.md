# OCR in MinerU — and how PaddleOCR-VL-0.9B differs

This note records two findings:

1. **What OCR engine the vendored MinerU package (`references/MinerU`) actually uses.**
2. **What PaddleOCR-VL-0.9B is**, and how it relates to / differs from MinerU's approach.

---

## 1. What MinerU uses

### 1.1 It is a PyTorch port of PaddleOCR, not the Paddle library

MinerU does **not** depend on the `paddleocr` / `paddle` runtime. It vendors a
complete **PyTorch re-implementation** of the PaddleOCR (PP-OCR) model family and
runs it on Torch.

Evidence:

- No `import paddle` / `import paddleocr` anywhere in the tree.
- Weights are `.pth` (PyTorch `state_dict`), not Paddle's `.pdparams`.
- All inference is `torch.no_grad()` over `nn.Module` graphs.
- The port lives under [`mineru/model/utils/pytorchocr/`](references/MinerU/mineru/model/utils/pytorchocr/);
  weights load via plain `torch.load(...)` / `load_state_dict(...)` in
  [base_ocr_v20.py](references/MinerU/mineru/model/utils/pytorchocr/base_ocr_v20.py).

This is the long-standing "pytorch-paddle-ocr" pattern: PaddleOCR's *model
architectures and weights* are reused, but the Paddle framework dependency is
dropped.

### 1.2 The three classic PP-OCR stages are all present

| Stage | Algorithm | Class | File |
|---|---|---|---|
| Text **detection** | DB (Differentiable Binarization) | `TextDetector` | [predict_det.py:13](references/MinerU/mineru/model/utils/pytorchocr/tools/infer/predict_det.py#L13) |
| Text **recognition** | SVTR_LCNet / CRNN | `TextRecognizer` | [predict_rec.py:16](references/MinerU/mineru/model/utils/pytorchocr/tools/infer/predict_rec.py#L16) |
| Angle **classification** | 180° line orientation (off by default) | `TextClassifier` | [predict_cls.py:13](references/MinerU/mineru/model/utils/pytorchocr/tools/infer/predict_cls.py#L13) |

Models are **PP-OCRv5** (detection + lite recognition) with **PP-OCRv4 server**
recognition variants available for higher accuracy:

- `ch_PP-OCRv5_det_infer.pth`
- `ch_PP-OCRv5_rec_infer.pth`
- `ch_PP-OCRv4_rec_server_infer.pth` / `..._doc_infer.pth`

77 language configs (including a dedicated `seal` stamp model) are declared in
[models_config.yml](references/MinerU/mineru/model/utils/pytorchocr/utils/resources/models_config.yml).
Weights auto-download from HuggingFace / ModelScope into
`models/OCR/paddleocr_torch`.

### 1.3 Public interface

The wrapper class is **`PytorchPaddleOCR`**
([pytorch_paddle.py:154](references/MinerU/mineru/model/ocr/pytorch_paddle.py#L154),
extends `TextSystem`):

- `ocr(img, det=True, rec=True, mfd_res=None, ...)` — full or partial pipeline.
  `mfd_res` masks math-formula regions so OCR doesn't garble equations.
- `__call__(img, mfd_res=None)` → `(filter_boxes, filter_rec_res)` with
  confidence filtering and seal polygon-cropping.
- `.text_detector` / `.text_recognizer` — direct access for batched calls.

Instantiated via `ocr_model_init(...)` in
[model_init.py:131](references/MinerU/mineru/backend/pipeline/model_init.py#L131).

### 1.4 How the pipeline invokes it

In [batch_analyze.py](references/MinerU/mineru/backend/pipeline/batch_analyze.py),
OCR is split detection-then-recognition so it can batch efficiently and reuse
layout info:

1. Batch-detect whole pages — `text_detector.batch_predict(...)`.
2. Detect on cropped regions with formula masking — `ocr(..., rec=False, mfd_res=...)`.
3. Batch-recognize the cropped text boxes — `ocr(img_crop_list, det=False)`.
4. Seal OCR — separate polygon-warp path
   ([seal_det_warp.py](references/MinerU/mineru/model/ocr/seal_det_warp.py)).

All calls go through `run_ocr_inference()`, which holds
`PIPELINE_OCR_INFERENCE_LOCK` for thread safety
([model_init.py:55](references/MinerU/mineru/backend/pipeline/model_init.py#L55)).

### 1.5 Page-level OCR-vs-TEXT routing

Before OCR runs, MinerU decides per-document whether a PDF needs OCR at all, via
`classify(pdf_bytes) -> "txt" | "ocr"` in
[pdf_classify.py](references/MinerU/mineru/utils/pdf_classify.py). It samples up
to 10 pages and returns `"ocr"` when the text layer is missing or garbled
(too few extractable chars, ToUnicode/CID mapping errors, high image coverage,
etc.). Used by the **pipeline** (`_get_ocr_enable`) and **hybrid**
(`ocr_classify`) backends.

### 1.6 The VLM backend uses no PaddleOCR

[vlm_analyze.py](references/MinerU/mineru/backend/vlm/vlm_analyze.py) contains no
OCR imports or calls. The VLM path does end-to-end recognition with a
vision-language model. PaddleOCR (the PyTorch port above) is only reached on the
**pipeline** or **hybrid** backends.

### Summary of MinerU's stack

- **Detect → recognize** with separate specialist CNN/transformer models
  (DB + SVTR/CRNN), PP-OCRv4/v5 weights, run on **PyTorch**.
- A classical OCR cascade: layout/MFD models feed boxes into det+rec; formulas,
  tables and reading order are handled by *other* models, not the OCR engine.
- A VLM backend exists as an alternative, but it replaces the OCR cascade rather
  than using it.

---

## 2. What PaddleOCR-VL-0.9B is

PaddleOCR-VL is Baidu/PaddlePaddle's **end-to-end document-parsing
vision-language model** (paper *"PaddleOCR-VL: Boosting Multilingual Document
Parsing via a 0.9B Ultra-Compact Vision-Language Model"*, arXiv:2510.14528,
Oct 2025). The "0.9B" is the recognition VLM at the heart of the system.

### 2.1 Architecture (~0.9B params total)

- **Vision encoder:** a **NaViT-style dynamic / native-resolution** encoder,
  initialized from **Keye-VL**. Processing images at native resolution avoids
  distortion, reduces hallucination, and helps on text-dense pages.
- **Projector:** a 2-layer MLP with GELU bridging vision → language.
- **Language model:** **ERNIE-4.5-0.3B**, a compact decoder.

So it is one neural network that takes a (cropped) image and *generates* the
content — text, table markup, formula LaTeX, chart data — as tokens. There is no
separate DB-detection / CRNN-recognition split.

### 2.2 The full PaddleOCR-VL system is two-stage

1. **Layout stage — PP-DocLayoutV2:** detects element regions (text, title,
   table, formula, figure, chart…) and predicts **reading order**.
2. **Recognition stage — PaddleOCR-VL-0.9B:** the VLM recognizes the *content* of
   each detected element.

It supports **109 languages** and handles text, tables, formulas, and charts,
reporting SOTA on OmniDocBench-style page- and element-level benchmarks at small
model size.

### 2.3 How it contrasts with MinerU's current OCR

| Dimension | MinerU pipeline (PyTorch-PaddleOCR) | PaddleOCR-VL-0.9B |
|---|---|---|
| Paradigm | Classical cascade: detect boxes → recognize text | Generative VLM: image → content tokens |
| Models | Many specialist models (DB det, SVTR/CRNN rec, layout, MFD, table) | Layout model (PP-DocLayoutV2) + one VLM for all element content |
| Framework | PyTorch (`.pth`), Paddle dropped | Released for Paddle/PaddleX; also a HF `transformers` model class |
| Formulas/tables/charts | Separate downstream models | Recognized directly by the same VLM |
| Languages | 77 lang configs | 109 languages |
| Resolution handling | Fixed preprocessing per crop | NaViT native dynamic resolution |
| Strength | Fast, modular, mature, low memory per stage | Higher accuracy on complex/multilingual docs, fewer moving parts |
| Weakness | Many models to wire; brittle on hard layouts/scripts | Heavier (VLM inference), newer/less battle-tested in this repo |

### 2.4 Relationship to *this* repo

- This vendored MinerU **does not** reference PaddleOCR-VL anywhere
  (no `PaddleOCR-VL`, `NaViT`, or `ERNIE` matches in `references/`).
- MinerU's own **VLM backend** plays the analogous role to PaddleOCR-VL — an
  end-to-end VLM replacing the OCR cascade — but it is a *different* model, not
  PaddleOCR-VL.
- Conceptually, **PaddleOCR-VL-0.9B is to the classic PP-OCR cascade what
  MinerU's VLM backend is to MinerU's PyTorch-PaddleOCR cascade**: a single
  generative model superseding detect+recognize+table+formula specialists.

---

## Sources

- [PaddleOCR-VL paper page (HuggingFace)](https://huggingface.co/papers/2510.14528)
- [arXiv:2510.14528](https://arxiv.org/abs/2510.14528)
- [PaddlePaddle/PaddleOCR-VL model card](https://huggingface.co/PaddlePaddle/PaddleOCR-VL)
- [transformers model doc: paddleocr_vl](https://huggingface.co/docs/transformers/model_doc/paddleocr_vl)
- [ERNIE blog: PaddleOCR-VL](https://ernie.baidu.com/blog/posts/paddleocr-vl/)
- MinerU source: `references/MinerU/` (paths cited inline above)
