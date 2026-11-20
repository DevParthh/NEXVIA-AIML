"""Export the DCN ranker into a Triton model repository (tensorflow_savedmodel backend)."""
import os
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
import shutil
import tensorflow as tf
import tensorflow_recommenders  # noqa: F401
from config import ART, ROOT

repo = ROOT / "triton_models" / "ranker"
dst = repo / "1" / "model.savedmodel"
if dst.exists():
    shutil.rmtree(dst)
dst.parent.mkdir(parents=True, exist_ok=True)
shutil.copytree(ART / "ranker", dst)

sig = tf.saved_model.load(str(dst)).signatures["serving_default"]
(in_name, in_spec), = sig.structured_input_signature[1].items()
(out_name, out_spec), = sig.structured_outputs.items()
cfg = f'''name: "ranker"
platform: "tensorflow_savedmodel"
max_batch_size: 8192
input [ {{ name: "{in_name}" data_type: TYPE_FP32 dims: [ {in_spec.shape[-1]} ] }} ]
output [ {{ name: "{out_name}" data_type: TYPE_FP32 dims: [ 1 ] }} ]
'''
(repo / "config.pbtxt").write_text(cfg)
print("Triton repo ready at", repo, "| input:", in_name, "| output:", out_name)
