import sys
import os
import torch
import torch.fx
from torch.fx import symbolic_trace
from torch.fx.passes.shape_prop import ShapeProp
from tabulate import tabulate

# --- 1. Import Your Local PaliGemma Model ---
# Using the local implementation as requested.
from modeling_gemma import PaliGemmaForConditionalGeneration
from utils import load_hf_model
print("[Info] Using local PaliGemma implementation.")


# --- Helper Function for Shape Extraction ---
def _get_shape_str(meta) -> str:
    if meta is None:
        return "N/A"
    if hasattr(meta, 'shape'):
        return str(tuple(meta.shape))
    elif isinstance(meta, (tuple, list)):
        shapes = [tuple(m.shape) for m in meta if hasattr(m, 'shape')]
        return str(shapes)
    return "N/A"

def main():
    # --- 2. Load the Model from Local Path ---
    # Using the local model path from your launch_inference.sh script
    model_path = "/export/home/nikhil/parikshit/research/ViTs-performance-modelling/src/models/Paligemma/paligemma-3b-mix-224"
    print(f"[1/4] Loading PaliGemma model from local path: {model_path}...")

    # Use the local loader from utils.py.
    # We load on CPU for tracing.
    full_model, _ = load_hf_model(model_path, device="cpu")
    full_model = full_model.float() # Ensure model is float32 for FX
    full_model.eval()

    # --- 3. ISOLATE THE VISION TOWER ---
    # The structure is the same, we isolate the .vision_tower
    print("[2/4] Isolating Vision Tower (SigLIP)")
    vision_tower = full_model.vision_tower
    
    # Wrapper to ensure clean Tracing.
    class VisionWrapper(torch.nn.Module):
        def __init__(self, tower):
            super().__init__()
            self.tower = tower
        
        def forward(self, x):
            # The local implementation returns a single tensor, so we can directly return it.
            # The previous wrapper's control flow confused the FX tracer.
            out = self.tower(x)
            return out

    model_to_trace = VisionWrapper(vision_tower)
    
    # --- 4. Trace the Graph ---
    print("[3/4] Symbolically Tracing the Vision Tower...")
    traced_graph = symbolic_trace(model_to_trace)

    # --- 5. Shape Propagation ---
    print("[4/4] Propagating Shapes...")
    # PaliGemma 224 takes (Batch, 3, 224, 224).
    dummy_input = torch.randn(1, 3, 224, 224)
    
    try:
        ShapeProp(traced_graph).propagate(dummy_input)
    except Exception as e:
        print(f"[Error] Shape propagation failed. This can happen with complex control flow.")
        print(f"Error: {e}")
        # Even if shape prop fails, we can print the graph without shapes.
    
    # --- 6. Extract & Visualize ---
    extracted_data = []
    print(f"\n--- NeuSight Graph Extractor (PaliGemma ViT / SigLIP) ---")

    for node in traced_graph.graph.nodes:
        if node.op in ['call_module', 'call_function']:
            
            # Output Shape
            out_shape = "N/A"
            if 'tensor_meta' in node.meta:
                out_shape = _get_shape_str(node.meta.get('tensor_meta'))

            # Input Shape(s)
            in_shapes = []
            for arg in node.args:
                if isinstance(arg, torch.fx.Node):
                    if 'tensor_meta' in arg.meta:
                        in_shapes.append(_get_shape_str(arg.meta.get('tensor_meta')))
                elif isinstance(arg, (list, tuple)):
                    for sub_arg in arg:
                        if isinstance(sub_arg, torch.fx.Node) and 'tensor_meta' in sub_arg.meta:
                            in_shapes.append(_get_shape_str(sub_arg.meta.get('tensor_meta')))
            
            in_shape_str = ", ".join(in_shapes) if in_shapes else "N/A"

            # Operator Type
            op_type = "Func"
            if node.op == 'call_module':
                sub_mod = traced_graph.get_submodule(node.target)
                op_type = type(sub_mod).__name__
            else:
                op_type = str(node.target)

            extracted_data.append([node.name, op_type, in_shape_str, out_shape])

    print(tabulate(extracted_data[:40], headers=["Node Name", "Op Type", "Input Shape", "Output Shape"], tablefmt="grid"))

if __name__ == "__main__":
    main()
