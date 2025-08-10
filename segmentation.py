import torch
import torchvision.transforms.v2 as T
import torch.nn.functional as F
from .utils import expand_mask

class LoadCLIPSegModels:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {},
        }

    RETURN_TYPES = ("CLIP_SEG",)
    FUNCTION = "execute"
    CATEGORY = "essentials/segmentation"

    def execute(self):
        from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
        processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
        model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined")

        return ((processor, model),)

class ApplyCLIPSeg:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "clip_seg": ("CLIP_SEG",),
                "image": ("IMAGE",),
                "prompt": ("STRING", { "multiline": False, "default": "" }),
                "threshold": ("FLOAT", { "default": 0.4, "min": 0.0, "max": 1.0, "step": 0.05 }),
                "smooth": ("INT", { "default": 9, "min": 0, "max": 32, "step": 1 }),
                "dilate": ("INT", { "default": 0, "min": -32, "max": 32, "step": 1 }),
                "blur": ("INT", { "default": 0, "min": 0, "max": 64, "step": 1 }),
            },
        }

    RETURN_TYPES = ("MASK",)
    FUNCTION = "execute"
    CATEGORY = "essentials/segmentation"

    def execute(self, image, clip_seg, prompt, threshold, smooth, dilate, blur):
        processor, model = clip_seg

        imagenp = image.mul(255).clamp(0, 255).byte().cpu().numpy()

        outputs = []
        prompt = [p.strip() for p in prompt.split(",") if p.strip()]
        prompt = [""] if len(prompt) == 0 else prompt
        for i in imagenp:
            inputs = processor(text=prompt, images=[i] * len(prompt), return_tensors="pt", padding=True)
            out = model(**inputs).logits  # shape: (num_prompts, H, W)
            out = torch.sigmoid(out)
            out = out.max(dim=0).values
            out = (out > threshold)
            outputs.append(out)

        del imagenp

        outputs = torch.stack(outputs, dim=0)

        if smooth > 0:
            if smooth % 2 == 0:
                smooth += 1
            outputs = T.functional.gaussian_blur(outputs, smooth)

        outputs = outputs.float()

        if dilate != 0:
            outputs = expand_mask(outputs, dilate, True)

        if blur > 0:
            if blur % 2 == 0:
                blur += 1
            outputs = T.functional.gaussian_blur(outputs, blur)
        
        # resize to original size
        outputs = F.interpolate(outputs.unsqueeze(1), size=(image.shape[1], image.shape[2]), mode='bicubic').squeeze(1)

        return (outputs,)

SEG_CLASS_MAPPINGS = {
    "ApplyCLIPSeg+": ApplyCLIPSeg,
    "LoadCLIPSegModels+": LoadCLIPSegModels,
}

SEG_NAME_MAPPINGS = {
    "ApplyCLIPSeg+": "🔧 Apply CLIPSeg",
    "LoadCLIPSegModels+": "🔧 Load CLIPSeg Models",
}