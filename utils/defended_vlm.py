from .vlm import VLM, SMOL_VLMS, QWEN_VLMS, VLMS_WITH_FAST_PROCESSOR
from .image_utils import process_image
import torch
import torchvision.transforms as T

class DefendedVLM(VLM):

    prompt_image = None

    """
    This class implements a version of the VLM which adds another learnable image to the prompt to make it more resilient
    """

    def get_user_message(self, user_query):
        return {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": user_query},
                {"type": "image"},
            ],
        }


    def create_vlm_inputs(self, image: torch.tensor, formatted_prompt, prompt_image=None, overwrite: bool = False):
        """
        NOTE: padding_side should be set to "left", otherwise this will interfere with the attack optimization
        """
        if isinstance(formatted_prompt, str): formatted_prompt = [formatted_prompt]

        if prompt_image is None:
            prompt_image = self.prompt_image
        mock_images = [T.ToPILImage()(prompt_image), T.ToPILImage()(image)]
        if self.name in VLMS_WITH_FAST_PROCESSOR:
            mock_images = [prompt_image, image]
        
        # process
        inputs = self.processor(text=formatted_prompt, images=mock_images, return_tensors="pt", truncation=True, padding=True, padding_side="left").to(self.device)

        # overwrite with exact adversarial image tensor if needed
        if overwrite and self.name not in VLMS_WITH_FAST_PROCESSOR:
            image_ppd = process_image(image, self)
            prompt_image_ppd = process_image(prompt_image, self)
            for i in range(inputs['pixel_values'].shape[0]):
                inputs['pixel_values'][i][0] = prompt_image_ppd
                inputs['pixel_values'][i][1] = image_ppd
        
        return inputs
    
    @torch.no_grad()
    def generate(self, image: torch.tensor, formatted_prompt, prompt_image=None, overwrite: bool = False, max_new_tokens=30, do_sample=True, temperature=0.5):
        inputs = self.create_vlm_inputs(image, formatted_prompt, prompt_image, overwrite)
        generated_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=do_sample, temperature=temperature)
        generated_texts = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
        return generated_texts

    def forward(self, image, formatted_prompt, prompt_image=None, overwrite: bool = False):
        inputs = self.create_vlm_inputs(image, formatted_prompt, prompt_image, overwrite)
        out = self.model(**inputs, use_cache=False, output_attentions=False, output_hidden_states=False)
        return out