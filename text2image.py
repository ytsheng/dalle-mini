import jax
import jax.numpy as jnp

# Load models & tokenizer
from dalle_mini.model import DalleBart
from vqgan_jax.modeling_flax_vqgan import VQModel
from transformers import BartTokenizer, CLIPProcessor, FlaxCLIPModel
import wandb

from flax.jax_utils import replicate
from functools import partial
from dalle_mini.text import TextNormalizer
from dalle_mini.model.modeling import FlaxBartModule, FlaxBartForConditionalGenerationModule
from flax.training.common_utils import shard

from flax.training.common_utils import shard_prng_key
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from tqdm.notebook import trange
import time 
import random

import jax
import flax.linen as nn

from transformers.models.bart.modeling_flax_bart import (
    FlaxBartDecoder,
    FlaxBartEncoder,
    FlaxBartForConditionalGeneration,
    FlaxBartForConditionalGenerationModule,
    FlaxBartModule,
)
from transformers import BartTokenizer, FlaxBartForConditionalGeneration, BartConfig


OUTPUT_VOCAB_SIZE = 16384 + 1  # encoded image token space + 1 for bos
OUTPUT_LENGTH = 256 + 1  # number of encoded tokens + 1 for bos
BOS_TOKEN_ID = 16384
BASE_MODEL = 'facebook/bart-large'

class CustomFlaxBartModule(FlaxBartModule):
    def setup(self):
        # we keep shared to easily load pre-trained weights
        self.shared = nn.Embed(
            self.config.vocab_size,
            self.config.d_model,
            embedding_init=jax.nn.initializers.normal(self.config.init_std, self.dtype),
            dtype=self.dtype,
        )
        # a separate embedding is used for the decoder
        self.decoder_embed = nn.Embed(
            OUTPUT_VOCAB_SIZE,
            self.config.d_model,
            embedding_init=jax.nn.initializers.normal(self.config.init_std, self.dtype),
            dtype=self.dtype,
        )
        self.encoder = FlaxBartEncoder(self.config, dtype=self.dtype, embed_tokens=self.shared)

        # the decoder has a different config
        decoder_config = BartConfig(self.config.to_dict())
        decoder_config.max_position_embeddings = OUTPUT_LENGTH
        decoder_config.vocab_size = OUTPUT_VOCAB_SIZE
        self.decoder = FlaxBartDecoder(decoder_config, dtype=self.dtype, embed_tokens=self.decoder_embed)

class CustomFlaxBartForConditionalGenerationModule(FlaxBartForConditionalGenerationModule):
    def setup(self):
        self.model = CustomFlaxBartModule(config=self.config, dtype=self.dtype)
        self.lm_head = nn.Dense(
            OUTPUT_VOCAB_SIZE,
            use_bias=False,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(self.config.init_std, self.dtype),
        )
        self.final_logits_bias = self.param("final_logits_bias", self.bias_init, (1, OUTPUT_VOCAB_SIZE))

class CustomFlaxBartForConditionalGeneration(FlaxBartForConditionalGeneration):
    module_class = CustomFlaxBartForConditionalGenerationModule

class Text2Image():
    def __init__(self, DALLE_MODEL, VQGAN_REPO, VQGAN_COMMIT_ID, CLIP_REPO, DALLE_COMMIT_ID=None, CLIP_COMMIT_ID=None):
        tokenizer = BartTokenizer.from_pretrained(DALLE_MODEL, revision=DALLE_COMMIT_ID)
        model = CustomFlaxBartForConditionalGeneration.from_pretrained(
            DALLE_MODEL, revision=DALLE_COMMIT_ID
        )
        self.model = model
        self.tokenizer = tokenizer

        # Load VQGAN
        self.vqgan = VQModel.from_pretrained(VQGAN_REPO, revision=VQGAN_COMMIT_ID)

        # Load CLIP
        self.clip = FlaxCLIPModel.from_pretrained(CLIP_REPO, revision=CLIP_COMMIT_ID)
        self.processor = CLIPProcessor.from_pretrained(CLIP_REPO, revision=CLIP_COMMIT_ID)

        self.bart_params = replicate(model.params)
        self.vqgan_params = replicate(self.vqgan.params)
        self.clip_params = replicate(self.clip.params)

    def inference(self, prompt, num_candidates = 32, k = 8):
        prompt = [prompt] * jax.device_count()
        inputs = self.tokenizer(
            prompt,
            return_tensors="jax",
            padding="max_length",
            truncation=True,
            max_length=128,
        ).data
        inputs = shard(inputs)
        model = self.model
        vqgan = self.vqgan

        def generate(input, rng, params):
            return model.generate(
                **input,
                max_length=257,
                num_beams=1,
                do_sample=True,
                prng_key=rng,
                eos_token_id=50000,
                pad_token_id=50000,
                params=params,
            )
        p_generate = jax.pmap(generate, "batch")

        def get_images(indices, params):
            return vqgan.decode_code(indices, params=params)
        p_get_images = jax.pmap(get_images, "batch")

        def custom_to_pil(x):
            x = np.clip(x, 0.0, 1.0)
            x = (255 * x).astype(np.uint8)
            x = Image.fromarray(x)
            if not x.mode == "RGB":
                x = x.convert("RGB")
            return x

        # inference
        all_images = []
        for _ in range(num_candidates // jax.device_count()):
            key = random.randint(0, 1e7)
            rng = jax.random.PRNGKey(key)
            rngs = jax.random.split(rng, jax.local_device_count())
            indices = p_generate(inputs, rngs, self.bart_params).sequences
            indices = indices[:, :, 1:]

            images = p_get_images(indices, self.vqgan_params)
            images = np.squeeze(np.asarray(images), 1)
            for image in images:
                all_images.append(custom_to_pil(image))

        sorted_images = self.clip_top_k(all_images, prompt, k=k)
        return sorted_images

    def clip_top_k(self, images, prompt, k=8):
        inputs = self.processor(text=prompt, images=images, return_tensors="np", padding=True)
        outputs = self.clip(**inputs)
        logits = outputs.logits_per_text
        scores = np.array(logits[0]).argsort()[-k:][::-1]
        return [images[score] for score in scores]

    def captioned_strip(self, images, caption=None, rows=1):
        increased_h = 0 if caption is None else 48
        w, h = images[0].size[0], images[0].size[1]
        img = Image.new("RGB", (len(images) * w // rows, h * rows + increased_h))
        for i, img_ in enumerate(images):
            img.paste(img_, (i // rows * w, increased_h + (i % rows) * h))

        if caption is not None:
            draw = ImageDraw.Draw(img)
            font = ImageFont.truetype(
                "/usr/share/fonts/truetype/liberation2/LiberationMono-Bold.ttf", 40
            )
            draw.text((20, 3), caption, (255, 255, 255), font=font)
        return img

if __name__ == "__main__":
    # Model references

    # dalle-mini
    # DALLE_MODEL = "dalle-mini/dalle-mini/model-3bqwu04f:latest"  # can be wandb artifact or 🤗 Hub or local folder
    # DALLE_MODEL = "tools/inference/model"
    DALLE_MODEL = "flax-community/dalle-mini"
    DALLE_COMMIT_ID = "4d34126d0df8bc4a692ae933e3b902a1fa8b6114"  # used only with 🤗 hub

    # VQGAN model
    # VQGAN_REPO = "dalle-mini/vqgan_imagenet_f16_16384"
    VQGAN_REPO = "dalle-mini/vqgan_imagenet_f16_16384"
    VQGAN_COMMIT_ID = "90cc46addd2dd8f5be21586a9a23e1b95aa506a9"

    # CLIP model
    CLIP_REPO = "openai/clip-vit-base-patch32"
    CLIP_COMMIT_ID = None

    gen = Text2Image(DALLE_MODEL, VQGAN_REPO=VQGAN_REPO, VQGAN_COMMIT_ID=VQGAN_COMMIT_ID, CLIP_REPO=CLIP_REPO, DALLE_COMMIT_ID=DALLE_COMMIT_ID)
    ranked_images = gen.inference("A woman standing in a room with a computer and other items")