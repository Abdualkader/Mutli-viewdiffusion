from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from diffusers.callbacks import MultiPipelineCallbacks, PipelineCallback
except:

    class MultiPipelineCallbacks:
        ...

    class PipelineCallback:
        ...


from diffusers.image_processor import PipelineImageInput
from diffusers.models import AutoencoderKL, UNet2DConditionModel
from diffusers.models.attention import Attention
from diffusers.models.attention_processor import AttnProcessor2_0
from diffusers.pipelines.stable_diffusion.pipeline_output import (
    StableDiffusionPipelineOutput,
)
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import (
    StableDiffusionPipeline,
    rescale_noise_cfg,
    retrieve_timesteps,
)
from diffusers.pipelines.stable_diffusion.safety_checker import (
    StableDiffusionSafetyChecker,
)
from diffusers.schedulers import KarrasDiffusionSchedulers
from diffusers.utils import deprecate
from transformers import (
    CLIPImageProcessor,
    CLIPTextModel,
    CLIPTokenizer,
    CLIPVisionModel,
)


class MVDiffusionPipeline(StableDiffusionPipeline):
    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: UNet2DConditionModel,
        scheduler: KarrasDiffusionSchedulers,
        safety_checker: StableDiffusionSafetyChecker,
        feature_extractor: Optional[CLIPImageProcessor] = None,
        image_encoder: Optional[CLIPVisionModel] = None,
        requires_safety_checker: bool = False,
    ) -> None:
        super().__init__(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=add_mv_attn_processor(unet),
            scheduler=scheduler,
            safety_checker=safety_checker,
            feature_extractor=feature_extractor,
            image_encoder=image_encoder,
            requires_safety_checker=requires_safety_checker,
        )
        self.num_views = 8

    def load_ip_adapter(
        self,
        pretrained_model_name_or_path_or_dict: Union[
            str, List[str], Dict[str, torch.Tensor]
        ] = "kiigii/imagedream-ipmv-diffusers",
        subfolder: Union[str, List[str]] = "ip_adapter",
        weight_name: Union[str, List[str]] = "ip-adapter-plus_imagedream.bin",
        image_encoder_folder: Optional[str] = "image_encoder",
        **kwargs,
    ) -> None:
        super().load_ip_adapter(
            pretrained_model_name_or_path_or_dict=pretrained_model_name_or_path_or_dict,
            subfolder=subfolder,
            weight_name=weight_name,
            image_encoder_folder=image_encoder_folder,
            **kwargs,
        )
        print("IP-Adapter Loaded.")

        if weight_name == "ip-adapter-plus_imagedream.bin":
            setattr(self.image_encoder, "visual_projection", nn.Identity())
            add_mv_attn_processor(self.unet)
            set_num_views(self.unet, self.num_views + 1)

    def unload_ip_adapter(self) -> None:
        super().unload_ip_adapter()
        set_num_views(self.unet, self.num_views)

    def encode_image_to_latents(
        self,
        image: PipelineImageInput,
        height: int,
        width: int,
        device: torch.device,
        num_images_per_prompt: int = 1,
    ):
        dtype = next(self.vae.parameters()).dtype

        if isinstance(image, torch.Tensor):
            image = F.interpolate(
                image,
                (height, width),
                mode="bilinear",
                align_corners=False,
                antialias=True,
            )
        else:
            image = self.image_processor.preprocess(image, height, width)

        # image should be in range [-1, 1]
        image = image.to(device=device, dtype=dtype)

        def vae_encode(image):
            posterior = self.vae.encode(image).latent_dist
            latents = posterior.sample() * self.vae.config.scaling_factor
            latents = latents.repeat_interleave(num_images_per_prompt, dim=0)
            return latents

        latents = vae_encode(image)
        uncond_latents = vae_encode(torch.zeros_like(image))
        return latents, uncond_latents

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        elevation: float = 0.0,
        timesteps: List[int] = None,
        sigmas: List[float] = None,
        guidance_scale: float = 5.0,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        ip_adapter_image: Optional[PipelineImageInput] = None,
        # StableDiffusion support `ip_adapter_image_embeds` but we don't use, and raise ValueError.
        ip_adapter_image_embeds: Optional[List[torch.Tensor]] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        guidance_rescale: float = 0.0,
        clip_skip: Optional[int] = None,
        callback_on_step_end: Optional[
            Union[
                Callable[[int, int, Dict], None],
                PipelineCallback,
                MultiPipelineCallbacks,
            ]
        ] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        **kwargs,
    ):
        if ip_adapter_image_embeds is not None:
            raise ValueError(
                "do not use `ip_adapter_image_embeds` in ImageDream, use `ip_adapter_image`"
            )

        callback = kwargs.pop("callback", None)
        callback_steps = kwargs.pop("callback_steps", None)

        if callback is not None:
            deprecate(
                "callback",
                "1.0.0",
                "Passing `callback` as an input argument to `__call__` is deprecated, consider using `callback_on_step_end`",
            )
        if callback_steps is not None:
            deprecate(
                "callback_steps",
                "1.0.0",
                "Passing `callback_steps` as an input argument to `__call__` is deprecated, consider using `callback_on_step_end`",
            )

        if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
            callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs

        # ImageDream number of views
        if cross_attention_kwargs is None:
            num_views = self.num_views
        else:
            cross_attention_kwargs.pop("num_views", self.num_views)

        # 0. Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor
        # to deal with lora scaling and other possible forward hooks

        # 1. Check inputs. Raise error if not correct
        if prompt is None:
            prompt = ""
        self.check_inputs(
            prompt,
            height,
            width,
            callback_steps,
            negative_prompt,
            prompt_embeds,
            negative_prompt_embeds,
            ip_adapter_image,
            None,  # ip_adapter_image_embeds,
            callback_on_step_end_tensor_inputs,
        )

        self._guidance_scale = guidance_scale
        self._guidance_rescale = guidance_rescale
        self._clip_skip = clip_skip
        self._cross_attention_kwargs = cross_attention_kwargs
        self._interrupt = False

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device

        # 3. Encode input prompt
        lora_scale = (
            self.cross_attention_kwargs.get("scale", None)
            if self.cross_attention_kwargs is not None
            else None
        )

        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            self.do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            lora_scale=lora_scale,
            clip_skip=self.clip_skip,
        )

        # camera parameter for ImageDream
        camera = get_camera(
            num_views, elevation=elevation, extra_view=ip_adapter_image is not None
        ).to(dtype=prompt_embeds.dtype, device=device)
        camera = camera.repeat(batch_size * num_images_per_prompt, 1)

        if ip_adapter_image is not None:
            image_embeds = self.prepare_ip_adapter_image_embeds(
                ip_adapter_image,
                None,  # ip_adapter_image_embeds,
                device,
                batch_size * num_images_per_prompt,
                self.do_classifier_free_guidance,
            )
            # ImageDream
            image_latents, negative_image_latents = self.encode_image_to_latents(
                ip_adapter_image,
                height,
                width,
                device,
                batch_size * num_images_per_prompt,
            )
            num_views += 1

        # For classifier free guidance, we need to do two forward passes.
        # Here we concatenate the unconditional and text embeddings into a single batch
        # to avoid doing two forward passes
        if self.do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])
            camera = torch.cat([camera] * 2)
            if ip_adapter_image is not None:
                image_latents = torch.cat([negative_image_latents, image_latents])

        # Multi-view inputs for ImageDream.
        prompt_embeds = prompt_embeds.repeat_interleave(num_views, dim=0)
        if ip_adapter_image is not None:
            image_embeds = [i.repeat_interleave(num_views, dim=0) for i in image_embeds]

        # 4. Prepare timesteps
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler, num_inference_steps, device, timesteps, sigmas
        )

        # 5. Prepare latent variables
        num_channels_latents = self.unet.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt * num_views,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 6.1 Add image embeds for IP-Adapter
        if ip_adapter_image is not None:
            added_cond_kwargs = {"image_embeds": image_embeds}
        else:
            added_cond_kwargs = None

        # 6.2 Optionally get Guidance Scale Embedding
        timestep_cond = None
        if self.unet.config.time_cond_proj_dim is not None:
            guidance_scale_tensor = torch.tensor(self.guidance_scale - 1).repeat(
                batch_size * num_images_per_prompt
            )
            timestep_cond = self.get_guidance_scale_embedding(
                guidance_scale_tensor, embedding_dim=self.unet.config.time_cond_proj_dim
            ).to(device=device, dtype=latents.dtype)

        set_num_views(self.unet, num_views)

        # fmt: off
        # 7. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        self._num_timesteps = len(timesteps)
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                if ip_adapter_image is not None:
                    latent_model_input[num_views - 1 :: num_views, :, :, :] = image_latents
                # predict the noise residual
                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    class_labels=camera,
                    encoder_hidden_states=prompt_embeds,
                    timestep_cond=timestep_cond,
                    cross_attention_kwargs=self.cross_attention_kwargs,
                    added_cond_kwargs=added_cond_kwargs,
                    return_dict=False,
                )[0]

                # perform guidance
                if self.do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = torch.lerp(noise_pred_uncond, noise_pred_text, self.guidance_scale)

                if self.do_classifier_free_guidance and self.guidance_rescale > 0.0:
                    # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
                    noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale=self.guidance_rescale)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                    negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        step_idx = i // getattr(self.scheduler, "order", 1)
                        callback(step_idx, t, latents)
        # fmt: on
        if not output_type == "latent":
            image = self.vae.decode(
                latents / self.vae.config.scaling_factor,
                return_dict=False,
                generator=generator,
            )[0]
            image, has_nsfw_concept = self.run_safety_checker(
                image, device, prompt_embeds.dtype
            )
        else:
            image = latents
            has_nsfw_concept = None

        if has_nsfw_concept is None:
            do_denormalize = [True] * image.shape[0]
        else:
            do_denormalize = [not has_nsfw for has_nsfw in has_nsfw_concept]

        image = self.image_processor.postprocess(
            image, output_type=output_type, do_denormalize=do_denormalize
        )

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (image, has_nsfw_concept)

        return StableDiffusionPipelineOutput(
            images=image, nsfw_content_detected=has_nsfw_concept
        )


# fmt: off
# Copied from ImageDream
# https://github.com/bytedance/ImageDream/blob/main/extern/ImageDream/imagedream/camera_utils.py


def create_camera_to_world_matrix(elevation, azimuth):
    elevation = np.radians(elevation)
    azimuth = np.radians(azimuth)
    # Convert elevation and azimuth angles to Cartesian coordinates on a unit sphere
    x = np.cos(elevation) * np.sin(azimuth)
    y = np.sin(elevation)
    z = np.cos(elevation) * np.cos(azimuth)

    # Calculate camera position, target, and up vectors
    camera_pos = np.array([x, y, z])
    target = np.array([0, 0, 0])
    up = np.array([0, 1, 0])

    # Construct view matrix
    forward = target - camera_pos
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    new_up = np.cross(right, forward)
    new_up /= np.linalg.norm(new_up)
    cam2world = np.eye(4)
    cam2world[:3, :3] = np.array([right, new_up, -forward]).T
    cam2world[:3, 3] = camera_pos
    return cam2world


def convert_opengl_to_blender(camera_matrix):
    if isinstance(camera_matrix, np.ndarray):
        # Construct transformation matrix to convert from OpenGL space to Blender space
        flip_yz = np.array([[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
        camera_matrix_blender = np.dot(flip_yz, camera_matrix)
    else:
        # Construct transformation matrix to convert from OpenGL space to Blender space
        flip_yz = torch.tensor(
            [[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]]
        )
        if camera_matrix.ndim == 3:
            flip_yz = flip_yz.unsqueeze(0)
        camera_matrix_blender = torch.matmul(flip_yz.to(camera_matrix), camera_matrix)
    return camera_matrix_blender


def normalize_camera(camera_matrix):
    """normalize the camera location onto a unit-sphere"""
    if isinstance(camera_matrix, np.ndarray):
        camera_matrix = camera_matrix.reshape(-1, 4, 4)
        translation = camera_matrix[:, :3, 3]
        translation = translation / (
            np.linalg.norm(translation, axis=1, keepdims=True) + 1e-8
        )
        camera_matrix[:, :3, 3] = translation
    else:
        camera_matrix = camera_matrix.reshape(-1, 4, 4)
        translation = camera_matrix[:, :3, 3]
        translation = translation / (
            torch.norm(translation, dim=1, keepdim=True) + 1e-8
        )
        camera_matrix[:, :3, 3] = translation
    return camera_matrix.reshape(-1, 16)


def get_camera(
    num_frames,
    elevation=15,
    azimuth_start=0,
    azimuth_span=180,
    blender_coord=True,
    extra_view=False,
):
    angle_gap = azimuth_span / num_frames
    cameras = []
    for azimuth in np.arange(azimuth_start, azimuth_span + azimuth_start, angle_gap):
        camera_matrix = create_camera_to_world_matrix(elevation, azimuth)
        if blender_coord:
            camera_matrix = convert_opengl_to_blender(camera_matrix)
        cameras.append(camera_matrix.flatten())

    if extra_view:
        dim = len(cameras[0])
        cameras.append(np.zeros(dim))
    return torch.tensor(np.stack(cameras, 0)).float()
# fmt: on


def add_mv_attn_processor(unet: UNet2DConditionModel, num_views: int = 4) -> UNet2DConditionModel:
    attn_procs = {}
    for key, attn_processor in unet.attn_processors.items():
        if "attn1" in key:
            attn_procs[key] = MVAttnProcessor2_0(num_views)
        else:
            attn_procs[key] = attn_processor
    unet.set_attn_processor(attn_procs)
    return unet


def set_num_views(unet: UNet2DConditionModel, num_views: int) -> UNet2DConditionModel:
    for key, attn_processor in unet.attn_processors.items():
        if isinstance(attn_processor, MVAttnProcessor2_0):
            attn_processor.num_views = num_views
    return unet


class MVAttnProcessor2_0(AttnProcessor2_0):
    def __init__(self, num_views: int = 4):
        super().__init__()
        self.num_views = num_views

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
        *args,
        **kwargs,
    ):
        if self.num_views == 1:
            return super().__call__(
                attn=attn,
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=attention_mask,
                temb=temb,
                *args,
                **kwargs,
            )

        input_ndim = hidden_states.ndim
        B = hidden_states.size(0)
        if B % self.num_views:
            raise ValueError(
                f"`batch_size`(got {B}) must be a multiple of `num_views`(got {self.num_views})."
            )
        real_B = B // self.num_views
        if input_ndim == 4:
            H, W = hidden_states.shape[2:]
            hidden_states = hidden_states.reshape(real_B, -1, H, W).transpose(1, 2)
        else:
            hidden_states = hidden_states.reshape(real_B, -1, hidden_states.size(-1))
        hidden_states = super().__call__(
            attn=attn,
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=attention_mask,
            temb=temb,
            *args,
            **kwargs,
        )
        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(B, -1, H, W)
        else:
            hidden_states = hidden_states.reshape(B, -1, hidden_states.size(-1))
        return hidden_states
