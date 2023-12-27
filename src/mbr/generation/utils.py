import copy
import inspect
import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, List, Optional, Union, Tuple

import torch
import torch.distributed as dist
from tqdm import tqdm
from transformers import GenerationMixin, GenerationConfig
from transformers.generation.logits_process import LogitsProcessorList
from transformers.generation.stopping_criteria import StoppingCriteriaList
from transformers.generation.utils import GenerationMode
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
from transformers.utils import logging, ModelOutput

from mbr.generation.configuration_utils import MBRGenerationConfig
from mbr.metrics.base import MetricRunner, MetricOutput

if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizer
    from transformers.generation.streamers import BaseStreamer

logger = logging.get_logger(__name__)


@dataclass
class MBROutput(ModelOutput):
    """
    Base class for outputs of generation models when using MBR decoding.

    Args:
        sequences (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            The generated sequences. The second dimension (sequence_length) is either equal to `max_length` or shorter
            if all batches finished early due to the `eos_token_id`.
        all_samples (`tuple(ModelOutput)`), *optional*, returned when `output_all_samples=True` is passed or when
        `config.output_all_samples=True`):
            The model outputs for all sampled sequences (scores, hidden states, attentions etc.). The tuple contains
            `num_samples` output instances, each of which contains one sample per batch item.
        selected_samples_indices (`torch.LongTensor` of shape `(batch_size,)`), *optional*, returned when
        `output_all_samples=True` is passed or when `config.output_all_samples=True`):
            The indices (in `all_samples`) of the selected sequences for each batch item.
        references (`tuple(ModelOutput)`), *optional*, returned when `output_all_samples=True` is passed or when
        `config.output_all_samples=True`):
        metric_scores (`MetricOutput`), *optional*, returned when `output_metric_scores=True` is passed or when
        `config.output_metric_scores=True`):
            The output of the metric.
    """

    sequences: torch.LongTensor = None
    all_samples: Optional[Tuple[ModelOutput]] = None
    selected_samples_indices: Optional[torch.LongTensor] = None
    references: Optional[Tuple[ModelOutput]] = None
    metric_scores: Optional[MetricOutput] = None


class MBRGenerationMixin(GenerationMixin):

    @torch.no_grad()
    def generate(
            self,
            inputs: Optional[torch.Tensor] = None,
            generation_config: Optional[GenerationConfig] = None,
            references_config: Optional[GenerationConfig] = None,
            mbr_config: Optional[MBRGenerationConfig] = None,
            tokenizer: Optional["PreTrainedTokenizer"] = None,
            metric_runner: Optional[MetricRunner] = None,
            logits_processor: Optional[LogitsProcessorList] = None,
            stopping_criteria: Optional[StoppingCriteriaList] = None,
            prefix_allowed_tokens_fn: Optional[Callable[[int, torch.Tensor], List[int]]] = None,
            synced_gpus: Optional[bool] = None,
            assistant_model: Optional["PreTrainedModel"] = None,
            streamer: Optional["BaseStreamer"] = None,
            negative_prompt_ids: Optional[torch.Tensor] = None,
            negative_prompt_attention_mask: Optional[torch.Tensor] = None,
            progress_bar: bool = False,
            **kwargs,
    ) -> Union[MBROutput, torch.LongTensor]:
        r"""

        Generates sequences of token ids for models with a language modeling head, using MBR decoding.

        <Tip warning={true}>

        Most sampling-controlling parameters are set in `generation_config` which, if not passed, will be set to the
        model's default generation configuration. You can override any `generation_config` by passing the corresponding
        parameters to generate(), e.g. `.generate(inputs, epsilon_cutoff=3e-4)`.

        For an overview of sampling strategies and code examples, check out the [following
        guide](../generation_strategies).

        </Tip>

        Parameters:
            inputs (`torch.Tensor` of varying shape depending on the modality, *optional*):
                The sequence used as a prompt for the generation or as model inputs to the encoder. If `None` the
                method initializes it with `bos_token_id` and a batch size of 1. For decoder-only models `inputs`
                should be in the format of `input_ids`. For encoder-decoder models *inputs* can represent any of
                `input_ids`, `input_values`, `input_features`, or `pixel_values`.
            generation_config (`GenerationConfig`, *optional*):
                The generation configuration to be used as base parametrization for sampling. `**kwargs`
                passed to generate matching the attributes of `generation_config` will override them. If
                `generation_config` is not provided, the default will be used, which had the following loading
                priority: 1) from the `generation_config.json` model file, if it exists; 2) from the model
                configuration. Please note that unspecified parameters will inherit [`~generation.GenerationConfig`]'s
                default values, whose documentation should be checked to parameterize sampling.
            references_config (`GenerationConfig`, *optional*):
                The generation configuration to be used for the generation of pseudo-references.
                If `None`, `generation_config` will be used.
            mbr_config (`~mbr.generation.MBRGenerationConfig`, *optional*):
                The generation configuration to be used as base parametrization for MBR decoding. If `None`, the
                default will be used, which had the following loading priority: 1) from the `mbr_config.json` model
                file, if it exists; 2) from the model configuration. Please note that unspecified parameters will
                inherit [`~mbr.generation.MBRGenerationConfig`]'s default values.
            tokenizer (`PreTrainedTokenizer`, *optional*):
                A tokenizer that can be used to convert the generated token ids to strings, which is usually
                required for computing the metric.
            metric_runner (`~mbr.metrics.MetricRunner`, *optional*):
                An instance of a metric runner that will be used to compute the metric. If `None`, the default
                metric runner will be used.
            logits_processor (`LogitsProcessorList`, *optional*):
                Custom logits processors that complement the default logits processors built from arguments and
                generation config. If a logit processor is passed that is already created with the arguments or a
                generation config an error is thrown. This feature is intended for advanced users.
            stopping_criteria (`StoppingCriteriaList`, *optional*):
                Custom stopping criteria that complement the default stopping criteria built from arguments and a
                generation config. If a stopping criteria is passed that is already created with the arguments or a
                generation config an error is thrown. This feature is intended for advanced users.
            prefix_allowed_tokens_fn (`Callable[[int, torch.Tensor], List[int]]`, *optional*):
                If provided, this function constraints the beam search to allowed tokens only at each step. If not
                provided no constraint is applied. This function takes 2 arguments: the batch ID `batch_id` and
                `input_ids`. It has to return a list with the allowed tokens for the next generation step conditioned
                on the batch ID `batch_id` and the previously generated tokens `inputs_ids`. This argument is useful
                for constrained generation conditioned on the prefix, as described in [Autoregressive Entity
                Retrieval](https://arxiv.org/abs/2010.00904).
            synced_gpus (`bool`, *optional*):
                Whether to continue running the while loop until max_length. Unless overridden this flag will be set to
                `True` under DeepSpeed ZeRO Stage 3 multiple GPUs environment to avoid hanging if one GPU finished
                generating before other GPUs. Otherwise it'll be set to `False`.
            assistant_model (`PreTrainedModel`, *optional*):
                An assistant model that can be used to accelerate generation. The assistant model must have the exact
                same tokenizer. The acceleration is achieved when forecasting candidate tokens with the assistent model
                is much faster than running generation with the model you're calling generate from. As such, the
                assistant model should be much smaller.
            streamer (`BaseStreamer`, *optional*):
                Streamer object that will be used to stream the generated sequences. Generated tokens are passed
                through `streamer.put(token_ids)` and the streamer is responsible for any further processing.
            negative_prompt_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                The negative prompt needed for some processors such as CFG. The batch size must match the input batch
                size. This is an experimental feature, subject to breaking API changes in future versions.
            negative_prompt_attention_mask (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Attention_mask for `negative_prompt_ids`.
            progress_bar (`bool`, defaults to `False`):
                Whether or not to show a progress bar while generating the samples.
            kwargs (`Dict[str, Any]`, *optional*):
                Ad hoc parametrization of `generate_config` and/or additional model-specific kwargs that will be
                forwarded to the `forward` function of the model. If the model is an encoder-decoder model, encoder
                specific kwargs should not be prefixed and decoder specific kwargs should be prefixed with *decoder_*.

        Return:
            [`~utils.ModelOutput`] or `torch.LongTensor`: A [`~utils.ModelOutput`] (if `return_dict_in_generate=True`
            or when `config.return_dict_in_generate=True`) or a `torch.FloatTensor`.

                If the model is *not* an encoder-decoder model (`model.config.is_encoder_decoder=False`), the possible
                [`~utils.ModelOutput`] types are:

                    - [`~generation.MBRDecoderOnlyOutput`],

                If the model is an encoder-decoder model (`model.config.is_encoder_decoder=True`), the possible
                [`~utils.ModelOutput`] types are:

                    - [`~generation.MBREncoderDecoderOutput`],
        """
        if streamer is not None:
            raise NotImplementedError("streamer is not supported with MBR decoding.")

        if generation_config is not None and not isinstance(generation_config, GenerationConfig):
            raise ValueError(
                f"`generation_config` has to be of type `GenerationConfig`, but is {type(generation_config)}"
            )
        if references_config is not None and not isinstance(references_config, GenerationConfig):
            raise ValueError(
                f"`references_config` has to be of type `GenerationConfig`, but is {type(references_config)}"
            )
        if mbr_config is not None and not isinstance(mbr_config, MBRGenerationConfig):
            raise ValueError(f"`mbr_config` has to be of type `MBRGenerationConfig`, but is {type(mbr_config)}")
        if mbr_config is None:
            raise ValueError("`mbr_config` must be passed to `generate()`.")
        if tokenizer is None and metric_runner is None:
            raise ValueError("`tokenizer` must be passed to `generate()`, because computing the metric usually "
                             "requires decoding the generated token ids.")

        if synced_gpus is None:
            if is_deepspeed_zero3_enabled() and dist.get_world_size() > 1:
                synced_gpus = True
            else:
                synced_gpus = False

        # 1. Handle `generation_config` and kwargs that might update it, and validate the `.generate()` call
        self._validate_model_class()

        # priority: `generation_config` argument > `model.generation_config` (the default generation config)
        if generation_config is None:
            # legacy: users may modify the model configuration to control generation. To trigger this legacy behavior,
            # two conditions must be met
            # 1) the generation config must have been created from the model config (`_from_model_config` field);
            # 2) the generation config must have seen no modification since its creation (the hash is the same).
            if self.generation_config._from_model_config and self.generation_config._original_object_hash == hash(
                    self.generation_config
            ):
                new_generation_config = GenerationConfig.from_model_config(self.config)
                if new_generation_config != self.generation_config:
                    warnings.warn(
                        "You have modified the pretrained model configuration to control generation. This is a"
                        " deprecated strategy to control generation and will be removed soon, in a future version."
                        " Please use and modify the model generation configuration (see"
                        " https://huggingface.co/docs/transformers/generation_strategies#default-text-generation-configuration )"
                    )
                    self.generation_config = new_generation_config
            generation_config = self.generation_config

        generation_config = copy.deepcopy(generation_config)
        model_kwargs = generation_config.update(**kwargs)  # All unused kwargs must be model kwargs
        generation_config.validate()
        self._validate_model_kwargs(model_kwargs.copy())

        if references_config is not None:
            references_config.validate()

        # 2. Set generation parameters if not already defined
        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
        stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()

        if generation_config.pad_token_id is None and generation_config.eos_token_id is not None:
            if model_kwargs.get("attention_mask", None) is None:
                logger.warning(
                    "The attention mask and the pad token id were not set. As a consequence, you may observe "
                    "unexpected behavior. Please pass your input's `attention_mask` to obtain reliable results."
                )
            eos_token_id = generation_config.eos_token_id
            if isinstance(eos_token_id, list):
                eos_token_id = eos_token_id[0]
            logger.warning(f"Setting `pad_token_id` to `eos_token_id`:{eos_token_id} for open-end generation.")
            generation_config.pad_token_id = eos_token_id
        if references_config is not None:
            references_config.pad_token_id = generation_config.pad_token_id
            references_config.eos_token_id = generation_config.eos_token_id

        # 3. Define model inputs
        # inputs_tensor has to be defined
        # model_input_name is defined if model-specific keyword input is passed
        # otherwise model_input_name is None
        # all model-specific keyword inputs are removed from `model_kwargs`
        inputs_tensor, model_input_name, model_kwargs = self._prepare_model_inputs(
            inputs, generation_config.bos_token_id, model_kwargs
        )
        batch_size = inputs_tensor.shape[0]

        # 4. Define other model kwargs
        # synchronize generation configs w.r.t. returned data
        if mbr_config.return_dict_in_generate:
            generation_config.return_dict_in_generate = True
            if references_config is not None:
                references_config.return_dict_in_generate = True
        if mbr_config.output_scores:
            generation_config.output_scores = True
            if references_config is not None:
                references_config.output_scores = True
        if mbr_config.output_attentions:
            generation_config.output_attentions = True
            if references_config is not None:
                references_config.output_attentions = True
        if mbr_config.output_hidden_states:
            generation_config.output_hidden_states = True
            if references_config is not None:
                references_config.output_hidden_states = True

        model_kwargs["output_attentions"] = generation_config.output_attentions
        model_kwargs["output_hidden_states"] = generation_config.output_hidden_states
        # decoder-only models with inputs_embeds forwarding must use caching (otherwise we can't detect whether we are
        # generating the first new token or not, and we only want to use the embeddings for the first new token)
        if not self.config.is_encoder_decoder and model_input_name == "inputs_embeds":
            model_kwargs["use_cache"] = True
        else:
            model_kwargs["use_cache"] = generation_config.use_cache

        accepts_attention_mask = "attention_mask" in set(inspect.signature(self.forward).parameters.keys())
        requires_attention_mask = "encoder_outputs" not in model_kwargs

        if model_kwargs.get("attention_mask", None) is None and requires_attention_mask and accepts_attention_mask:
            model_kwargs["attention_mask"] = self._prepare_attention_mask_for_generation(
                inputs_tensor, generation_config.pad_token_id, generation_config.eos_token_id
            )

        # decoder-only models should use left-padding for generation
        if not self.config.is_encoder_decoder:
            # If `input_ids` was given, check if the last id in any sequence is `pad_token_id`
            # Note: If using, `inputs_embeds` this check does not work, because we want to be more hands-off.
            if (
                    generation_config.pad_token_id is not None
                    and len(inputs_tensor.shape) == 2
                    and torch.sum(inputs_tensor[:, -1] == generation_config.pad_token_id) > 0
            ):
                logger.warning(
                    "A decoder-only architecture is being used, but right-padding was detected! For correct "
                    "generation results, please set `padding_side='left'` when initializing the tokenizer."
                )

        if self.config.is_encoder_decoder and "encoder_outputs" not in model_kwargs:
            # if model is encoder decoder encoder_outputs are created
            # and added to `model_kwargs`
            model_kwargs = self._prepare_encoder_decoder_kwargs_for_generation(
                inputs_tensor, model_kwargs, model_input_name
            )

        # 5. Prepare `input_ids` which will be used for auto-regressive generation
        if self.config.is_encoder_decoder:
            input_ids, model_kwargs = self._prepare_decoder_input_ids_for_generation(
                batch_size=batch_size,
                model_input_name=model_input_name,
                model_kwargs=model_kwargs,
                decoder_start_token_id=generation_config.decoder_start_token_id,
                bos_token_id=generation_config.bos_token_id,
                device=inputs_tensor.device,
            )
        else:
            input_ids = inputs_tensor if model_input_name == "input_ids" else model_kwargs.pop("input_ids")

        # 6. Prepare `max_length` depending on other stopping criteria.
        input_ids_length = input_ids.shape[-1]
        has_default_max_length = kwargs.get("max_length") is None and generation_config.max_length is not None
        if generation_config.max_new_tokens is not None:
            if not has_default_max_length and generation_config.max_length is not None:
                logger.warning(
                    f"Both `max_new_tokens` (={generation_config.max_new_tokens}) and `max_length`(="
                    f"{generation_config.max_length}) seem to have been set. `max_new_tokens` will take precedence. "
                    "Please refer to the documentation for more information. "
                    "(https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)"
                )
            generation_config.max_length = generation_config.max_new_tokens + input_ids_length
        self._validate_generated_length(generation_config, input_ids_length, has_default_max_length)

        # 7. determine generation mode
        generation_mode = self._get_generation_mode(generation_config, assistant_model)
        if references_config is not None:
            references_mode = self._get_generation_mode(references_config, assistant_model)
        else:
            references_mode = generation_mode
        supported_modes = {GenerationMode.SAMPLE}
        if generation_mode not in supported_modes or references_mode not in supported_modes:
            raise NotImplementedError(
                f"MBR decoding is not implemented with generation modes '{generation_mode}/{references_mode}'."
                f"Only '{GenerationMode.SAMPLE}' is currently supported. "
                f"Set `do_sample=True` and `num_beams=1` to enable sampling."
            )

        if streamer is not None and (generation_config.num_beams > 1):
            raise ValueError(
                "`streamer` cannot be used with beam search (yet!). Make sure that `num_beams` is set to 1."
            )

        if self.device.type != input_ids.device.type:
            warnings.warn(
                "You are calling .generate() with the `input_ids` being on a device type different"
                f" than your model's device. `input_ids` is on {input_ids.device.type}, whereas the model"
                f" is on {self.device.type}. You may experience unexpected behaviors or slower generation."
                " Please make sure that you have put `input_ids` to the"
                f" correct device by calling for example input_ids = input_ids.to('{self.device.type}') before"
                " running `.generate()`.",
                UserWarning,
            )

        # 8. prepare distribution pre_processing samplers
        if references_config is not None:
            references_logits_processor = self._get_logits_processor(
                generation_config=references_config,
                input_ids_seq_length=input_ids_length,
                encoder_input_ids=inputs_tensor,
                prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
                logits_processor=copy.deepcopy(logits_processor),
                model_kwargs=model_kwargs,
                negative_prompt_ids=negative_prompt_ids,
                negative_prompt_attention_mask=negative_prompt_attention_mask,
            )
        logits_processor = self._get_logits_processor(
            generation_config=generation_config,
            input_ids_seq_length=input_ids_length,
            encoder_input_ids=inputs_tensor,
            prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
            logits_processor=logits_processor,
            model_kwargs=model_kwargs,
            negative_prompt_ids=negative_prompt_ids,
            negative_prompt_attention_mask=negative_prompt_attention_mask,
        )

        # 9. prepare stopping criteria
        if references_config is not None:
            references_stopping_criteria = self._get_stopping_criteria(
                generation_config=references_config, stopping_criteria=copy.deepcopy(stopping_criteria)
            )
        stopping_criteria = self._get_stopping_criteria(
            generation_config=generation_config, stopping_criteria=stopping_criteria
        )

        # 10. run generation
        samples = []  # num_samples x batch_size
        references = []  # num_references x batch_size

        if generation_mode == GenerationMode.SAMPLE:
            # 11. prepare logits warper
            logits_warper = self._get_logits_warper(generation_config)

            # 12. expand input_ids with `num_return_sequences` additional sequences per batch
            input_ids, model_kwargs = self._expand_inputs_for_generation(
                input_ids=input_ids,
                expand_size=generation_config.num_return_sequences,
                is_encoder_decoder=self.config.is_encoder_decoder,
                **model_kwargs,
            )

            # 13. run sampling

            # Implementation note: We use a loop here instead of `num_return_sequences`. As a result, `samples` is a
            # list of model outputs of length `num_samples`. Each of the model outputs covers the whole batch.
            # A disadvantage of this approach is that the batch dimension is now the second dimension, not the first.
            # However, it allows us to easily maintain the batch size chosen by the user, whereas `num_return_sequences`
            # would use a batch size of `batch_size`*`num_samples`.
            for i in tqdm(list(range(mbr_config.num_samples)), disable=not progress_bar):
                samples.append(self.sample(
                    input_ids,
                    logits_processor=logits_processor,
                    logits_warper=logits_warper,
                    stopping_criteria=stopping_criteria,
                    pad_token_id=generation_config.pad_token_id,
                    eos_token_id=generation_config.eos_token_id,
                    output_scores=generation_config.output_scores,
                    return_dict_in_generate=generation_config.return_dict_in_generate,
                    synced_gpus=synced_gpus,
                    streamer=streamer,
                    **model_kwargs,
                ))

            # 14. references
            if references_config is None:
                # Use samples as references
                references = samples[:mbr_config.num_references]
            else:
                # Generate references
                logits_warper = self._get_logits_warper(references_config)

                input_ids, model_kwargs = self._expand_inputs_for_generation(
                    input_ids=input_ids,
                    expand_size=references_config.num_return_sequences,
                    is_encoder_decoder=self.config.is_encoder_decoder,
                    **model_kwargs,
                )

                for i in tqdm(list(range(mbr_config.num_references)), disable=not progress_bar):
                    references.append(self.sample(
                        input_ids,
                        logits_processor=references_logits_processor,
                        logits_warper=logits_warper,
                        stopping_criteria=references_stopping_criteria,
                        pad_token_id=references_config.pad_token_id,
                        eos_token_id=references_config.eos_token_id,
                        output_scores=references_config.output_scores,
                        return_dict_in_generate=references_config.return_dict_in_generate,
                        synced_gpus=synced_gpus,
                        streamer=streamer,
                        **model_kwargs,
                    ))

        # 15. apply metric to samples
        if metric_runner is None:
            metric_runner = MetricRunner(mbr_config, tokenizer)

        if isinstance(samples[0], ModelOutput):
            sample_ids = tuple(sample.sequences for sample in samples)
        else:
            sample_ids = samples
        if isinstance(references[0], ModelOutput):
            reference_ids = tuple(reference.sequences for reference in references)
        else:
            reference_ids = references

        metric_output = metric_runner(input_ids, sample_ids, reference_ids)
        if not mbr_config.lower_is_better:
            top_metric_scores, top_metric_indices = metric_output.scores.max(dim=-1)
        else:
            top_metric_scores, top_metric_indices = metric_output.scores.min(dim=-1)

        # Copy top samples into a tensor of shape (batch_size, max_length)
        max_length = max(sample.shape[1] for sample in sample_ids)
        output = MBROutput(
            sequences=generation_config.pad_token_id * torch.ones((batch_size, max_length), dtype=torch.long),
            all_samples=(tuple(samples) if mbr_config.output_all_samples else None),
            selected_samples_indices=(top_metric_indices if mbr_config.output_all_samples else None),
            references=(tuple(references) if mbr_config.output_all_samples else None),
            metric_scores=(metric_output if mbr_config.output_metric_scores else None),
        )
        for batch_idx, sample_idx in enumerate(top_metric_indices):
            output.sequences[batch_idx][:sample_ids[sample_idx].shape[1]] = sample_ids[sample_idx][batch_idx]

        if mbr_config.return_dict_in_generate:
            return output
        else:
            return output.sequences
