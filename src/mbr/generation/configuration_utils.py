import os
from typing import Union, Optional

from transformers import GenerationConfig

from transformers import __version__ as transformers_version
from transformers.utils import logging

from mbr.generation import MBR_GENERATION_CONFIG_NAME

logger = logging.get_logger(__name__)


class MBRGenerationConfig(GenerationConfig):
    r"""
    Class that holds a configuration for a generation task. A `generate` call supports the following generation method
    for text-decoder, text-to-text, speech-to-text, and vision-to-text models:

        - *minimum Bayes risk decoding* by calling [`~generation.MBRGenerationMixin.mbr_decoding`]

    You do not need to call the above method directly; call '.generate()' instead.

    Arg:
        num_samples (`int`, *optional*, defaults to 10):
            Number of samples generated. 1 means no MBR decoding.
        num_references (`int`, *optional*, defaults to `num_samples`):
            Number of pseudo-references used for MBR decoding.
        metric (`str` or `~evaluate.Metric`, *optional*, defaults to 'chrf'):
            Metric used for MBR decoding.
        metric_config_name (`str`, *optional*, defaults to None):
            Metric configuration to pass to `evaluate.load` (e.g., the model for a trained metric, such as
            "eamt22-cometinho-da"). If not specified, the default configuration is used.
        metric_output_field (`str`, *optional*, defaults to 'score'):
            Field of the metric output that is used
        metric_kwargs (optional):
            Additional arguments for the metric's `compute` method. The default MetricRunner requires it to be hashable.
        lower_is_better (`bool`, *optional*, defaults to `False`):
            Set to true if lower metric scores are better (e.g., perplexity).

        > Parameters that define the output variables of `generate`

        output_attentions (`bool`, *optional*, defaults to `False`):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more details.
        output_hidden_states (`bool`, *optional*, defaults to `False`):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more details.
        output_scores (`bool`, *optional*, defaults to `False`):
            Whether or not to return the prediction scores. See `scores` under returned tensors for more details.
        output_all_samples (`bool`, *optional*, defaults to `False`):
            Whether or not to return all sampled sequences. See `all_sampled_sequences` under returned tensors for more
            details.
        output_reference_sequences (`bool`, *optional*, defaults to `False`):
            Whether or not to return the reference sequences. See `reference_sequences` under returned tensors for more
            details.
        output_metric_scores (`bool`, *optional*, defaults to `False`):
            Whether or not to return the metric scores. See `metric_scores` under returned tensors for more details.
        return_dict_in_generate (`bool`, *optional*, defaults to `False`):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
    """

    def __init__(self, **kwargs):
        # Parameters that control the generation strategy used
        self.num_samples = kwargs.pop("num_samples", 10)
        self.num_references = kwargs.pop("num_references", self.num_samples)
        self.metric = kwargs.pop("metric", "chrf")
        self.metric_config_name = kwargs.pop("metric_config_name", None)
        self.metric_output_field = kwargs.pop("metric_output_field", "score")
        self.metric_kwargs = kwargs.pop("metric_kwargs", {})
        self.lower_is_better = kwargs.pop("lower_is_better", False)

        # Parameters that define the output variables of `generate`
        self.output_attentions = kwargs.pop("output_attentions", False)
        self.output_hidden_states = kwargs.pop("output_hidden_states", False)
        self.output_scores = kwargs.pop("output_scores", False)
        self.output_all_samples = kwargs.pop("output_all_samples", False)
        self.output_reference_sequences = kwargs.pop("output_reference_sequences", False)
        self.output_metric_scores = kwargs.pop("output_metric_scores", False)
        self.return_dict_in_generate = kwargs.pop("return_dict_in_generate", False)

        # The remaining attributes do not parametrize `.generate()`, but are informative and/or used by the hub
        # interface.
        self._from_model_config = kwargs.pop("_from_model_config", False)
        self._commit_hash = kwargs.pop("_commit_hash", None)
        self.transformers_version = kwargs.pop("transformers_version", transformers_version)
        import mbr
        self.mbr_version = kwargs.pop("mbr_version", mbr.__version__)

        # Additional attributes without default values
        if not self._from_model_config:
            # we don't want to copy values from the model config if we're initializing an `MBRGenerationConfig` from a
            # model's default configuration file
            for key, value in kwargs.items():
                try:
                    setattr(self, key, value)
                except AttributeError as err:
                    logger.error(f"Can't set {key} with value {value} for {self}")
                    raise err

        # Validate the values of the attributes
        self.validate(is_init=True)

    def save_pretrained(
            self,
            save_directory: Union[str, os.PathLike],
            config_file_name: Optional[Union[str, os.PathLike]] = MBR_GENERATION_CONFIG_NAME,
            **kwargs,
    ):
        super().save_pretrained(save_directory, config_file_name=config_file_name, **kwargs)

    @classmethod
    def from_pretrained(
            cls,
            pretrained_model_name: Union[str, os.PathLike],
            config_file_name: Optional[Union[str, os.PathLike]] = MBR_GENERATION_CONFIG_NAME,
            **kwargs,
    ) -> "MBRGenerationConfig":
        generation_config = super().from_pretrained(pretrained_model_name, config_file_name=config_file_name, **kwargs)
        return cls.from_dict(generation_config.to_dict())
