from transformers import __version__ as transformers_version
from transformers.utils import logging

logger = logging.get_logger(__name__)


class MBRConfig:
    r"""
    Class that holds a configuration for minimum Bayes risk decoding (MBR). Pass this config when calling
    `MBRGenerationMixin.generate()`:

        Example:

        ```python
        >>> config = MBRConfig(num_samples=10, num_references=10, metric="fastchrf")
        >>> model.generate(..., mbr_config=config)
        ```

    The class is inspired by `transformers.GenerationConfig`.
    Note that `MBRConfig` does not control the sampling strategy. Pass separate `GenerationConfig` objects to control
    sampling:

        ```python
        >>> generation_config = GenerationConfig(do_sample=True, num_beams=1, top_p=0.9)
        >>> references_config = GenerationConfig(do_sample=True, num_beams=1, epsilon_cutoff=0.02)
        >>> model.generate(..., mbr_config=config, generation_config=generation_config, references_config=references_config)
        ```

    Arg:
        num_samples (`int`, *optional*, defaults to 10):
            Number of samples generated. 1 means no MBR decoding.
        num_references (`int`, *optional*, defaults to `num_samples`):
            Number of pseudo-references used for MBR decoding.
        metric (`str` or `~evaluate.Metric`, *optional*, defaults to 'fastchrf'):
            Metric used for MBR decoding.
        metric_config_name (`str`, *optional*, defaults to None):
            Metric configuration to pass to `evaluate.load` (e.g., the model for a trained metric, such as
            "eamt22-cometinho-da"). If not specified, the default configuration is used.
        metric_output_field (`str`, *optional*, defaults to 'score'):
            Field of the metric output that is used
        metric_kwargs (optional):
            Additional arguments for the metric's `compute` method. The default MetricRunner requires it to be hashable.
        metric_cache_size (`int`, *optional*, defaults to `num_samples` * `num_references`):
            Size of the cache for the metric. Set to `None` to disable caching (not recommended).
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
        self.metric = kwargs.pop("metric", "fastchrf")
        self.metric_config_name = kwargs.pop("metric_config_name", None)
        self.metric_output_field = kwargs.pop("metric_output_field", "score")
        self.metric_kwargs = kwargs.pop("metric_kwargs", {})
        self.metric_cache_size = kwargs.pop("metric_cache_size", self.num_samples * self.num_references)
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
            # we don't want to copy values from the model config if we're initializing an `MBRConfig` from a
            # model's default configuration file
            for key, value in kwargs.items():
                try:
                    setattr(self, key, value)
                except AttributeError as err:
                    logger.error(f"Can't set {key} with value {value} for {self}")
                    raise err

        # Validate the values of the attributes
        self.validate(is_init=True)

    def validate(self, is_init=False):
        """
        Validates the values of the attributes of the [`GenerationConfig`] instance. Raises exceptions in the presence
        of parameterization that can be detected as incorrect from the configuration instance alone.

        Note that some parameters are best validated at generate runtime, as they may depend on other inputs and/or the
        model, such as parameters related to the generation length.
        """
        pass
