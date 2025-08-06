from transformers import PretrainedConfig

class NTConfig(PretrainedConfig):
    def __init__(
        self,
        emb_layer_norm_before=False,
        esmfold_config=None,
        hidden_dropout_prob=0.0,
        hidden_size=512,
        initializer_range=0.02,
        intermediate_size=2048,
        is_folding_model=False,
        layer_norm_eps=1e-12,
        mask_token_id=2,
        max_position_embeddings=2050,
        num_attention_heads=16,
        num_hidden_layers=12,
        pad_token_id=1,
        position_embedding_type='rotary',
        tie_word_embeddings=False,
        token_dropout=False,
        torch_dtype='float32',
        transformers_version='4.32.0.dev0',
        use_cache=False,
        vocab_list=None,
        vocab_size=4107,
        **kwargs,
    ):  
        self.emb_layer_norm_before = emb_layer_norm_before
        self.esmfold_config = esmfold_config
        self.hidden_dropout_prob = hidden_dropout_prob
        self.hidden_size = hidden_size
        self.initializer_range = initializer_range
        self.intermediate_size = intermediate_size
        self.is_folding_model = is_folding_model
        self.layer_norm_eps = layer_norm_eps
        self.mask_token_id = mask_token_id
        self.max_position_embeddings = max_position_embeddings
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.pad_token_id = pad_token_id
        self.position_embedding_type = position_embedding_type
        self.tie_word_embeddings = tie_word_embeddings
        self.token_dropout = token_dropout
        self.torch_dtype = torch_dtype
        self.transformers_version = transformers_version
        self.use_cache = use_cache
        self.vocab_list = vocab_list
        self.vocab_size = vocab_size
        super().__init__(**kwargs)

class DnaBert2Config(PretrainedConfig):
    def __init__(
        self,
        classifier_dropout=None,
        gradient_checkpointing=False,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        hidden_size=768,
        initializer_range=0.02,
        intermediate_size=3072,
        layer_norm_eps=1e-12,
        max_position_embeddings=512,
        num_attention_heads=12,
        num_hidden_layers=12,
        position_embedding_type="absolute",
        torch_dtype="float32",
        transformers_version="4.28.0",
        type_vocab_size=2,
        use_cache=True,
        vocab_size=4096,
        **kwargs,
    ):
        self.classifier_dropout = classifier_dropout
        self.gradient_checkpointing = gradient_checkpointing
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.hidden_size = hidden_size
        self.initializer_range = initializer_range
        self.intermediate_size = intermediate_size
        self.layer_norm_eps = layer_norm_eps
        self.max_position_embeddings = max_position_embeddings
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.position_embedding_type = position_embedding_type
        self.torch_dtype = torch_dtype
        self.transformers_version = transformers_version
        self.type_vocab_size = type_vocab_size
        self.use_cache = use_cache
        self.vocab_size = vocab_size
        
        super().__init__(**kwargs)

from transformers import PretrainedConfig

class EVOConfig(PretrainedConfig):
    def __init__(
        self,
        _commit_hash="1cc23830f62c268082475776fb449af8428eb703",
        _name_or_path="togethercomputer/evo-1-131k-base",
        architectures=["StripedHyenaModelForCausalLM"],
        attn_layer_idxs=[8, 16, 24],
        auto_map={
            "AutoConfig": "togethercomputer/evo-1-131k-base--configuration_hyena.StripedHyenaConfig",
            "AutoModelForCausalLM": "togethercomputer/evo-1-131k-base--modeling_hyena.StripedHyenaModelForCausalLM",
            "AutoTokenizer": ["togethercomputer/evo-1-131k-base--tokenizer.ByteTokenizer", None]
        },
        column_split=False,
        column_split_hyena=True,
        eps=1e-06,
        final_norm=True,
        hidden_size=4096,
        hyena_filter_groups=1,
        hyena_layer_idxs=[
            0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 17, 18, 19, 20, 21, 22, 23, 25, 26, 27, 28, 29, 30, 31
        ],
        inference_mode=False,
        inner_mlp_size=10928,
        log_intermediate_values=False,
        make_vocab_size_divisible_by=8,
        max_seqlen=8192,
        mha_out_proj_bias=True,
        mlp_activation="gelu",
        model_parallel_size=1,
        model_type="stripedhyena",
        num_attention_heads=32,
        num_filters=4096,
        num_layers=32,
        pipe_parallel_size=1,
        prefill_style="fft",
        proj_groups=1,
        qkv_proj_bias=True,
        rotary_emb_base=10000,
        rotary_emb_scaling_factor=1,
        short_filter_bias=True,
        short_filter_length=3,
        smeared_gqa=False,
        split_k0=True,
        state_size=8,
        tie_embeddings=True,
        torch_dtype="bfloat16",
        transformers_version=None,
        use_cache=True,
        use_flash_attn=True,
        use_flash_depthwise=False,
        use_flash_rmsnorm=False,
        use_flashfft=False,
        use_interpolated_rotary_pos_emb=False,
        vocab_size=512,
        **kwargs,
    ):
        self._commit_hash = _commit_hash
        self._name_or_path = _name_or_path
        self.architectures = architectures
        self.attn_layer_idxs = attn_layer_idxs
        self.auto_map = auto_map
        self.column_split = column_split
        self.column_split_hyena = column_split_hyena
        self.eps = eps
        self.final_norm = final_norm
        self.hidden_size = hidden_size
        self.hyena_filter_groups = hyena_filter_groups
        self.hyena_layer_idxs = hyena_layer_idxs
        self.inference_mode = inference_mode
        self.inner_mlp_size = inner_mlp_size
        self.log_intermediate_values = log_intermediate_values
        self.make_vocab_size_divisible_by = make_vocab_size_divisible_by
        self.max_seqlen = max_seqlen
        self.mha_out_proj_bias = mha_out_proj_bias
        self.mlp_activation = mlp_activation
        self.model_parallel_size = model_parallel_size
        self.model_type = model_type
        self.num_attention_heads = num_attention_heads
        self.num_filters = num_filters
        self.num_layers = num_layers
        self.pipe_parallel_size = pipe_parallel_size
        self.prefill_style = prefill_style
        self.proj_groups = proj_groups
        self.qkv_proj_bias = qkv_proj_bias
        self.rotary_emb_base = rotary_emb_base
        self.rotary_emb_scaling_factor = rotary_emb_scaling_factor
        self.short_filter_bias = short_filter_bias
        self.short_filter_length = short_filter_length
        self.smeared_gqa = smeared_gqa
        self.split_k0 = split_k0
        self.state_size = state_size
        self.tie_embeddings = tie_embeddings
        self.torch_dtype = torch_dtype
        self.transformers_version = transformers_version
        self.use_cache = use_cache
        self.use_flash_attn = use_flash_attn
        self.use_flash_depthwise = use_flash_depthwise
        self.use_flash_rmsnorm = use_flash_rmsnorm
        self.use_flashfft = use_flashfft
        self.use_interpolated_rotary_pos_emb = use_interpolated_rotary_pos_emb
        self.vocab_size = vocab_size

        super().__init__(**kwargs)



class EnhanceRepresentationConfig(PretrainedConfig):
    model_type = "enhance_representation"

    def __init__(
        self,
        num_heads_omics_cross_attention=8,
        num_tokens_per_seq_nuctf=2048,
        num_tokens_per_seq_nuctf_rna=2048,
        num_protein_tokens_per_seq=2048,
        pool_window_end=1,
        pool_window_start=0,
        torch_dtype="float32",
        transformers_version="4.29.2",
        **kwargs,
    ):
        self.num_heads_omics_cross_attention = num_heads_omics_cross_attention
        self.num_tokens_per_seq_nuctf = num_tokens_per_seq_nuctf
        self.num_tokens_per_seq_nuctf_rna = num_tokens_per_seq_nuctf_rna
        self.num_protein_tokens_per_seq = num_protein_tokens_per_seq
        self.pool_window_end = pool_window_end
        self.pool_window_start = pool_window_start
        self.torch_dtype = torch_dtype
        self.transformers_version = transformers_version
        self.nt_config = NTConfig()
        self.dnabert2_config = DnaBert2Config()
        self.evo_config = EVOConfig()
        super().__init__(**kwargs)
