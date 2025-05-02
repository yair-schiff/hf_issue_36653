import transformers

from .backbone import MyBackbone


class MyModelConfig(transformers.PretrainedConfig):

    model_type = "my_model"
    auto_map = {
        "AutoConfig": "modeling.MyModelConfig",
        "AutoModel": "modeling.MyModel",
    }

    def __init__(
        self,
        num_layers: int = 2,
        input_dim: int = 2,
        hidden_dim: int = 128,
        output_dim: int = 2,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.num_layers = num_layers
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim


class MyModel(transformers.PreTrainedModel):

    config_class = MyModelConfig

    def __init__(self, config: MyModelConfig):
        super().__init__(config)
        self.config = config
        self.backbone = MyBackbone(
            num_layers=config.num_layers,
            input_dim=config.input_dim,
            hidden_dim=config.hidden_dim,
            output_dim=config.output_dim,
        )

    def forward(self, inputs):
        # Forward pass through the backbone
        outputs = self.backbone(inputs)
        return outputs
