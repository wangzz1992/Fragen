from .interaction import TransformerFeatureMixer, TransformerFeatureMixerForPos


def get_encoder_vn(config):
    if config.name == 'cftfm':
        return TransformerFeatureMixer(             # ./models/interaction/interaction.py
            hidden_channels = [config.hidden_channels, config.hidden_channels_vec],
            edge_channels = config.edge_channels,
            key_channels = config.key_channels,  # not use
            num_heads = config.num_heads,  # not use
            num_interactions = config.num_interactions,
            k = config.knn,
            cutoff = config.cutoff,
        )
    else:
        raise NotImplementedError('Unknown encoder: %s' % config.name)
    
def get_encoder_vn_for_pos(config):
    if config.name == 'cftfm':
        return TransformerFeatureMixerForPos(             # ./models/interaction/interaction.py
            hidden_channels = [config.hidden_channels, config.hidden_channels_vec],
            edge_channels = config.edge_channels,
            key_channels = config.key_channels,  # not use
            num_heads = config.num_heads,  # not use
            num_interactions = config.num_interactions,
            k = config.knn,
            cutoff = config.cutoff,
        )
    else:
        raise NotImplementedError('Unknown encoder: %s' % config.name)
