from modeling_retro import *


def get_test_retro_config():
    return RetroConfig(
        num_embeddings = 100,
        chunk_size = 8,
        dropout_p = 0.0,

        # Encoder
        enc_hidden_dim = 16,
        enc_num_layers = 2,
        enc_ffn_dim = 16 * 4,
        enc_num_heads = 4,
        enc_ca_layers= [0, 1],

        # Decoder
        dec_hidden_dim = 32,
        dec_num_layers = 2,
        dec_ffn_dim = 32 * 4,
        dec_num_heads = 4,
        dec_cca_layers = [1]
    )


def get_test_retro_input(
    batch_size: int = 1, 
    input_len: int = 16,
    chunk_size: int = 8,
    num_neighbours: int = 2, 
    neighbour_len: int = 16,
    num_embeddings: int = 100
):
    input_ids = torch.randint(0, num_embeddings, (batch_size, input_len))
    input_attention_mask = torch.ones_like(input_ids)
    neighbour_ids = torch.randint(0, num_embeddings, (batch_size, input_len // chunk_size, num_neighbours, neighbour_len))
    neighbour_attention_mask = torch.ones_like(neighbour_ids)

    return input_ids, input_attention_mask, neighbour_ids, neighbour_attention_mask


def set_gradient(obj, name):
    def _set_fn(grad):
        obj[name] = grad
    return _set_fn


def test_batch_example_independence():
    """
    Test that any output hidden state in one batch example has a
    zero gradient wrt input hidden states of another batch example
    """

    config = get_test_retro_config()
    model = RetroModel(config)

    input_ids, input_attention_mask, neighbour_ids, neighbour_attention_mask = get_test_retro_input(batch_size=2)
    output = model(input_ids, neighbour_ids, input_attention_mask, neighbour_attention_mask, return_all_hidden_states=True)

    input_hidden_states = output.hidden_states[0]
    output_hidden_states = output.hidden_states[-1]

    gradients = {}
    input_hidden_states.register_hook(set_gradient(gradients, "input_hidden_states"))

    l = output_hidden_states[0].sum()
    l.backward()

    gradient_magnitudes_per_batch_ex = gradients["input_hidden_states"].abs().sum(dim=[1,2])

    assert gradient_magnitudes_per_batch_ex[0] > 0., "Gradient should be greater than zero for first example"
    assert gradient_magnitudes_per_batch_ex[1] == 0., "Gradient should be equal to zero for second example"


def test_causality():
    """
    Test that any output hidden state has zero gradient wrt input hidden states of following positions
    """

    config = get_test_retro_config()
    model = RetroModel(config)

    input_ids, input_attention_mask, neighbour_ids, neighbour_attention_mask = get_test_retro_input()
    output = model(input_ids, neighbour_ids, input_attention_mask, neighbour_attention_mask, return_all_hidden_states=True)

    input_hidden_states = output.hidden_states[0]
    output_hidden_states = output.hidden_states[-1]

    gradients = {}
    input_hidden_states.register_hook(set_gradient(gradients, "input_hidden_states"))

    l = output_hidden_states[0, 5, :].sum() # Take the sixth position's output hidden states
    l.backward()

    gradient_magnitudes_per_pos = gradients["input_hidden_states"][0, :, :].abs().sum(-1)

    assert all(gradient_magnitudes_per_pos[:6] > 0.), "Gradient magnitudes should be greater than zero for previous and current positions"
    assert all(gradient_magnitudes_per_pos[6:] == 0.), "Gradient magnitudes should be equal to zero for future positions"


def test_neighbour_masking():
    """
    Test that:
     1. output hidden states has zero gradient wrt input neighbour hidden states that are masked
     2. positions in output neighbour hidden states has zero gradient wrt input neighbour hidden states that are masked
    """

    config = get_test_retro_config()
    model = RetroModel(config)

    input_ids, input_attention_mask, neighbour_ids, neighbour_attention_mask = get_test_retro_input()

    # Mask last position in first neighbour for the first attending chunk
    neighbour_attention_mask[0, 0, 0, -1] = 0.0

    output = model(
        input_ids, 
        neighbour_ids, 
        input_attention_mask, 
        neighbour_attention_mask, 
        return_all_hidden_states=True, 
        return_all_neighbour_hidden_states=True
    )

    output_hidden_states = output.hidden_states[-1]
    input_neighbour_hidden_states = output.neighbour_hidden_states[0]

    gradients = {}
    input_neighbour_hidden_states.register_hook(set_gradient(gradients, "input_neighbour_hidden_states"))

    # 1.
    l = output_hidden_states[0, 7, :].sum() # Take the eigth position, i.e first token in the first attending chunk
    l.backward()

    neighbour_gradient_magnitudes = gradients["input_neighbour_hidden_states"][0, 0, 0, :, :].abs().sum(-1)

    assert all(neighbour_gradient_magnitudes[:-1] > 0.), "Gradients should be non-zero for unmasked positions"
    assert neighbour_gradient_magnitudes[-1] == 0., "Gradient should be zero for masked positions"

    # Reset for second test
    model.zero_grad()
    output = model(
        input_ids, 
        neighbour_ids, 
        input_attention_mask, 
        neighbour_attention_mask, 
        return_all_hidden_states=True, 
        return_all_neighbour_hidden_states=True
    )

    output_hidden_states = output.hidden_states[-1]
    input_neighbour_hidden_states = output.neighbour_hidden_states[0]
    output_neighbour_hidden_states = output.neighbour_hidden_states[-1]

    gradients = {}
    input_neighbour_hidden_states.register_hook(set_gradient(gradients, "input_neighbour_hidden_states"))

    # 2.
    l = output_neighbour_hidden_states[0, 0, 0, 0, :].sum()  # Take first position of first neighbour in first attending chunk
    l.backward()

    neighbour_gradient_magnitudes = gradients["input_neighbour_hidden_states"][0, 0, 0, :, :].abs().sum(-1)

    assert all(neighbour_gradient_magnitudes[:-1] > 0.), "Gradients should be non-zero for unmasked positions"
    assert neighbour_gradient_magnitudes[-1] == 0., "Gradient should be zero for masked positions"


def test_input_masking():
    """
    Test that gradients of neighbour output hidden states wrt masked input hidden states are zero.
    I.e. neighbour tokens should not be able to attend to masked input tokens
    """

    config = get_test_retro_config()
    model = RetroModel(config)

    input_ids, input_attention_mask, neighbour_ids, neighbour_attention_mask = get_test_retro_input()

    # Mask last input token
    input_attention_mask[0, -1] = 0.0

    output = model(
        input_ids, 
        neighbour_ids, 
        input_attention_mask, 
        neighbour_attention_mask, 
        return_all_hidden_states=True, 
        return_all_neighbour_hidden_states=True
    )

    input_hidden_states = output.hidden_states[0]
    output_neighbour_hidden_states = output.neighbour_hidden_states[-1]

    gradients = {}
    input_hidden_states.register_hook(set_gradient(gradients, "input_hidden_states"))

    l = output_neighbour_hidden_states[0, 1, 0, :, :].sum()  # Take all output hidden states for first neighbour in second chunk
    l.backward()

    input_gradient_magnitudes = gradients["input_hidden_states"][0, :, :].abs().sum(-1)

    assert all(input_gradient_magnitudes[:-1] > 0.), "Gradients for all non-masked input positions should be non-zero"
    assert input_gradient_magnitudes[-1] == 0., "Gradients for masked input positions should be zero"


def test_complete_neighbours_masked():
    """
    Test special case when all positions of all neighbours of a chunk are masked, and make sure gradient of
    output hidden states wrt any masked neighbour hidden state, is zero
    """

    config = get_test_retro_config()
    model = RetroModel(config)

    input_ids, input_attention_mask, neighbour_ids, neighbour_attention_mask = get_test_retro_input()

    # Mask all neighbours completely for the first chunk
    neighbour_attention_mask[0, 0, :, :] = 0

    output = model(
        input_ids, 
        neighbour_ids, 
        input_attention_mask, 
        neighbour_attention_mask, 
        return_all_hidden_states=True, 
        return_all_neighbour_hidden_states=True
    )

    output_hidden_states = output.hidden_states[-1]
    input_neighbour_hidden_states = output.neighbour_hidden_states[0]

    gradients = {}
    input_neighbour_hidden_states.register_hook(set_gradient(gradients, "input_neighbour_hidden_states"))

    l = output_hidden_states[0, 7, :].sum() # Take the eigth position, i.e first token in the first attending chunk
    l.backward()

    neighbour_gradient_magnitudes = gradients["input_neighbour_hidden_states"][0, 0, 0, :, :].abs().sum(-1)

    assert all(neighbour_gradient_magnitudes == 0.), "Gradient should be zero for masked positions"


def test_relative_position_bucket_linear():

    qlen = 5
    klen = 5

    context_position = torch.arange(qlen, dtype=torch.long)[:, None]
    memory_position = torch.arange(klen, dtype=torch.long)[None, :]
    relative_position = memory_position - context_position # shape (qlen, klen)
    
    # bidirectional = True
    res = RelativePositionBias._relative_position_bucket_linear(relative_position, bidirectional=True, num_buckets=2)
    expected = torch.tensor([
        [0, 1, 1, 1, 1],
        [0, 0, 1, 1, 1],
        [0, 0, 0, 1, 1],
        [0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0]
    ])
    assert torch.all(res == expected)

    res = RelativePositionBias._relative_position_bucket_linear(relative_position, bidirectional=True, num_buckets=3)
    expected = torch.tensor([
        [1, 2, 2, 2, 2],
        [0, 1, 2, 2, 2],
        [0, 0, 1, 2, 2],
        [0, 0, 0, 1, 2],
        [0, 0, 0, 0, 1]
    ])
    assert torch.all(res == expected)

    # Bidirectional=False
    res = RelativePositionBias._relative_position_bucket_linear(relative_position, bidirectional=False, num_buckets=2)
    expected = torch.tensor([
        [1, 1, 1, 1, 1],
        [0, 1, 1, 1, 1],
        [0, 0, 1, 1, 1],
        [0, 0, 0, 1, 1],
        [0, 0, 0, 0, 1]
    ])
    assert torch.all(res == expected)

    res = RelativePositionBias._relative_position_bucket_linear(relative_position, bidirectional=False, num_buckets=3)
    expected = torch.tensor([
        [2, 2, 2, 2, 2],
        [1, 2, 2, 2, 2],
        [0, 1, 2, 2, 2],
        [0, 0, 1, 2, 2],
        [0, 0, 0, 1, 2]
    ])
    assert torch.all(res == expected)


def test_incomplete_input_chunk():
    """
    Test whether the RetroModel returns the same output hidden states for a token in an incomplete chunk as for a complete chunk
    """

    config = get_test_retro_config()
    model = RetroModel(config)

    input_ids, input_attention_mask, neighbour_ids, neighbour_attention_mask = get_test_retro_input()

    # Slice up to the first token in the second chunk
    input_ids_incomplete = input_ids[:, :9]
    input_attention_mask_incomplete = input_attention_mask[:, :9]
    neighbour_ids_incomplete = torch.clone(neighbour_ids)
    neighbour_ids_incomplete[:, -1, :, :] = 0
    neighbour_attention_mask_incomplete = torch.clone(neighbour_attention_mask)
    neighbour_attention_mask_incomplete[:, -1, :, :] = 0

    model.eval()
    complete_output = model(
        input_ids, 
        neighbour_ids, 
        input_attention_mask, 
        neighbour_attention_mask
    )
    incomplete_output = model(
        input_ids_incomplete, 
        neighbour_ids_incomplete, 
        input_attention_mask_incomplete, 
        neighbour_attention_mask_incomplete
    )

    assert torch.all(torch.abs(complete_output.hidden_states[0, :8, :] - incomplete_output.hidden_states[0, :8, :]) < 1e-5), \
        "The RetroModel gives different predictions for complete and incomplete chunks that are otherwise equal"


if __name__ == "__main__":
    test_incomplete_input_chunk()