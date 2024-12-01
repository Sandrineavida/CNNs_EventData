
def load_clean_state_dict(model, state_dict):
    # Get the keys that are present in the model
    model_keys = model.state_dict().keys()
    # Filter out the extra keys from the state_dict
    filtered_state_dict = {k: v for k, v in state_dict.items() if k in model_keys}
    return filtered_state_dict