import torch

def retrieve_current_accelerator_type() -> str:
    """
        Returns: The gpu accelerator for the current system if it exists. Otherwise "".
    """
    current_device = torch.accelerator.current_accelerator(check_available=True)
    if current_device is not None:
        return current_device.type
    return ""
