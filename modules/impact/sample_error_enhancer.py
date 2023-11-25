import comfy.sample
import traceback

original_sample = comfy.sample.sample


def informative_sample(*args, **kwargs):
    try:
        return original_sample(*args, **kwargs)  # This code helps interpret error messages that occur within exceptions but does not have any impact on other operations.
    except RuntimeError as e:
        is_model_mix_issue = False
        try:
            if 'mat1 and mat2 shapes cannot be multiplied' in e.args[0]:
                if 'torch.nn.functional.linear' in traceback.format_exc().strip().split('\n')[-3]:
                    is_model_mix_issue = True
        except:
            pass

        if is_model_mix_issue:
            raise RuntimeError("\n\n#### It seems that models and clips are mixed and interconnected between SDXL Base, SDXL Refiner, SD1.x, and SD2.x. Please verify. ####\n\n")
        else:
            raise e


comfy.sample.sample = informative_sample
