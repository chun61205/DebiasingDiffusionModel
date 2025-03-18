import argparse
import os
import torch

from diffusers import (
    StableDiffusionPipeline,
)
from peft import PeftModel
from PIL import Image
from typing import Optional

# Args
def parseArgs():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--modelDir",
        type=str,
        default=None,
        required=True,
        help="The pretrained model",
    )
    parser.add_argument(
        "--pretrainedModelNameOrPath",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        required=True,
        help="The prompt of the t2i model",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=None,
        required=True,
        help="The resolution of the images",
    )
    parser.add_argument(
        "--numImagePerIteration",
        type=int,
        default=None,
        required=True,
        help="The number of images generated in one iteration. The total number of images is args.numImagePerIteration * args.iterations.",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=None,
        required=True,
        help="The number of images generation iteration. The total number of images is args.numImagePerIteration * args.iterations.",
    )
    parser.add_argument(
        "--outputDir",
        type=str,
        default=None,
        required=True,
        help="The output directory of the generated images",
    )

    args = parser.parse_args()

    return args

def mergingLoraWithBase(pipe, modelDir, adapterName = "default" ) -> StableDiffusionPipeline:
    """
    This function merge the stable diffusion model with the lora.
    
    Args:
    - pipe (StableDiffusionPipeline): The stored weight
    - modelDir (str): The directory of the unet and the textEncoder
    - adapterName (str): The adapter name

    Return:
    - StableDiffusionPipeline: The stable diffusion pipeline with lora
    """

    unetSubDir = os.path.join(modelDir, "unet")
    textEncoderSubDir = os.path.join(modelDir, "text_encoder")
    if isinstance(pipe.unet, PeftModel):
        pipe.unet.set_adapter(adapterName)
    else:
        pipe.unet = PeftModel.from_pretrained(pipe.unet, unetSubDir, adapter_name=adapterName)
    pipe.unet = pipe.unet.merge_and_unload()
    print('peft-lora-pipeline, merging-lora-with-base: load unet')

    if os.path.exists(textEncoderSubDir):
        if isinstance(pipe.text_encoder, PeftModel):
            pipe.text_encoder.set_adapter(adapterName)
        else:
            pipe.text_encoder = PeftModel.from_pretrained(
                pipe.text_encoder, textEncoderSubDir, adapter_name=adapterName
            )
        pipe.text_encoder = pipe.text_encoder.merge_and_unload()
        print( 'peft-lora-pipeline, merging-lora-with-base: load text encoder' )

    return pipe 

def pipeline( 
    modelDir: str,
    pretrainedModelNameOrPath: str,
    adapterName: Optional[str] = 'adapter',
    device: Optional[torch.device] = 'cuda',
    prompt: Optional[str] = 'A human face',
    resolution: Optional[int] = 512,
    numImagePerIteration: Optional[int] = 1,
    iterations: Optional[int] = 1000,
    dtype: Optional[torch.dtype] = torch.float32,
    randomSeed: Optional[int] = 0,
    outputDir: Optional[str] = 'img2/dataset7_debiasing_0_2_4500',
):
    # Initializeng a normal pipeline
    pipe = StableDiffusionPipeline.from_pretrained(pretrained_model_name_or_path = pretrainedModelNameOrPath)
    pipe = mergingLoraWithBase( 
        pipe = pipe,
        modelDir = modelDir,
        adapterName = adapterName
    )
    pipe = pipe.to(device)

    os.makedirs(args.outputDir, exist_ok=True)

    for i in range(iterations):
        seed = i
        torch.manual_seed(seed)
        generator = torch.Generator(device = device).manual_seed(seed)

        out = pipe( 
            prompt = prompt,
            height = resolution, 
            width = resolution,
            num_inference_steps = 40,
            num_images_per_prompt = numImagePerIteration,
            generator = generator
        )
        
        for j in range(numImagePerIteration):
            result : Image.Image = out[0][j]
            result.convert("RGB").save(os.path.join(outputDir, f"img{i * numImagePerIteration + j}.png"))

    return

def main(args):
    dtype = torch.float32
    if torch.cuda.is_available() is True:
        device = 'cuda'
    else:
        device = 'cpu'
    print('peft_lora_pipeline, device: {}'.format(device))

    paramPipeline = {
        'modelDir' : args.modelDir,
        'pretrainedModelNameOrPath' : args.pretrainedModelNameOrPath,
        'adapterName': 'adapter',
        'device' : device,
        'prompt': args.prompt,
        'resolution': args.resolution,
        'numImagePerIteration': args.numImagePerIteration,
        'iterations': args.iterations,
        'dtype' : dtype,
        'outputDir': args.outputDir,
    }
    pipeline(**paramPipeline)

if __name__ == '__main__':
    args = parseArgs()
    main(args)