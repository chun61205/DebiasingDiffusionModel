import os
import pickle
import torch
import sys
sys.path.append("src")
import ddm

import numpy as np
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from transformers import AutoTokenizer
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    UNet2DConditionModel,
)
from peft import PeftModel

class ClassifierEvaluationDataset(Dataset):
    def __init__(self, datasetDir, prompt, tokenizer, transform=None):
        self.datasetDir = datasetDir
        self.datasetPaths = [os.path.join(datasetDir, img) for img in os.listdir(datasetDir) 
                             if img.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]
        # Loading prompts
        self.prompts = prompt * len(self.datasetPaths)
        
        self.tokenizer = tokenizer
        
        # Default transform if none provided
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((128, 128)),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5])
            ])
        else:
            self.transform = transform

    def __len__(self):
        return len(self.datasetPaths)

    def __getitem__(self, idx):
        imgPath = self.datasetPaths[idx]
        prompt = self.prompts[idx]
        
        image = Image.open(imgPath).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        # Tokenizing prompt
        inputIds = self.tokenizer(
            prompt,
            truncation=True,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids.squeeze(0)
        
        return {
            'image': image,
            'inputIds': inputIds,
            'path': imgPath
        }

def evaluateIndicator(
    datasetDir, 
    prompt,
    modelPath, 
    indicatorClass, 
    pretrainedModelNameOrPath,
    revision=None
):
    """
    This function evaluate a single indicator using the full pipeline
    
    Args:
    - datasetDir (str): Directory containing images to evaluate
    - modelPath (str): Directory containing saved models
    - indicatorClass (torch.nn.Module): Indicator model class
    - pretrainedModelNameOrPath (str): Path to pretrained model
    - revision (str, optional): Model revision
    
    Returns:
    - dict: Evaluation results including entropies
    """
    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        pretrainedModelNameOrPath, 
        subfolder="tokenizer", 
        revision=revision,
        use_fast=False
    )
    
    # Load models
    vae = AutoencoderKL.from_pretrained(
        pretrainedModelNameOrPath, 
        subfolder="vae", 
        revision=revision
    ).to(device)

    textEncoderCls = importModelClassFromModelNameOrPath(pretrainedModelNameOrPath, revision)

    baseTextEncoder = textEncoderCls.from_pretrained(
        pretrainedModelNameOrPath,
        subfolder="text_encoder"
    ).to(device)

    loraPath = os.path.join(modelPath, "text_encoder")
    textEncoder = PeftModel.from_pretrained(baseTextEncoder, loraPath).to(device)

    textEncoder.print_trainable_parameters()

    baseUnet = UNet2DConditionModel.from_pretrained(
        pretrainedModelNameOrPath,
        subfolder="unet"
    ).to(device)

    # LoRA adapter
    loraUnetPath = os.path.join(modelPath, "unet")
    unet = PeftModel.from_pretrained(baseUnet, loraUnetPath).to(device)
    
    noiseScheduler = DDPMScheduler(
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        num_train_timesteps=1000,
    )
    
    # Preparing dataset and dataloader
    dataset = ClassifierEvaluationDataset(datasetDir, prompt, tokenizer)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    # Prepare models
    vae.requires_grad_(False)
    textEncoder.requires_grad_(False)
    unet.requires_grad_(False)
    
    # Results storage
    entropies = []
    detailedResults = []
    
    # Set models to eval mode
    vae.eval()
    textEncoder.eval()
    unet.eval()
    
    # Loading indicator
    indicator = indicatorClass(size=128)
    indicator.load_state_dict(torch.load(os.path.join(modelPath, "indicator")))
    indicator = indicator.to(device).eval()
    
    # Evaluate on dataset
    with torch.no_grad():
        for batch in dataloader:
            images = batch['image'].to(device)
            inputIds = batch['inputIds'].to(device)
            imagePaths = batch['path']
            
            # Convert images to latent space
            latents = vae.encode(images).latent_dist.sample()
            latents = latents * 0.18215
            
            # Sample noise
            noise = torch.randn_like(latents)
            bsz = latents.shape[0]
            
            # Sample a random timestep
            timesteps = torch.randint(
                0, noiseScheduler.config.num_train_timesteps, (bsz,), device=latents.device
            ).long()
            
            # Add noise to latents
            noisyLatents = noiseScheduler.add_noise(latents, noise, timesteps)
            
            # Get text embedding
            encoderHiddenStates = textEncoder(inputIds)[0]
            
            # Predict noise residual
            modelPred = unet(noisyLatents, timesteps, encoderHiddenStates).sample
            
            # Get classifier output
            classifierOutput = indicator(modelPred)
            
            # Calculate probabilities and entropy
            probs = torch.softmax(classifierOutput, dim=1)
            entropy = -torch.sum(probs * torch.log(probs + 1e-9), dim=1)
            
            entropies.append(entropy.cpu().item())
            
            # Store detailed results
            detailedResults.append({
                'path': imagePaths[0],
                'probabilities': probs.cpu().numpy()[0],
                'entropy': entropy.cpu().item()
            })
    
    # Compute statistics
    results = {
        'detailedResults': detailedResults,
        'entropyStats': {
            'modelPath': modelPath,
            'meanEntropy': np.mean(entropies),
            'stdEntropy': np.std(entropies)
        }
    }
    
    return results

def importModelClassFromModelNameOrPath(pretrainedModelNameOrPath: str, revision: str):
    from transformers import PretrainedConfig
    
    textEncoderConfig = PretrainedConfig.from_pretrained(
        pretrainedModelNameOrPath,
        subfolder="textEncoder",
        revision=revision,
    )
    modelClass = textEncoderConfig.architectures[0]

    if modelClass == "CLIPTextModel":
        from transformers import CLIPTextModel
        return CLIPTextModel
    elif modelClass == "RobertaSeriesModelWithTransformation":
        from diffusers.pipelines.alt_diffusion.modeling_roberta_series import RobertaSeriesModelWithTransformation
        return RobertaSeriesModelWithTransformation
    else:
        raise ValueError(f"{modelClass} is not supported.")

def main(datasetDir, prompt, pretrainedModelNameOrPath, modelPath):
    # Evaluating classifier
    results = evaluateIndicator(
        datasetDir, 
        prompt,
        modelPath, 
        ddm.models.Indicator(), 
        pretrainedModelNameOrPath
    )
    
    # Results
    print("Entropy Statistics:")
    print(f"Model Path: {results['entropy_stats']['model_path']}")
    print(f"  Mean Entropy: {results['entropy_stats']['mean_entropy']:.4f}")
    print(f"  Std Entropy: {results['entropy_stats']['std_entropy']:.4f}")
    
if __name__ == "__main__":
    modelPaths = [
        "Debiasing_Diffusion_Model/results2/mnist78_0005",
        "Debiasing_Diffusion_Model/results2/mnist78_001",
    ]
    prompt = "A human face"
    for modelPath in modelPaths:
        datasetDir = 'mixed_dataset/7'
        pretrainedModelNameOrPath = "sd-legacy/stable-diffusion-v1-5"
        main(datasetDir, prompt, pretrainedModelNameOrPath, modelPath) 