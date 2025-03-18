import argparse
import gc
import pickle
import itertools
import logging
import math
import os
import threading
import datasets
import diffusers
import psutil
import torch
import torch.utils.checkpoint
import transformers
import sys
sys.path.append("src")
import ddm

import numpy as np
import torch.nn.functional as F

from pathlib import Path
from glob import glob
from typing import Optional
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    DiffusionPipeline,
    DPMSolverMultistepScheduler,
    UNet2DConditionModel,
)
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version
from diffusers.utils.import_utils import is_xformers_available
from huggingface_hub import HfFolder, Repository, whoami
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import AutoTokenizer, PretrainedConfig
from peft import LoraConfig, get_peft_model

logger = logging.getLogger(__name__)

unetTargetModules = ["to_q", "to_v", "query", "value"]
textEncoderTargetModules = ["q_proj", "v_proj"]

# Getting model class
def importModelClassFromModelNameOrPath(pretrainedModelNameOrPath: str, revision: str):
    textEncoderConfig = PretrainedConfig.from_pretrained(
        pretrainedModelNameOrPath,
        subfolder="text_encoder",
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

# Args
def parseArgs(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    # Data
    parser.add_argument(
        "--instanceDataDir",
        type=str,
        default=None,
        required=True,
        help="A folder containing the training data of instance images.",
    )
    parser.add_argument(
        "--labelDir",
        type=str,
        default=None,
        required=True,
        help="A folder containing the training data of labels for images",
    )
    parser.add_argument(
        "--promptDir",
        type=str,
        default=None,
        required=True,
        help="A folder containing the training data of labels for images",
    )
    parser.add_argument(
        "--outputDir",
        type=str,
        default="text-inversion-model",
        help="The output directory where the model predictions and checkpoints will be written",
    )
    # Model
    parser.add_argument(
        "--pretrainedModelNameOrPath",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models",
    )
    parser.add_argument(
        "--tokenizerName",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--seed", 
        type=int, 
        default=None, 
        help="A seed for reproducible training."
    )
    # Data
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--trainBatchSize",
        type=int, default=4, 
        help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument(
        "--centerCrop",
        action="store_true",
        help="Whether to center crop images before resizing to resolution"
    )
    # Local rank
    parser.add_argument(
        "--localRank", 
        type=int, 
        default=-1, 
        help="For distributed training: localRank"
    )
    # Xformers memory efficient attention
    parser.add_argument(
        "--enableXformersMemoryEfficientAttention", 
        action="store_true", 
        help="Whether or not to use xformers."
    )
    # Lora args
    parser.add_argument(
        "--useLora", 
        action="store_true", 
        help="Whether to use Lora for parameter efficient tuning"
    )
    parser.add_argument(
        "--loraR", 
        type=int, 
        default=8, 
        help="Lora rank, only used if useLora is True"
    )
    parser.add_argument(
        "--loraAlpha", 
        type=int, 
        default=32, 
        help="Lora alpha, only used if useLora is True"
    )
    parser.add_argument(
        "--loraDropout", 
        type=float, 
        default=0.0, 
        help="Lora dropout, only used if useLora is True"
    )
    parser.add_argument(
        "--loraBias",
        type=str,
        default="none",
        help="Bias type for Lora. Can be 'none', 'all' or 'lora_only', only used if useLora is True",
    )
    parser.add_argument(
        "--loraTextEncoderR",
        type=int,
        default=8,
        help="Lora rank for text encoder, only used if `useLora` and `trainTextEncoder` are True",
    )
    parser.add_argument(
        "--loraTextEncoderAlpha",
        type=int,
        default=32,
        help="Lora alpha for text encoder, only used if `useLora` and `trainTextEncoder` are True",
    )
    parser.add_argument(
        "--loraTextEncoderDropout",
        type=float,
        default=0.0,
        help="Lora dropout for text encoder, only used if `useLora` and `trainTextEncoder` are True",
    )
    parser.add_argument(
        "--loraTextEncoderBias",
        type=str,
        default="none",
        help="Bias type for Lora. Can be 'none', 'all' or 'lora_only', only used if useLora and `trainTextEncoder` are True",
    )
    # Loss weights
    parser.add_argument(
        "--reconstructionWeight", 
        type=float, 
        default=1.0, 
        help="The weight of the reconstruction loss (mse)."
    )
    parser.add_argument(
        "--indicatorWeight", 
        type=float, 
        default=0, 
        help="The weight of the indicator loss."
    )
    # Text encoder args
    parser.add_argument(
        "--trainTextEncoder", 
        action="store_true", 
        help="Whether to train the text encoder"
    )
    # Epochs and steps
    parser.add_argument(
        "--numTrainEpochs", 
        type=int, 
        default=1
    )
    parser.add_argument(
        "--maxTrainSteps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides numTrainEpochs.",
    )
    parser.add_argument(
        "--checkpointingSteps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints can be used both as final"
            " checkpoints in case they are better than the last checkpoint, and are also suitable for resuming"
            " training using `--resumeFromCheckpoint`."
        ),
    )
    parser.add_argument(
        "--resumeFromCheckpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointingSteps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--gradientAccumulationSteps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradientCheckpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    # Scheduler args
    parser.add_argument(
        "--learningRate",
        type=float,
        default=5e-6,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scaleLr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lrScheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lrWarmupSteps", 
        type=int, 
        default=500, 
        help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--lrNumCycles",
        type=int,
        default=1,
        help="Number of hard resets of the lr in cosine_with_restarts scheduler.",
    )
    parser.add_argument(
        "--lrPower", 
        type=float, 
        default=1.0, 
        help="Power factor of the polynomial scheduler."
    )
    # Optimizer
    parser.add_argument(
        "--use8bitAdam", 
        action="store_true", 
        help="Whether or not to use 8-bit Adam from bitsandbytes."
    )
    parser.add_argument(
        "--adamBeta1", 
        type=float, 
        default=0.9,
        help="The beta1 parameter for the Adam optimizer."
    )
    parser.add_argument(
        "--adamBeta2", 
        type=float, 
        default=0.999, 
        help="The beta2 parameter for the Adam optimizer."
    )
    parser.add_argument(
        "--adamWeightDecay", 
        type=float, 
        default=1e-2, 
        help="Weight decay to use."
    )
    parser.add_argument(
        "--adamEpsilon", 
        type=float, 
        default=1e-08, 
        help="Epsilon value for the Adam optimizer"
    )
    # Max gradient norm
    parser.add_argument(
        "--maxGradNorm", 
        default=1.0, 
        type=float, 
        help="Max gradient norm."
    )
    # Validation
    parser.add_argument(
        "--validationPrompt",
        type=str,
        default=None,
        help="A prompt that is used during validation to verify that the model is learning.",
    )
    parser.add_argument(
        "--numValidationImages",
        type=int,
        default=4,
        help="Number of images that should be generated during validation with `validationPrompt`.",
    )
    parser.add_argument(
        "--validationSteps",
        type=int,
        default=100,
        help=(
            "Run dreambooth validation every X steps. Dreambooth validation consists of running the prompt"
            " `args.validationPrompt` multiple times: `args.numValidationImages`."
        ),
    )
    # Other args
    parser.add_argument(
        "--pushToHub", 
        action="store_true", 
        help="Whether or not to push the model to the Hub."
    )
    parser.add_argument(
        "--hubToken", 
        type=str, 
        default=None, 
        help="The token to use to push to the Model Hub."
    )
    parser.add_argument(
        "--hubModelId",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `outputDir`.",
    )
    
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    envLocalRank = int(os.environ.get("LOCAL_RANK", -1))
    if envLocalRank != -1 and envLocalRank != args.localRank:
        args.localRank = envLocalRank

    return args

# Converting Bytes to Megabytes
def b2mb(x):
    return int(x / 2**20)

def collate_fn(examples):
    inputIds = [example["instancePromptIds"] for example in examples]
    pixelValues = [example["instanceImages"] for example in examples]
    labels = [example["labels"] for example in examples]
    labels = torch.tensor(labels).long()
    
    pixelValues = torch.stack(pixelValues)
    pixelValues = pixelValues.to(memory_format=torch.contiguous_format).float()

    inputIds = torch.cat(inputIds, dim=0)

    batch = {
        "inputIds": inputIds,
        "pixelValues": pixelValues,
        "labels": labels,
    }
    return batch

# This context manager is used to track the peak memory usage of the process
class TorchTracemalloc:
    def __enter__(self):
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_max_memory_allocated()  # reset the peak gauge to zero
        self.begin = torch.cuda.memory_allocated()
        self.process = psutil.Process()

        self.cpuBegin = self.cpuMemUsed()
        self.peakMonitoring = True
        peakMonitorThread = threading.Thread(target=self.peakMonitorFunc)
        peakMonitorThread.daemon = True
        peakMonitorThread.start()
        return self

    def cpuMemUsed(self):
        return self.process.memory_info().rss

    def peakMonitorFunc(self):
        self.cpuPeak = -1

        while True:
            self.cpuPeak = max(self.cpuMemUsed(), self.cpuPeak)
            if not self.peakMonitoring:
                break

    def __exit__(self, *exc):
        self.peakMonitoring = False

        gc.collect()
        torch.cuda.empty_cache()
        self.end = torch.cuda.memory_allocated()
        self.peak = torch.cuda.max_memory_allocated()
        self.used = b2mb(self.end - self.begin)
        self.peaked = b2mb(self.peak - self.begin)

        self.cpuEnd = self.cpuMemUsed()
        self.cpuUsed = b2mb(self.cpuEnd - self.cpuBegin)
        self.cpuPeaked = b2mb(self.cpuPeak - self.cpuBegin)

class DreamBoothDataset(Dataset):
    """
    A dataset to prepare the instance and class images with the prompts for fine-tuning the model.
    It pre-processes the images and the tokenizes prompts.
    """

    def __init__(
        self,
        instanceDataDir,
        labelDir,
        promptDir,
        tokenizer,
        size=512,
        centerCrop=False,
    ):

        # Images
        self.instanceImagesPath = list(sorted(Path(instanceDataDir).iterdir(), key=lambda x: x.name))
        print('peft-train-dreambooth, dataset init, num of dicoms: {}'.format(len(self.instanceImagesPath)))
        
        # Labels
        with open(labelDir, "rb") as f:
            labels = pickle.load(f)
            self.labels = labels

        # Prompts
        with open(promptDir, 'r') as f:
            instancePrompt = f.readlines()
        instancePrompt = [x.strip() for x in instancePrompt]
        self.instancePrompt = instancePrompt
        
        self._length = len(self.instanceImagesPath)
        self.size = size
        self.centerCrop = centerCrop
        self.tokenizer = tokenizer

        self.imageTransforms = transforms.Compose(
            [
                transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(size) if centerCrop else transforms.RandomCrop(size),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5])
            ]
        )

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        example = {}
        instanceImage = Image.open(self.instanceImagesPath[index])
        if not instanceImage.mode == "RGB":
            instanceImage = instanceImage.convert("RGB")
        example["instanceImages"] = self.imageTransforms(instanceImage)
        example["instancePromptIds"] = self.tokenizer(
            self.instancePrompt[index],
            truncation=True,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids
        example["labels"] = self.labels[index]

        return example

def getFullRepoName(model_id: str, organization: Optional[str] = None, token: Optional[str] = None):
    if token is None:
        token = HfFolder.get_token()
    if organization is None:
        username = whoami(token)["name"]
        return f"{username}/{model_id}"
    else:
        return f"{organization}/{model_id}"

def main(args):
    # Logging
    os.makedirs(args.outputDir, exist_ok=True)
    loggingDir = Path(args.outputDir, "log.log")

    logging.basicConfig(
        filename=loggingDir,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    datasets.utils.logging.set_verbosity_error()
    transformers.utils.logging.set_verbosity_error()
    diffusers.utils.logging.set_verbosity_error()

    # Handling the repository creation
    if args.pushToHub:
        if args.hubModelId is None:
            repoName = getFullRepoName(Path(args.outputDir).name, token=args.hubToken)
        else:
            repoName = args.hubModelId
        repo = Repository(args.outputDir, clone_from=repoName)  # noqa: F841

        with open(os.path.join(args.outputDir, ".gitignore"), "w+") as gitignore:
            if "step_*" not in gitignore:
                gitignore.write("step_*\n")
            if "epoch_*" not in gitignore:
                gitignore.write("epoch_*\n")

    # Loading the tokenizer
    if args.tokenizerName:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizerName, revision=args.revision, use_fast=False)
    elif args.pretrainedModelNameOrPath:
        tokenizer = AutoTokenizer.from_pretrained(
            args.pretrainedModelNameOrPath,
            subfolder="tokenizer",
            revision=args.revision,
            use_fast=False,
        )

    # Importing correct text encoder class
    textEncoderCls = importModelClassFromModelNameOrPath(args.pretrainedModelNameOrPath, args.revision)

    # Load scheduler and models
    noiseScheduler = DDPMScheduler(
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        num_train_timesteps=1000,
    )  # DDPMScheduler.from_pretrained(args.pretrainedModelNameOrPath, subfolder="scheduler")
    textEncoder = textEncoderCls.from_pretrained(
        args.pretrainedModelNameOrPath, subfolder="text_encoder", revision=args.revision
    )
    vae = AutoencoderKL.from_pretrained(args.pretrainedModelNameOrPath, subfolder="vae", revision=args.revision)
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrainedModelNameOrPath, subfolder="unet", revision=args.revision
    )
    indicator = ddm.models.Indicator(size = args.resolution)

    if args.useLora:
        config = LoraConfig(
            r=args.loraR,
            lora_alpha=args.loraAlpha,
            target_modules=unetTargetModules,
            lora_dropout=args.loraDropout,
            bias=args.loraBias,
        )
        unet = get_peft_model(unet, config)
        unet.print_trainable_parameters()
        print(unet)

    vae.requires_grad_(False)
    if not args.trainTextEncoder:
        textEncoder.requires_grad_(False)
    elif args.trainTextEncoder and args.useLora:
        config = LoraConfig(
            r=args.loraTextEncoderR,
            lora_alpha=args.loraTextEncoderAlpha,
            target_modules=textEncoderTargetModules,
            lora_dropout=args.loraTextEncoderDropout,
            bias=args.loraTextEncoderBias,
        )
        textEncoder = get_peft_model(textEncoder, config)
        textEncoder.print_trainable_parameters()
        print(textEncoder)

    if args.enableXformersMemoryEfficientAttention:
        if is_xformers_available():
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    if args.gradientCheckpointing:
        print('peft-train-dreambooth, gradientCheckpointing set to True')
        unet.gradientCheckpointing()
        # below fails when using lora so commenting it out
        if args.trainTextEncoder and not args.useLora:
            textEncoder.gradient_checkpointing_enable()
    else:
        print('peft-train-dreambooth, gradientCheckpointing set to False')

    # Enabling TF32 for faster training on Ampere GPUs https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scaleLr:
        args.learningRate = (
            args.learningRate * args.gradientAccumulationSteps * args.trainBatchSize #* accelerator.num_processes
        )

    # Using 8-bit Adam for lower memory usage or to fine-tune the model in 16GB GPUs
    if args.use8bitAdam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
            )

        optimizerClass = bnb.optim.AdamW8bit
    else:
        optimizerClass = torch.optim.AdamW

    # DM optimizer creation
    paramsToOptimize = (
        itertools.chain(unet.parameters(), textEncoder.parameters(), indicator.parameters()) if args.trainTextEncoder else itertools.chain(unet.parameters(), indicator.parameters())
    )
    optimizer = optimizerClass(
        paramsToOptimize,
        lr=args.learningRate,
        betas=(args.adamBeta1, args.adamBeta2),
        weight_decay=args.adamWeightDecay,
        eps=args.adamEpsilon,
    )

    # Dataset and DataLoaders creation:
    trainDataset = DreamBoothDataset(
        instanceDataDir=args.instanceDataDir,
        labelDir=args.labelDir,
        promptDir=args.promptDir,
        tokenizer=tokenizer,
        size=args.resolution,
        centerCrop=args.centerCrop,
    )

    trainDataloader = torch.utils.data.DataLoader(
        trainDataset,
        batch_size=args.trainBatchSize,
        shuffle=True,
        collate_fn=lambda examples: collate_fn(examples),
        num_workers=1,
    )

    # Scheduler and math around the number of training steps
    overrodeMaxTrainSteps = False
    numUpdateStepsPerEpoch = math.ceil(len(trainDataloader) / args.gradientAccumulationSteps)
    if args.maxTrainSteps is None:
        args.maxTrainSteps = args.numTrainEpochs * numUpdateStepsPerEpoch
        overrodeMaxTrainSteps = True

    lrScheduler = get_scheduler(
        args.lrScheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lrWarmupSteps * args.gradientAccumulationSteps,
        num_training_steps=args.maxTrainSteps * args.gradientAccumulationSteps,
        num_cycles=args.lrNumCycles,
        power=args.lrPower,
    )

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        print("CUDA not available!")
        device = torch.device("cpu")

    weightDtype = torch.float32
    vae.to(device, dtype=weightDtype)
    if not args.trainTextEncoder:
        textEncoder.to(device, dtype=weightDtype)
    else:
        textEncoder.to(device)
    unet.to(device)
    indicator.to(device)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    numUpdateStepsPerEpoch = math.ceil(len(trainDataset) / args.gradientAccumulationSteps)
    if overrodeMaxTrainSteps:
        args.maxTrainSteps = args.numTrainEpochs * numUpdateStepsPerEpoch
    # Afterwards we recalculate our number of training epochs
    args.numTrainEpochs = math.ceil(args.maxTrainSteps / numUpdateStepsPerEpoch)

    # Train!
    totalBatchSize = args.trainBatchSize * args.gradientAccumulationSteps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(trainDataset)}")
    logger.info(f"  Num batches each epoch = {len(trainDataset)}")
    logger.info(f"  Num Epochs = {args.numTrainEpochs}")
    logger.info(f"  Instantaneous batch size per device = {args.trainBatchSize}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {totalBatchSize}")
    logger.info(f"  Gradient Accumulation steps = {args.gradientAccumulationSteps}")
    logger.info(f"  Total optimization steps = {args.maxTrainSteps}")
    globalStep = 0
    firstEpoch = 0

    # Potentially loading in the weights and states from a previous save
    if args.resumeFromCheckpoint:
        if args.resumeFromCheckpoint != "latest":
            path = os.path.basename(args.resumeFromCheckpoint)
        else:
            # Getting the mos recent checkpoint
            dirs = os.listdir(args.outputDir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1]
        print(f"Resuming from checkpoint {path}")
        globalStep = int(path.split("-")[1])

        resumeGlobalStep = globalStep * args.gradientAccumulationSteps
        firstEpoch = resumeGlobalStep // numUpdateStepsPerEpoch
        resumeStep = resumeGlobalStep % numUpdateStepsPerEpoch

    # Only show the progress bar once on each machine.
    progressBar = tqdm(range(globalStep, args.maxTrainSteps))
    progressBar.set_description("Steps")

    for epoch in range(firstEpoch, args.numTrainEpochs):
        unet.train()
        indicator.train()
        if args.trainTextEncoder:
            textEncoder.train()
        with TorchTracemalloc() as tracemalloc:
            for step, batch in enumerate(trainDataloader):
                # Skip steps until we reach the resumed step
                if args.resumeFromCheckpoint and epoch == firstEpoch and step < resumeStep:
                    if step % args.gradientAccumulationSteps == 0:
                        progressBar.update(1)
                        print(progressBar)
                    continue

                # Convertting images to latent space
                latents = vae.encode(batch["pixelValues"].to(device, dtype=weightDtype))
                latents = latents.latent_dist.sample()
                latents = latents * 0.18215

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                # Sample a random timestep for each image
                timesteps = torch.randint(
                    0, noiseScheduler.config.num_train_timesteps, (bsz,), device=latents.device
                )
                timesteps = timesteps.long()

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisyLatents = noiseScheduler.add_noise(latents, noise, timesteps)

                # Get the text embedding for conditioning
                encoderHiddenStates = textEncoder(batch["inputIds"].to(device))[0]
                # Predict the noise residual
                modelPred = unet(noisyLatents, timesteps, encoderHiddenStates).sample

                # Get the target for loss depending on the prediction type
                if noiseScheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noiseScheduler.config.prediction_type == "v_prediction":
                    target = noiseScheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noiseScheduler.config.prediction_type}")

                reconstructionLoss = F.mse_loss(modelPred.float(), target.float(), reduction="mean")

                # Get the output from the indicator
                indicatorOutput = indicator(modelPred)

                # Compute indicator loss
                indicatorLoss = F.cross_entropy(indicatorOutput, batch["labels"].to(device))
                
                reconstructionLoss = args.reconstructionWeight * reconstructionLoss
                indicatorLoss = args.indicatorWeight * indicatorLoss

                totalLoss = reconstructionLoss + indicatorLoss

                progressBar.set_postfix(ReconstructionLoss=f"{reconstructionLoss.item():.4f}", IndicatorLoss=f"{indicatorLoss.item():.4f}")

                totalLoss = totalLoss / args.gradientAccumulationSteps
                totalLoss.backward()

                paramsToClipDm = (
                    itertools.chain(unet.parameters(), textEncoder.parameters(), indicator.parameters()) if args.trainTextEncoder else itertools.chain(unet.parameters(), indicator.parameters())
                )
                torch.nn.utils.clip_grad_value_(paramsToClipDm, args.maxGradNorm)
                if (step + 1) % args.gradientAccumulationSteps == 0:
                    optimizer.step()
                    lrScheduler.step()
                    optimizer.zero_grad()

                progressBar.update(1)
                print(progressBar)
                globalStep += 1
                if globalStep % args.checkpointingSteps == 0:
                    savePath = os.path.join(args.outputDir, f"checkpoint-{globalStep}")

                    logger.info(f"Saved state to {savePath}")

                if(args.validationPrompt is not None and (step + numUpdateStepsPerEpoch * epoch) % args.validationSteps == 0):
                    logger.info(
                        f"Running validation... \n Generating {args.numValidationImages} images with prompt:"
                        f" {args.validationPrompt}."
                    )
                    # create pipeline
                    pipeline = DiffusionPipeline.from_pretrained(
                        args.pretrainedModelNameOrPath,
                        safety_checker=None,
                        revision=args.revision,
                    )
                    # set `keep_fp32_wrapper` to True because we do not want to remove
                    # mixed precision hooks while we are still training
                    pipeline.unet = unet
                    pipeline.textEncoder = textEncoder
                    pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
                    pipeline.set_progress_bar_config(disable=True)

                    # run inference
                    if args.seed is not None:
                        generator = torch.Generator().manual_seed(args.seed)
                    else:
                        generator = None
                    images = []
                    for _ in range(args.numValidationImages):
                        image = pipeline(args.validationPrompt, num_inference_steps=25, generator=generator).images[0]
                        images.append(image)

                    del pipeline
                    torch.cuda.empty_cache()

                if globalStep >= args.maxTrainSteps:
                    break
        # Printing the GPU memory usage details such as allocated memory, peak memory, and total memory usage
        print("GPU Memory before entering the train : {}".format(b2mb(tracemalloc.begin)))
        print("GPU Memory consumed at the end of the train (end-begin): {}".format(tracemalloc.used))
        print("GPU Peak Memory consumed during the train (max-begin): {}".format(tracemalloc.peaked))
        print(
            "GPU Total Peak Memory consumed during the train (max): {}".format(
                tracemalloc.peaked + b2mb(tracemalloc.begin)
            )
        )

        print("CPU Memory before entering the train : {}".format(b2mb(tracemalloc.cpuBegin)))
        print("CPU Memory consumed at the end of the train (end-begin): {}".format(tracemalloc.cpuUsed))
        print("CPU Peak Memory consumed during the train (max-begin): {}".format(tracemalloc.cpuPeaked))
        print(
            "CPU Total Peak Memory consumed during the train (max): {}".format(
                tracemalloc.cpuPeaked + b2mb(tracemalloc.cpuBegin)
            )
        )

    # Create the pipeline using using the trained modules and save it.
    if args.useLora:
        unet.save_pretrained(
            os.path.join(args.outputDir, "unet"), state_dict=unet.state_dict()
        )
        if args.trainTextEncoder:
            textEncoder.save_pretrained(
                os.path.join(args.outputDir, "textEncoder"),
                state_dict=textEncoder.state_dict(),
            )
    else:
        pipeline = DiffusionPipeline.from_pretrained(
            args.pretrainedModelNameOrPath,
            unet=unet,
            text_encoder=textEncoder,
            revision=args.revision,
        )
        pipeline.save_pretrained(args.outputDir)

    if args.pushToHub:
        repo.pushToHub(commit_message="End of training", blocking=False, auto_lfs_prune=True)
    if args.useLora:
        unet.save_pretrained(
            os.path.join(args.outputDir, "unet"), state_dict=unet.state_dict()
        )
        if args.trainTextEncoder:
            textEncoder.save_pretrained(
                os.path.join(args.outputDir, "textEncoder"),
                state_dict=textEncoder.state_dict(),
            )
        torch.save(indicator.state_dict(), os.path.join(args.outputDir, "indicator"))

    else:
        pipeline = DiffusionPipeline.from_pretrained(
            args.pretrainedModelNameOrPath,
            unet=unet,
            text_encoder=textEncoder,
            revision=args.revision,
        )
        pipeline.save_pretrained(args.outputDir)

    if args.pushToHub:
        repo.pushToHub(commit_message="End of training", blocking=False, auto_lfs_prune=True)

if __name__ == "__main__":
    args = parseArgs()
    main(args)