from argparse import Namespace
from dotenv import load_dotenv
import numpy as np
import os
import timeit
import torch
import wandb

from cs336_basics import data, model, optimizer, nn_utils
from cs336_systems.util import constants, dataset_util

TIMING_WARMUP_STEPS = 5
TIMING_TIMED_STEPS = 10

class TrainingLoop:
    def __init__(self, args: Namespace):
        self.args = args
        self.device = torch.device("cuda")

        self.model = model.BasicsTransformerLM(
            vocab_size=constants.VOCAB_SIZE[args.dataset],
            context_length=args.context_length,
            d_model=args.d_model,
            num_layers=args.num_layers,
            num_heads=args.num_attention_heads,
            d_ff=args.d_ff,
            rope_theta=args.rope_theta
        )
        self.model.to(self.device)
        self.compiled_model = torch.compile(self.model)

        self.optimizer = optimizer.AdamW(
            params=self.model.parameters(),
            lr=args.learning_rate,
            betas=(args.beta1, args.beta2),
            eps=args.eps,
            weight_decay=args.weight_decay
        )

        self._setup_dataloader()
        self.training_ends = False
        self.step = 0


    def _setup_dataloader(self):
        self.dataset = dataset_util.load_dataset(self.args.dataset)

    def get_next_data_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        return data.get_batch(
            dataset=self.dataset,
            batch_size=self.args.batch_size,
            context_length=self.args.context_length,
            device="cuda"
        )

    def _mini_train_step(self):
        self.optimizer.zero_grad(set_to_none=True)
        x, y = self.get_next_data_batch()
        logits = self.compiled_model(x)

        loss = nn_utils.cross_entropy(inputs=logits, targets=y)

        # TODO: remove this when performance matters
        if self.args.forward_only:
            return loss.detach()

        loss.backward()
        self.optimizer.step()
        return loss.detach()

    def train(self):
        '''
        load_dotenv()
        wandb_api_key = os.getenv("WANDB_API_KEY")
        if wandb_api_key:
            wandb.login(key=wandb_api_key)

        wandb.init(
            project=self.args.wandb_project,
            name=self.args.wandb_name,
            config=vars(self.args)
        )
        '''
        self.timing_list = []

        while not self.training_ends:
            if self.step >= TIMING_WARMUP_STEPS:
                torch.cuda.synchronize()
                self.timing_list.append(timeit.default_timer())
            if self.step == TIMING_TIMED_STEPS + TIMING_WARMUP_STEPS:
                break
            self._mini_train_step()
            self.step += 1


    def calculate_timing_stats(self):
        durations = []
        for i in range(1, len(self.timing_list)):
            durations.append(self.timing_list[i] - self.timing_list[i-1])
        
        print(f"avg: {np.mean(durations)}; std: {np.std(durations)}")
    
    def run(self):
        self.train()
        self.calculate_timing_stats()