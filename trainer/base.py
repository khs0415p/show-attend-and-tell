import os
import gc
import time
import torch
import heapq
import pickle
import random
import torch.optim as optim
import torch.nn.functional as F

from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
from utils import (
    set_seed,
    LOGGER,
    get_metrics,
    save_loss_history,
    save_history,
    save_pred_figure,
    save_attention_figure
)
from utils.file_utils import make_dir, remove_file
from transformers import (
    get_linear_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
)
from models import Encoder, Decoder



class BaseTrainer:
    def __init__(
        self,
        config,
        device,
    ) -> None:
        self.config = config
        self.device = torch.device(device)
        self.ddp = config.ddp

        self.n_iter = 0

        self.start_epoch = 0
        self.epochs = config.epochs

        self.train_loss_history = []
        self.valid_loss_history = []
        self.encoder_learning_rates = []
        self.decoder_learning_rates = []
        self.best_val_loss = float("inf")

        self.save_total_limit = self.config.save_total_limit
        self.saved_path = []

        self.save_path = "results/{}/{}-{}"
        self.last_save_path = "results/{}"

        self.phases = ['train', 'valid'] if config.do_eval else ['train']

        set_seed(config.seed)

        self.world_size = len(self.config.device) if self.ddp else 1
        self.is_rank_zero = True if not self.ddp or (self.ddp and device == 0) else False


    def _init_trainer(self):
        # initialize model
        self.init_model()

        if self.config.mode == 'train':
            # initialize optimizer
            self.init_optimizer()
            # initialize scheduler
            self.init_scheduler()

            if self.config.fp16:
                from apex import amp
                self.encoder, self.encoder_optimizer = amp.initialize(
                    self.encoder, self.encoder_optimizer, opt_level=self.config.fp16_opt_level
                )

                self.decoder, self.decoder_optimizer = amp.initialize(
                    self.decoder, self.decoder_optimizer, opt_level=self.config.fp16_opt_level
                )
                if self.config.continuous:
                    import pickle
                    with open(os.path.join(self.config.checkpoint, 'checkpoint_info.pk', 'rb')) as f:
                        checkpoint_info = pickle.load(f)
                        amp.load_state_dict(checkpoint_info['amp'])
                    
                    self.encoder_learning_rates = checkpoint_info['encoder_learining_rates']
                    self.decoder_learning_rates = checkpoint_info['decoder_learining_rates']
                    self.best_val_loss = checkpoint_info['best_val_loss']
                    self.train_loss_history = checkpoint_info['train_losses']
                    self.valid_loss_history = checkpoint_info['valid_losses']
                    self.start_epoch = checkpoint_info['epoch_or_step']

            if self.ddp:
                self.encoder = torch.nn.parallel.DistributedDataParallel(self.encoder, device_ids=[self.device % torch.cuda.device_count()])
                self.decoder = torch.nn.parallel.DistributedDataParallel(self.decoder, device_ids=[self.device % torch.cuda.device_count()])

            if self.is_rank_zero:
                LOGGER.info(f"{'Initialize':<25} {self.__class__.__name__}")
                LOGGER.info(f"{'Batch Size':<25} {str(self.config.batch_size)}")
                LOGGER.info(f"{'Accumulation Step':<25} {str(self.config.gradient_accumulation_steps)}")
                LOGGER.info(f"{'Encoder Learning Rate':<25} {str(self.config.encoder_lr)}")
                LOGGER.info(f"{'Decoder Learning Rate':<25} {str(self.config.decoder_lr)}")

        self.encoder.to(self.device)
        self.decoder.to(self.device)

        torch.cuda.empty_cache()
        gc.collect()


    def init_model(self):
        
        encoder = Encoder(self.config)
        decoder = Decoder(self.config)
        if self.config.continuous or self.config.mode == 'test':
            checkpoint_path = self.config.checkpoint
            try:
                encoder_state = torch.load(os.path.join(checkpoint_path, 'encoder_pytorch_model.bin'), map_location=self.device)
                decoder_state = torch.load(os.path.join(checkpoint_path, 'decoder_pytorch_model.bin'), map_location=self.device)
                encoder.load_state_dict(encoder_state)
                decoder.load_state_dict(decoder_state)

                del encoder_state, decoder_state
                torch.cuda.empty_cache()
                LOGGER.info(f"Loaded checkpoint-({checkpoint_path})")
            except:
                raise ValueError(f"Not exists file : {checkpoint_path}/encoder_pytorch_model.bin/decoder_pytorch_model.bin")

        encoder.fine_tune(self.config.encoder_fine_tune)
        self.encoder = encoder
        self.decoder = decoder


    def init_scheduler(self):
        try:
            num_training_steps = len(self.dataloader['train']) * self.config.epochs
        except:
            raise ValueError("Don't exists dataloader")
        
        num_warmup_steps = int(num_training_steps * self.config.warmup_ratio)

        if self.config.scheduler_type == "linear":
            encoder_scheduler = get_linear_schedule_with_warmup(
                self.encoder_optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=num_training_steps,
            )

            decoder_scheduler = get_linear_schedule_with_warmup(
                self.decoder_optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=num_training_steps,
            )
        elif self.config.scheduler_type == "cosine":
            encoder_scheduler = get_cosine_schedule_with_warmup(
                self.encoder_optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=num_training_steps,
            )

            decoder_scheduler = get_cosine_schedule_with_warmup(
                self.decoder_optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=num_training_steps,
            )
        else:
            raise ValueError("You have to chioce scheduler type from [linear, cosine].")
        
        if self.config.continuous:
            checkpoint_path = self.config.checkpoint
            try:
                encoder_scheduler_state = torch.load(os.path.join(checkpoint_path, 'encoder_scheduler.pt'), map_location=self.device)
                decoder_scheduler_state = torch.load(os.path.join(checkpoint_path, 'decoder_scheduler.pt'), map_location=self.device)
                encoder_scheduler.load_state_dict(encoder_scheduler_state)
                decoder_scheduler.load_state_dict(decoder_scheduler_state)
                del encoder_scheduler_state, decoder_scheduler_state
                torch.cuda.empty_cache()
            except:
                raise ValueError(f"Not exists file : {checkpoint_path}/encoder_scheduler.pt/decoder_scheduler.pt")

        self.encoder_scheduler = encoder_scheduler
        self.decoder_scheduler = decoder_scheduler


    def init_optimizer(self):
        encoder_optimizer = optim.Adam(
            params=filter(lambda p: p.requires_grad, self.encoder.parameters()),lr=self.config.encoder_lr
            )

        decoder_optimizer = torch.optim.Adam(
            self.decoder.parameters(), lr=self.config.decoder_lr
            )

        if self.config.continuous:
            checkpoint_path = self.config.checkpoint
            try:
                encoder_optimizer_state = torch.load(os.path.join(checkpoint_path, 'encoder_optimizer.pt'), map_location=self.device)
                decoder_optimizer_state = torch.load(os.path.join(checkpoint_path, 'decoder_optimizer.pt'), map_location=self.device)
                encoder_optimizer.load_state_dict(encoder_optimizer_state)
                decoder_optimizer.load_state_dict(decoder_optimizer_state)
                del encoder_optimizer_state, decoder_optimizer_state
                torch.cuda.empty_cache()
            except:
                raise ValueError(f"Not exists file : {checkpoint_path}/encoder_optimizer.pt/decoder_optimizer.pt")

        self.encoder_optimizer = encoder_optimizer
        self.decoder_optimizer = decoder_optimizer


    def _backward_step(self, loss):
        if self.config.fp16:
            from apex import amp
            with amp.scale_loss(loss, self.encoder_optimizer) as scaled_loss:
                scaled_loss.backward()

            with amp.scale_loss(loss, self.decoder_optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()

        if self.n_iter % self.config.gradient_accumulation_steps == 0:
            if self.config.fp16:
                clip_grad_norm_(amp.master_params(self.encoder_optimizer), self.config.clip_max_norm)
                clip_grad_norm_(amp.master_params(self.decoder_optimizer), self.config.clip_max_norm)
            else:
                clip_grad_norm_(self.encoder.parameters(), self.config.clip_max_norm)
                clip_grad_norm_(self.decoder.parameters(), self.config.clip_max_norm)

            self.encoder_optimizer.step()
            self.decoder_optimizer.step()
            self.encoder_scheduler.step()
            self.decoder_scheduler.step()

        self.n_iter += 1


    def training(self, epoch):
        epoch_loss = 0
        total_size = 0
        for i, batch in enumerate(tqdm(self.dataloader['train'], total=len(self.dataloader['train']), desc=f"Training...| Epoch {epoch+1}")):
            self.encoder_optimizer.zero_grad()
            self.decoder_optimizer.zero_grad()
            batch_size = -1
            model_inputs = {}
            for batch_key in batch.keys():
                if isinstance(batch[batch_key], dict):
                    for key in batch[batch_key].keys():
                        if batch_key not in model_inputs:
                            model_inputs[batch_key] = {}

                        model_inputs[batch_key][key] = batch[batch_key][key].to(self.device)
                        batch_size = batch[batch_key][key].size(0)
                else:
                    model_inputs[batch_key] = batch[batch_key].to(self.device)
                    batch_size = batch[batch_key].size(0)

            step = (epoch * len(self.dataloader['train'])) + i

            if self.is_rank_zero:
                self._save_learning_rate()
                loss = self._training_step(model_inputs)

                if self.config.save_strategy == "step":
                    if self.n_iter % self.config.save_step == 0 and self.is_rank_zero:
                        self.save_checkpoint(loss=loss, step=step)

                if i % self.config.log_step == 0:
                    LOGGER.info(f"{'Epoch':<25}{str(epoch + 1)}")
                    LOGGER.info(f"{'Step':<25}{str(step)}")
                    LOGGER.info(f"{'Phase':<25}Train")
                    LOGGER.info(f"{'Loss':<25}{str(loss)}")
                    self.train_loss_history.append([step, loss])

            epoch_loss += loss * batch_size
            total_size += batch_size

        epoch_loss = epoch_loss / total_size

        LOGGER.info(f"{'Epoch Loss':<15}{epoch_loss}")


    def validation(self, epoch):
        epoch_loss = 0
        total_size = 0
        epoch_bleu2_score, epoch_bleu4_score = 0, 0
        epoch_meteor_score = 0
        for i, batch in enumerate(tqdm(self.dataloader['valid'], total=len(self.dataloader['valid']), desc=f"Validation...| Epoch {epoch+1}")):
            model_inputs = {}
            for batch_key in batch.keys():
                if isinstance(batch[batch_key], dict):
                    for key in batch[batch_key].keys():
                        if batch_key not in model_inputs:
                            model_inputs[batch_key] = {}

                        model_inputs[batch_key][key] = batch[batch_key][key].to(self.device)
                        batch_size = batch[batch_key][key].size(0)
                else:
                    model_inputs[batch_key] = batch[batch_key].to(self.device)
                    batch_size = batch[batch_key].size(0)

            step = (epoch * len(self.dataloader['valid'])) + i

            if self.is_rank_zero:
                loss, logits = self._validation_step(model_inputs)

                model_inputs['captions'] = model_inputs['captions'].detach().cpu()
                logits = torch.argmax(logits.detach().cpu(), dim=-1)
                bleu2_score, bleu4_score, meteor_score = get_metrics(model_inputs['captions'], logits, self.tokenizer)

                epoch_bleu2_score += bleu2_score
                epoch_bleu4_score += bleu4_score
                epoch_meteor_score += meteor_score

                if i % self.config.log_step == 0:
                    LOGGER.info(f"{'Epoch':<25}{str(epoch + 1)}")
                    LOGGER.info(f"{'Step':<25}{str(step)}")
                    LOGGER.info(f"{'Phase':<25}Validation")
                    LOGGER.info(f"{'Loss':<25}{str(loss)}")
                    LOGGER.info(f"{'BLEU-(2)':<25}{bleu2_score:.4f}")
                    LOGGER.info(f"{'BLEU-(4)':<25}{bleu4_score:.4f}")
                    LOGGER.info(f"{'METEOR':<25}{meteor_score:.4f}")
                    self.valid_loss_history.append([step, loss])

            epoch_loss += loss * batch_size
            total_size += batch_size

        epoch_loss = epoch_loss / total_size

        # bleu score
        self.bleu_score_list.append([epoch_bleu2_score / (i + 1), epoch_bleu4_score / (i + 1)])
        self.meteor_score_list.apppend(epoch_meteor_score / (i+1))

        if self.config.save_strategy == 'epoch' and self.is_rank_zero:
            self.save_checkpoint(epoch_loss, epoch + 1)

        LOGGER.info(f"{'Epoch Loss':<15}{epoch_loss}")


    def train(self):
        for epoch in range(self.start_epoch, self.start_epoch + self.epochs):
            for phase in self.phases:
                if phase == 'train':
                    self.encoder.train()
                    self.decoder.train()
                    self.training(epoch)
                else:
                    self.encoder.eval()
                    self.decoder.eval()
                    self.validation(epoch)

                torch.cuda.empty_cache()
                gc.collect()

        if self.is_rank_zero:
            self.save_checkpoint(step=epoch+1, last_save=True)
            LOGGER.info(f"{'Completed training.'}")

    def test(self):
        self.encoder.eval()
        self.decoder.eval()
        labels, preds, all_scores = [], [], []
        for i, batch in enumerate(tqdm(self.dataloader['test'], total=len(self.dataloader['test']), desc="Test...")):
            model_inputs = {}
            for batch_key in batch.keys():
                if isinstance(batch[batch_key], dict):
                    for key in batch[batch_key].keys():
                        if batch_key not in model_inputs:
                            model_inputs[batch_key] = {}

                        model_inputs[batch_key][key] = batch[batch_key][key].to(self.device)
                        batch_size = batch[batch_key][key].size(0)
                else:
                    model_inputs[batch_key] = batch[batch_key].to(self.device)
                    batch_size = batch[batch_key].size(0)

            if self.is_rank_zero:
                loss, logits, attention_scores = self._test_step(model_inputs)
                model_inputs['captions'] = model_inputs['captions'].detach().cpu()
                logits = torch.argmax(logits.detach().cpu(), dim=-1)
                labels.append(model_inputs['captions'])
                preds.append(logits)
                all_scores.append(attention_scores.detach().cpu())
        
        # random sampling
        lengths = [label.size(1) for label in labels]
        labels = [F.pad(label, (0, max(lengths) - label.size(1)), value=self.tokenizer.pad_token_id) for label in labels]
        preds = [F.pad(pred, (0, max(lengths) - pred.size(1)), value=self.tokenizer.pad_token_id) for pred in preds]
        all_scores = [F.pad(score, (0, 0, 0, max(lengths) - score.size(1)), value=self.tokenizer.pad_token_id) for score in all_scores]
        labels = torch.cat(labels, dim=0)
        preds = torch.cat(preds, dim=0)
        all_scores = torch.cat(all_scores, dim=0)
        
        random.seed(int(1000*time.time())%(2**32))
        random_idx = random.sample(range(len(self.dataloader['test'].dataset)), 5)

        image_names = self.dataloader['test'].dataset.captions['image']

        sampled_images = [self.config.img_data_path+image_names[index] for index in random_idx]
        sampled_scores = [all_scores[index] for index in random_idx]
        sampled_labels = []
        sampled_preds = []
        for i, index in enumerate(random_idx):
            label = self.tokenizer.decode(labels[index])
            pred = self.tokenizer.decode(preds[index])
            sampled_labels.append(label)
            sampled_preds.append(pred)
            print(f"========= Sample {i+1} =========")
            print(f"Label : {label}")
            print(f"Predict : {pred}\n")

        save_pred_figure(self.config.checkpoint, sampled_images, sampled_labels, sampled_preds)
        save_attention_figure(self.config.checkpoint, self.config.encoded_image_size, self.test_transform, sampled_images, sampled_scores, sampled_preds)


    def _save_checkpoint(
        self,
        base_path: str = None,
        loss: float = None,
        step: int = None,
        last_save: bool = False
    ):
        # Save model & training state
        torch.save(self.encoder.state_dict(), f'{base_path}/encoder_pytorch_model.bin')
        torch.save(self.decoder.state_dict(), f'{base_path}/decoder_pytorch_model.bin')
        torch.save(self.encoder_optimizer.state_dict(), f'{base_path}/encoder_optimizer.pt')
        torch.save(self.decoder_optimizer.state_dict(), f'{base_path}/decoder_optimizer.pt')
        torch.save(self.encoder_scheduler.state_dict(), f'{base_path}/encoder_scheduler.pt')
        torch.save(self.decoder_scheduler.state_dict(), f'{base_path}/decoder_scheduler.pt')

        # Save yaml file
        with open(f'{base_path}/config.yaml', 'w') as f:
            f.write(self.config.dumps(modified_color=None, quote_str=True))

        save_items = {
            'train_losses': self.train_loss_history,
            'valid_losses': self.valid_loss_history,
            'encoder_learning_rates': self.encoder_learning_rates,
            'decoder_learning_rates': self.decoder_learning_rates,
            'best_val_loss': loss,
            'epoch_or_step': step,
            'bleu_score_list': self.bleu_score_list,
            'meteor_score_list': self.meteor_score_list,
        }

        if self.config.fp16:
            from apex import amp
            save_items['amp'] = amp.state_dict()
        
        with open(f'{base_path}/checkpoint-info.pk', 'wb') as f:
            pickle.dump(save_items, f)

        if last_save:
            save_loss_history(base_path, "Train", self.config.model_type, self.train_loss_history, step)
            save_loss_history(base_path, "Validation", self.config.model_type, self.valid_loss_history, step)

            save_history(base_path, self.bleu_score_list)
            save_history(base_path, self.meteor_score_list, 'meteor')

        LOGGER.info(f"{'Saved model...'}")


    def save_checkpoint(
        self,
        loss: float = float("inf"),
        step: int = 0,
        last_save: bool = False,
    ):
        cur_time = time.strftime("%m-%d")
        base_path = self.save_path.format(cur_time, step, self.config.save_strategy)
        base_path = self.last_save_path.format(cur_time) if last_save else base_path
        make_dir(base_path)

        if last_save:
            self._save_checkpoint(
                base_path,
                loss,
                step,
                last_save
            )
            return

        elif not self.config.compare_best:
            if len(self.saved_path) >= self.save_total_limit:
                remove_item = heapq.heappop(self.saved_path)
                remove_file(remove_item[1])
            self._save_checkpoint(base_path, loss, step)
            heapq.heappush(self.saved_path, (-loss, base_path))

        elif self.best_val_loss > loss or loss == float("inf"):
            if len(self.saved_path) >= self.save_total_limit:
                remove_item = heapq.heappop(self.saved_path)
                remove_file(remove_item[1])
            self._save_checkpoint(base_path, loss, step)
            heapq.heappush(self.saved_path, (-loss, base_path))

            self.best_val_loss = loss


    def _save_learning_rate(self):
        self.encoder_learning_rates.append(self.encoder_optimizer.param_groups[0]['lr'])
        self.decoder_learning_rates.append(self.decoder_optimizer.param_groups[0]['lr'])