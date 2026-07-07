import time, os
import torch
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from losses.loss import loss_mossformer2_sr_48k 
from utils.misc import power_compress, power_uncompress, stft, istft, EPS

import warnings
warnings.filterwarnings("ignore")
class Solver(object):
    def __init__(self, args, models, optimizer_g, optimizer_m, discriminators, optimizer_discs, train_data, validation_data, test_data):
        self.train_data = train_data
        self.validation_data = validation_data
        self.test_data = test_data
        self.args = args
        self.device = self.args.device

        self.print = False
        if (self.args.distributed and self.args.local_rank ==0) or not self.args.distributed:
            self.print = True
            self.writer = SummaryWriter('%s/tensorboard/' % args.checkpoint_dir)

        self.models = models
        self.optimizer_g = optimizer_g
        self.optimizer_m = optimizer_m
        self.discriminators = discriminators
        self.optimizer_discs = optimizer_discs
        if self.args.distributed:
            for i in range(len(self.models)):
                self.models[i] = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.models[i])
                self.models[i] = DDP(self.models[i], device_ids=[self.args.local_rank],find_unused_parameters=True)
            if self.discriminators is not None:
                for j in range(len(self.discriminators)):
                    self.discriminators[j] = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.discriminators[j])
                    self.discriminators[j] = DDP(self.discriminators[j], device_ids=[self.args.local_rank],find_unused_parameters=True)
        self._init()
 
        if self.args.network == 'MossFormer2_SR_48K':
            self._run_one_epoch = self._run_one_epoch_mossformer2_sr_48k
        else:
            print(f'_run_one_epoch is not implemented for {self.args.network}!')


    def _init(self):
        self.halving = False
        self.epoch = 0
        self.step = 0
        self.best_val_loss = float("inf")
        self.val_no_impv = 0

        if self.args.train_from_last_checkpoint:
            flag = self._load_model()
        else:
            if self.print: print('Start new training')

        for i in range(len(self.models)):
            self.models[i].to(self.device)
        if self.discriminators is not None:
            for j in range(len(self.discriminators)):
                self.discriminators[j].to(self.device)

    def _load_model(self, mode='last_checkpoint', use_cuda=True, strict=True):
        ckpt_name = os.path.join(self.args.checkpoint_dir, mode)
        if not os.path.isfile(ckpt_name):
            mode = 'last_best_checkpoint'
            ckpt_name = os.path.join(self.args.checkpoint_dir, mode)
            if not os.path.isfile(ckpt_name):
                print('[!] Last checkpoints are not found. Start new training ...')
                self.epoch = 0
                self.step = 0
        else:
            print(f'Loading checkpoint: {ckpt_name}')
            model_names = []
            with open(ckpt_name, 'r') as f:
                for i in range(len(self.models)):
                    model_names.append(f.readline().strip())
                disc_name = f.readline().strip()
            checkpoint_path = os.path.join(self.args.checkpoint_dir, model_names[0])
            self._load_model_w_key(self.models[0], checkpoint_path, model_key='mossformer')
            checkpoint_path = os.path.join(self.args.checkpoint_dir, model_names[1])
            self._load_model_w_key(self.models[1], checkpoint_path, model_key='generator')
            if self.discriminators is not None and disc_name is not None:
                checkpoint_path = os.path.join(self.args.checkpoint_dir, disc_name)
                #state_dict_do = load_checkpoint(cp_do, device)
                checkpoint = self.load_checkpoint(checkpoint_path)                 
                self._load_model_w_key(self.discriminators[0], checkpoint_path, 'mpd', checkpoint)
                self._load_model_w_key(self.discriminators[1], checkpoint_path, 'msd', checkpoint)
                self._load_model_w_key(self.discriminators[2], checkpoint_path, 'mbd', checkpoint)
                #self.discriminators[0].load_state_dict(checkpoint['mpd'])
                #self.discriminators[1].load_state_dict(checkpoint['msd'])
                #self.discriminators[2].load_state_dict(checkpoint['mbd'])

                self.epoch = checkpoint['epoch']
                self.step = checkpoint['steps']

            print('=> Reloaded previous model and optimizer. Continue training ...')

    def _load_model_w_key(self, model, checkpoint_path, model_key=None, checkpoint=None):
        # Load the checkpoint file into memory (map_location ensures compatibility with different devices)
        if checkpoint is None:
            checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
        # Load the model's state dictionary (weights and biases) into the current model
        if model_key in checkpoint:
            pretrained_model = checkpoint[model_key]
        else:
            pretrained_model = checkpoint
        state = model.state_dict()
        for key in state.keys():
            if key in pretrained_model and state[key].shape == pretrained_model[key].shape:
                state[key] = pretrained_model[key]
            elif key.replace('module.', '') in pretrained_model and state[key].shape == pretrained_model[key.replace('module.', '')].shape:
                 state[key] = pretrained_model[key.replace('module.', '')]
            elif 'module.'+key in pretrained_model and state[key].shape == pretrained_model['module.'+key].shape:
                 state[key] = pretrained_model['module.'+key]
            elif self.print: print(f'{key} not loaded')
        model.load_state_dict(state)

    def load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(
            checkpoint_path, map_location=lambda storage, loc: storage)
        return checkpoint
        
    def _load_pretrained_model(self, checkpoint_path, load_optimizer=False, load_training_stat=False):
        if os.path.isfile(checkpoint_path):
            print(f'Loading checkpoint: {checkpoint_path}\n')
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            # load model weights
            if 'model' in checkpoint:
                pretrained_model = checkpoint['model']
            else:
                pretrained_model = checkpoint
            state = self.model.state_dict()
            for key in state.keys():
                if key in pretrained_model and state[key].shape == pretrained_model[key].shape:
                     state[key] = pretrained_model[key]
                elif key.replace('module.', '') in pretrained_model and state[key].shape == pretrained_model[key.replace('module.', '')].shape:     
                     state[key] = pretrained_model[key.replace('module.', '')]
                elif 'module.'+key in pretrained_model and state[key].shape == pretrained_model['module.'+key].shape:
                     state[key] = pretrained_model['module.'+key]
                elif self.print: print(f'{key} not loaded')
            self.model.load_state_dict(state)
            for g in self.optimizer.param_groups:
                g['lr'] = self.args.finetune_learning_rate 
            if self.discriminator is not None:
                discriminator_path = checkpoint_path.replace('.pt', '.disc.pt')
                checkpoint = self.load_checkpoint(discriminator_path)
                self.discriminator.load_state_dict(checkpoint['discriminator'], strict=False)
                self.optimizer_disc.load_state_dict(checkpoint['optimizer_disc'])
            print(f'==> Done model init from {checkpoint_path}. Start finetune training ...')
        else:
            print(f'{checkpoint_path} is not found. Start new training ...')
            self.epoch = 0
            self.step = 0

        # load optimizer only
        if load_optimizer:
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            for g in self.optimizer.param_groups:
                g['lr'] = self.args.learning_rate

        # load the training states
        if load_training_stat:
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.epoch=checkpoint['epoch']
            self.step = checkpoint['step']
            if self.print: print("Resume training from epoch: {}".format(self.epoch))
        
    def save_checkpoint(self, mode='last_checkpoint'):
        checkpoint_path = "{}/g_{:08d}".format(self.args.checkpoint_dir, self.step)
        torch.save({'generator': (self.models[1].module if self.args.num_gpus > 1 else self.models[1]).state_dict()}, checkpoint_path)

        checkpoint_path = "{}/m_{:08d}".format(self.args.checkpoint_dir, self.step)
        torch.save({'mossformer': (self.models[0].module if self.args.num_gpus > 1 else self.models[0]).state_dict()}, checkpoint_path)

        checkpoint_path = "{}/do_{:08d}".format(self.args.checkpoint_dir, self.step)
        torch.save({'mpd': (self.discriminators[0].module if self.args.num_gpus > 1
                                      else self.discriminators[0]).state_dict(),
                   'msd': (self.discriminators[1].module if self.args.num_gpus > 1
                                      else self.discriminators[1]).state_dict(),
                   'mbd': (self.discriminators[2].module if self.args.num_gpus > 1
                                      else self.discriminators[2]).state_dict(),
                   'optim_g': self.optimizer_g.state_dict(), 
                   'optim_m': self.optimizer_m.state_dict(), 
                   'optim_d': self.optimizer_discs.state_dict(), 
                   'steps': self.step,
                   'epoch': self.epoch}, 
                   checkpoint_path)

        with open(os.path.join(self.args.checkpoint_dir, mode), 'w') as f:
            f.write('m_{:08d}\n'.format(self.step))
            f.write('g_{:08d}\n'.format(self.step))
            if self.discriminators is not None:
                f.write('do_{:08d}\n'.format(self.step))
        print("=> Save checkpoint:", checkpoint_path)

    def train(self):
        start_epoch = self.epoch
        for epoch in range(start_epoch, self.args.max_epoch+1):
            if self.args.distributed: self.args.train_sampler.set_epoch(epoch)
            # Train
            for i in range(len(self.models)):
                self.models[i].train()
            for j in range(len(self.discriminators)):
                self.discriminators[j].train()

            start = time.time()            
            tr_loss = self._run_one_epoch(data_loader = self.train_data)
            if self.args.distributed: tr_loss = self._reduce_tensor(tr_loss.to(self.device))
            if self.print: print(f'Train Summary | End of Epoch {epoch} | Time {time.time() - start:2.3f}s | Train Loss {tr_loss:2.4f}')

            # Validation
            
            for i in range(len(self.models)):
                self.models[i].eval()
            for j in range(len(self.discriminators)):
                self.discriminators[j].eval()

            start = time.time()
            with torch.no_grad():
                val_loss = self._run_one_epoch(data_loader = self.validation_data, state='val')
                if self.args.distributed: val_loss = self._reduce_tensor(val_loss.to(self.device))
            if self.print: print(f'Valid Summary | End of Epoch {epoch} | Time {time.time() - start:2.3f}s | Valid Loss {val_loss:2.4f}')

            if self.test_data is not None:
                # Test
                for i in range(len(self.models)):
                    self.models[i].eval()
                for j in range(len(self.discriminators)):
                    self.discriminators[j].eval()
                start = time.time()
                with torch.no_grad():
                    test_loss = self._run_one_epoch(data_loader = self.test_data, state='test')
                    if self.args.distributed: test_loss = self._reduce_tensor(test_loss.to(self.device))
                if self.print: print(f'Test Summary | End of Epoch {epoch} | Time {time.time() - start:2.3f}s | Test Loss {test_loss:2.4f}')

            # Check whether to early stop and to reduce learning rate
            find_best_model = False
            if val_loss >= self.best_val_loss:
                self.best_val_loss = val_loss
                self.val_no_impv += 1
                if self.val_no_impv == 5:
                    self.halving = True
                elif self.val_no_impv >= 10:
                    if self.print: print("No imporvement for 10 epochs, early stopping.")
                    break
            else:
                self.val_no_impv = 0
                self.best_val_loss = val_loss
                find_best_model=True

            # Halfing the learning rate
            if self.halving:
                self.halving = False
                self._load_model(mode='last_best_checkpoint')
                if self.print: print('reload from last best checkpoint')

                optim_state = self.optimizer.state_dict()
                optim_state['param_groups'][0]['lr'] *= 0.5
                self.optimizer.load_state_dict(optim_state)
                if self.print: print('Learning rate adjusted to: {lr:.6f}'.format(lr=optim_state['param_groups'][0]['lr']))
                

            if self.print:
                # Tensorboard logging
                self.writer.add_scalar('Train_loss', tr_loss, epoch)
                self.writer.add_scalar('Validation_loss', val_loss, epoch)
                if self.test_data is not None:
                    self.writer.add_scalar('Test_loss', test_loss, epoch)

            # Save model
            self.save_checkpoint()
            if find_best_model:
                self.save_checkpoint(mode='last_best_checkpoint')
                print("Found new best model, dict saved")
            self.epoch = self.epoch + 1

    def _run_one_epoch_mossformer2_sr_48k(self, data_loader, state='train'):
        total_loss = 0.0
        gen_loss_print = 0.0
        disc_loss_print = 0.0
        self.accu_count = 0
        num_batch = len(data_loader)
        self.optimizer_m.zero_grad()
        self.optimizer_g.zero_grad()
        self.optimizer_discs.zero_grad()
        for i, batch in enumerate(data_loader):
            stime = time.time()
            x, y, y_mel = batch
            x = torch.autograd.Variable(x.to(self.device, non_blocking=True))
            y = torch.autograd.Variable(y.to(self.device, non_blocking=True))
            y_mel = torch.autograd.Variable(y_mel.to(self.device, non_blocking=True))
            y = y.unsqueeze(1)

            #mossformer process 
            x = self.models[0](x)
            #convolutional-generator process
            y_g_hat = self.models[1](x)

            loss_gen, loss_disc = loss_mossformer2_sr_48k(self.args, self.discriminators, y, y_mel, y_g_hat)

            if state=='train':
                if self.args.accu_grad:
                    if loss_gen is not None:
                        self.accu_count += 1
                        loss_gen_scaled = loss_gen/(self.args.effec_batch_size / self.args.batch_size)
                        loss_gen_scaled.backward()
                        for m in range(len(self.models)):
                            torch.nn.utils.clip_grad_norm_(self.models[m].parameters(), self.args.clip_grad_norm)
                        loss_disc_scaled = loss_disc/(self.args.effec_batch_size / self.args.batch_size)
                        loss_disc_scaled.backward()
                        for d in range(len(self.discriminators)):
                            torch.nn.utils.clip_grad_norm_(self.discriminators[d].parameters(), self.args.clip_grad_norm)
                        if self.accu_count == (self.args.effec_batch_size / self.args.batch_size):
                            self.optimizer_m.step()
                            self.optimizer_g.step()
                            self.optimizer_m.zero_grad()
                            self.optimizer_g.zero_grad()
                            self.optimizer_discs.step()
                            self.optimizer_discs.zero_grad()
                            self.accu_count = 0
                else:
                    if loss_gen is not None:
                        loss_gen.backward()
                        for m in range(len(self.models)):
                            torch.nn.utils.clip_grad_norm_(self.models[m].parameters(), self.args.clip_grad_norm)
                        self.optimizer_m.step()
                        self.optimizer_g.step()
                        self.optimizer_m.zero_grad()
                        self.optimizer_g.zero_grad()

                        for d in range(len(self.discriminators)):
                            torch.nn.utils.clip_grad_norm_(self.discriminators[d].parameters(), self.args.clip_grad_norm)
                        self.optimizer_discs.step()
                        self.optimizer_discs.zero_grad()
                self.step += 1
                ##cal losses for printing
                if loss_gen is not None:
                    gen_loss_print += loss_gen.data.cpu()
                    disc_loss_print += loss_disc.data.cpu()
                if (i + 1) % self.args.print_freq == 0:
                    eplashed = time.time() - stime
                    speed_avg = eplashed / (i+1)
                    gen_loss_print_avg = gen_loss_print / self.args.print_freq
                    disc_loss_print_avg = disc_loss_print / self.args.print_freq
                    print('Train Epoch: {}/{} Step: {}/{} | {:2.3f}s/batch | lr {:1.4e} |'
                      '| Gen_Loss {:2.4f}'
                      '| Disc_Loss {:2.4f}'
                      .format(self.epoch, self.args.max_epoch,
                          i+1, num_batch, speed_avg, self.optimizer_g.param_groups[0]["lr"],
                          gen_loss_print_avg,
                          disc_loss_print_avg,
                        ))
                    gen_loss_print = 0.0
                    disc_loss_print = 0.0
                if (i + 1) % self.args.checkpoint_save_freq == 0:
                    self.save_checkpoint()

            if loss_gen is not None:
                total_loss += loss_gen.data.cpu()

        return total_loss / (i+1)

    def _reduce_tensor(self, tensor):
        if not self.args.distributed: return tensor
        rt = tensor.clone()
        dist.all_reduce(rt, op=dist.ReduceOp.SUM)
        rt /= self.args.world_size
        return rt
