import torch
import json
import os
import tqdm

class Trainer(object):
    def __init__(self, model, pg, optimizer, args, distribution=None):
        self.model = model
        self.pg = pg
        self.optimizer = optimizer
        self.args = args
        self.distribution = distribution

    def train_epoch(self, dataloader, ntriple):
        self.model.train()
        total_loss = 0.0
        total_reward = 0.0
        counter = 0
        with tqdm.tqdm(total=ntriple, unit='ex') as bar:
            bar.set_description('Train')
            for src_batch, rel_batch, dst_batch, time_batch in dataloader:
                if self.args.cuda:
                    src_batch = src_batch.cuda()
                    rel_batch = rel_batch.cuda()
                    dst_batch = dst_batch.cuda()
                    time_batch = time_batch.cuda()

                all_loss, all_logits, _, current_entities, current_time = self.model(src_batch, time_batch, rel_batch)

                reward = self.pg.get_reward(current_entities, dst_batch)
                if self.args.reward_shaping:
                    # reward shaping
                    delta_time = time_batch - current_time
                    p_dt = []

                    for i in range(rel_batch.shape[0]):
                        rel = rel_batch[i].item()
                        dt = delta_time[i].item()
                        p_dt.append(self.distribution(rel, dt // self.args.time_span))

                    p_dt = torch.tensor(p_dt)
                    if self.args.cuda:
                        p_dt = p_dt.cuda()
                    shaped_reward = (1 + p_dt) * reward
                    cum_discounted_reward = self.pg.calc_cum_discounted_reward(shaped_reward)
                else:
                    cum_discounted_reward = self.pg.calc_cum_discounted_reward(reward)
                reinfore_loss = self.pg.calc_reinforce_loss(all_loss, all_logits, cum_discounted_reward)
                self.pg.baseline.update(torch.mean(cum_discounted_reward))
                self.pg.now_epoch += 1

                self.optimizer.zero_grad()
                reinfore_loss.backward()
                if self.args.clip_gradient:
                    total_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip_gradient)
                self.optimizer.step()

                total_loss += reinfore_loss
                total_reward += torch.mean(reward)
                counter += 1
                bar.update(self.args.batch_size)
                bar.set_postfix(loss='%.4f' % reinfore_loss, reward='%.4f' % torch.mean(reward).item())
        return total_loss / counter, total_reward / counter

    def save_model(self, checkpoint_path='checkpoint.pth'):
        """Save the parameters of the model and the optimizer,"""
        argparse_dict = vars(self.args)
        with open(os.path.join(self.args.save_path, 'config.json'), 'w') as fjson:
            json.dump(argparse_dict, fjson)

        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()},
            os.path.join(self.args.save_path, checkpoint_path)
        )
