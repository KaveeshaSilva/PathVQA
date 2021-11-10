from logging import NullHandler
import re
from tarfile import NUL
from torch.utils.tensorboard import SummaryWriter
import os
import collections

import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
from torch.autograd import Variable


from param import args

from pretrain.qa_answer_table import load_lxmert_qa
from tasks.pvqa_model import PVQAModel
from tasks.pvqa_model_adv import PVQAAdvModel
from tasks.pvqa_model_adv import Discriminator
from tasks.pvqa_data import PVQADataset, PVQATorchDataset, PVQAEvaluator
baseUrl = 'drive/MyDrive/PathVQA'
checkpoint_dir = baseUrl+"/checkpoint_with_LXRT_1.pth"
load_dir = baseUrl+"/checkpoint"
temp_checkpoint_save_dir = baseUrl+"/checkpointtemp_LXRT.pth"

startFrom = 'B'  # M - middle ,   B - beginning

# default `log_dir` is "runs" - we'll be more specific here
writer = SummaryWriter(baseUrl+'runs/Pathvqa_experiment_1')

DataTuple = collections.namedtuple("DataTuple", 'dataset loader evaluator')
valid_bs = 256

cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor


def get_data_tuple(splits: str, bs: int, shuffle=False, drop_last=False) -> DataTuple:
    dset = PVQADataset(splits)
    tset = PVQATorchDataset(dset)
    evaluator = PVQAEvaluator(dset)
    data_loader = DataLoader(
        tset, batch_size=bs, shuffle=shuffle, num_workers=args.num_workers,
        drop_last=drop_last, pin_memory=True
    )
    return DataTuple(dataset=dset, loader=data_loader, evaluator=evaluator)


class PVQAAdv:
    def __init__(self):
        # datasets

        self.train_tuple = get_data_tuple(
            splits=args.train, bs=args.batch_size, shuffle=True, drop_last=True
        )
        if args.valid != "":
            self.valid_tuple = get_data_tuple(
                splits=args.valid, bs=valid_bs,
                shuffle=False, drop_last=False
            )
        else:
            self.valid_tuple = None

        # Model
        self.q_i_model = PVQAAdvModel(self.train_tuple.dataset.num_answers)
        self.q_a_model = PVQAAdvModel(self.train_tuple.dataset.num_answers)
        # self.q_a_model = self.q_a_full_model.lxrt_encoder
        # self.q_a_model.load_state_dict(self.newLoadModel()['model_lxrt'])
        self.q_a_model.lxrt_encoder = self.newLoadModel()['model_lxrt']
        # self.q_a_model = PVQAAdvModel(
        #     self.train_tuple.dataset.num_answers)  # load model
        self.discriminator = Discriminator()

        # Load pre-trained weights
        if args.load_lxmert is not None:
            print(args.load_lxmert)
            self.q_i_model.lxrt_encoder.load(args.load_lxmert)
            # self.q_a_model.lxrt_encoder.load(args.load_lxmert)

        # if args.load_lxmert_qa is not None:
        #     print(args.load_lxmert_qa)

        #     load_lxmert_qa(args.load_lxmert_qa, self.model,
        #                    label2ans=self.train_tuple.dataset.label2ans)
        # GPU options
        self.discriminator = self.discriminator.cuda()
        self.q_i_model = self.q_i_model.cuda()
        self.q_a_model = self.q_a_model.cuda()
        if args.multiGPU:
            self.q_i_model.lxrt_encoder.multi_gpu()
            # self.q_a_model.lxrt_encoder.multi_gpu()

        # Loss and Optimizer
        # self.bce_loss = nn.BCEWithLogitsLoss()
        # if 'bert' in args.optim:
        #     batch_per_epoch = len(self.train_tuple.loader)
        #     t_total = int(batch_per_epoch * args.epochs)
        #     print("BertAdam Total Iters: %d" % t_total)
        #     from lxrt.optimization import BertAdam
        #     self.optim = BertAdam(list(self.model.parameters()),
        #                           lr=args.lr,
        #                           warmup=0.1,
        #                           t_total=t_total)
        # else:
            # self.optim = args.optimizer(self.model.parameters(), args.lr)
        self.optimizer_G = torch.optim.Adam(
            self.q_i_model.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.optimizer_D = torch.optim.Adam(
            self.discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.adversarial_loss = torch.nn.BCELoss()

        # Output Directory
        self.output = args.output
        os.makedirs(self.output, exist_ok=True)

    def train(self, train_tuple, eval_tuple):
        dset, loader, evaluator = train_tuple
        iter_wrapper = (lambda x: tqdm(x, total=len(loader))
                        ) if args.tqdm else (lambda x: x)

        best_valid = 0.

        lastEpoch = -1
        running_loss_g = 0
        running_loss_d = 0
        # print('loss and epoch reset')
        # if(startFrom == "M"):
        #     print('loading from saved model..')
        #     checkpoint = self.getLastEpoch()
        #     lastEpoch = checkpoint['epoch']
        #     running_loss = checkpoint['loss']
        #     print("last epoch :" + str(lastEpoch))
        #     print('loss')
        #     print(running_loss)

        start_epoch = lastEpoch + 1  # new

        # new to add model graph to tensorboard
        # dataiter = enumerate(loader)
        # ques_id, feats, boxes, sent, target = dataiter.next()
        # writer.add_graph(self.model, (feats, boxes, sent))
        ####################################################
        print('start running epochs')
        for epoch in range(start_epoch, args.epochs):
            print("Start new epoch - epoch number : "+str(epoch))
            quesid2ans = {}
            for i, (ques_id, feats, boxes, sent, target) in iter_wrapper(enumerate(loader)):

                # Adversarial ground truths
                valid = Variable(Tensor(32, 1).fill_(1.0),  # 32 is the batch size
                                 requires_grad=False)
                fake = Variable(Tensor(32, 1).fill_(0.0),  # 32 is the batch size
                                requires_grad=False)

                self.q_i_model.train()

                # -----------------
                #  Train Generator
                # -----------------

                self.optimizer_G.zero_grad()

                feats, boxes, target = feats.cuda(), boxes.cuda(), target.cuda()  # target 32 size

                target_answers = []
                for target_label in (target.max(1)[1]).cpu().numpy():
                    target_ans = dset.label2ans[target_label]
                    target_answers.append(target_ans)

                q_i_embeeeding = self.q_i_model(
                    feats, boxes, sent, target_answers)
                q_a_embeeeding = self.q_a_model(
                    feats, boxes, sent, target_answers)  # answer from trained model

                assert q_i_embeeeding.dim() == q_a_embeeeding.dim()
                print("q_a_embeeeding.dim() :" + str(q_a_embeeeding.dim()))

                dis_output_q_i = self.discriminator(q_i_embeeeding)

                # Loss measures generator's ability to fool the discriminator
                g_loss = self.adversarial_loss(dis_output_q_i, valid)
                g_loss.backward()
                self.optimizer_G.step()

                # ---------------------
                #  Train Discriminator
                # ---------------------

                self.optimizer_D.zero_grad()
                dis_output_q_a = self.discriminator(q_a_embeeeding)

                # Measure discriminator's ability to classify real from generated samples
                real_loss = self.adversarial_loss(dis_output_q_a, valid)
                fake_loss = self.adversarial_loss(
                    self.discriminator(q_i_embeeeding.detach()), fake)
                d_loss = (real_loss + fake_loss) / 2

                # loss = self.adversarial_loss(q_i_embeeeding, target)

                # loss = loss * q_i_embeeeding.size(1)

                # loss.backward()

                # /////////////////////////////////////////// #new
                running_loss_g += g_loss.item()
                running_loss_d += d_loss.item()

                if i % 100 == 99:    # every 1000 mini-batches...

                    # ...log the running loss
                    writer.add_scalar('training g loss',
                                      running_loss_g / 100,
                                      epoch * len(loader) + i)
                    running_loss_g = 0

                    writer.add_scalar('training d loss',
                                      running_loss_d / 100,
                                      epoch * len(loader) + i)
                    running_loss_d = 0

                    # ...log the validation loss
                    # if self.valid_tuple is not None:
                    #     valid_score = self.evaluate(eval_tuple)
                    #     writer.add_scalar('validation score',
                    #                       valid_score,
                    #                       epoch * len(loader) + i)
                # //////////////////////////////////////////          continue from here

                nn.utils.clip_grad_norm_(self.model.parameters(), 5.)
                self.optim.step()

                # score, label = q_i_embeeeding.max(1)
                # for qid, l in zip(ques_id, label.cpu().numpy()):
                #     ans = dset.label2ans[l]
                #     quesid2ans[qid.item()] = ans

            # log_str = "\nEpoch- %d: Train %0.2f\n" % (
            #     epoch, evaluator.evaluate(quesid2ans) * 100.)

            # if self.valid_tuple is not None:  # Do Validation
            #     valid_score = self.evaluate(eval_tuple)
            #     if valid_score > best_valid:
            #         best_valid = valid_score
            #         print('model checkpoint saved  epoch:'+str(epoch))
            #         self.save("BEST")
            #         # self.newSave(epoch, running_loss_)

            #     log_str += "Epoch- %d: Valid %0.2f\n" % (epoch, valid_score * 100.) + \
            #                "Epoch- %d: Best %0.2f\n" % (
            #                    epoch, best_valid * 100.)

            # print(log_str, end='')

            # with open(self.output + "/log.log", 'a') as f:
            #     print('logged the epoch result')
            #     f.write(log_str)
            #     f.flush()

        writer.flush()  # new
        writer.close()

        self.save("LAST")

    def predict(self, eval_tuple: DataTuple, dump=None):
        """
        Predict the answers to questions in a data split.

        :param eval_tuple: The data tuple to be evaluated.
        :param dump: The path of saved file to dump results.
        :return: A dict of question_id to answer.
        """
        self.model.eval()
        dset, loader, evaluator = eval_tuple
        quesid2ans = {}

        for i, datum_tuple in enumerate(loader):
            # Avoid seeing ground truth
            ques_id, feats, boxes, sent, target = datum_tuple
            with torch.no_grad():
                feats, boxes = feats.cuda(), boxes.cuda()

                target_answers = []
                for target_label in (target.max(1)[1]).cpu().numpy():
                    target_ans = dset.label2ans[target_label]
                    target_answers.append(target_ans)

                logit = self.model(feats, boxes, sent, target_answers)
                score, label = logit.max(1)
                for qid, l in zip(ques_id, label.cpu().numpy()):
                    ans = dset.label2ans[l]
                    quesid2ans[qid.item()] = ans
        if dump is not None:
            evaluator.dump_result(quesid2ans, dump)
        return quesid2ans

    def evaluate(self, eval_tuple: DataTuple, dump=None):
        """Evaluate all data in data_tuple."""
        quesid2ans = self.predict(eval_tuple, dump)
        return eval_tuple.evaluator.evaluate(quesid2ans)

    @staticmethod
    def oracle_score(data_tuple):
        dset, loader, evaluator = data_tuple
        quesid2ans = {}
        for i, (ques_id, feats, boxes, sent, target) in enumerate(loader):
            _, label = target.max(1)
            for qid, l in zip(ques_id, label.cpu().numpy()):
                ans = dset.label2ans[l]
                quesid2ans[qid.item()] = ans
        return evaluator.evaluate(quesid2ans)

    def save(self, name):
        print(os.path.join(self.output, "%s.pth" % name))
        torch.save(self.model.state_dict(),
                   os.path.join(self.output, "%s.pth" % name))

    def load(self, path):
        print("Load model from %s" % path)
        state_dict = torch.load("%s.pth" % path)
        self.model.load_state_dict(state_dict)

    def newLoadModel(self):
        PATH = checkpoint_dir
        checkpoint = torch.load(PATH)
        return checkpoint
        # self.q_a_model.load_state_dict(checkpoint['model_state_dict'])
        # self.optim.load_state_dict(checkpoint['optimizer_state_dict'])

    def getLastEpoch(self):
        PATH = checkpoint_dir
        checkpoint = torch.load(PATH)
        return checkpoint

    def newSave(self, EPOCH, LOSS):
        # PATH = checkpoint_dir
        PATH = temp_checkpoint_save_dir
        torch.save({
            'epoch': EPOCH,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optim.state_dict(),
            'loss': LOSS,
        }, PATH)


if __name__ == '__main__':

    pvqa = PVQAAdv()

    # print('first')
    # if(startFrom == "M"):
    #     pvqa.newLoadModel()
    # # Load VQA model weights
    # # Note: It is different from loading LXMERT pre-trained weights.
    # if args.load is not None:
    #     # pvqa.load(args.load)
    #     pvqa.newLoadModel()
    #     print('second')
    # else:
    #     print('third')

    # Test or Train
    # if args.test is not None:
    #     args.fast = args.tiny = False  # Always loading all data in test
    #     if 'test' in args.test:
    #         result = pvqa.evaluate(
    #             get_data_tuple(args.test, bs=valid_bs,
    #                            shuffle=False, drop_last=False),
    #             dump=os.path.join(args.output, 'test_predict.json')
    #         )
    #         print(result)
    #         with open(args.output + "/log.log", 'a') as f:
    #             f.write('test result=' + str(result))
    #             f.flush()

    #     elif 'val' in args.test:
    #         # NOT USED
    #         # Since part of valididation data are used in pre-training/fine-tuning,
    #         # only validate on the minival set.
    #         result = pvqa.evaluate(
    #             get_data_tuple('test', bs=valid_bs,
    #                            shuffle=False, drop_last=False),
    #             dump=os.path.join(args.output, 'test_predict.json')
    #         )
    #         print(result)
    #         with open(args.output + "/log.log", 'a') as f:
    #             f.write('test result=' + str(result))
    #             f.flush()

    #     else:
    #         assert False, "No such test option for %s" % args.test
    if(1 == 0):
        print('NO')
    else:
        print('Splits in Train data:', pvqa.train_tuple.dataset.splits)
        if pvqa.valid_tuple is not None:
            print('Splits in Valid data:', pvqa.valid_tuple.dataset.splits)
            print("Valid Oracle: %0.2f" %
                  (pvqa.oracle_score(pvqa.valid_tuple) * 100))
        else:
            print("DO NOT USE VALIDATION")
        pvqa.train(pvqa.train_tuple, pvqa.valid_tuple)
